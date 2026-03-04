# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Docker-based distributed executor for vLLM.

This executor spawns each worker in a separate Docker container instead of
a separate OS process. It uses network-based communication (ZMQ) for both
the control plane (MessageQueue) and data plane (NCCL).
"""

import base64
import os
import pickle
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Sequence

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

# Set to True to use subprocess instead of Docker for faster debugging
_USE_SUBPROCESS = os.environ.get("VLLM_DOCKER_EXECUTOR_USE_SUBPROCESS", "0") == "1"


@dataclass
class DockerWorkerHandle:
    """Handle for a Docker container worker."""
    container_name: str
    rank: int
    worker_response_mq: MessageQueue | None
    death_monitor_thread: threading.Thread | None = None
    process: subprocess.Popen | None = None  # For subprocess mode


class DockerDistributedExecutor(Executor):
    """Docker-based distributed executor for TP workers.

    Spawns each worker in a separate Docker container instead of
    a separate process. Uses network-based communication for both
    control plane (ZMQ) and data plane (NCCL).

    In Docker mode, uses a shared volume for handle exchange.
    In subprocess mode (VLLM_DOCKER_EXECUTOR_USE_SUBPROCESS=1), uses /tmp files.
    """

    supports_pp: bool = True

    # Shared volume for handle exchange in Docker mode
    _DOCKER_SHARED_VOLUME = "/tmp/vllm_docker_shared"

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        self.container_handles: list[DockerWorkerHandle] = []
        self.host_ip = self._get_host_ip()
        self._finalizer: weakref.finalize | None = None
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: FailureCallback | None = None
        super().__init__(vllm_config)

    def _get_host_ip(self) -> str:
        """Get IP address accessible from Docker containers."""
        import socket
        # Try to get the default route IP address
        try:
            # Connect to a remote address to determine the local IP
            # used for external communication
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                s.connect(('10.254.254.254', 1))  # Arbitrary non-routable address
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()
            return ip
        except Exception:
            return '127.0.0.1'

    def _init_executor(self) -> None:
        """Initialize executor by starting Docker containers."""
        # Register shutdown at exit
        self._finalizer = weakref.finalize(self, self.shutdown)

        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        self.world_size = tp_size * pp_size * pcp_size

        # Get distributed init method for NCCL
        from vllm.utils.network_utils import get_distributed_init_method
        self.distributed_init_method = get_distributed_init_method(
            self.host_ip, get_open_port()
        )

        # Create network-based MessageQueue (no shared memory)
        # Force remote communication via ZMQ TCP
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(
            n_reader=self.world_size,
            n_local_reader=0,  # No local shared memory readers - all network
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=10,
            connect_ip=self.host_ip,
        )

        # Get scheduler output handle for passing to workers
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Start worker containers
        success = False
        try:
            for rank in range(self.world_size):
                local_rank = rank % self.parallel_config.local_world_size
                is_driver_worker = self._is_driver_worker(rank)
                handle = self._start_worker(
                    rank=rank,
                    local_rank=local_rank,
                    scheduler_output_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                )
                self.container_handles.append(handle)

            # Wait for message queues to be ready
            logger.info("Waiting for RPC broadcast MQ to be ready...")
            self.rpc_broadcast_mq.wait_until_ready()
            logger.info("RPC broadcast MQ is ready")

            # Collect response message queues from all workers
            self.response_mqs: Sequence[MessageQueue] = [
                handle.worker_response_mq for handle in self.container_handles
                if handle.worker_response_mq is not None
            ]

            # TODO: We intentionally do NOT call wait_until_ready() on response_mqs.
            # This would create a circular wait with the worker:
            # - Reader (executor) waits for writer's READY signal
            # - Writer (worker) waits for reader subscription
            # - Both wait for each other = deadlock
            #
            # Instead, we trust that the connection is established when we
            # receive the handle file from the worker. Synchronization happens
            # naturally when we try to dequeue responses.
            #
            # For the real Docker backend, implement proper handshake:
            # 1. Worker creates response MQ and exports handle
            # 2. Executor connects and sends confirmation to worker
            # 3. Both proceed knowing connection is established
            logger.info(f"Collected {len(self.response_mqs)} response MQs (sync deferred)")

            self.futures_queue = deque[tuple[Future, Any]]()

            # Start monitoring thread if enabled
            if self.monitor_workers:
                self.start_worker_monitor()

            success = True
        finally:
            if not success:
                self.shutdown()

        self.output_rank = self._get_output_rank()

    def _is_driver_worker(self, rank: int) -> bool:
        """Determine if this worker should be the driver worker."""
        return rank % self.parallel_config.tensor_parallel_size == 0

    def _get_output_rank(self) -> int:
        """Get the rank that should return the model output."""
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    def _get_worker_response_handle(self, rank: int) -> Handle:
        """Create a handle for worker response MQ."""
        # Workers will create their own response MQ and we need to connect to it
        port = get_open_port()
        return Handle(
            local_reader_ranks=[],
            buffer_handle=None,
            local_subscribe_addr=None,
            remote_subscribe_addr=f"tcp://{self.host_ip}:{port}",
            remote_addr_ipv6=False,
        )

    def _start_worker(
        self,
        rank: int,
        local_rank: int,
        scheduler_output_handle: Handle,
        is_driver_worker: bool,
    ) -> DockerWorkerHandle:
        """Start a worker - either in Docker container or subprocess for debugging."""
        if _USE_SUBPROCESS:
            return self._start_worker_subprocess(
                rank, local_rank, scheduler_output_handle, is_driver_worker
            )
        else:
            return self._start_worker_container(
                rank, local_rank, scheduler_output_handle, is_driver_worker
            )

    def _start_worker_subprocess(
        self,
        rank: int,
        local_rank: int,
        scheduler_output_handle: Handle,
        is_driver_worker: bool,
    ) -> DockerWorkerHandle:
        """Start a worker in a subprocess for faster debugging (no Docker rebuild needed)."""
        import json

        worker_name = f"vllm-worker-{rank}"

        # Serialize handle for passing to worker
        handle_bytes = pickle.dumps(scheduler_output_handle)
        handle_b64 = base64.b64encode(handle_bytes).decode('utf-8')

        # Serialize vllm_config for passing to worker
        config_dict = self._serialize_vllm_config()
        config_b64 = base64.b64encode(json.dumps(config_dict).encode()).decode('utf-8')

        # Prepare environment for subprocess
        env = os.environ.copy()
        env["VLLM_WORKER_RANK"] = str(rank)
        env["VLLM_WORKER_LOCAL_RANK"] = str(local_rank)
        env["VLLM_WORLD_SIZE"] = str(self.world_size)
        env["VLLM_SCHEDULER_HANDLE"] = handle_b64
        env["VLLM_MASTER_ADDR"] = self.host_ip
        env["VLLM_DISTRIBUTED_INIT_METHOD"] = self.distributed_init_method
        env["VLLM_CONFIG"] = config_b64
        env["VLLM_IS_DRIVER_WORKER"] = str(is_driver_worker).lower()
        env["NCCL_SOCKET_IFNAME"] = "eth0"
        env["NCCL_DEBUG"] = "WARN"
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(local_rank)  # Restrict GPU access

        # Add vllm-source to PYTHONPATH and set shared volume for handle exchange
        vllm_source_path = "/home/thd/repositories/vllm-dev/vllm-source"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{vllm_source_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = vllm_source_path
        env["VLLM_DOCKER_SHARED_VOLUME"] = self._DOCKER_SHARED_VOLUME

        logger.info(f"Starting worker {rank} in subprocess (debug mode) '{worker_name}'")

        # Start worker as subprocess - use module execution
        cmd = [
            sys.executable, "-m", "vllm.v1.executor.docker_worker_entrypoint"
        ]

        logger.debug(f"Subprocess command: {' '.join(cmd)}")
        logger.debug(f"Subprocess env: CUDA_VISIBLE_DEVICES={local_rank}, "
                    f"VLLM_WORKER_RANK={rank}, VLLM_WORLD_SIZE={self.world_size}")

        # Redirect stdout/stderr to files to avoid pipe buffer deadlock
        log_dir = f"{self._DOCKER_SHARED_VOLUME}/logs"
        os.makedirs(log_dir, exist_ok=True)
        stdout_file = open(f"{log_dir}/worker_{rank}_stdout.log", "w")
        stderr_file = open(f"{log_dir}/worker_{rank}_stderr.log", "w")

        # Start subprocess with file redirection
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )

        logger.info(f"Worker {rank} subprocess started: PID {process.pid}")

        # Wait a moment and check if process is still running
        time.sleep(2)

        if process.poll() is not None:
            # Process exited immediately - capture output from files
            stdout_file.close()
            stderr_file.close()
            with open(f"{self._DOCKER_SHARED_VOLUME}/logs/worker_{rank}_stdout.log", "r") as f:
                stdout = f.read()
            with open(f"{self._DOCKER_SHARED_VOLUME}/logs/worker_{rank}_stderr.log", "r") as f:
                stderr = f.read()
            raise RuntimeError(
                f"Worker {rank} subprocess (PID: {process.pid}) "
                f"exited immediately with code: {process.returncode}.\n"
                f"STDOUT:\n{stdout[-2000:]}\n"
                f"STDERR:\n{stderr[-2000:]}"
            )

        logger.info(f"Worker {rank} subprocess confirmed running (PID: {process.pid})")

        # Wait for worker to export its response handle
        # The worker writes the handle to a file after initialization
        handle_file = f"{self._DOCKER_SHARED_VOLUME}/worker_response_handle_{rank}.txt"
        response_handle_b64 = None
        for _ in range(60):  # Wait up to 60 seconds
            if os.path.exists(handle_file):
                with open(handle_file, 'r') as f:
                    response_handle_b64 = f.read().strip()
                break
            time.sleep(1)

        if response_handle_b64 is None:
            # Worker didn't write handle file - check if process died
            if process.poll() is not None:
                stdout_file.close()
                stderr_file.close()
                log_base = f"{self._DOCKER_SHARED_VOLUME}/logs"
                with open(f"{log_base}/worker_{rank}_stdout.log", "r") as f:
                    stdout = f.read()
                with open(f"{log_base}/worker_{rank}_stderr.log", "r") as f:
                    stderr = f.read()
                raise RuntimeError(
                    f"Worker {rank} subprocess exited before exporting handle.\n"
                    f"Exit code: {process.returncode}\n"
                    f"STDOUT:\n{stdout[-2000:]}\n"
                    f"STDERR:\n{stderr[-2000:]}"
                )
            raise RuntimeError(
                f"Timeout waiting for worker {rank} to export response handle"
            )

        # Close log files - they're now owned by the subprocess
        stdout_file.close()
        stderr_file.close()

        # Deserialize the handle
        response_handle_bytes = base64.b64decode(response_handle_b64)
        response_handle = pickle.loads(response_handle_bytes)
        logger.info(f"Got response handle from worker {rank}: {response_handle.remote_subscribe_addr}")

        # Create a connection to the worker's response MQ
        # The worker creates the MQ with n_reader=1 (expecting the executor to read)
        # We connect as rank 0 (the reader) since we dequeue responses
        logger.info(f"Connecting to worker {rank} response MQ as reader...")
        response_mq = MessageQueue.create_from_handle(response_handle, 0)
        logger.info(f"Connected to worker {rank} response MQ")

        return DockerWorkerHandle(
            container_name=worker_name,
            rank=rank,
            worker_response_mq=response_mq,
            process=process,
        )

    def _start_worker_container(
        self,
        rank: int,
        local_rank: int,
        scheduler_output_handle: Handle,
        is_driver_worker: bool,
    ) -> DockerWorkerHandle:
        """Start a worker in a Docker container.

        Uses a shared Docker volume for handle exchange between the executor
        (host) and worker (container). This allows bidirectional communication
        without requiring the executor to be in a container.
        """
        import json

        container_name = f"vllm-worker-{rank}"

        # Serialize handle for passing to container
        handle_bytes = pickle.dumps(scheduler_output_handle)
        handle_b64 = base64.b64encode(handle_bytes).decode('utf-8')

        # Serialize vllm_config for passing to container
        config_dict = self._serialize_vllm_config()
        config_b64 = base64.b64encode(json.dumps(config_dict).encode()).decode('utf-8')

        # Create shared volume path for this worker
        shared_volume = self._DOCKER_SHARED_VOLUME
        os.makedirs(shared_volume, exist_ok=True)

        # Build docker run command
        # Note: We intentionally don't use --rm so that failed containers can be
        # inspected with 'docker logs' for debugging purposes
        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            # "--rm",  # REMOVED: Keep containers for debugging failed starts
            "--name", container_name,
            "--gpus", f"device={local_rank}",
            "--network", "host",  # Use host network for simplicity
            "-v", f"{shared_volume}:{shared_volume}",  # Shared volume for handle exchange
            "-e", f"VLLM_WORKER_RANK={rank}",
            "-e", f"VLLM_WORKER_LOCAL_RANK={local_rank}",
            "-e", f"VLLM_WORLD_SIZE={self.world_size}",
            "-e", f"VLLM_SCHEDULER_HANDLE={handle_b64}",
            "-e", f"VLLM_MASTER_ADDR={self.host_ip}",
            "-e", f"VLLM_DISTRIBUTED_INIT_METHOD={self.distributed_init_method}",
            "-e", f"VLLM_CONFIG={config_b64}",
            "-e", f"VLLM_IS_DRIVER_WORKER={str(is_driver_worker).lower()}",
            "-e", f"VLLM_DOCKER_SHARED_VOLUME={shared_volume}",
            "-e", "NCCL_SOCKET_IFNAME=eth0",  # Adjust as needed
            "-e", "NCCL_DEBUG=WARN",
            "-e", "PYTHONUNBUFFERED=1",
        ]

        # Add model cache directory mount if specified
        if hasattr(self.load_config, 'model_cache_dir') and self.load_config.model_cache_dir:
            cmd.extend(["-v", f"{self.load_config.model_cache_dir}:/models"])

        # Add the image and command
        # Use vllm/vllm-docker-executor as the default image since it contains
        # the custom DockerDistributedExecutor files (docker_worker_entrypoint.py)
        image_name = envs.VLLM_DOCKER_IMAGE or "vllm/vllm-docker-executor:latest"
        cmd.extend([
            image_name,
            "python", "-m", "vllm.v1.executor.docker_worker_entrypoint"
        ])

        logger.info(f"Starting worker {rank} in Docker container '{container_name}'")
        logger.debug(f"Docker command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Docker run failed - container was never created
            raise RuntimeError(
                f"Failed to start Docker container for worker {rank}: {result.stderr}\n"
                f"Docker command: {' '.join(cmd)}"
            )

        container_id = result.stdout.strip()
        logger.info(f"Worker {rank} container started: {container_id[:12]}")

        # Wait a moment and check if container is still running
        time.sleep(2)
        check_result = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Status}}', container_name],
            capture_output=True, text=True
        )

        if check_result.returncode != 0:
            # Container doesn't exist or inspect failed
            raise RuntimeError(
                f"Container {container_name} (ID: {container_id[:12]}) "
                f"failed immediately after start. "
                f"Check 'docker logs {container_id}' for details."
            )

        container_status = check_result.stdout.strip()
        if container_status != "running":
            # Container exited immediately - get logs
            logs_result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True, text=True
            )
            logs = logs_result.stdout if logs_result.returncode == 0 else "No logs available"
            raise RuntimeError(
                f"Container {container_name} (ID: {container_id[:12]}) "
                f"exited immediately with status: {container_status}.\n"
                f"Container logs:\n{logs[-2000:]}"  # Last 2000 chars
            )

        logger.info(f"Worker {rank} container confirmed running")

        # Wait for worker to export its response handle via shared volume
        handle_file = f"{shared_volume}/worker_response_handle_{rank}.txt"
        response_handle_b64 = None
        for _ in range(60):  # Wait up to 60 seconds
            if os.path.exists(handle_file):
                with open(handle_file, 'r') as f:
                    response_handle_b64 = f.read().strip()
                break
            time.sleep(1)

        if response_handle_b64 is None:
            # Worker didn't write handle file - check container status
            check_result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Status}}', container_name],
                capture_output=True, text=True
            )
            container_status = check_result.stdout.strip() if check_result.returncode == 0 else "unknown"
            raise RuntimeError(
                f"Timeout waiting for worker {rank} to export response handle.\n"
                f"Container status: {container_status}\n"
                f"Check 'docker logs {container_name}' for details."
            )

        # Deserialize the handle
        response_handle_bytes = base64.b64decode(response_handle_b64)
        response_handle = pickle.loads(response_handle_bytes)
        logger.info(f"Got response handle from worker {rank}: {response_handle.remote_subscribe_addr}")

        # Create a connection to the worker's response MQ
        # The worker creates the MQ with n_reader=1 (expecting the executor to read)
        # We connect as rank 0 (the reader) since we dequeue responses
        logger.info(f"Connecting to worker {rank} response MQ as reader...")
        response_mq = MessageQueue.create_from_handle(response_handle, 0)
        logger.info(f"Connected to worker {rank} response MQ")

        return DockerWorkerHandle(
            container_name=container_name,
            rank=rank,
            worker_response_mq=response_mq,
        )

    def _serialize_vllm_config(self) -> dict:
        """Serialize VllmConfig to a dictionary for passing to workers."""
        from vllm.config import ModelConfig, CacheConfig, ParallelConfig

        config = self.vllm_config

        # Helper to convert dtype to string that ModelConfig accepts
        def dtype_to_str(dtype):
            if hasattr(dtype, 'name'):
                return dtype.name
            dtype_str = str(dtype)
            # Handle torch.dtype strings like 'torch.float16' -> 'float16'
            if dtype_str.startswith('torch.'):
                return dtype_str[6:]  # Remove 'torch.' prefix
            return dtype_str

        # Serialize the key configuration components
        config_dict = {
            "model_config": {
                "model": config.model_config.model,
                "tokenizer": config.model_config.tokenizer,
                "tokenizer_mode": config.model_config.tokenizer_mode,
                "trust_remote_code": config.model_config.trust_remote_code,
                "dtype": dtype_to_str(config.model_config.dtype),
                "seed": config.model_config.seed,
            },
            "parallel_config": {
                "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
                "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
                # Note: world_size and local_world_size are computed, not constructor args
            },
            "cache_config": {
                "block_size": config.cache_config.block_size,
                "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
            },
            "scheduler_config": {
                "max_num_seqs": config.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
                "max_model_len": config.model_config.max_model_len,
                "is_encoder_decoder": config.model_config.is_encoder_decoder,
                "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
            },
        }

        return config_dict

    def collective_rpc(
        self,
        method,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute RPC on all workers via MessageQueue."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called before initialization"
        )

        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        output_rank = unique_reply_rank

        # Serialize method if it's a callable
        import cloudpickle
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)

        # Broadcast to all workers
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(
                        timeout=dequeue_timeout, cancel=self.shutdown_event
                    )
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e

                if status != 0:  # ResponseStatus.SUCCESS = 0
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        if non_block:
            # Return a Future
            from vllm.v1.executor.multiproc_executor import FutureWrapper
            future = FutureWrapper(self.futures_queue)
            self.futures_queue.appendleft((future, get_response))
            return future

        # First drain any pending futures in the queue
        while self.futures_queue:
            future, get_fut_response = self.futures_queue.pop()
            future.wait_for_response(get_fut_response)

        return get_response()

    def execute_model(self, scheduler_output, non_block: bool = False):
        """Execute model on all workers."""
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def sample_tokens(self, grammar_output, non_block: bool = False):
        """Sample tokens from model output."""
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def start_worker_monitor(self) -> None:
        """Start a thread to monitor worker container/process health."""
        containers = self.container_handles
        self_ref = weakref.ref(self)

        def monitor_workers():
            """Monitor worker health (Docker or subprocess)."""
            while True:
                _self = self_ref()
                if not _self or getattr(_self, "shutting_down", False):
                    return

                for handle in containers:
                    if _USE_SUBPROCESS and handle.process is not None:
                        # Check subprocess health
                        if handle.process.poll() is not None:
                            logger.error(
                                f"Worker subprocess {handle.container_name} "
                                f"(PID: {handle.process.pid}) has exited unexpectedly "
                                f"with code {handle.process.returncode}"
                            )
                            _self.is_failed = True
                            _self.shutdown()
                            callback = _self.failure_callback
                            if callback is not None:
                                _self.failure_callback = None
                                callback()
                            return
                    else:
                        # Check Docker container health
                        result = subprocess.run(
                            ["docker", "inspect", "-f", "{{.State.Status}}", handle.container_name],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode != 0 or result.stdout.strip() not in ("running", "restarting"):
                            logger.error(
                                f"Worker container {handle.container_name} has stopped unexpectedly"
                            )
                            _self.is_failed = True
                            _self.shutdown()
                            callback = _self.failure_callback
                            if callback is not None:
                                _self.failure_callback = None
                                callback()
                            return

                time.sleep(1)  # Check every second

        threading.Thread(
            target=monitor_workers, daemon=True, name="DockerWorkerMonitor"
        ).start()

    def register_failure_callback(self, callback: FailureCallback):
        """Register a callback to be called if the executor fails."""
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def check_health(self) -> None:
        """Check if all workers are running (containers or subprocesses)."""
        for handle in self.container_handles:
            if _USE_SUBPROCESS and handle.process is not None:
                # Check subprocess health
                if handle.process.poll() is not None:
                    raise RuntimeError(
                        f"Worker subprocess {handle.container_name} (PID: {handle.process.pid}) "
                        f"has exited with code {handle.process.returncode}"
                    )
            else:
                # Check Docker container health
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Status}}", handle.container_name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Cannot check health of container {handle.container_name}: {result.stderr}"
                    )
                if result.stdout.strip() != "running":
                    raise RuntimeError(
                        f"Worker container {handle.container_name} is not running: {result.stdout.strip()}"
                    )

        # Also run health check on workers via RPC
        self.collective_rpc("check_health", timeout=10)

    def shutdown(self) -> None:
        """Stop all worker containers/processes."""
        if getattr(self, "shutting_down", False):
            return
        self.shutting_down = True
        self.shutdown_event.set()

        # Stop all workers (containers or subprocesses)
        for handle in self.container_handles:
            if _USE_SUBPROCESS and handle.process is not None:
                logger.info(f"Stopping worker subprocess {handle.container_name} (PID: {handle.process.pid})")
                # Try graceful shutdown first
                handle.process.terminate()
                try:
                    handle.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Worker {handle.container_name} did not terminate gracefully, killing...")
                    handle.process.kill()
                    handle.process.wait()
            else:
                logger.info(f"Stopping worker container {handle.container_name}")
                subprocess.run(
                    ["docker", "stop", "-t", "30", handle.container_name],
                    capture_output=True,
                )

        # Clean up
        self.rpc_broadcast_mq = None
        self.response_mqs = []
