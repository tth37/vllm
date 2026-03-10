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
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Sequence

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.executor.abstract import Executor, FailureCallback

logger = init_logger(__name__)


@dataclass
class DockerWorkerHandle:
    """Handle for a Docker container worker."""
    container_name: str
    rank: int
    worker_response_mq: MessageQueue | None
    death_monitor_thread: threading.Thread | None = None


class DockerDistributedExecutor(Executor):
    """Docker-based distributed executor for TP workers.

    Spawns each worker in a separate Docker container instead of
    a separate process. Uses network-based communication for both
    control plane (ZMQ) and data plane (NCCL).

    Uses a shared volume for handle exchange between the executor (host)
    and worker containers.
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

        # Clean up any stale state from previous runs before initializing
        self._cleanup_stale_resources()

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        super().__init__(vllm_config)

    def _cleanup_stale_resources(self) -> None:
        """Clean up stale containers and shared directory from previous runs.

        This is called before initialization to ensure a clean state,
        especially important when restarting after a crash or unclean shutdown.
        """
        logger.info("Cleaning up stale Docker executor resources...")

        # Clean up shared volume directory
        shared_volume = self._DOCKER_SHARED_VOLUME
        if os.path.exists(shared_volume):
            try:
                shutil.rmtree(shared_volume)
                logger.info(f"Cleaned up shared volume: {shared_volume}")
            except Exception as e:
                logger.warning(f"Failed to clean up shared volume: {e}")

        # Clean up any leftover vllm-worker containers
        try:
            result = subprocess.run(
                ["docker", "ps", "-aq", "--filter", "name=vllm-worker-"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                for cid in container_ids:
                    if cid:
                        # Force remove the container (stop if running, then remove)
                        rm_result = subprocess.run(
                            ["docker", "rm", "-f", cid],
                            capture_output=True, timeout=10
                        )
                        if rm_result.returncode == 0:
                            logger.info(f"Removed stale container: {cid[:12]}")
                        else:
                            logger.warning(f"Failed to remove stale container {cid[:12]}")
        except Exception as e:
            logger.warning(f"Failed to clean up stale containers: {e}")

        logger.info("Stale resource cleanup complete")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        This ensures containers are properly cleaned up when the process
        receives SIGTERM (e.g., from systemd, Kubernetes) or SIGINT (Ctrl+C).
        """
        # Use a class-level flag to avoid registering multiple times
        if getattr(DockerDistributedExecutor, '_signal_handlers_registered', False):
            return
        DockerDistributedExecutor._signal_handlers_registered = True

        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            # The finalizer will call shutdown() which will clean up containers
            if self._finalizer is not None:
                self._finalizer()
            # Call original handler if it wasn't default
            if signum == signal.SIGTERM and original_sigterm not in (signal.SIG_DFL, signal.SIG_IGN):
                original_sigterm(signum, frame)
            elif signum == signal.SIGINT and original_sigint not in (signal.SIG_DFL, signal.SIG_IGN):
                original_sigint(signum, frame)
            # Exit cleanly
            import sys
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.debug("Registered signal handlers for graceful shutdown")

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
        # IMPORTANT: All containers must start simultaneously for NCCL initialization.
        # NCCL requires all ranks to participate in init_distributed_environment()
        # at roughly the same time. If we start them sequentially and wait for each
        # to fully initialize, the first worker will timeout waiting for others.
        success = False
        temp_handles: list[tuple[int, str]] = []  # (rank, container_name)
        try:
            # Phase 1: Launch all containers first (don't wait for handles yet)
            logger.info(f"Launching {self.world_size} worker containers...")
            for rank in range(self.world_size):
                local_rank = rank % self.parallel_config.local_world_size
                is_driver_worker = self._is_driver_worker(rank)
                container_name = self._launch_worker_container(
                    rank=rank,
                    local_rank=local_rank,
                    scheduler_output_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                )
                temp_handles.append((rank, container_name))
                logger.info(f"Launched container {container_name} for rank {rank}")

            # Phase 2: Wait for all workers to export their handles
            # This happens in parallel as all containers are now running NCCL init
            logger.info("All containers launched. Waiting for workers to export handles...")
            for rank, container_name in temp_handles:
                handle = self._wait_for_worker_handle(rank, container_name)
                self.container_handles.append(handle)
                logger.info(f"Worker {rank} handle collected")

            # Wait for message queues to be ready
            logger.info("Waiting for RPC broadcast MQ to be ready...")
            self.rpc_broadcast_mq.wait_until_ready()
            logger.info("RPC broadcast MQ is ready")

            # Collect response message queues from all workers
            self.response_mqs: Sequence[MessageQueue] = [
                handle.worker_response_mq for handle in self.container_handles
                if handle.worker_response_mq is not None
            ]

            # Note: We intentionally do NOT call wait_until_ready() on response_mqs.
            # This would create a circular wait with the worker:
            # - Reader (executor) waits for writer's READY signal
            # - Writer (worker) waits for reader subscription
            # - Both wait for each other = deadlock
            #
            # Instead, we trust that the connection is established when we
            # receive the handle file from the worker. Synchronization happens
            # naturally when we try to dequeue responses.
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

    def _launch_worker_container(
        self,
        rank: int,
        local_rank: int,
        scheduler_output_handle: Handle,
        is_driver_worker: bool,
    ) -> str:
        """Launch a Docker container for a worker.

        Returns the container name. The container is started but we don't wait
        for it to fully initialize yet. This allows all containers to start
        simultaneously for NCCL initialization.
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

        # Get total GPUs needed for CUDA_VISIBLE_DEVICES
        # All containers see all GPUs, but each worker only uses its assigned GPU
        total_gpus = self.parallel_config.tensor_parallel_size * self.parallel_config.pipeline_parallel_size
        all_gpus = ",".join(str(i) for i in range(total_gpus))

        # Build docker run command
        # Note: We intentionally don't use --rm so that failed containers can be
        # inspected with 'docker logs' for debugging purposes
        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--rm",  # Auto-remove container when it stops (safeguard)
            "--name", container_name,
            "--gpus", "all",  # All GPUs visible to all containers for NCCL
            "--network", "host",  # Use host network for NCCL socket communication
            "--ipc", "host",  # Share host IPC namespace for CUDA IPC/P2P memory
            "--pid", "host",  # Share host PID namespace for NCCL process visibility
            "--shm-size=8g",  # Increase shared memory for NCCL (default 64MB is too small)
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
            # ZMQ IPC socket path: must be on the shared volume so that
            # ZMQ IPC sockets created by one container are accessible from
            # other containers. Without this, in_the_same_node_as() detects
            # co-location (via --ipc host shared /dev/shm) and creates
            # local MessageQueues using ZMQ IPC, but the default IPC path
            # (/tmp) is container-local, causing wait_until_ready() deadlock.
            "-e", f"VLLM_RPC_BASE_PATH={shared_volume}",
            # GPU visibility: NVIDIA_VISIBLE_DEVICES controls driver-level GPU exposure
            # All containers must see ALL GPUs for NCCL topology discovery to detect NVLink
            "-e", f"NVIDIA_VISIBLE_DEVICES={all_gpus}",
            "-e", f"CUDA_VISIBLE_DEVICES={all_gpus}",  # All GPUs visible for NCCL
            "-e", f"LOCAL_RANK={local_rank}",  # Each worker uses its assigned GPU
            "-e", "HF_HOME=/root/.cache/huggingface",  # Use mounted HF cache
            # NCCL host ID: Force same hostHash across containers so NCCL treats
            # them as a single node (required for NVLink P2P/CUMEM transport)
            # "-e", "NCCL_HOSTID=vllm-docker-executor",
            # NCCL socket interface: let NCCL auto-detect, or exclude loopback
            # Using '^lo' tells NCCL to use any interface except loopback
            "-e", "NCCL_SOCKET_IFNAME=^lo",
            "-e", "NCCL_DEBUG=INFO",
            "-e", "NCCL_DEBUG_SUBSYS=INIT,GRAPH",
            "-e", "PYTHONUNBUFFERED=1",
        ]

        # Add HuggingFace cache mount to avoid re-downloading models
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            cmd.extend(["-v", f"{hf_cache}:/root/.cache/huggingface"])
            logger.debug(f"Mounted HuggingFace cache: {hf_cache}")

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

        logger.info(f"Launching worker {rank} container '{container_name}'")
        logger.debug(f"Docker command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Docker run failed - container was never created
            raise RuntimeError(
                f"Failed to start Docker container for worker {rank}: {result.stderr}\n"
                f"Docker command: {' '.join(cmd)}"
            )

        container_id = result.stdout.strip()
        logger.info(f"Worker {rank} container launched: {container_id[:12]}")

        # Quick check that container is still running (but don't wait for full init)
        time.sleep(1)
        check_result = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Status}}', container_name],
            capture_output=True, text=True
        )

        if check_result.returncode != 0:
            raise RuntimeError(
                f"Container {container_name} (ID: {container_id[:12]}) "
                f"failed immediately after start. "
                f"Check 'docker logs {container_id}' for details."
            )

        container_status = check_result.stdout.strip()
        if container_status != "running":
            logs_result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True, text=True
            )
            logs = logs_result.stdout if logs_result.returncode == 0 else "No logs available"
            raise RuntimeError(
                f"Container {container_name} (ID: {container_id[:12]}) "
                f"exited immediately with status: {container_status}.\n"
                f"Container logs:\n{logs[-2000:]}"
            )

        return container_name

    def _wait_for_worker_handle(
        self,
        rank: int,
        container_name: str,
    ) -> DockerWorkerHandle:
        """Wait for a worker to export its response handle.

        This is called after all containers have been launched, allowing
        NCCL initialization to happen in parallel across all workers.
        """
        shared_volume = self._DOCKER_SHARED_VOLUME
        handle_file = f"{shared_volume}/worker_response_handle_{rank}.txt"

        logger.info(f"Waiting for worker {rank} to export handle to {handle_file}...")

        response_handle_b64 = None
        for i in range(180):  # Wait up to 180 seconds (model loading + NCCL init can take time)
            if os.path.exists(handle_file):
                with open(handle_file, 'r') as f:
                    response_handle_b64 = f.read().strip()
                logger.info(f"Got handle file for worker {rank} after {i+1} seconds")
                break

            # Check if container is still running
            if i % 5 == 0:
                check_result = subprocess.run(
                    ['docker', 'inspect', '-f', '{{.State.Status}}', container_name],
                    capture_output=True, text=True
                )
                if check_result.returncode != 0 or check_result.stdout.strip() != "running":
                    logs_result = subprocess.run(
                        ['docker', 'logs', container_name],
                        capture_output=True, text=True
                    )
                    logs = logs_result.stdout if logs_result.returncode == 0 else "No logs available"
                    raise RuntimeError(
                        f"Container {container_name} stopped while waiting for handle.\n"
                        f"Container status: {check_result.stdout.strip() if check_result.returncode == 0 else 'unknown'}\n"
                        f"Container logs:\n{logs[-3000:]}"
                    )

            if i % 10 == 0:
                logger.debug(f"Still waiting for handle file for worker {rank}... ({i}s elapsed)")
            time.sleep(1)

        if response_handle_b64 is None:
            # Worker didn't write handle file - check container status
            check_result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Status}}', container_name],
                capture_output=True, text=True
            )
            container_status = check_result.stdout.strip() if check_result.returncode == 0 else "unknown"
            logs_result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True, text=True
            )
            logs = logs_result.stdout if logs_result.returncode == 0 else "No logs available"
            raise RuntimeError(
                f"Timeout waiting for worker {rank} to export response handle to {handle_file}.\n"
                f"Container status: {container_status}\n"
                f"Container logs:\n{logs[-3000:]}"
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
        """Start a thread to monitor worker container health."""
        containers = self.container_handles
        self_ref = weakref.ref(self)

        def monitor_workers():
            """Monitor worker container health."""
            while True:
                _self = self_ref()
                if not _self or getattr(_self, "shutting_down", False):
                    return

                for handle in containers:
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
        """Check if all worker containers are running."""
        for handle in self.container_handles:
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
        """Stop and remove all worker containers.

        This method is called:
        1. On normal exit via weakref.finalize
        2. On signal reception (SIGTERM/SIGINT)
        3. On initialization failure
        4. Explicitly by the user
        """
        if getattr(self, "shutting_down", False):
            return
        self.shutting_down = True
        self.shutdown_event.set()

        logger.info("Shutting down Docker executor...")

        # Stop and remove all worker containers
        for handle in self.container_handles:
            container_name = handle.container_name
            logger.info(f"Cleaning up worker container {container_name}")

            # Step 1: Try graceful stop (docker stop sends SIGTERM)
            try:
                result = subprocess.run(
                    ["docker", "stop", "-t", "10", container_name],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    logger.info(f"Stopped container {container_name}")
                else:
                    # Container might already be stopped or removed
                    logger.debug(f"Stop returned {result.returncode} for {container_name}: {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout stopping container {container_name}")
            except Exception as e:
                logger.debug(f"Error stopping container {container_name}: {e}")

            # Step 2: Force kill if still running (SIGKILL)
            try:
                result = subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Killed container {container_name}")
            except Exception:
                # Ignore errors - container might already be stopped
                pass

            # Step 3: Remove the container
            try:
                result = subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Removed container {container_name}")
                else:
                    err = result.stderr.strip()
                    if "No such container" in err or "is not running" in err:
                        logger.debug(f"Container {container_name} already removed")
                    else:
                        logger.warning(f"Failed to remove container {container_name}: {err}")
            except Exception as e:
                logger.debug(f"Error removing container {container_name}: {e}")

        # Clean up shared volume
        shared_volume = self._DOCKER_SHARED_VOLUME
        if os.path.exists(shared_volume):
            try:
                shutil.rmtree(shared_volume)
                logger.info(f"Cleaned up shared volume: {shared_volume}")
            except Exception as e:
                logger.warning(f"Failed to clean up shared volume: {e}")

        # Clean up Python objects
        self.rpc_broadcast_mq = None
        self.response_mqs = []
        self.container_handles = []

        logger.info("Docker executor shutdown complete")
