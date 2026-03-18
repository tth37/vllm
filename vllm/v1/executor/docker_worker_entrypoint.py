#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry point for Docker worker containers.

This module is executed inside Docker containers spawned by DockerDistributedExecutor.
It initializes the worker and connects to the executor via MessageQueue.
"""

import base64
import json
import os
import pickle
import queue
import signal
import threading
import time
import traceback
from functools import partial

import cloudpickle

from vllm.config import VllmConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
)
from vllm.logger import init_logger
from vllm.utils.rpc_profiling import (
    flush_rpc_profile,
    get_rpc_profiler,
    set_rpc_profile_metadata,
)
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class ResponseStatus:
    """Response status codes matching MultiprocExecutor."""
    SUCCESS = 0
    FAILURE = 1


def get_env_var(name: str, required: bool = True) -> str | None:
    """Get environment variable with optional requirement."""
    value = os.environ.get(name)
    if required and value is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def deserialize_scheduler_handle(handle_b64: str) -> Handle:
    """Deserialize the scheduler broadcast MQ handle from base64."""
    handle_bytes = base64.b64decode(handle_b64)
    return pickle.loads(handle_bytes)


def deserialize_vllm_config(config_b64: str) -> VllmConfig:
    """Deserialize VllmConfig from base64-encoded JSON."""
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
    )

    config_dict = json.loads(base64.b64decode(config_b64).decode("utf-8"))

    model_config = ModelConfig(**config_dict["model_config"])

    parallel_config_dict = config_dict["parallel_config"].copy()
    parallel_config_dict.pop("world_size", None)
    parallel_config_dict.pop("local_world_size", None)
    parallel_config = ParallelConfig(**parallel_config_dict)

    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(**config_dict["cache_config"]),
        parallel_config=parallel_config,
        scheduler_config=SchedulerConfig(**config_dict["scheduler_config"]),
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
    )


def init_distributed_for_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    distributed_init_method: str,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> None:
    """Initialize distributed environment for the worker."""
    from vllm.distributed import init_distributed_environment
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized

    # Initialize distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="nccl",
    )

    # Initialize model parallel with proper sizes
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
    )

    logger.info(
        f"Distributed initialized: rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, tp={tensor_parallel_size}, pp={pipeline_parallel_size}"
    )


def _make_mq_world_accessible(mq: MessageQueue) -> None:
    """Chmod SHM segment and ZMQ IPC socket so the host user can access them.

    Docker containers run as root, so SharedMemory segments (mode 0600) and
    IPC sockets are owned by UID 0.  The host executor runs as a non-root
    user and needs read/write access to both.
    """
    import stat

    if mq.buffer is not None:
        shm_path = f"/dev/shm/{mq.buffer.shared_memory.name}"
        target_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        os.chmod(shm_path, target_mode)
        logger.debug("chmod 0666 %s", shm_path)

    handle = mq.export_handle()
    if handle.local_subscribe_addr:
        ipc_path = handle.local_subscribe_addr.removeprefix("ipc://")
        os.chmod(ipc_path, 0o777)
        logger.debug("chmod 0777 %s", ipc_path)


def main():
    """Main entry point for Docker worker containers."""
    rank = int(get_env_var("VLLM_WORKER_RANK"))
    local_rank = int(get_env_var("VLLM_WORKER_LOCAL_RANK"))
    world_size = int(get_env_var("VLLM_WORLD_SIZE"))
    handle_b64 = get_env_var("VLLM_SCHEDULER_HANDLE")
    master_addr = get_env_var("VLLM_MASTER_ADDR")
    distributed_init_method = get_env_var("VLLM_DISTRIBUTED_INIT_METHOD")
    config_b64 = get_env_var("VLLM_CONFIG")
    is_driver_worker = get_env_var("VLLM_IS_DRIVER_WORKER", required=False) == "true"
    rpc_profiler = get_rpc_profiler()
    set_rpc_profile_metadata(
        role=f"docker_worker_rank{rank}",
        executor="docker",
        process_kind="worker_container",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_driver_worker=is_driver_worker,
    )

    logger.info(
        "Starting Docker worker: rank=%d, local_rank=%d, "
        "world_size=%d, is_driver=%s",
        rank, local_rank, world_size, is_driver_worker,
    )

    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        shutdown_requested.set()
        raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker_wrapper = None
    async_output_queue: queue.Queue[tuple[str, float, object] | None] | None = None
    async_output_copy_thread: threading.Thread | None = None

    try:
        vllm_config = deserialize_vllm_config(config_b64)
        async_output_copy_enabled = (
            vllm_config.scheduler_config.async_scheduling
            and os.environ.get("VLLM_DOCKER_ASYNC_OUTPUT_COPY", "1") != "0"
        )
        set_rpc_profile_metadata(
            async_output_copy_enabled=async_output_copy_enabled,
        )

        init_distributed_for_worker(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed_init_method=distributed_init_method,
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        )

        scheduler_handle = deserialize_scheduler_handle(handle_b64)
        rpc_broadcast_mq = MessageQueue.create_from_handle(
            scheduler_handle, rank
        ).set_profile_label(f"docker.worker.rank{rank}.rpc_broadcast")
        logger.info("Connected to RPC broadcast message queue")

        # Each worker creates its own response MQ; the executor connects as
        # the single reader.  With --ipc host the host shares /dev/shm, so
        # we use SHM (n_local_reader=1) for zero-copy message passing.
        # Python's SharedMemory creates segments with mode 0600 (owner-only),
        # and the container runs as root, so we chmod afterwards to let the
        # host user access the segment and ZMQ IPC socket.
        worker_response_mq = MessageQueue(
            n_reader=1,
            n_local_reader=1,
            max_chunk_bytes=24 * 1024 * 1024,
            max_chunks=10,
            connect_ip=master_addr,
        ).set_profile_label(f"docker.worker.rank{rank}.response")
        _make_mq_world_accessible(worker_response_mq)
        logger.info("Worker %d created response MQ (SHM)", rank)

        # Initialize worker via WorkerWrapperBase (matches multiproc pattern)
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": None,
        }
        wrapper.init_worker(all_kwargs)
        worker_wrapper = wrapper

        wrapper.init_device()
        wrapper.load_model()
        logger.info("Worker %d initialized and ready", rank)

        # Export response handle to shared volume for the executor to pick up
        response_handle = worker_response_mq.export_handle()
        handle_bytes = pickle.dumps(response_handle)
        shared_volume = os.environ.get("VLLM_DOCKER_SHARED_VOLUME", "/tmp")
        handle_file = f"{shared_volume}/worker_response_handle_{rank}.txt"
        with open(handle_file, "w") as f:
            f.write(base64.b64encode(handle_bytes).decode("utf-8"))
        logger.info("Exported response handle to %s", handle_file)

        rpc_broadcast_mq.wait_until_ready()
        worker_response_mq.wait_until_ready()
        logger.info("Worker %d entering main loop", rank)

        def enqueue_output(output: object, method_name: str) -> None:
            if isinstance(output, AsyncModelRunnerOutput):
                materialize_start = time.perf_counter()
                output = output.get_output()
                if rpc_profiler is not None:
                    rpc_profiler.record(
                        f"rpc.docker.worker.rank{rank}.{method_name}.output_materialize_s",
                        time.perf_counter() - materialize_start,
                    )

            if isinstance(output, Exception):
                result = (ResponseStatus.FAILURE, str(output))
            else:
                result = (ResponseStatus.SUCCESS, output)

            enqueue_start = time.perf_counter()
            worker_response_mq.enqueue(result)
            if rpc_profiler is not None:
                rpc_profiler.record(
                    f"rpc.docker.worker.rank{rank}.{method_name}.response_enqueue_s",
                    time.perf_counter() - enqueue_start,
                )

        use_async_scheduling = async_output_copy_enabled
        if use_async_scheduling:
            async_output_queue = queue.Queue()

            def async_output_busy_loop() -> None:
                assert async_output_queue is not None
                while True:
                    item = async_output_queue.get()
                    if item is None:
                        return
                    method_name, queued_at, output = item
                    if rpc_profiler is not None:
                        rpc_profiler.record(
                            f"rpc.docker.worker.rank{rank}.{method_name}.async_queue_wait_s",
                            time.perf_counter() - queued_at,
                        )
                    enqueue_output(output, method_name)

            async_output_copy_thread = threading.Thread(
                target=async_output_busy_loop,
                daemon=True,
                name=f"DockerWorkerAsyncOutputCopy-{rank}",
            )
            async_output_copy_thread.start()

        while not shutdown_requested.is_set():
            try:
                method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue(
                    cancel=shutdown_requested, indefinite=True
                )
            except RuntimeError:
                if shutdown_requested.is_set():
                    break
                raise

            try:
                method_name = method if isinstance(method, str) else "callable"
                dispatch_start = time.perf_counter()
                if isinstance(method, str):
                    func = getattr(worker_wrapper, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), worker_wrapper)
                else:
                    raise TypeError(f"Unknown method type: {type(method)}")
                if rpc_profiler is not None:
                    rpc_profiler.record(
                        f"rpc.docker.worker.rank{rank}.{method_name}.dispatch_s",
                        time.perf_counter() - dispatch_start,
                    )

                execute_start = time.perf_counter()
                output = func(*args, **kwargs)
                if rpc_profiler is not None:
                    rpc_profiler.record(
                        f"rpc.docker.worker.rank{rank}.{method_name}.execute_s",
                        time.perf_counter() - execute_start,
                    )

                if output_rank is None or rank == output_rank:
                    if use_async_scheduling:
                        assert async_output_queue is not None
                        async_output_queue.put(
                            (method_name, time.perf_counter(), output)
                        )
                    else:
                        enqueue_output(output, method_name)

                if rpc_profiler is not None:
                    rpc_profiler.record(
                        f"rpc.docker.worker.rank{rank}.{method_name}.call_count",
                        1.0,
                    )

            except Exception as e:
                logger.exception("Worker %d encountered an error", rank)
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                error_msg = RuntimeError(f"{e}\n{traceback.format_exc()}")
                if output_rank is None or rank == output_rank:
                    if use_async_scheduling:
                        assert async_output_queue is not None
                        async_output_queue.put(
                            (method_name, time.perf_counter(), error_msg)
                        )
                    else:
                        enqueue_output(error_msg, method_name)
                if rpc_profiler is not None:
                    rpc_profiler.record(
                        f"rpc.docker.worker.rank{rank}.{method_name}.error_count",
                        1.0,
                    )

    except SystemExit:
        logger.info("Worker %d received shutdown signal", rank)
        raise

    except Exception:
        logger.exception("Worker %d failed with error", rank)
        raise

    finally:
        logger.info("Worker %d shutting down", rank)
        if async_output_queue is not None:
            async_output_queue.put(None)
        if async_output_copy_thread is not None:
            async_output_copy_thread.join(timeout=5)
        flush_rpc_profile()
        if worker_wrapper is not None:
            worker_wrapper.shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        logger.info("Worker %d shutdown complete", rank)


if __name__ == "__main__":
    main()
