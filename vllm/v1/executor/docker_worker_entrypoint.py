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
import signal
import threading
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

    try:
        vllm_config = deserialize_vllm_config(config_b64)

        init_distributed_for_worker(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed_init_method=distributed_init_method,
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        )

        scheduler_handle = deserialize_scheduler_handle(handle_b64)
        rpc_broadcast_mq = MessageQueue.create_from_handle(scheduler_handle, rank)
        logger.info("Connected to RPC broadcast message queue")

        # Each worker creates its own response MQ; the executor connects as reader
        worker_response_mq = MessageQueue(
            n_reader=1,
            n_local_reader=0,
            max_chunk_bytes=24 * 1024 * 1024,
            max_chunks=10,
            connect_ip=master_addr,
        )
        logger.info("Worker %d created response MQ", rank)

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
        # NOTE: We do NOT call wait_until_ready() on worker_response_mq here.
        # Doing so would deadlock: this writer waits for the reader's
        # subscription while the executor (reader) waits for our READY signal.
        logger.info("Worker %d entering main loop", rank)

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
                if isinstance(method, str):
                    func = getattr(worker_wrapper, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), worker_wrapper)
                else:
                    raise TypeError(f"Unknown method type: {type(method)}")

                output = func(*args, **kwargs)

                if isinstance(output, AsyncModelRunnerOutput):
                    output = output.get_output()

                if output_rank is None or rank == output_rank:
                    worker_response_mq.enqueue((ResponseStatus.SUCCESS, output))

            except Exception as e:
                logger.exception("Worker %d encountered an error", rank)
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                error_msg = f"{e}\n{traceback.format_exc()}"
                if output_rank is None or rank == output_rank:
                    worker_response_mq.enqueue(
                        (ResponseStatus.FAILURE, error_msg)
                    )

    except SystemExit:
        logger.info("Worker %d received shutdown signal", rank)
        raise

    except Exception:
        logger.exception("Worker %d failed with error", rank)
        raise

    finally:
        logger.info("Worker %d shutting down", rank)
        if worker_wrapper is not None:
            worker_wrapper.shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        logger.info("Worker %d shutdown complete", rank)


if __name__ == "__main__":
    main()
