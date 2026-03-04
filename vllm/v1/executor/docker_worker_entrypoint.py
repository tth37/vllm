#!/usr/bin/env python3
# SPDX-File-Identifier: Apache-2.0
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
import sys
import threading
import traceback
from functools import partial

import cloudpickle

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
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


def deserialize_scheduler_handle(handle_b64: str) -> MessageQueue:
    """Deserialize the scheduler output handle from base64."""
    handle_bytes = base64.b64decode(handle_b64)
    handle = pickle.loads(handle_bytes)
    return handle


def deserialize_vllm_config(config_b64: str) -> VllmConfig:
    """Deserialize VllmConfig from base64-encoded JSON."""
    config_json = base64.b64decode(config_b64).decode('utf-8')
    config_dict = json.loads(config_json)

    from vllm.config import (
        ModelConfig,
        CacheConfig,
        ParallelConfig,
        SchedulerConfig,
        DeviceConfig,
        LoadConfig,
    )

    # Reconstruct config objects
    model_config = ModelConfig(**config_dict["model_config"])

    # ParallelConfig doesn't accept world_size/local_world_size as kwargs
    # They are calculated properties, so filter them out
    parallel_config_dict = config_dict["parallel_config"].copy()
    parallel_config_dict.pop("world_size", None)
    parallel_config_dict.pop("local_world_size", None)
    parallel_config = ParallelConfig(**parallel_config_dict)

    cache_config = CacheConfig(**config_dict["cache_config"])
    scheduler_config = SchedulerConfig(**config_dict["scheduler_config"])

    # Create VllmConfig
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
    )

    return vllm_config


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
    # Get configuration from environment
    rank = int(get_env_var("VLLM_WORKER_RANK"))
    local_rank = int(get_env_var("VLLM_WORKER_LOCAL_RANK"))
    world_size = int(get_env_var("VLLM_WORLD_SIZE"))
    handle_b64 = get_env_var("VLLM_SCHEDULER_HANDLE")
    master_addr = get_env_var("VLLM_MASTER_ADDR")
    distributed_init_method = get_env_var("VLLM_DISTRIBUTED_INIT_METHOD")
    config_b64 = get_env_var("VLLM_CONFIG")
    is_driver_worker = get_env_var("VLLM_IS_DRIVER_WORKER", required=False) == "true"

    logger.info(
        f"Starting Docker worker: rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, is_driver={is_driver_worker}"
    )

    # Signal handler for graceful shutdown
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested.set()
        raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker_wrapper = None
    rpc_broadcast_mq = None
    worker_response_mq = None

    try:
        # Deserialize config first to get parallel sizes
        vllm_config = deserialize_vllm_config(config_b64)

        # Initialize distributed environment for NCCL
        init_distributed_for_worker(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed_init_method=distributed_init_method,
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        )

        # Deserialize scheduler output handle and create MessageQueue
        scheduler_handle = deserialize_scheduler_handle(handle_b64)
        rpc_broadcast_mq = MessageQueue.create_from_handle(scheduler_handle, rank)
        logger.info("Connected to RPC broadcast message queue")

        # Create or connect to response MessageQueue for sending results back
        response_handle_b64 = get_env_var("VLLM_RESPONSE_MQ_HANDLE", required=False)
        if response_handle_b64:
            # Connect to executor's response MQ
            response_handle_bytes = base64.b64decode(response_handle_b64)
            response_handle = pickle.loads(response_handle_bytes)
            worker_response_mq = MessageQueue.create_from_handle(response_handle, -1)
            logger.info(f"Connected to executor response message queue: {response_handle.remote_subscribe_addr}")
        else:
            # Create our own response MQ (for backward compatibility)
            logger.info(f"Worker {rank} creating response MQ with n_reader=1...")
            worker_response_mq = MessageQueue(
                n_reader=1,  # Executor reads
                n_local_reader=0,  # Remote only
                max_chunk_bytes=24 * 1024 * 1024,  # 24MB default
                max_chunks=10,
                connect_ip=master_addr,
            )
            response_handle = worker_response_mq.export_handle()
            logger.info(f"Worker {rank} created response MQ: {response_handle.remote_subscribe_addr}")

        # Deserialize VllmConfig
        vllm_config = deserialize_vllm_config(config_b64)

        # Create WorkerWrapperBase
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)

        # Prepare kwargs for worker initialization
        all_kwargs = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": None,  # Not used in Docker mode
        }

        # Initialize worker
        wrapper.init_worker(all_kwargs)
        worker_wrapper = wrapper

        # Initialize device
        wrapper.init_device()

        # Load model
        wrapper.load_model()

        logger.info(f"Worker {rank} initialized and ready")

        # Export response handle and write to file for executor to pick up
        response_handle = worker_response_mq.export_handle()
        handle_bytes = pickle.dumps(response_handle)
        handle_b64 = base64.b64encode(handle_bytes).decode('utf-8')

        # Write handle to file that executor can read
        handle_file = f"/tmp/vllm_worker_response_handle_{rank}.txt"
        with open(handle_file, 'w') as f:
            f.write(handle_b64)
        logger.info(f"Exported response handle to {handle_file}")

        # Wait for message queues to be ready
        logger.info(f"Worker {rank} waiting for RPC broadcast MQ to be ready...")
        rpc_broadcast_mq.wait_until_ready()
        logger.info(f"Worker {rank} RPC broadcast MQ is ready")

        # TODO: We intentionally do NOT call wait_until_ready() on worker_response_mq.
        # Calling it would create a circular wait:
        # - Writer (worker) waits for reader subscription in wait_until_ready()
        # - Reader (executor) waits for writer's READY signal in its wait_until_ready()
        # - Both wait for each other = deadlock
        #
        # The executor (reader) will call wait_until_ready() on the response MQ,
        # which properly waits for the writer's READY signal. When we enqueue
        # responses later, the synchronization happens naturally via acquire_write().
        #
        # For the real Docker backend, consider using explicit handshake:
        # 1. Executor connects and sends HTTP POST to worker's health endpoint
        # 2. Worker then knows executor is ready and can proceed
        # 3. Or use ZMQ's subscription forwarding to detect connection
        logger.info(f"Worker {rank} entering main loop (response MQ sync deferred)")

        # Main worker loop
        while not shutdown_requested.is_set():
            try:
                method, args, kwargs, output_rank = rpc_broadcast_mq.dequeue(
                    cancel=shutdown_requested, indefinite=True
                )
            except RuntimeError as e:
                if shutdown_requested.is_set():
                    break
                raise

            try:
                # Resolve method
                if isinstance(method, str):
                    func = getattr(worker_wrapper, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), worker_wrapper)
                else:
                    raise TypeError(f"Unknown method type: {type(method)}")

                # Execute method
                output = func(*args, **kwargs)

                # Handle AsyncModelRunnerOutput - need to call get_output() before pickling
                if isinstance(output, AsyncModelRunnerOutput):
                    output = output.get_output()

                # Send response if needed
                if output_rank is None or rank == output_rank:
                    worker_response_mq.enqueue((ResponseStatus.SUCCESS, output))

            except Exception as e:
                logger.exception(f"Worker {rank} encountered an error")
                # Add traceback note if available (Python 3.11+)
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())

                if output_rank is None or rank == output_rank:
                    worker_response_mq.enqueue((ResponseStatus.FAILURE, str(e)))

    except SystemExit:
        logger.info(f"Worker {rank} received shutdown signal")
        raise

    except Exception as e:
        logger.exception(f"Worker {rank} failed with error")
        raise

    finally:
        logger.info(f"Worker {rank} shutting down")

        # Clean up
        if worker_wrapper is not None:
            worker_wrapper.shutdown()

        destroy_model_parallel()
        destroy_distributed_environment()

        logger.info(f"Worker {rank} shutdown complete")


if __name__ == "__main__":
    main()
