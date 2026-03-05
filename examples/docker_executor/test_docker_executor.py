#!/usr/bin/env python3
"""Example/test script for Docker executor using actual Docker containers.

This script tests the DockerDistributedExecutor with real Docker containers.
Make sure to build the Docker image first:
    ./build-docker-executor.sh

Requirements:
- Docker installed and running
- NVIDIA Container Toolkit installed for GPU support
- vllm/vllm-docker-executor:latest image built

Usage:
    python examples/docker_executor/test_docker_executor.py [--tp TP] [--pp PP]

Options:
    --tp, --tensor-parallel-size    Tensor parallel size (default: 1)
    --pp, --pipeline-parallel-size  Pipeline parallel size (default: 1)
    --model                         Model name to use (default: facebook/opt-125m)

Or from the repo root:
    python -m examples.docker_executor.test_docker_executor --tp 2 --pp 1
"""
import argparse
import os
import sys
import time
import shutil
import subprocess

# Add vllm-source to path if running directly
if __name__ == "__main__":
    repo_root = os.path.join(os.path.dirname(__file__), "../..")
    sys.path.insert(0, os.path.abspath(repo_root))

import torch
from vllm import LLM, SamplingParams


def cleanup():
    """Clean up any stale files and stopped containers."""
    try:
        # Clean up shared volume directory
        shared_volume = "/tmp/vllm_docker_shared"
        if os.path.exists(shared_volume):
            shutil.rmtree(shared_volume, ignore_errors=True)

        # Clean up any leftover vllm-worker containers
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=vllm-worker-"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            for cid in container_ids:
                if cid:
                    subprocess.run(
                        ["docker", "rm", "-f", cid],
                        capture_output=True, timeout=10
                    )
                    print(f"Cleaned up container: {cid[:12]}")
    except Exception as e:
        print(f"Warning: Cleanup had issues: {e}")


def check_docker():
    """Check if Docker is available and image exists."""
    try:
        # Check Docker is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: Docker is not running or not accessible")
            return False

        # Check NVIDIA runtime is available
        result = subprocess.run(
            ["docker", "info", "-f", "{{.Runtimes.nvidia}}"],
            capture_output=True, text=True, timeout=5
        )
        if "nvidia" not in result.stdout.lower():
            print("WARNING: NVIDIA Container Toolkit may not be configured")
            print("GPU access in containers may not work")

        # Check image exists
        result = subprocess.run(
            ["docker", "images", "vllm/vllm-docker-executor:latest", "-q"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            print("ERROR: Docker image vllm/vllm-docker-executor:latest not found")
            print("Please build it first: ./build-docker-executor.sh")
            return False

        print("Docker check passed")
        return True
    except Exception as e:
        print(f"ERROR: Docker check failed: {e}")
        return False


def test_basic_inference(tensor_parallel_size: int = 1,
                         pipeline_parallel_size: int = 1,
                         model: str = "facebook/opt-125m"):
    """Test basic inference with docker executor using real containers.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of stages for pipeline parallelism
        model: Model name to use for testing
    """
    # Clean up any stale state first
    cleanup()

    print("=" * 60)
    print("Docker Executor Test (Real Containers)")
    print("=" * 60)
    print()

    # Check Docker
    if not check_docker():
        cleanup()
        return False
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        cleanup()
        return False

    gpu_count = torch.cuda.device_count()
    print(f"GPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Calculate required GPUs
    total_gpus_needed = tensor_parallel_size * pipeline_parallel_size
    if gpu_count < total_gpus_needed:
        print(
            f"ERROR: Not enough GPUs. Need {total_gpus_needed} (TP={tensor_parallel_size} x PP={pipeline_parallel_size}), have {gpu_count}"
        )
        cleanup()
        return False

    # Test configuration
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    print(f"Model: {model}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Pipeline parallel size: {pipeline_parallel_size}")
    print(f"Prompts: {prompts}")
    print()

    llm = None
    success = False
    try:
        # Create LLM
        print("Creating LLM with docker executor backend (real containers)...")
        start_time = time.time()

        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend="docker",
            gpu_memory_utilization=0.5,
            max_model_len=512,
            disable_log_stats=True,
        )
        init_time = time.time() - start_time
        print(f"LLM created successfully in {init_time:.1f}s")
        print()

        # Run inference
        print("Running inference...")
        start_time = time.time()

        outputs = llm.generate(prompts, sampling_params)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.1f}s")
        print()

        # Print results
        print("Results:")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"  Prompt {i+1}: {prompt!r}")
            print(f"  Generated: {generated_text!r}")
            print()

        success = True

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    finally:
        # Cleanup - always run this
        print("Cleaning up...")
        if llm is not None:
            del llm
        cleanup()

    if success:
        print()
        print("=" * 60)
        print("SUCCESS: All tests passed!")
        print("=" * 60)

    return success


def main():
    """Parse arguments and run the test."""
    parser = argparse.ArgumentParser(
        description="Test DockerDistributedExecutor for vLLM")
    parser.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Tensor parallel size (number of GPUs per pipeline stage) (default: 1)",
    )
    parser.add_argument(
        "--pp",
        "--pipeline-parallel-size",
        type=int,
        default=1,
        dest="pipeline_parallel_size",
        help="Pipeline parallel size (number of pipeline stages) (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name to use for testing (default: facebook/opt-125m)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.tensor_parallel_size < 1:
        print("ERROR: Tensor parallel size must be >= 1")
        sys.exit(1)
    if args.pipeline_parallel_size < 1:
        print("ERROR: Pipeline parallel size must be >= 1")
        sys.exit(1)

    try:
        success = test_basic_inference(
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            model=args.model,
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
