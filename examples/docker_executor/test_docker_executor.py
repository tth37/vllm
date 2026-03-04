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
    python examples/docker_executor/test_docker_executor.py

Or from the repo root:
    python -m examples.docker_executor.test_docker_executor
"""
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


def test_basic_inference():
    """Test basic inference with docker executor using real containers."""
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

    # Test configuration
    model = "facebook/opt-125m"
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    print(f"Model: {model}")
    print(f"Tensor parallel size: 1")
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
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
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


if __name__ == "__main__":
    try:
        success = test_basic_inference()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cleanup()
        sys.exit(1)
