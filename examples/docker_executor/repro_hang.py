#!/usr/bin/env python3
"""Reproduce the NCCL SHM hang bug with Docker executor TP=2.

When --ipc host and --pid host are set, NCCL uses SHM/direct/direct transport.
The worker containers complete NCCL init but then hang during worker
initialization (init_worker/init_device/load_model), never writing their
response handle files.

Usage:
    python examples/docker_executor/repro_hang.py
    python examples/docker_executor/repro_hang.py --timeout 120
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import threading

if __name__ == "__main__":
    repo_root = os.path.join(os.path.dirname(__file__), "../..")
    sys.path.insert(0, os.path.abspath(repo_root))


def cleanup():
    """Clean up containers and shared volume."""
    shared_volume = "/tmp/vllm_docker_shared"
    if os.path.exists(shared_volume):
        shutil.rmtree(shared_volume, ignore_errors=True)

    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=vllm-worker-"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0 and result.stdout.strip():
        for cid in result.stdout.strip().split('\n'):
            if cid:
                subprocess.run(["docker", "rm", "-f", cid],
                               capture_output=True, timeout=10)
                print(f"  Removed container: {cid[:12]}")


def collect_docker_logs(container_name: str, tail: int = 80) -> str:
    """Collect logs from a docker container."""
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        combined = result.stdout
        if result.stderr:
            combined += "\n" + result.stderr
        return combined
    return f"(failed to get logs: {result.stderr})"


def check_container_status(container_name: str) -> str:
    """Check if a container is running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "not_found"


def main():
    parser = argparse.ArgumentParser(description="Reproduce Docker executor NCCL SHM hang")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Timeout in seconds to wait for engine init (default: 90)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to use (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallel size (default: 2)")
    args = parser.parse_args()

    print("=" * 70)
    print("REPRO: Docker Executor NCCL SHM Hang Bug")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"TP: {args.tp}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Clean up first
    print("[1/4] Cleaning up stale state...")
    cleanup()
    print()

    # Check prerequisites
    print("[2/4] Checking prerequisites...")
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"  GPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    if gpu_count < args.tp:
        print(f"  ERROR: Need {args.tp} GPUs, only have {gpu_count}")
        return 1
    print()

    # Try to create the LLM with a timeout
    print(f"[3/4] Creating LLM with docker executor (timeout={args.timeout}s)...")
    print(f"  distributed_executor_backend='docker', tensor_parallel_size={args.tp}")
    print()

    hang_detected = threading.Event()
    init_done = threading.Event()
    init_error = [None]

    def try_init():
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tp,
                distributed_executor_backend="docker",
                gpu_memory_utilization=0.5,
                max_model_len=512,
                disable_log_stats=True,
            )
            init_done.set()
            # If we somehow succeed, run a quick test
            outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.0))
            print(f"  Inference succeeded: {outputs[0].outputs[0].text!r}")
            del llm
        except Exception as e:
            init_error[0] = e
            init_done.set()

    init_thread = threading.Thread(target=try_init, daemon=True)
    init_thread.start()

    # Monitor progress while waiting
    start_time = time.time()
    shared_volume = "/tmp/vllm_docker_shared"
    last_status_time = 0

    while not init_done.is_set():
        elapsed = time.time() - start_time

        if elapsed > args.timeout:
            hang_detected.set()
            break

        # Print periodic status updates
        if elapsed - last_status_time >= 10:
            last_status_time = elapsed
            print(f"  [{elapsed:.0f}s] Still waiting for initialization...")

            # Check handle files
            if os.path.exists(shared_volume):
                files = os.listdir(shared_volume)
                handle_files = [f for f in files if f.startswith("worker_response_handle_")]
                print(f"    Handle files in shared volume: {handle_files if handle_files else '(none)'}")

            # Check container statuses
            for rank in range(args.tp):
                name = f"vllm-worker-{rank}"
                status = check_container_status(name)
                if status != "not_found":
                    print(f"    Container {name}: {status}")

        time.sleep(2)

    elapsed = time.time() - start_time

    if hang_detected.is_set():
        print()
        print("!" * 70)
        print(f"HANG DETECTED after {elapsed:.0f}s - engine initialization did not complete")
        print("!" * 70)
        print()

        # Check handle files
        print("[4/4] Collecting diagnostic information...")
        print()
        if os.path.exists(shared_volume):
            files = os.listdir(shared_volume)
            handle_files = [f for f in files if f.startswith("worker_response_handle_")]
            print(f"Handle files written: {handle_files if handle_files else 'NONE (workers never exported handles)'}")
        else:
            print(f"Shared volume {shared_volume} does not exist")
        print()

        # Collect docker logs from each container
        for rank in range(args.tp):
            container_name = f"vllm-worker-{rank}"
            status = check_container_status(container_name)
            print(f"--- Container {container_name} (status: {status}) ---")
            if status != "not_found":
                logs = collect_docker_logs(container_name, tail=60)
                print(logs)
            else:
                print("(container not found)")
            print()

        print("=" * 70)
        print("CONCLUSION: Bug reproduced successfully.")
        print()
        print("The workers complete NCCL init (SHM/direct/direct transport)")
        print("but hang during subsequent worker initialization, never writing")
        print("their response handle files to the shared volume.")
        print("=" * 70)

        # Clean up
        print()
        print("Cleaning up...")
        cleanup()
        return 1

    elif init_error[0]:
        print(f"\nInitialization failed with error: {init_error[0]}")
        cleanup()
        return 1

    else:
        print(f"\nInitialization completed in {elapsed:.1f}s (bug NOT reproduced)")
        cleanup()
        return 0


if __name__ == "__main__":
    sys.exit(main())
