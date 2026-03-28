#!/usr/bin/env python3
"""
P2P Upper Bound Benchmark — runs via torchrun with all GPUs visible.
Each rank uses cuda:LOCAL_RANK for true CUDA P2P transport.
"""
import os
import sys
import json
import torch
import torch.distributed as dist
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import CommBenchmark


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    smoke = os.environ.get("SMOKE_TEST", "").lower() == "true"

    if smoke:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"P2P Upper Bound Smoke Test (torchrun, {world_size} GPUs)")
            print(f"NCCL Version: {torch.cuda.nccl.version()}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"{'='*60}")

        tensor = torch.ones(1024, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        if rank == 0:
            expected = world_size
            actual = tensor[0].item()
            status = "PASSED" if abs(actual - expected) < 0.01 else "FAILED"
            print(f"✓ All-Reduce test {status}: {actual:.0f} == {expected}")
            print(f"{'='*60}\n")
    else:
        if rank == 0:
            print(f"\nP2P Upper Bound Benchmark")
            print(f"  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
            print(f"  GPU: {torch.cuda.get_device_name(device)}")
            print(f"  NCCL: {torch.cuda.nccl.version()}")
            print(f"  Ranks: {world_size}, Device: cuda:{local_rank}")

        benchmark = CommBenchmark(rank, world_size, device)
        results = benchmark.run_all_benchmarks()

        if rank == 0:
            output_file = os.environ.get(
                "OUTPUT_FILE", "/results/p2p_upperbound_results.json"
            )
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "world_size": world_size,
                "backend": dist.get_backend(),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(device),
                "nccl_version": str(torch.cuda.nccl.version()),
                "deployment": "torchrun_p2p",
                "results": results,
            }
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
