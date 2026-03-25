#!/usr/bin/env python3
"""
GPU Communication Benchmark - Docker Container Version
Single process per container (no multiprocessing spawn).
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from datetime import datetime

# Import the benchmark class from the main benchmark.py
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import CommBenchmark


def smoke_test(rank: int, world_size: int, master_addr: str, master_port: str):
    """Quick smoke test for Docker container - minimal NCCL operation."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'='*60}")
        print("NCCL Smoke Test - Docker Container")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"NCCL Version: {torch.cuda.nccl.version()}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"{'='*60}")

    tensor = torch.ones(1024, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world_size
    actual = tensor[0].item()

    if rank == 0:
        if abs(actual - expected) < 0.01:
            print(f"✓ All-Reduce test PASSED: {actual:.0f} == {expected}")
        else:
            print(f"✗ All-Reduce test FAILED: {actual:.0f} != {expected}")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


def main():
    """Main function for Docker container (single process)."""
    if os.environ.get('SMOKE_TEST', '').lower() == 'true':
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        smoke_test(rank, world_size, master_addr, master_port)
        return

    # Get environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    output_file = os.environ.get('OUTPUT_FILE', '/results/benchmark_results.json')
    
    # Initialize process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    
    # In Docker, NVIDIA_VISIBLE_DEVICES is set to 0 (mapped to host GPU)
    # So we always use cuda:0 inside the container
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Get system info
    if rank == 0:
        print(f"\nDocker Container Benchmark")
        print(f"System Information:")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  NCCL Version: {torch.cuda.nccl.version()}")
        print(f"  Container Rank: {rank}/{world_size}")
    
    # Run benchmarks
    benchmark = CommBenchmark(rank, world_size, device)
    results = benchmark.run_all_benchmarks()
    
    # Save results from rank 0
    if rank == 0:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'world_size': world_size,
            'backend': dist.get_backend(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'gpu_name': torch.cuda.get_device_name(device),
            'nccl_version': str(torch.cuda.nccl.version()),
            'deployment': 'docker',
            'results': results,
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
