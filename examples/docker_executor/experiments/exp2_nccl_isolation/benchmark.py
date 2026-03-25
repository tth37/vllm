#!/usr/bin/env python3
"""
GPU Communication Benchmark
Profiles performance of distributed communication operations.
"""

import os
import time
import json
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple
from datetime import datetime


class CommBenchmark:
    """Benchmark for distributed communication operations."""
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.results = {}
        
    def warmup(self, iterations: int = 10):
        """Warmup to stabilize GPU clocks."""
        tensor = torch.randn(1024, device=self.device)
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
    def benchmark_all_reduce(self, sizes: List[int], iterations: int = 100) -> Dict:
        """Benchmark all-reduce operation with different tensor sizes."""
        results = {}
        
        for size in sizes:
            # Create tensor
            tensor = torch.randn(size, device=self.device, dtype=torch.float32)
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            # Warmup
            for _ in range(10):
                dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(iterations):
                tensor_copy = tensor.clone()
                start.record()
                dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time_ms = sum(times) / len(times)
            std_time_ms = torch.tensor(times).std().item()
            bandwidth_gbps = (size_mb / 1024) / (avg_time_ms / 1000)
            
            results[f"{size_mb:.2f}MB"] = {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "bandwidth_gbps": bandwidth_gbps,
                "size_elements": size,
            }
            
            if self.rank == 0:
                print(f"  All-Reduce {size_mb:8.2f} MB: {avg_time_ms:7.3f} ± {std_time_ms:6.3f} ms, "
                      f"{bandwidth_gbps:6.2f} GB/s")
        
        return results
    
    def benchmark_all_gather(self, sizes: List[int], iterations: int = 100) -> Dict:
        """Benchmark all-gather operation."""
        results = {}
        
        for size in sizes:
            # Input tensor (each rank has its own)
            tensor = torch.randn(size, device=self.device, dtype=torch.float32)
            # Output tensor (concatenated from all ranks)
            output = torch.empty(size * self.world_size, device=self.device, dtype=torch.float32)
            
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            total_size_mb = size_mb * self.world_size
            
            # Warmup
            for _ in range(10):
                tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, tensor)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(iterations):
                tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
                start.record()
                dist.all_gather(tensor_list, tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time_ms = sum(times) / len(times)
            std_time_ms = torch.tensor(times).std().item()
            bandwidth_gbps = (total_size_mb / 1024) / (avg_time_ms / 1000)
            
            results[f"{size_mb:.2f}MB"] = {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "bandwidth_gbps": bandwidth_gbps,
                "size_elements": size,
                "total_size_mb": total_size_mb,
            }
            
            if self.rank == 0:
                print(f"  All-Gather {size_mb:8.2f} MB: {avg_time_ms:7.3f} ± {std_time_ms:6.3f} ms, "
                      f"{bandwidth_gbps:6.2f} GB/s")
        
        return results
    
    def benchmark_reduce_scatter(self, sizes: List[int], iterations: int = 100) -> Dict:
        """Benchmark reduce-scatter operation."""
        results = {}
        
        for size in sizes:
            # Input must be divisible by world_size
            total_size = (size // self.world_size) * self.world_size
            per_rank_size = total_size // self.world_size
            
            tensor = torch.randn(total_size, device=self.device, dtype=torch.float32)
            output = torch.empty(per_rank_size, device=self.device, dtype=torch.float32)
            
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            # Warmup
            for _ in range(10):
                output_list = [torch.empty(per_rank_size, device=self.device) 
                              for _ in range(self.world_size)]
                dist.reduce_scatter(output, output_list, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(iterations):
                output_list = [torch.empty(per_rank_size, device=self.device) 
                              for _ in range(self.world_size)]
                start.record()
                dist.reduce_scatter(output, output_list, op=dist.ReduceOp.SUM)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time_ms = sum(times) / len(times)
            std_time_ms = torch.tensor(times).std().item()
            bandwidth_gbps = (size_mb / 1024) / (avg_time_ms / 1000)
            
            results[f"{size_mb:.2f}MB"] = {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "bandwidth_gbps": bandwidth_gbps,
                "size_elements": total_size,
            }
            
            if self.rank == 0:
                print(f"  Reduce-Scatter {size_mb:8.2f} MB: {avg_time_ms:7.3f} ± {std_time_ms:6.3f} ms, "
                      f"{bandwidth_gbps:6.2f} GB/s")
        
        return results
    
    def benchmark_broadcast(self, sizes: List[int], iterations: int = 100) -> Dict:
        """Benchmark broadcast operation."""
        results = {}
        
        for size in sizes:
            tensor = torch.randn(size, device=self.device, dtype=torch.float32)
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            # Warmup
            for _ in range(10):
                dist.broadcast(tensor.clone(), src=0)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(iterations):
                tensor_copy = tensor.clone()
                start.record()
                dist.broadcast(tensor_copy, src=0)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time_ms = sum(times) / len(times)
            std_time_ms = torch.tensor(times).std().item()
            bandwidth_gbps = (size_mb / 1024) / (avg_time_ms / 1000)
            
            results[f"{size_mb:.2f}MB"] = {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "bandwidth_gbps": bandwidth_gbps,
                "size_elements": size,
            }
            
            if self.rank == 0:
                print(f"  Broadcast {size_mb:8.2f} MB: {avg_time_ms:7.3f} ± {std_time_ms:6.3f} ms, "
                      f"{bandwidth_gbps:6.2f} GB/s")
        
        return results
    
    def benchmark_p2p(self, sizes: List[int], iterations: int = 100) -> Dict:
        """Benchmark point-to-point send/recv."""
        results = {}
        
        # Only rank 0 and 1 participate
        if self.rank >= 2:
            dist.barrier()
            return results
        
        for size in sizes:
            tensor = torch.randn(size, device=self.device, dtype=torch.float32)
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            # Warmup
            for _ in range(10):
                if self.rank == 0:
                    dist.send(tensor, dst=1)
                elif self.rank == 1:
                    dist.recv(tensor, src=0)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(iterations):
                start.record()
                if self.rank == 0:
                    dist.send(tensor, dst=1)
                elif self.rank == 1:
                    dist.recv(tensor, src=0)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time_ms = sum(times) / len(times)
            std_time_ms = torch.tensor(times).std().item()
            bandwidth_gbps = (size_mb / 1024) / (avg_time_ms / 1000)
            
            results[f"{size_mb:.2f}MB"] = {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "bandwidth_gbps": bandwidth_gbps,
                "size_elements": size,
            }
            
            if self.rank == 0:
                print(f"  P2P Send/Recv {size_mb:8.2f} MB: {avg_time_ms:7.3f} ± {std_time_ms:6.3f} ms, "
                      f"{bandwidth_gbps:6.2f} GB/s")
        
        dist.barrier()
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks and collect results."""
        # Test sizes: 1KB to 100MB
        sizes = [
            256,           # 1 KB
            2560,          # 10 KB
            25600,         # 100 KB
            256000,        # 1 MB
            2560000,       # 10 MB
            25600000,      # 100 MB
        ]
        
        if self.rank == 0:
            print("\n" + "="*80)
            print("Running Communication Benchmarks")
            print("="*80)
            print(f"World Size: {self.world_size}")
            print(f"Device: {self.device}")
            print(f"Backend: {dist.get_backend()}")
            print("="*80 + "\n")
        
        # Warmup
        if self.rank == 0:
            print("Warming up...")
        self.warmup()
        dist.barrier()
        
        # All-Reduce
        if self.rank == 0:
            print("\nBenchmarking All-Reduce:")
        self.results['all_reduce'] = self.benchmark_all_reduce(sizes)
        dist.barrier()
        
        # All-Gather
        if self.rank == 0:
            print("\nBenchmarking All-Gather:")
        self.results['all_gather'] = self.benchmark_all_gather(sizes)
        dist.barrier()
        
        # Reduce-Scatter
        if self.rank == 0:
            print("\nBenchmarking Reduce-Scatter:")
        self.results['reduce_scatter'] = self.benchmark_reduce_scatter(sizes)
        dist.barrier()
        
        # Broadcast
        if self.rank == 0:
            print("\nBenchmarking Broadcast:")
        self.results['broadcast'] = self.benchmark_broadcast(sizes)
        dist.barrier()
        
        # P2P
        if self.rank == 0:
            print("\nBenchmarking Point-to-Point (Rank 0 ↔ Rank 1):")
        self.results['p2p'] = self.benchmark_p2p(sizes)
        dist.barrier()
        
        return self.results


def worker(rank: int, world_size: int, output_file: str):
    """Worker process that runs benchmarks."""
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Get system info
    if rank == 0:
        print(f"\nSystem Information:")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  NCCL Version: {torch.cuda.nccl.version()}")
    
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
            'deployment': 'host',
            'results': results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
    
    # Cleanup
    dist.destroy_process_group()


def smoke_test_worker(rank: int, world_size: int):
    """Quick smoke test worker - minimal NCCL operation for verification."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'='*60}")
        print("NCCL Smoke Test - Quick Communication Check")
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


if __name__ == '__main__':
    import torch.multiprocessing as mp
    import argparse

    parser = argparse.ArgumentParser(description='GPU Communication Benchmark')
    parser.add_argument('--smoke', action='store_true',
                        help='Run quick smoke test instead of full benchmark')
    args = parser.parse_args()
    
    world_size = int(os.environ.get('WORLD_SIZE', 2))
    output_file = os.environ.get('OUTPUT_FILE', 'benchmark_results.json')
    
    # Set environment
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    
    mp.set_start_method('spawn', force=True)
    if args.smoke:
        mp.spawn(smoke_test_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        mp.spawn(worker, args=(world_size, output_file), nprocs=world_size, join=True)
