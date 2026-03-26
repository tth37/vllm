#!/usr/bin/env python3
"""Dump NCCL topology XML for use with NCCL_TOPO_FILE.

Run this inside a container with all GPUs visible to generate the topology
XML that can then be provided to per-GPU containers.

Usage:
    docker run --rm --gpus all -v $(pwd):/workspace gpu-comm-benchmark:latest \
        python3 /workspace/dump_topo.py /workspace/results/nccl_topo.xml
"""
import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: dump_topo.py <output_xml_path>")
        sys.exit(1)

    output_path = sys.argv[1]

    # Use a minimal NCCL init to trigger topology detection and dump
    # NCCL_TOPO_DUMP_FILE triggers the dump during ncclCommInitRank
    env = os.environ.copy()
    env["NCCL_TOPO_DUMP_FILE"] = output_path
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "INIT"
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Only need one GPU to trigger topo detection

    # Simple PyTorch script that initializes NCCL (single rank)
    script = """
import torch
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

dist.init_process_group(backend='nccl', world_size=1, rank=0)
print(f"NCCL topology dumped to {os.environ.get('NCCL_TOPO_DUMP_FILE', 'unknown')}")
dist.destroy_process_group()
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=False
    )

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"Topology XML written: {output_path} ({size} bytes)")
    else:
        print(f"ERROR: Topology file not created at {output_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
