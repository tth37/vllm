# Exp2: NCCL Transport Under Per-GPU Container Isolation (NVLink)

## Problem Statement

In exp1, we showed that vLLM's DockerDistributedExecutor achieves bare-metal NCCL performance when containers share the host's IPC and network namespaces (all GPUs visible, `ipc: host`, `network_mode: host`). However, this provides **zero isolation** between GPU workers.

When we isolate containers (1 GPU per container, private IPC, private network), NCCL loses access to high-speed P2P transport over NVLink, falling back to SHM (host memory) or TCP sockets — causing **7-12x bandwidth degradation**.

**Goal:** Characterize the isolation penalty on NVLink-connected GPUs and explore paths to recover NVLink P2P performance under per-GPU container isolation.

## Testbed

| Property | Value |
|----------|-------|
| Machine | node192 (10.0.2.192) |
| GPUs | 2x NVIDIA A100-SXM4-40GB |
| GPU Topology | **NV12** (12x NVLink between GPU0 and GPU1) |
| NUMA | Both GPUs on NUMA node 0 |
| OS | Ubuntu 24.04 LTS |
| Docker | 27.3.1 with nvidia runtime |
| NCCL | 2.21.5 |
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |

### GPU Topology

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV12    SYS     SYS
GPU1    NV12     X      SYS     SYS
GPU2    SYS     SYS      X      PIX
GPU3    SYS     SYS     PIX      X
```

GPU0 and GPU1 are A100-SXM4 connected via 12x NVLink (~600 GB/s bidirectional theoretical).
This experiment uses only GPU0-GPU1.

## Configurations

Three Docker Compose configurations test the isolation spectrum:

| Config | File | GPU Visibility | IPC | Network | NCCL Transport |
|--------|------|---------------|-----|---------|----------------|
| **Baseline** | `compose.baseline.yml` | all GPUs | host | host | P2P/CUMEM/read (NVLink) |
| **SHM Isolation** | `compose.shm_isolation.yml` | all (NVML), 1 (CUDA) | private | bridge | SHM/direct/direct |
| **Naive Isolation** | `compose.p2p_isolation.yml` | all (NVML), 1 (CUDA) | private | bridge | NET/Socket (TCP) |

### Baseline (`compose.baseline.yml`)
- `network_mode: host`, `ipc: host`, `pid: host`
- `NVIDIA_VISIBLE_DEVICES=0,1` — both GPUs visible
- `CUDA_VISIBLE_DEVICES=0` or `1` — per-rank GPU selection
- No namespace isolation — CUDA IPC works natively, NCCL selects P2P

### SHM Isolation (`compose.shm_isolation.yml`)
- Bridge network, private IPC namespace
- `NVIDIA_VISIBLE_DEVICES=all` — NVML sees all GPUs for topology discovery
- `CUDA_VISIBLE_DEVICES=0` or `1` — per-container GPU
- `NCCL_P2P_DISABLE=1` — force SHM instead of failing P2P
- `NCCL_HOSTID=exp2-node192` — override hostname for same-node detection
- Shared `/dev/shm` (4 GB tmpfs) and `/tmp` (200 MB tmpfs) volumes

### Naive Isolation (`compose.p2p_isolation.yml`)
- Bridge network, private IPC namespace
- `NVIDIA_VISIBLE_DEVICES=all`, `CUDA_VISIBLE_DEVICES=0` or `1`
- No P2P disable, no shared `/dev/shm`
- NCCL detects `nNodes=2`, falls back to NET/Socket

## Results

### Bandwidth Comparison at 100 MB

| Operation | Baseline (P2P) | SHM Isolation | Naive (Socket) | SHM vs Baseline |
|-----------|----------------|---------------|----------------|-----------------|
| All-Reduce | 156.64 GB/s | 13.34 GB/s | 4.23 GB/s | **-91.5%** |
| All-Gather | 174.85 GB/s | 22.80 GB/s | 8.50 GB/s | **-87.0%** |
| Reduce-Scatter | 260.78 GB/s | 23.21 GB/s | 8.58 GB/s | **-91.1%** |
| Broadcast | 218.65 GB/s | 23.29 GB/s | 5.75 GB/s | **-89.3%** |
| P2P Send/Recv | 219.28 GB/s | 22.06 GB/s | 5.11 GB/s | **-89.9%** |

### Full Size Sweep — Baseline (P2P/CUMEM/read over NVLink)

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s |
| 10 KB | 0.38 GB/s | 0.50 GB/s | 0.43 GB/s | 0.44 GB/s | 0.44 GB/s |
| 100 KB | 3.87 GB/s | 4.86 GB/s | 4.11 GB/s | 4.17 GB/s | 3.90 GB/s |
| 1 MB | 19.32 GB/s | 32.56 GB/s | 32.34 GB/s | 28.30 GB/s | 35.26 GB/s |
| 10 MB | 97.38 GB/s | 119.43 GB/s | 136.60 GB/s | 144.00 GB/s | 148.19 GB/s |
| 100 MB | 156.64 GB/s | 174.85 GB/s | 260.78 GB/s | 218.65 GB/s | 219.28 GB/s |

### Full Size Sweep — SHM Isolation (SHM/direct/direct)

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s |
| 10 KB | 0.37 GB/s | 0.51 GB/s | 0.40 GB/s | 0.45 GB/s | 0.45 GB/s |
| 100 KB | 2.02 GB/s | 3.31 GB/s | 2.50 GB/s | 2.77 GB/s | 3.82 GB/s |
| 1 MB | 8.00 GB/s | 14.86 GB/s | 11.72 GB/s | 13.73 GB/s | 11.44 GB/s |
| 10 MB | 12.54 GB/s | 21.67 GB/s | 21.40 GB/s | 22.01 GB/s | 21.05 GB/s |
| 100 MB | 13.34 GB/s | 22.80 GB/s | 23.21 GB/s | 23.29 GB/s | 22.06 GB/s |

### Full Size Sweep — Naive Isolation (NET/Socket)

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.02 GB/s | 0.04 GB/s | 0.03 GB/s | 0.04 GB/s | 0.03 GB/s |
| 10 KB | 0.17 GB/s | 0.33 GB/s | 0.25 GB/s | 0.37 GB/s | 0.34 GB/s |
| 100 KB | 0.73 GB/s | 1.67 GB/s | 1.13 GB/s | 2.83 GB/s | 2.32 GB/s |
| 1 MB | 2.68 GB/s | 5.51 GB/s | 4.42 GB/s | 6.67 GB/s | 4.49 GB/s |
| 10 MB | 3.95 GB/s | 7.77 GB/s | 6.78 GB/s | 5.80 GB/s | 5.09 GB/s |
| 100 MB | 4.23 GB/s | 8.50 GB/s | 8.58 GB/s | 5.75 GB/s | 5.11 GB/s |

## Analysis

### Why P2P Breaks with Per-GPU Containers

NCCL's P2P transport uses **CUDA IPC** (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) to share GPU memory between processes. For P2P to work, the transport selection code (`p2pCanConnect`) checks:

1. Both ranks are on the same node (`sameNode`)
2. `cudaDeviceCanAccessPeer(myGpu, peerGpu)` returns true

With per-GPU container isolation (`CUDA_VISIBLE_DEVICES=0` in one container, `CUDA_VISIBLE_DEVICES=1` in the other), each container only sees **one GPU** as `cuda:0`. The peer GPU is not visible to CUDA, so `cudaDeviceCanAccessPeer` fails — there is no peer to check.

Even though `NVIDIA_VISIBLE_DEVICES=all` makes both GPUs visible to NVML (for topology discovery), CUDA runtime isolation prevents the P2P data path.

```
+-----------------+     +-----------------+
|  Container 0    |     |  Container 1    |
|  cuda:0 = GPU0  |     |  cuda:0 = GPU1  |
|  (no peer GPU)  |     |  (no peer GPU)  |
+--------+--------+     +--------+--------+
         |                       |
         |   NVLink (unused!)    |
    GPU0 ===================== GPU1
         |                       |
    +----+---------------------------+----+
    |         Host Memory (DRAM)          |
    |     Shared /dev/shm (tmpfs)         |
    |    <- SHM transport goes here ->    |
    +-------------------------------------+
```

### Performance Summary

| Scenario | Transport | 100 MB Bandwidth | Relative |
|----------|-----------|------------------|----------|
| Baseline (no isolation) | P2P/CUMEM/read (NVLink) | 157-261 GB/s | 100% |
| Best isolation (SHM) | SHM/direct/direct (host memory) | 13-23 GB/s | **~9-12%** |
| Naive isolation | NET/Socket (TCP) | 4-9 GB/s | **~2-5%** |

The SHM workaround recovers from ~3% (Socket) to ~10% (SHM) of baseline bandwidth. This still represents a **~7-12x degradation** compared to NVLink P2P — a fundamental architectural limitation, not a configuration problem.

### Potential Recovery Paths

To preserve NVLink performance with per-GPU container isolation:

1. **Cross-namespace CUDA IPC via `pidfd_getfd()`** (kernel 5.6+) — pass CUDA IPC file descriptors across PID namespaces. Requires NCCL P2P transport patch.
2. **CUDA VMM API** (`cuMemExportToShareableHandle` with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`) — export GPU memory as POSIX FDs that can be shared via Unix domain sockets on a shared volume.
3. **NCCL NVLS** (NVLink SHARP) — hardware-accelerated multicast over NVLink, but requires multi-GPU visibility per container.

The `compose.patched_nccl.yml`, `build_nccl.sh`, `Dockerfile.nccl-dev`, and `patches/` directory provide the scaffold for testing custom NCCL builds with these approaches.

## Quick Start

```bash
cd examples/docker_executor/experiments/exp2_nccl_isolation

# Build benchmark image
docker build -t gpu-comm-benchmark:latest -f Dockerfile .

# Run all 3 configs
./run_all.sh

# Run a single config
docker compose -f compose.baseline.yml up
docker compose -f compose.shm_isolation.yml up
docker compose -f compose.p2p_isolation.yml up

# Analyze results
python3 analyze_exp2.py
```

## Environment Variables Reference

### NCCL Transport Control
| Variable | Values | Effect |
|----------|--------|--------|
| `NCCL_P2P_DISABLE` | `0`/`1` | Disable P2P (CUDA IPC) transport |
| `NCCL_SHM_DISABLE` | `0`/`1` | Disable SHM transport |
| `NCCL_NET_DISABLE_INTRA` | `0`/`1` | Prefer SHM over NET for intra-node |
| `NCCL_HOSTID` | string | Override hostname-based hostHash (force single-node) |

### NCCL Debug
| Variable | Values | Effect |
|----------|--------|--------|
| `NCCL_DEBUG` | `WARN`/`INFO`/`TRACE` | Debug verbosity |
| `NCCL_DEBUG_SUBSYS` | `INIT,P2P,SHM,NET,...` | Filter debug subsystems |

### GPU Visibility (Docker)
| Variable | Scope | Effect |
|----------|-------|--------|
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA runtime | GPUs visible to driver (topology discovery) |
| `CUDA_VISIBLE_DEVICES` | CUDA runtime | GPU used by application |

## NCCL Transport Detection

```bash
# P2P (best -- NVLink):
grep "via P2P" results/*_log.txt

# SHM (fallback -- host memory):
grep "via SHM" results/*_log.txt

# NET/Socket (worst -- TCP):
grep "via NET" results/*_log.txt
```

## Directory Structure

```
exp2_nccl_isolation/
├── README.md                     # This file
├── NCCL_SHM_MECHANISM.md        # Deep-dive: NCCL SHM transport internals
├── Dockerfile                    # Benchmark Docker image
├── benchmark.py                  # Core NCCL benchmark (all-reduce, all-gather, etc.)
├── benchmark_docker.py           # Container entry point
├── benchmark_p2p.py              # P2P-specific benchmark
├── diagnose_nccl.py              # NCCL namespace diagnostic tool
├── dump_topo.py                  # GPU topology dumper
├── analyze_exp2.py               # Multi-config results comparison
├── run_all.sh                    # Run all 3 configs sequentially
├── compose.baseline.yml          # No isolation (P2P reference)
├── compose.shm_isolation.yml     # Per-GPU isolation + SHM workaround
├── compose.p2p_isolation.yml     # Per-GPU isolation, naive (TCP fallback)
├── compose.patched_nccl.yml      # Custom NCCL build (future)
├── build_nccl.sh                 # Clone + patch + build NCCL
├── Dockerfile.nccl-dev           # NCCL build environment
├── patches/                      # NCCL patch files (future)
├── results/                      # Benchmark JSON results
│   ├── node192_baseline.json
│   ├── node192_isolated_shm.json
│   └── node192_isolated_p2p.json
└── nccl/                         # NCCL source (git submodule)
```
