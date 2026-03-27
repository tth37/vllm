# Node192 2-GPU Experiment: NVLink P2P vs Container Isolation

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

GPU0 and GPU1 are A100-SXM4 connected via 12x NVLink (theoretical ~600 GB/s bidirectional).
GPU2 and GPU3 are on NUMA node 1 (GPU3 is A100-PCIe). This experiment uses only GPU0-GPU1.

## Key Question

On NVLink-connected GPUs, the baseline uses **P2P transport** (CUDA IPC over NVLink).
Can we achieve the same performance with per-GPU container isolation?

## Results Summary

### Configuration Comparison at 100MB

| Operation | Baseline (P2P) | Isolated (SHM) | Naive Isolation (Socket) | SHM vs Baseline |
|-----------|----------------|-----------------|--------------------------|------------------|
| All-Reduce | 156.64 GB/s | 13.34 GB/s | 4.23 GB/s | **-91.5%** |
| All-Gather | 174.85 GB/s | 22.80 GB/s | 8.50 GB/s | **-87.0%** |
| Reduce-Scatter | 260.78 GB/s | 23.21 GB/s | 8.58 GB/s | **-91.1%** |
| Broadcast | 218.65 GB/s | 23.29 GB/s | 5.75 GB/s | **-89.3%** |
| P2P Send/Recv | 219.28 GB/s | 22.06 GB/s | 5.11 GB/s | **-89.9%** |

### Interpretation

**This is a fundamentally different result from node191 (PCIe-only GPUs).**

On node191 (A100-PCIe, NODE topology), the baseline already used SHM transport because
there was no NVLink. Container isolation achieved identical performance because both
baseline and isolated used the same SHM data path.

On node192 (A100-SXM4, NV12 topology), the baseline uses **P2P/CUMEM/read** — a direct
GPU-to-GPU DMA path over NVLink via CUDA IPC. Per-GPU container isolation **breaks P2P**
because each container can only see one GPU. The best fallback is SHM, which routes
data through host memory (CPU DRAM), giving ~13-23 GB/s instead of ~157-261 GB/s.

## Detailed Results

### Baseline: No Isolation (P2P/CUMEM/read over NVLink)

Containers use `ipc: host`, `network_mode: host`, and all GPUs visible.
NCCL selects **P2P/CUMEM/read** with 24 channels.

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s |
| 10 KB | 0.38 GB/s | 0.50 GB/s | 0.43 GB/s | 0.44 GB/s | 0.44 GB/s |
| 100 KB | 3.87 GB/s | 4.86 GB/s | 4.11 GB/s | 4.17 GB/s | 3.90 GB/s |
| 1 MB | 19.32 GB/s | 32.56 GB/s | 32.34 GB/s | 28.30 GB/s | 35.26 GB/s |
| 10 MB | 97.38 GB/s | 119.43 GB/s | 136.60 GB/s | 144.00 GB/s | 148.19 GB/s |
| 100 MB | 156.64 GB/s | 174.85 GB/s | 260.78 GB/s | 218.65 GB/s | 219.28 GB/s |

### Isolated: Per-GPU Containers with SHM (SHM/direct/direct)

Each container gets one GPU via `CUDA_VISIBLE_DEVICES`. P2P is disabled
(`NCCL_P2P_DISABLE=1`). Shared `/dev/shm` and `/tmp` volumes enable SHM transport.
NCCL correctly detects `nNodes=1, localRanks=2` (via `NCCL_HOSTID`).

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s | 0.05 GB/s | 0.04 GB/s |
| 10 KB | 0.37 GB/s | 0.51 GB/s | 0.40 GB/s | 0.45 GB/s | 0.45 GB/s |
| 100 KB | 2.02 GB/s | 3.31 GB/s | 2.50 GB/s | 2.77 GB/s | 3.82 GB/s |
| 1 MB | 8.00 GB/s | 14.86 GB/s | 11.72 GB/s | 13.73 GB/s | 11.44 GB/s |
| 10 MB | 12.54 GB/s | 21.67 GB/s | 21.40 GB/s | 22.01 GB/s | 21.05 GB/s |
| 100 MB | 13.34 GB/s | 22.80 GB/s | 23.21 GB/s | 23.29 GB/s | 22.06 GB/s |

### Naive Isolation: Per-GPU Containers without SHM (NET/Socket)

Without shared `/dev/shm`, NCCL detects `nNodes=2` and falls back to TCP sockets
over Docker bridge network. Only 2 channels.

| Size | All-Reduce | All-Gather | Reduce-Scatter | Broadcast | P2P |
|------|-----------|-----------|----------------|-----------|-----|
| 1 KB | 0.02 GB/s | 0.04 GB/s | 0.03 GB/s | 0.04 GB/s | 0.03 GB/s |
| 10 KB | 0.17 GB/s | 0.33 GB/s | 0.25 GB/s | 0.37 GB/s | 0.34 GB/s |
| 100 KB | 0.73 GB/s | 1.67 GB/s | 1.13 GB/s | 2.83 GB/s | 2.32 GB/s |
| 1 MB | 2.68 GB/s | 5.51 GB/s | 4.42 GB/s | 6.67 GB/s | 4.49 GB/s |
| 10 MB | 3.95 GB/s | 7.77 GB/s | 6.78 GB/s | 5.80 GB/s | 5.09 GB/s |
| 100 MB | 4.23 GB/s | 8.50 GB/s | 8.58 GB/s | 5.75 GB/s | 5.11 GB/s |

## Why P2P Breaks with Per-GPU Containers

NCCL's P2P transport uses **CUDA IPC** (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)
to share GPU memory between processes. For P2P to work, the transport selection code
(`p2pCanConnect`) checks:

1. Both ranks are on the same node (`sameNode`)
2. `cudaDeviceCanAccessPeer(myGpu, peerGpu)` returns true

With per-GPU container isolation (`CUDA_VISIBLE_DEVICES=0` in one container,
`CUDA_VISIBLE_DEVICES=1` in the other), each container only sees **one GPU** as
`cuda:0`. The peer GPU is not visible to CUDA, so `cudaDeviceCanAccessPeer` fails —
there is no peer to check.

Even though `NVIDIA_VISIBLE_DEVICES=all` makes both GPUs visible to NVML (for
topology discovery), CUDA runtime isolation prevents the P2P data path.

```
┌─────────────────┐     ┌─────────────────┐
│  Container 0    │     │  Container 1    │
│  cuda:0 = GPU0  │     │  cuda:0 = GPU1  │
│  (no peer GPU)  │     │  (no peer GPU)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │   NVLink (unused!)    │
    GPU0 ═══════════════════ GPU1
         │                       │
    ┌────┴───────────────────────┴────┐
    │         Host Memory (DRAM)       │
    │     Shared /dev/shm (tmpfs)      │
    │    ← SHM transport goes here →   │
    └──────────────────────────────────┘
```

## Isolation Mechanism

### Baseline (compose.baseline.yml)
- `network_mode: host`, `ipc: host`, `pid: host`
- `NVIDIA_VISIBLE_DEVICES=0,1` — both GPUs visible
- `CUDA_VISIBLE_DEVICES=0` or `1` — per-rank GPU selection
- No namespace isolation — CUDA IPC works natively

### SHM Isolation (compose.shm_isolation.yml)
- Bridge network, private IPC namespace
- `NVIDIA_VISIBLE_DEVICES=all` — NVML sees all GPUs for topology
- `CUDA_VISIBLE_DEVICES=0` or `1` — per-container GPU
- `NCCL_P2P_DISABLE=1` — force SHM instead of failing P2P
- `NCCL_HOSTID=exp2-node192` — override hostname for same-node detection
- Shared `/dev/shm` (tmpfs 4GB) and `/tmp` (tmpfs 200MB) volumes

### Naive Isolation (compose.p2p_isolation.yml)
- Bridge network, private IPC namespace
- `NVIDIA_VISIBLE_DEVICES=all`, `CUDA_VISIBLE_DEVICES=0` or `1`
- No P2P disable, no shared `/dev/shm`
- NCCL detects `nNodes=2`, falls back to NET/Socket

## Key Takeaway

**Per-GPU container isolation on NVLink machines incurs a significant performance penalty.**

| Scenario | Transport | 100MB Bandwidth | Relative to Baseline |
|----------|-----------|-----------------|---------------------|
| Baseline (no isolation) | P2P/CUMEM/read (NVLink) | 157-261 GB/s | 100% |
| Best isolation (SHM workaround) | SHM/direct/direct (host memory) | 13-23 GB/s | **~9-12%** |
| Naive isolation (no workaround) | NET/Socket (TCP) | 4-9 GB/s | **~2-5%** |

The SHM workaround recovers from ~3% (Socket) to ~10% (SHM) of baseline bandwidth.
However, this still represents an **~7-12x degradation** compared to the NVLink P2P path.

### Comparison with Node191 (PCIe-only)

| Machine | GPU | Baseline Transport | Isolation Penalty |
|---------|-----|-------------------|-------------------|
| node191 | A100-PCIe (NODE) | SHM/direct/direct | **~0%** (same transport) |
| node192 | A100-SXM4 (NV12) | P2P/CUMEM/read | **~88-91%** (forced SHM) |

**Conclusion:** The viability of per-GPU container isolation depends entirely on the
GPU interconnect. On PCIe-only machines, isolation is free. On NVLink machines,
isolation forces a downgrade from NVLink to host memory, which is a fundamental
architectural limitation — not a configuration problem.

To preserve NVLink performance with containers, one would need to either:
1. Give each container visibility to all GPUs (defeats per-GPU isolation)
2. Implement cross-PID-namespace CUDA IPC (requires kernel/driver support)
3. Use NVLink-aware shared memory (NCCL NVLS, but requires multi-GPU visibility)
