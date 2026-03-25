# Exp2: NCCL High-Bandwidth Communication with Per-GPU Container Isolation

## Problem Statement

In exp1, we showed that vLLM's DockerDistributedExecutor achieves bare-metal NCCL performance when containers share the host's IPC and network namespaces (all GPUs visible, `ipc: host`, `network_mode: host`). However, this provides **zero isolation** between GPU workers.

When we isolate containers (1 GPU per container, private IPC, private network), NCCL falls back to TCP sockets (`NET/Socket/2`), causing **2-10x latency degradation** for multi-GPU inference.

**Goal:** Find a configuration or NCCL modification that preserves high-speed GPU communication (PCIe P2P or SHM transport) while maintaining per-GPU container isolation.

## Background: Why Isolation Breaks NCCL

NCCL selects transport based on what the system makes available:

| Transport | Mechanism | Bandwidth (A30 PCIe) | Requirements |
|-----------|-----------|---------------------|--------------|
| **P2P** | CUDA IPC (`cuIpcGetMemHandle`) + NVLink/PCIe | ~20-25 GB/s per direction | Shared IPC namespace(?), all GPUs visible, NCCL UDS at `/tmp/` |
| **SHM** | POSIX shared memory via `/dev/shm` | ~10-15 GB/s | Shared `/dev/shm` |
| **NET/Socket** | TCP sockets | ~3 GB/s | Always works |

Three barriers cause TCP fallback:
1. **Different hostnames** → NCCL sees containers as separate nodes → solved with `NCCL_HOSTID`
2. **GPU topology invisible** → NCCL can't detect PCIe/NVLink paths → solved with `NVIDIA_VISIBLE_DEVICES=all`
3. **Namespace isolation** → CUDA IPC and NCCL Unix domain sockets don't cross namespace boundaries

This experiment systematically tests which namespace causes the performance loss and whether we can work around it.

## Experiment Structure

### Phase 1: Namespace Isolation Matrix (zero code changes)

Six Docker Compose configurations testing each isolation dimension independently:

| Config | GPU Visibility | IPC | Network | Shared Volumes | Key Question |
|--------|---------------|-----|---------|----------------|--------------|
| **A: baseline** | all GPUs | host | host | - | Known good (P2P) |
| **B: private_ipc** | all GPUs | **private** | host | - | Does CUDA IPC work across IPC namespaces? |
| **C: private_net** | all GPUs | host | **bridge** | /tmp (tmpfs) | Do NCCL UDS work via shared volume? |
| **D: private_both** | all GPUs | **private** | **bridge** | /tmp (tmpfs) | Combined isolation |
| **E: single_gpu** | 1 GPU each | **private** | **bridge** | - | Full isolation (TCP, known bad) |
| **F: shared_shm** | 1 GPU each | **private** | **bridge** | /dev/shm + /tmp | Does shared SHM volume enable SHM transport? |

**Critical hypothesis (Config B):** CUDA IPC is implemented in the NVIDIA kernel module (`nvidia.ko`), not through POSIX IPC. Private IPC namespace isolates System V IPC (semaphores, shared memory, message queues), but CUDA IPC uses `ioctl()` on `/dev/nvidia*`. If this is true, P2P transport works despite private IPC namespace.

### Phase 2: SHM Transport with Shared Volumes

If Phase 1 shows that no simple namespace config achieves P2P, force NCCL to use its SHM transport:

| Config | Description |
|--------|-------------|
| **shm_forced** | `NCCL_P2P_DISABLE=1`, shared `/dev/shm` volume, per-GPU containers |
| **shm_p2p_combo** | All GPUs visible, shared `/dev/shm` + `/tmp`, try P2P |

### Phase 3: NCCL Patching (if needed)

If Phases 1-2 fail:
- Clone NCCL v2.21.5 source
- Patch P2P transport to use `pidfd_getfd()` (kernel 5.6+) for cross-namespace CUDA IPC FD passing
- Or use CUDA VMM API (`cuMemExportToShareableHandle` with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`)
- Patch bootstrap sockets to use filesystem-based UDS on shared volume
- Build and test with `LD_PRELOAD`

## Quick Start

```bash
# On node196 (4x A30)
cd gpu_comm_benchmark/exp2

# Smoke test all Phase 1 configs (~2 min)
./run_all.sh --smoke

# Full benchmark (Phase 1 + Phase 2, ~30 min)
./run_all.sh

# Run a single Phase 1 config
cd phase1_isolation_matrix
./run_matrix.sh --config=private_ipc

# Analyze results
python3 analyze_exp2.py
```

## Environment Variables Reference

### NCCL Transport Control
| Variable | Values | Effect |
|----------|--------|--------|
| `NCCL_P2P_DISABLE` | `0`/`1` | Disable P2P (CUDA IPC) transport |
| `NCCL_SHM_DISABLE` | `0`/`1` | Disable SHM transport |
| `NCCL_SHM_USE_CUDA_MEMCPY` | `0`/`1` | Use CUDA memcpy for SHM data path |
| `NCCL_NET_DISABLE` | `0`/`1` | Disable NET (TCP/IB) transport |
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
| `device_ids` (compose) | Docker | GPUs accessible to container |

## NCCL Transport Detection

Check NCCL logs for transport selection:
```bash
# P2P (best — PCIe/NVLink):
grep "via P2P" results/*_log.txt

# SHM (good — host memory):
grep "via SHM" results/*_log.txt

# NET/Socket (bad — TCP):
grep "via NET" results/*_log.txt

# Topology info:
grep "nNodes\|localRanks" results/*_log.txt
```

## Machine

- **Node196** (10.0.2.196): 4x NVIDIA A30, PCIe Gen4 (no NVLink)
- **Kernel:** 5.15.0 (has `pidfd_open` 5.3+ and `pidfd_getfd` 5.6+)
- **Docker image:** `gpu-comm-benchmark:latest` (from `gpu_comm_benchmark/Dockerfile`)
- **NCCL:** v2.21.5 (bundled with PyTorch 2.5.1)

## Directory Structure

```
exp2/
├── README.md                           # This file
├── run_all.sh                          # Master orchestrator
├── analyze_exp2.py                     # Multi-config comparison tool
├── results/                            # Benchmark JSON + NCCL logs
│
├── phase1_isolation_matrix/
│   ├── run_matrix.sh                   # Phase 1 runner
│   ├── compose.baseline.yml            # A: all GPUs + host IPC + host net
│   ├── compose.private_ipc.yml         # B: all GPUs + private IPC + host net
│   ├── compose.private_net.yml         # C: all GPUs + host IPC + bridge net
│   ├── compose.private_both.yml        # D: all GPUs + private IPC + bridge net
│   ├── compose.single_gpu.yml          # E: 1 GPU + private IPC + bridge net
│   └── compose.shared_shm.yml         # F: 1 GPU + shared /dev/shm + bridge net
│
├── phase2_shm_transport/
│   ├── run_phase2.sh                   # Phase 2 runner
│   ├── compose.shm_forced.yml          # P2P disabled, force SHM
│   └── compose.shm_p2p_combo.yml       # All GPUs visible + shared SHM/tmp
│
└── phase3_nccl_patch/
    ├── Dockerfile.nccl-dev             # NCCL build environment
    ├── build_nccl.sh                   # Clone + patch + build
    ├── patches/                        # Patch files
    └── compose.patched_nccl.yml        # Test with patched NCCL

Bundled benchmark infrastructure:
├── benchmark.py                        # Core benchmark (all-reduce, all-gather, etc.)
├── benchmark_docker.py                 # Container entry point
├── Dockerfile                          # Base Docker image
├── diagnose_nccl.py                    # Namespace diagnostic tool
└── nccl/                               # NCCL source (git submodule)
```

## Results

*(To be filled after running experiments)*

### Phase 1 Summary

| Config | Transport | AllReduce 100MB | Status |
|--------|-----------|----------------|--------|
| A: baseline | | | |
| B: private_ipc | | | |
| C: private_net | | | |
| D: private_both | | | |
| E: single_gpu | | | |
| F: shared_shm | | | |

### Key Findings

*(To be filled)*
