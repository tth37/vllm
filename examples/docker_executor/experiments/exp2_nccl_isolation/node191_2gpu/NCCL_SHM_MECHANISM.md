# NCCL SHM Transport: Internal Mechanism

This document describes how NCCL implements SHM (shared memory) transport for inter-GPU communication, based on analysis of the NCCL 2.21.5 / 2.27.5 source code. Understanding this mechanism explains why per-GPU container isolation achieves zero performance overhead on PCIe-connected GPU systems.

## 1. Transport Selection: Why SHM?

NCCL evaluates transports in priority order for each GPU pair:

```
1. P2P     -- CUDA IPC (cuIpcGetMemHandle / cuMem API), requires shared IPC namespace
2. SHM     -- POSIX shared memory via /dev/shm
3. NET     -- TCP sockets or RDMA (InfiniBand)
4. COLLNET -- Collective network offload (e.g., SHARP)
```

The selection logic lives in `src/transport/shm.cc`, function `shmCanConnect`:

```c
static ncclResult_t shmCanConnect(int* ret, ...) {
  *ret = 0;

  // 1. Check if SHM is disabled
  if (ncclParamShmDisable()) return ncclSuccess;

  // 2. If there is a network path, prefer NET over SHM
  if (ncclTopoCheckNet(...) == 1) return ncclSuccess;

  // 3. Both ranks must be on the same host
  if (info->peerInfo->hostHash != info->myInfo->hostHash) return ncclSuccess;

  // 4. Both ranks must share the same /dev/shm filesystem
  if (info->peerInfo->shmDev != info->myInfo->shmDev) return ncclSuccess;

  *ret = 1;  // SHM transport is eligible
  return ncclSuccess;
}
```

Four conditions must ALL pass for SHM transport:

| Condition | What it checks | How containers satisfy it |
|-----------|---------------|--------------------------|
| `SHM_DISABLE=0` | Not explicitly disabled | `NCCL_SHM_DISABLE=0` (default) |
| No intra-node NET | `NCCL_NET_DISABLE_INTRA=1` avoids NET preference | Set via environment variable |
| Same `hostHash` | Hostname/machine identity match | `NCCL_HOSTID` set to same value |
| Same `shmDev` | `/dev/shm` is the same filesystem (same `st_dev`) | Shared Docker tmpfs volume mounted at `/dev/shm` |

The `shmDev` check is why a shared Docker volume is essential. Each container's `/dev/shm` must be backed by the same tmpfs instance. Docker achieves this with a named volume:

```yaml
volumes:
  shared-shm:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=4g
```

When both containers mount `shared-shm:/dev/shm`, `stat("/dev/shm").st_dev` returns the same device ID in both containers, satisfying NCCL's check.

## 2. Buffer Allocation

NCCL supports two SHM buffer allocation strategies, selected at compile/runtime:

### Strategy A: cuMem API (CUDA 12.2+, modern)

Used when the CUDA driver supports `cuMemCreate` with shareable handles:

```c
ncclCuMemHostAlloc(&ptr, size);                       // allocate device-accessible host memory
cuMemExportToShareableHandle(&handle, ptr, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
// handle is a POSIX file descriptor, sent to peer via Unix domain socket (SCM_RIGHTS)
```

The peer imports the buffer:

```c
cuMemImportFromShareableHandle(&remotePtr, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
```

This is the path used in our node191 baseline (`CUMEM allocated shareable buffer` in NCCL logs).

### Strategy B: Legacy mmap (fallback)

Uses POSIX shared memory files:

```c
fd = open("/dev/shm/nccl-XXXXXX", O_CREAT | O_RDWR);
fallocate(fd, 0, 0, size);
ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// Register with CUDA so GPU can access it
cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
cudaHostGetDevicePointer(&devPtr, ptr);
```

This is the path used in our isolated containers (`Allocated shareable buffer ... ipcDesc` in NCCL logs), because cuMem FD sharing requires UDS which works through the shared `/tmp` volume.

### Buffer Layout

Each SHM connection allocates two shared regions -- one for each direction:

```
Sender's shared region (sendMem):
  +------------------------------+
  | ncclSendMem header (128B)    |  <- head pointer, ptrExchange, offsFifo[8]
  +------------------------------+
  | SIMPLE protocol buffers      |  <- stepSize * NCCL_STEPS (8 slots)
  +------------------------------+
  | LL protocol buffers          |  <- low-latency protocol slots
  +------------------------------+
  | Reference count              |  <- for multi-process cleanup
  +------------------------------+

Receiver's shared region (recvMem):
  +------------------------------+
  | ncclRecvMem header (128B)    |  <- tail pointer, connFifo[8], flush flag
  +------------------------------+
  | SIMPLE protocol buffers      |  <- stepSize * NCCL_STEPS (8 slots)
  +------------------------------+
  | LL protocol buffers          |
  +------------------------------+
```

Key metadata structures:

```c
struct ncclSendMem {
  uint64_t head;                    // producer write pointer
  void* ptrExchange;                // for direct mode buffer exchange
  uint64_t redOpArgExchange[2];     // reduction operation arguments
  int offsFifo[NCCL_STEPS];         // offset FIFO for 8 pipeline stages
};

struct ncclRecvMem {
  uint64_t tail;                    // consumer read pointer
  struct ncclConnFifo connFifo[NCCL_STEPS];  // per-slot size metadata
  int flush;                        // GDRCopy flush flag
};
```

## 3. Data Transfer: Two Modes

The NCCL log message `via SHM/direct/direct` encodes the send-side and receive-side modes:

```
via SHM/{send_mode}/{recv_mode}
         ^^^^^^^^    ^^^^^^^^
         "direct"    "direct"   = GPU accesses SHM directly
         "CE"        "CE"       = Copy Engine (proxy cudaMemcpyAsync)
```

### Mode 1: Direct (our case)

In `direct` mode, the GPU reads from and writes to the SHM buffer directly via PCIe DMA. No CPU proxy thread is involved in the data path.

```
GPU 0                                        GPU 1
  |                                            |
  |  cudaMemcpy (GPU kernel)                   |
  +--- PCIe DMA write --->  /dev/shm buffer    |
  |                          (shared tmpfs)     |
  |                                  |          |
  |                        PCIe DMA read -------+
  |                                   cudaMemcpy (GPU kernel)
```

How it works:

1. **Setup**: During connection, each side maps the peer's SHM buffer into its GPU address space via `cudaHostGetDevicePointer()` or cuMem import. This gives each GPU a device-accessible pointer to the shared host memory.

2. **Send**: The GPU kernel on rank 0 writes data directly to the SHM buffer using the device pointer. The write traverses: GPU 0 VRAM -> PCIe bus -> CPU memory controller -> tmpfs page in RAM.

3. **Signal**: After writing, the GPU kernel updates the `head` pointer (also in shared memory) to notify the receiver.

4. **Receive**: The GPU kernel on rank 1 polls the `tail`/`head` pointer, sees new data, and reads from the SHM buffer. The read traverses: tmpfs page in RAM -> CPU memory controller -> PCIe bus -> GPU 1 VRAM.

The "direct" in `SHM/direct/direct` does **not** mean GPU Direct RDMA (GDR). GDR is an InfiniBand/network concept. Here, "direct" simply means the GPU accesses the SHM host memory directly through PCIe, without an intermediate CPU-side `cudaMemcpyAsync` from a proxy thread.

### Mode 2: CE (Copy Engine)

Enabled via `NCCL_SHM_USE_CUDA_MEMCPY=1`. A proxy thread orchestrates `cudaMemcpyAsync` calls:

```
GPU 0                Proxy Thread 0        /dev/shm         Proxy Thread 1           GPU 1
  |                       |                   |                    |                    |
  | write to devFifo      |                   |                    |                    |
  +-----> devFifo ------->|                   |                    |                    |
  |       (GPU mem)       | cudaMemcpyAsync   |                    |                    |
  |                       | (D2H)             |                    |                    |
  |                       +-------> shmFifo --+                    |                    |
  |                       |         (host)    |                    |                    |
  |                       |                   +----> shmFifo ----->|                    |
  |                       |                   |      (mapped)      | cudaMemcpyAsync    |
  |                       |                   |                    | (H2D)              |
  |                       |                   |                    +-------> devFifo --->+
  |                       |                   |                    |         (GPU mem)   |
```

CE mode adds one level of staging (devFifo in GPU memory) but allows the proxy to batch and pipeline copies more efficiently. It is typically used for specific network topologies and is not the default for intra-node SHM.

## 4. Pipeline Architecture

NCCL SHM transport uses an **8-stage pipeline** (`NCCL_STEPS = 8`). Each stage has its own buffer slot, allowing overlapped sends and receives.

### Pipeline State Machine

Each proxy operation tracks progress through 5 watermarks:

```c
struct ncclProxySubArgs {
  uint64_t base;         // starting step number
  uint64_t posted;       // GPU kernel has posted data
  uint64_t transmitted;  // copy to/from SHM initiated
  uint64_t done;         // copy completed, peer notified
  uint64_t end;          // final step number
};
```

The invariant is: `base <= posted <= transmitted <= done <= end`

### Pipeline in Action (8 slots)

For a large transfer broken into 8 chunks:

```
Time -->

Slot 0: [GPU write] [in SHM, peer reading] [done, reusable]
Slot 1:   [GPU write] [in SHM, peer reading] [done, reusable]
Slot 2:     [GPU write] [in SHM, peer reading] [done]
Slot 3:       [GPU write] [in SHM, peer reading]
Slot 4:         [GPU write] [in SHM]
Slot 5:           [GPU write]
Slot 6:             [GPU write]
Slot 7:               [GPU write]
```

The slot index wraps around: `slot = (base + step) % NCCL_STEPS`

This pipelining is critical for achieving high bandwidth. While GPU 0 writes chunk N to slot N%8, GPU 1 can simultaneously read chunk N-4 from slot (N-4)%8. The 8-slot depth keeps the PCIe bus saturated in both directions.

### Step Size

Each slot holds `stepSize = totalBufferSize / NCCL_STEPS` bytes. For a 4MB total SHM buffer (typical), each slot is 512KB. Larger transfers are broken into multiple rounds through all 8 slots.

## 5. Synchronization

NCCL SHM transport uses **lock-free, poll-based synchronization** with no kernel-level primitives (no futex, no semaphore, no mutex).

### Head/Tail Protocol

```
Sender (GPU 0)                              Receiver (GPU 1)

1. Poll: wait for tail > base + transmitted
   (receiver has consumed previous data)

2. Write data to shmFifo[slot]

3. Atomic fence (seq_cst)

4. Update connFifo[slot].size = N

5. Update recvMem->tail = base + done       3. Poll: wait for connFifo[slot].size > 0
                                               (sender has written new data)

                                            4. Read data from shmFifo[slot]

                                            5. Update sendMem->head = base + done
                                               (signals sender that slot is reusable)
```

### Memory Ordering

The critical ordering constraint: data must be fully written before the size/tail update is visible to the peer. NCCL ensures this with:

```c
// After writing data and size
__atomic_thread_fence(__ATOMIC_SEQ_CST);
resources->recvMem->tail = sub->base + sub->done;
```

The `seq_cst` fence ensures all prior stores (data + connFifo.size) are globally visible before the tail update. The receiver polls `tail` or `connFifo[slot].size` and is guaranteed to see the data once it sees the updated pointer.

### Why Polling (Not Blocking)?

NCCL's proxy thread busy-polls because:
1. **Latency**: Futex/semaphore require kernel transitions (~1-5us). Polling catches updates in nanoseconds.
2. **Throughput**: With 8 pipeline slots, there is almost always work to do. Idle polling is rare during active transfers.
3. **Predictability**: No scheduling jitter from kernel wakeup delays.

The proxy thread yields when idle:

```c
// In proxy.cc progress loop
if (no_work) {
  std::this_thread::yield();  // give CPU slice back, but stay in userspace
}
```

## 6. Proxy Thread Architecture

Each NCCL rank runs two proxy threads:

### Service Thread

Handles control-plane operations via a Unix domain socket:

```
Main Thread              Service Thread
    |                        |
    |-- setup request ------>|  (allocate SHM buffers)
    |                        |
    |-- connect request ---->|  (exchange handles with peer)
    |                        |
    |-- free request ------->|  (cleanup buffers)
```

The UDS socket is created in `/tmp`, which is why our isolated containers must share `/tmp` via a Docker tmpfs volume. Without it, the service threads in different containers cannot communicate.

### Progress Thread

Drives data-plane operations in a tight polling loop:

```c
void ncclProxyProgress(struct ncclProxyState* state) {
  while (!state->stop) {
    for (each active operation in state->opsPool) {
      // Call transport-specific progress function
      transport->proxyProgress(op);  // e.g., shmSendProxyProgress

      if (op->state == ncclProxyOpNone) {
        // Operation complete, remove from active list
        removeOp(op);
      }
    }

    if (no_active_ops) {
      std::this_thread::yield();
    }
  }
}
```

In `direct` mode (our case), the progress thread has minimal work -- it mainly handles setup/teardown. The GPU kernels themselves do the data transfer and signaling via the shared head/tail pointers. The progress thread only becomes active in `CE` mode where it orchestrates `cudaMemcpyAsync` calls.

## 7. End-to-End: What Happens During an All-Reduce

Here is the complete flow for a 100MB All-Reduce between GPU 0 and GPU 1 using `SHM/direct/direct`:

```
1. INITIALIZATION (once per NCCL communicator)
   - Rank 0: shm_open("/dev/shm/nccl-XXX") -> mmap -> cudaHostRegister -> cudaHostGetDevicePointer
   - Rank 1: open same file -> mmap -> cudaHostRegister -> cudaHostGetDevicePointer
   - Exchange device pointers via proxy service thread UDS
   - Both GPUs now have device pointers to each other's SHM buffers

2. RING ALL-REDUCE (for 100MB tensor, ring algorithm with 2 ranks)

   Phase 1: Reduce-Scatter (each rank sends half, receives half)

   GPU 0 kernel:                          GPU 1 kernel:
   for chunk in 0..N:                     for chunk in 0..N:
     slot = chunk % 8                       slot = chunk % 8
     poll(peer_tail > chunk)                poll(peer_tail > chunk)
     DMA write my_half[chunk]               DMA write my_half[chunk]
       -> peer's shmFifo[slot]                -> peer's shmFifo[slot]
     fence + update head                    fence + update head

     poll(my connFifo[slot].size)            poll(my connFifo[slot].size)
     DMA read peer's data from              DMA read peer's data from
       my shmFifo[slot]                       my shmFifo[slot]
     reduce: local[chunk] += received       reduce: local[chunk] += received
     update tail (slot reusable)            update tail (slot reusable)

   Phase 2: All-Gather (each rank sends its reduced half to peer)

   [Same pattern, but without the reduction step]

3. RESULT
   Both GPUs have the fully reduced 100MB tensor
```

Total data movement for 100MB All-Reduce with 2 ranks:
- Phase 1: 50MB sent + 50MB received per rank = 100MB through SHM per rank
- Phase 2: 50MB sent + 50MB received per rank = 100MB through SHM per rank
- Total: 200MB through SHM per rank, achieving ~13 GB/s (limited by PCIe bandwidth)

## 8. Why Isolation Has No Overhead

The analysis above reveals exactly why per-GPU container isolation is zero-cost:

1. **The data path is purely hardware.** In `direct` mode, data moves via PCIe DMA between GPU VRAM and host memory pages. These are physical memory operations that are unaffected by Linux namespace boundaries. The GPU's DMA engine doesn't know or care about container namespaces.

2. **The shared resource is a tmpfs page.** Both containers' `/dev/shm` point to the same tmpfs mount. When GPU 0 DMA-writes to a page in this tmpfs, GPU 1 reads from the exact same physical page. No data copying, no namespace translation.

3. **Synchronization is in shared memory.** The head/tail pointers live in the same SHM region as the data. Both GPUs poll these pointers directly via PCIe -- no system calls, no kernel involvement, no namespace crossings.

4. **The proxy thread is mostly idle.** In `direct` mode, the proxy service thread is only used during setup (handle exchange via UDS in `/tmp`). During data transfer, no proxy is involved. The GPU kernels drive the entire pipeline.

5. **NVML topology is preserved.** `NVIDIA_VISIBLE_DEVICES=0,1` ensures NCCL sees the full PCIe topology. `CUDA_VISIBLE_DEVICES` restricts which GPU each container uses for computation, but NCCL's topology discovery (via NVML, not CUDA) still sees both GPUs and correctly identifies them as co-located.

The only overhead comes from container startup and NCCL initialization. Once the SHM buffers are established, the steady-state data path is physically identical to the non-containerized case.

## References

- NCCL source: `src/transport/shm.cc` -- SHM transport implementation
- NCCL source: `src/misc/shmutils.cc` -- SHM buffer allocation (mmap/cuMem)
- NCCL source: `src/proxy.cc` -- Proxy thread management and progress loop
- NCCL source: `src/transport.cc` -- Transport selection logic (`selectTransport`)
- NCCL source: `src/include/comm.h` -- `ncclSendMem` / `ncclRecvMem` structures
- NCCL source: `src/include/proxy.h` -- Proxy operation state machine
