# SHMEM Barrier Synchronization Guide

This guide explains the usage and details of `barrier` and `barrier_all` in the CANN-SHMEM library, with a focus on high-performance collective synchronization for Ascend NPUs.

## 1. Core Concepts

A **barrier** is a collective synchronization point. When a PE (Processing Element) calls a barrier routine, it blocks until all other participating PEs have also called the same routine.

### Key Guarantees
- **Execution Synchronization**: No PE proceeds past the barrier until all PEs have arrived.
- **Memory Visibility**: All memory stores, Puts, and AMOs issued before the barrier are completed and visible to all other PEs after the barrier.

---

## 2. API Overview

### `shmem_barrier_all()`
Synchronizes all PEs in the global communication domain (`SHMEM_TEAM_WORLD`).

**Usage:**
```cpp
// On Device (AI Core)
shmem_barrier_all();

// On Host
shm::shmem_barrier_all();
```

### `shmem_barrier(shmem_team_t team)`
Synchronizes only the PEs belonging to a specific `team`.

**Usage:**
```cpp
shmem_barrier(my_team);
```

### `shmemx_barrier_all_vec()` (Extension)
A specialized barrier that only synchronizes **Vector (VEC) cores**. This is crucial for **compute-communication overlap**, allowing CUBE cores (doing matrix math) to continue working while VEC cores (doing data movement) synchronize.

---

## 3. MoE Dispatch/Combine Example

In Mixture of Experts (MoE), barriers are used to ensure that tokens are fully dispatched before experts start computation, and that expert results are fully combined before the next layer starts.

### Dispatch Phase Logic
1. **Prepare**: Calculate which tokens go to which expert.
2. **Sync**: Ensure all PEs have finished preparation.
3. **Dispatch**: Use `shmem_put` to send tokens to remote experts.
4. **Barrier**: Ensure all remote writes are completed.

### Code Example (Simplified)

```cpp
#include "shmem_api.h"

// Example MIX Kernel for MoE Dispatch
__aicore__ void moe_dispatch_combine_kernel(GM_ADDR symmetric_heap, ...) {
    // 1. Initialize Sync Base (Mandatory for Ascend NPUs)
    shmemx_set_ffts_config(shmemx_get_ffts_config());

    // --- DISPATCH PHASE ---
    // Prepare token buffers...
    
    // Ensure all data is ready before sending
    shmem_barrier_all();

    if (is_dispatcher) {
        // Send tokens to remote expert on PE 'target_rank'
        shmem_put_nbi(remote_buffer, local_tokens, size, target_rank);
    }

    // CRITICAL: Wait for all PEs to finish dispatching
    // This ensures experts can safely read their input buffers
    shmem_barrier_all();

    // --- EXPERT COMPUTATION ---
    // Expert logic here...
    
    // --- COMBINE PHASE ---
    // Ensure experts finished before pulling results back
    shmem_barrier_all();
    
    if (is_combiner) {
        shmem_get_nbi(local_output, remote_expert_result, size, expert_rank);
    }
    
    // Final sync before exiting kernel
    shmem_barrier_all();
}
```

---

## 4. Internal Implementation Levels

The library uses a hierarchical synchronization strategy:

1.  **Level 1 (Intra-Device)**: Synchronizes AI Cores within a single NPU using hardware `SyncAll`.
2.  **Level 2 (Intra-Host/Cluster)**: Synchronizes NPUs across the network.
    - **Centralized Pull (v3)**: Fast for small clusters. PE A writes a signal to its own memory; PE B polls PE A's memory.
    - **Dissemination (v1/v2)**: Scalable $O(\log N)$ algorithm for large clusters.
3.  **Control Barrier (Host)**: Used for slow-path synchronization like memory allocation (`shmem_malloc`), utilizing a TCP/Socket based control network.

## 5. Important Restrictions

- **MIX Kernel Requirement**: Barrier APIs must be used inside kernels configured as `KERNEL_TYPE_MIX_AIC_...`.
- **Conflict with `SyncAll`**: Avoid mixing `shmem_barrier` with manual `AscendC::SyncAll` calls to prevent internal state corruption.
- **Cache Consistency**: The library automatically handles `dcci` (Data Cache Clean and Invalidate) during barriers to ensure Global Memory (GM) visibility.
