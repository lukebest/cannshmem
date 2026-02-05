# SHMEM NPU Synchronization Mechanisms Report

This report provides a detailed analysis of the NPU-to-NPU synchronization schemes implemented in the SHMEM library. It covers the **Centralized (Pull Mode)** and **Dissemination (Push Mode)** algorithms, along with the low-level signal primitives.

## 1. Overview

The SHMEM library implements multiple synchronization algorithms to handle NPU-level barriers (Level 2 synchronization). These algorithms ensure that all ranks within a team reach a synchronization point before proceeding.

*   **v1 (Standard Dissemination)**: A scalable O(log N) algorithm using a "Push" model.
*   **v2 (Group Dissemination)**: An optimized "Push" model leveraging vector cores for higher radix.
*   **v3 (Centralized)**: A low-latency "Pull" model optimized for small scales (e.g., intra-server), using parallel reads.

## 2. Centralized Barrier (v3) - "Pull Mode"

This is the default algorithm for small-scale synchronization (e.g., 8 ranks within a server). It minimizes write contention by having each rank write only to its local memory, while other ranks poll remotely.

### 2.1 Concept
*   **Write Local**: A rank only updates its own synchronization flag in its local memory.
*   **Poll Remote**: A rank reads the synchronization flags of other ranks from remote memory.
*   **Parallelism**: The polling process is parallelized across the NPU's Vector Cores (VEC), allowing multiple remote flags to be checked simultaneously.

### 2.2 Algorithm Flow (`shmemi_barrier_npu_v3`)

1.  **Preparation**:
    *   Calculate `count = sync_counter + 1`. This unique value distinguishes the current barrier round from previous ones.
    *   Determine the subset of ranks to handle based on `vec_id` (Vector Core ID) and `k` (parallelism factor, default 8).

2.  **Execution Loop**:
    *   Iterate through all ranks `i` in the team (stride `k`).
    *   **If `i` is self**:
        *   Write `count` to the **local** `sync_array`.
        *   Call `shmemi_signal_set`.
    *   **If `i` is remote**:
        *   Calculate the remote PE's ID: `remote_pe = start + i * stride`.
        *   Calculate the remote address: `shmemi_ptr(sync_array, remote_pe)`.
        *   **Poll**: Call `shmemi_signal_wait_until_eq_for_barrier` to wait until the remote memory location equals `count`.

3.  **Completion**:
    *   Once all assigned remote ranks have been verified, update the local `sync_counter` to `count`.

### 2.3 Pros & Cons
*   **Pros**: Excellent for small N (e.g., N=8). Zero write contention on remote memory.
*   **Cons**: Linear number of reads O(N) if not parallelized. Bandwidth intensive for large N.

---

## 3. Dissemination Barrier (v1/v2) - "Push Mode"

These algorithms use a "Push" model where a rank actively signals (writes to) a remote rank and waits on its own local memory. This is scalable for large clusters.

### 3.1 v1: Standard Dissemination (`shmemi_barrier_npu_v1`)

*   **Structure**: Executes in `ceil(log2(N))` rounds.
*   **Distance**: In round `r`, rank `i` signals rank `(i + 2^r) % N`.
*   **Flow**:
    1.  **Loop**: Iterate `shift` from 1 up to `size` (shift doubles each round).
    2.  **Signal (Push)**:
        *   Calculate target: `next_pe`.
        *   Write `count` to `next_pe`'s `sync_array` at a specific `offset` (round index).
        *   Use `shmemi_signal_set(addr, next_pe, count)`.
    3.  **Wait (Local)**:
        *   Wait for `count` to appear in **local** `sync_array` at `offset`.
        *   Use `shmemi_signal_wait_until_eq_for_barrier`.
    4.  **Advance**: `shift *= 2`, `offset++`.

### 3.2 v2: Group Dissemination (`shmemi_barrier_npu_v2`)

*   **Optimization**: Uses multiple vector cores to perform a **Radix-K** dissemination step instead of Radix-2.
*   **Flow**:
    1.  **Parallel Signal**:
        *   Instead of sending 1 signal per round, use Vector Cores to send up to `k` signals in parallel.
        *   Each `vec_id` handles a subset of targets: `next_pe = (my_pe + i * shift) % size`.
    2.  **Parallel Wait**:
        *   Similarly, wait on `k` incoming signals in parallel.
    3.  **Efficiency**: Reduces the number of rounds to `O(log_k N)`, lowering synchronization latency on devices with high parallelism.

---

## 4. Low-Level Primitives

The barrier algorithms rely on two fundamental atomic-like operations tailored for the NPU memory consistency model.

### 4.1 Signal Set (`shmemi_signal_set`)
Defined in `include/internal/device/sync/shmemi_device_p2p.h`.

```cpp
SHMEM_DEVICE void shmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    shmemi_store(addr, val);
    // CRITICAL: Flush data cache to Global Memory (GM)
    dcci_cacheline((__gm__ uint8_t *)addr);
}
```
*   **Mechanism**: Writes value to memory and immediately executes `dcci_cacheline`. This ensures the write is pushed from the L1/L2 cache to the global memory, making it visible to other NPU cores or remote devices.

### 4.2 Signal Wait (`shmemi_signal_wait_until_eq_for_barrier`)

```cpp
SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq_for_barrier(...)
{
    do {
        // CRITICAL: Invalidate cache before reading
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
        
        if (*sig_addr == cmp_val) return *sig_addr;
        
        // Handle race condition where peer already moved to next barrier
        if (*sig_addr == cmp_val + 1) return *sig_addr;
        
    } while (true);
}
```
*   **Mechanism**: Implements a spin-wait loop.
*   **Coherence**: Inside the loop, it **must** call `dcci_cacheline` before every read. This invalidates the local cache line, forcing the NPU to re-fetch the value from Global Memory (where the remote rank or other core wrote it). Without this, the NPU would spin on a stale cached value forever.
