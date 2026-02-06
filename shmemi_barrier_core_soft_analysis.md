# shmemi_barrier_core_soft 函数详解与共享内存信号量同步机制

## 1. 概述

`shmemi_barrier_core_soft` 是 SHMEM 库中用于 **设备内（Intra-Device）** 多个 Vector Core（AIV）之间进行同步的核心函数。

在 Ascend NPU 架构中，当硬件同步指令（如 `AscendC::SyncAll`）不可用或被禁用时（例如在混合算子模式下），系统会自动回退到该软件实现。该函数实现了基于共享内存（Global Memory, GM）的 **Dissemination Barrier（传播栅栏/蝴蝶栅栏）** 算法。

## 2. 核心数据结构

为了实现高效且无竞争的同步，该机制依赖以下关键数据结构：

### 2.1. `shmemi_sync_bit` (防止伪共享)
```c
// include/internal/host_device/shmemi_types.h
#define SCALAR_DATA_CACHELINE_SIZE 64
#define SHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE
typedef int32_t shmemi_sync_bit[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];
```
*   **定义**：一个大小为 64 字节（Cacheline 大小）的数组，实际上只使用第一个 `int32_t` 存储信号量。
*   **目的**：**防止伪共享（False Sharing）**。在多核并发读写同一块内存区域时，如果不同核的信号量位于同一个 Cacheline 中，会导致缓存一致性协议（MESI）频繁触发无效化，严重降低性能。强制对齐到 Cacheline 确保每个核操作独立的缓存行。

### 2.2. `core_sync_pool` (信号量池)
*   这是一个巨大的 `shmemi_sync_bit` 数组，存储在 Global Memory 中。
*   **寻址方式**：
    ```cpp
    sync_array + idx * SHMEM_LOG_MAX_AIV_PER_NPU + offset
    ```
    *   `idx`：当前核的索引。
    *   `SHMEM_LOG_MAX_AIV_PER_NPU`：常数 6（对应最大 48 核的 log2 值）。
    *   `offset`：当前栅栏同步的轮次（Round）。

### 2.3. `core_sync_counter` (代计数器)
*   全局唯一的计数器，记录当前 Barrier 执行了多少次（Generation Count）。
*   用于区分不同批次的 Barrier 调用，防止“过快”的核处理了上一轮的旧信号。

## 3. 算法流程：Dissemination Barrier

该函数使用的是 **Dissemination Barrier** 算法。这是一种去中心化的同步算法，适合核数较多的场景（如 NPU 的 30-40 个 AIV 核）。

### 3.1. 算法逻辑
对于 $N$ 个参与者，算法需要进行 $\lceil \log_2 N \rceil$ 轮同步。
在第 $k$ 轮（$k=0, 1, ...$），距离 `shift` 为 $2^k$：
1.  **Signal（通知）**：核 $i$ 通知核 $(i + 2^k) \pmod N$。
2.  **Wait（等待）**：核 $i$ 等待核 $(i - 2^k) \pmod N$ 的通知。

经过 $\log_2 N$ 轮后，每个核都间接“知道”了所有其他核都已经到达了当前步骤，从而实现全员同步。

### 3.2. 代码实现详解
```cpp
// include/internal/device/sync/shmemi_device_barrier.h

SHMEM_DEVICE void shmemi_barrier_core_soft()
{
#ifdef __DAV_C220_VEC__
    // 1. 获取同步资源
    auto sync_array = shmemi_get_core_sync_array();
    auto sync_counter = shmemi_get_core_sync_counter();

    // 2. 获取当前核 ID 和总核数
    int idx = AscendC::GetBlockIdx();
    int size = AscendC::GetBlockNum();
    
    // 3. 获取并更新 Barrier 代计数 (Generation Count)
    // 每一个新的 Barrier 调用，count 都会增加，区分不同批次的同步
    int count = shmemi_load((__gm__ int32_t *)(sync_counter)) + 1;

    int shift = 1;  // 初始距离 2^0
    int offset = 0; // 轮次偏移量，用于区分同一 Barrier 内的不同轮次
    
    // 4. Log(N) 轮同步循环
    while (shift < size) {
        // 计算目标核：(idx + shift) % size
        int next = (idx + shift) % size;

        // 【Signal】: 通知目标核 "next"
        // 注意：这里写入的是目标核 next 的内存槽位
        shmemi_signal_set((__gm__ int32_t *)(sync_array + next * SHMEM_LOG_MAX_AIV_PER_NPU + offset), count);
        
        // 【Wait】: 等待自己的内存槽位被通知
        // 等待直到 sync_array[idx] 的值变为 count
        // 实际上是等待 (idx - shift) 的核来写入
        shmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array +
            idx * SHMEM_LOG_MAX_AIV_PER_NPU + offset), count);

        // 更新距离和轮次
        shift *= SHIFT_MULTIPLIER; // shift = 1, 2, 4, 8...
        offset++;
    }

    // 5. 更新全局计数器（通常只有 rank 0 或部分核做，但这里所有核都写也没问题，因为值一样）
    shmemi_store((__gm__ int32_t *)(sync_counter), count);
#endif
}
```

## 4. 底层通信机制：信号量与缓存一致性

该实现的核心在于如何保证 Global Memory 的读写在不同核之间立即可见。这依赖于 `shmemi_device_p2p.h` 中的底层原语。

### 4.1. `shmemi_signal_set` (生产者：写 + 刷缓存)
```cpp
SHMEM_DEVICE void shmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    // 1. 写入内存 (此时可能只在 L1/L2 Cache 中)
    shmemi_store(addr, val);

    // 2. 刷缓存 (Data Cache Clean/Invalidate)
    // 强制将 Cache 中的脏数据写回 Global Memory，确保其他核能看到最新值
    dcci_cacheline((__gm__ uint8_t *)addr);
}
```

### 4.2. `shmemi_signal_wait_until_eq_for_barrier` (消费者：无效化 + 读)
```cpp
SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        // 1. 无效化缓存 (Data Cache Invalidate)
        // 强制废弃当前 Cache 中的数据，下次读取必须从 Global Memory 拉取
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        // 2. 读取并比较
        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // 3. 处理“超前”情况
        // 如果对端核跑得太快，已经进入了下一个 Barrier (count + 1)，也视为本轮通过
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true); // 自旋等待
}
```

## 5. 总结

`shmemi_barrier_core_soft` 的设计体现了高性能并行编程的几个关键原则：
1.  **去中心化**：通过 Dissemination 算法，避免了单一计数器的热点竞争（Hot-spot contention）。
2.  **缓存友好**：通过 `shmemi_sync_bit` 的 Padding 设计，彻底消除了伪共享。
3.  **显式一致性**：利用 `dcci_cacheline` 指令，手动管理 NPU 复杂的存储层次结构，确保了数据的可见性。
