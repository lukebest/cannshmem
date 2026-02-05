# NPU 设备间 Barrier 同步机制分析报告

## 1. 概述
本代码库（SHMEM）为昇腾（Ascend）NPU平台提供了一套基于共享内存的多机多卡通信与同步机制。Barrier（屏障）是其中最核心的同步原语，用于确保通信域（Team）内的所有计算单元（PE）在继续执行前都到达同一个同步点。

## 2. Barrier 同步原语
SHMEM 提供了不同粒度和层级的 Barrier 原语，主要分为以下两类：

### 2.1 向量核（Vector Core）级 Barrier
用于同一设备（Device）内不同 Vector Core 之间的同步。
*   **API**: `shmemx_barrier_vec(shmem_team_t tid)`, `shmemx_barrier_all_vec()`
*   **实现**: `shmemi_barrier_core<bool is_aiv_only>`
    *   如果硬件支持（`__CCE_AICORE_ENABLE_MIX__`），直接调用 `AscendC::SyncAll`。
    *   否则使用软件实现 `shmemi_barrier_core_soft`，通过 `core_sync_pool` 和 `core_sync_counter` 进行基于共享内存的信号量同步。

### 2.2 设备（Device）级 Barrier
用于跨设备（不同 Rank）之间的同步。
*   **API**: `shmem_barrier(shmem_team_t tid)`, `shmem_barrier_all()`
*   **实现**: `shmemi_barrier`
    *   这是一个混合 Barrier，执行流程为：
        1.  `shmemi_barrier_core()`: 先同步本地所有 Core。
        2.  `shmemi_barrier_npu_v3(team)`: 执行设备间同步（默认使用 V3 集中式算法）。
        3.  `shmemi_barrier_core()`: 再次同步本地所有 Core，确保所有 Core 都感知到设备间同步完成。

## 3. 初始化与资源分配
Barrier 的正常运行依赖于初始化阶段分配的全局共享内存池。

### 3.1 初始化流程
在 `shmem_init_attr` -> `shmemi_team_init` 过程中，系统会预分配用于同步的内存资源：
*   **Sync Pool**: `shmemi_team_init_sync_pool` 分配 `SYNC_POOL_SIZE` 大小的内存，用于存放同步信号数组（Array）。
*   **Sync Counter**: `shmemi_team_init_sync_counter` 分配计数器内存，用于记录 Barrier 的轮次（Count）。
*   **Core Sync Pool**: 类似地，为 Device 内的 Core 级同步分配内存。

### 3.2 通信域（Team）设置
*   **World Team**: 初始化时默认创建 `SHMEM_TEAM_WORLD`，包含所有参与计算的 Rank。
*   **Team 管理**: `shmemi_team_t` 结构体维护了 Team 的拓扑信息：
    *   `mype`: 当前 Rank 在 Team 内的 ID。
    *   `start`: Team 起始的全局 PE ID。
    *   `stride`: Team 成员的全局 ID 跨度。
    *   `size`: Team 成员数量。
*   **Sync Array 映射**: 每个 Team 根据其 `team_idx` 在全局 Sync Pool 中拥有独立的偏移量，避免不同 Team 间的同步冲突。

## 4. Barrier 算法流程
代码库中实现了多种 Barrier 算法，针对不同规模和场景进行优化。

### 4.1 算法 V1: 基础 Dissemination
*   **复杂度**: O(N)
*   **逻辑**: 简单的环形同步，每个 PE 通知下一个 PE，并等待上一个 PE。

### 4.2 算法 V2: Group Dissemination (分组传播)
*   **复杂度**: O(log_k N)
*   **逻辑**: 
    *   将同步过程分为多轮（Round）。
    *   每轮中，Rank 之间的距离指数级增加。
    *   每个 Rank 向距离为 `stride * k^round` 的 Rank 发送信号。
    *   适合大规模集群，减少同步延迟。

### 4.3 算法 V3: Centralized (集中式/Pull Mode) - **当前默认**
*   **复杂度**: 时间 O(N/K)，空间 O(1)
*   **适用场景**: 小规模集群（如 8 卡），性能优于 Dissemination。
*   **逻辑**:
    *   所有 Rank（Vector Core 0）更新本地的 `sync_counter`。
    *   **Pull Mode**: 每个 Rank 主动去读取（Wait）其他 Rank 的信号，而不是等待别人写自己。
    *   利用 `shmem_ptr(sync_array, remote_pe)` 获取远端地址，通过 `shmemi_signal_wait_until_eq_for_barrier` 轮询等待。

## 5. 底层通信机制
Barrier 的底层实现依赖于对全局共享内存（Global Memory, GM）的原子读写和缓存一致性操作。

1.  **信号设置 (`Signal Set`)**: 
    *   使用 `shmemi_signal_set` 或 `gm_store` 将 Barrier 轮次计数（Count）写入目标地址。
2.  **信号等待 (`Signal Wait`)**:
    *   使用 `shmemi_signal_wait_until_eq_for_barrier`。
    *   **DCCI (Data Cache Clean and Invalidate)**: 在循环检查信号前，必须调用 `gm_dcci` 刷新数据缓存，确保读取到最新的全局内存值，防止读取到 L1/L2 缓存中的旧值。
3.  **地址计算**:
    *   `shmem_ptr(addr, pe)`: 计算远端 PE 的全局内存地址。这是 SHMEM 库的核心能力，基于初始化时建立的内存映射表。

## 6. MoE Combine Dispatch 中的 Barrier 应用示例

在 Mixture of Experts (MoE) 模型的 `Combine Dispatch` 阶段，需要对不同 Expert 的 Token 进行重排和分发，这是一个典型的 All-to-All 通信模式。

### 6.1 场景描述
在 `examples/dispatch_gmm_combine` 中，算子需要将本地 Token 发送到持有对应 Expert 的远端 Rank。

### 6.2 核心同步代码分析 (`sync_util.h` -> `CrossRankSync`)
该示例没有直接调用 `shmem_barrier` 高级接口，而是手动实现了基于共享内存的同步逻辑，以获得更细粒度的控制：

```cpp
FORCE_INLINE_AICORE void CrossRankSync(GM_ADDR symmetricPtr, int32_t rank, int32_t rankSize)
{
    // 1. 获取同步基地址和计数器
    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET;
    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET + 1024;
    
    // 2. 增加同步轮次 (Count)
    int count = gm_load(sync_base) + 1;
    
    // 3. 多核并行发送信号 (Signal)
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    
    // 每个 Core 负责通知一部分远端 Rank
    for (int i = vec_id; i < rankSize; i += vec_size) {
        // 计算远端 Rank 的同步标志位地址
        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(shmem_ptr(symmetricPtr, i)) + FLAG_OFFSET + rank * 16;
        
        // 写信号：通知 Rank i，"我 (Rank rank) 已经到达同步点"
        gm_store(sync_remote, count);
        
        // 刷新缓存，确保写入可见
        gm_dcci((__gm__ uint8_t*)sync_remote);
        
        // 4. 等待信号 (Wait)
        // 等待 Rank i 通知 "它也到达了同步点"
        auto sync_check = sync_counter + i * 16;
        gm_signal_wait_until_eq_for_barrier(sync_check, count);
    }

    // 5. 本地 Core 间同步，确保所有 Core 都完成了跨设备同步
    AscendC::SyncAll<true>();
    
    // 6. 更新本地同步轮次记录
    gm_store(sync_base, count);
}
```

### 6.3 流程总结
1.  **计算 Count**: 确定当前是第几次同步。
2.  **Notify Remote**: 利用 `shmem_ptr` 获得远端指针，写入当前 Count。
3.  **Wait Remote**: 轮询本地内存中的特定位置（该位置由远端 Rank 写入），直到值变为 Count。
4.  **Local Sync**: 使用 `AscendC::SyncAll` 确保 Device 内所有 Core 步调一致。

这种实现方式实现了全互连（All-to-All）的同步模式，确保了在开始数据交换前，所有 Rank 的所有 Core 都已就绪。
