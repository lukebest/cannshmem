# Dispatch GMM Combine 算子分析与同步机制解析

本文档详细解析 `examples/dispatch_gmm_combine` 目录下的 Dispatch-GMM-Combine 算子流程，重点分析其中的同步机制，并探讨使用 SHMEM 标准 Barrier 接口替代手写同步函数的可行性。

## 1. 算子流程详解

Dispatch-GMM-Combine 是混合专家模型（MoE）中的核心计算模块，其主要流程包含三个阶段：**Dispatch（分发）**、**GMM（计算）** 和 **Combine（聚合）**。该算子利用共享内存（SHMEM）技术实现了跨设备的高效数据传输与计算。

### 1.1 总体架构

代码入口位于 `examples/dispatch_gmm_combine/include/dispatch_gmm_combine.h`，核心类为 `DispatchGmmCombineKernel`。

*   **AIC (AI Core)**: 主要负责矩阵乘法（GMM1, GMM2）。
*   **AIV (AI Vector Core)**: 主要负责数据搬运、分发（Dispatch）、聚合（Combine）以及部分后处理（Epilogue）。

### 1.2 详细步骤

#### 阶段一：Dispatch (分发) - AIV 执行
该阶段的目标是将本地 Token 根据路由信息发送到持有对应 Expert 的远端 Rank。

1.  **MoE 初始化与路由 (`moe_init_routing_quant_v2`)**:
    *   计算每个 Token 应该发送到哪个 Expert。
    *   生成 Token 排序索引 (`expandedRowIdx`)。
    *   统计本地每个 Expert 的 Token 数量。
2.  **Token 数量全交换 (All-to-All for Token Counts)**:
    *   计算本地将要发送给远端每个 Expert 的 Token 数量。
    *   通过 `shmem_ptr` 直接将这些数量写入远端 Rank 的 `ptrPeerTokenPerExpert` 内存区域。
    *   **同步点**: 写入完成后需要同步，确保远端 Rank 能看到最新的 Token 数量。
3.  **计算接收偏移量 (`GetCumsumForMMAIV`)**:
    *   每个 Rank 统计自己将从所有其他 Rank 接收多少 Token，计算前缀和（Cumsum），为后续接收数据做准备。
4.  **数据分发 (Data Dispatch)**:
    *   根据计算出的偏移量，从本地 `ptrA` 读取 Token 数据。
    *   通过 `shmem_ptr` 将 Token 数据直接写入目标 Rank 的 `peermemInfo.ptrA`。
    *   **同步点**: 数据写入完成后需要同步。

#### 阶段二：GMM (通用矩阵乘) - AIC 执行
该阶段在 AIC 上执行，对接收到的 Token 进行计算。

1.  **GMM1 (第一次矩阵乘)**:
    *   等待 AIV 完成 Dispatch 阶段的数据准备。
    *   对每个 Expert 的数据执行 GEMM 运算：`Input * Weight1 = Result1`。
    *   使用 `CrossCoreWaitFlag` 等待 AIV 的信号。
2.  **GMM2 (第二次矩阵乘)**:
    *   对 Result1 执行激活函数等操作后，执行第二次 GEMM：`Result1 * Weight2 = Output`。
    *   计算结果写入本地 buffer。

#### 阶段三：Combine (聚合) - AIV 执行
该阶段将计算结果传回源 Rank 并恢复原始顺序。

1.  **结果回传 (Data Combine)**:
    *   每个 Rank 将计算结果（位于本地）发送回 Token 原本所属的 Rank。
    *   同样使用 `shmem_ptr` 直接写入源 Rank 的 `peermemInfo.ptrD`。
    *   **同步点**: 写入前需要确保目标缓冲区可用，写入后需要通知目标 Rank。
2.  **Token 反重排 (`MoeTokenUnpermute`)**:
    *   接收到所有结果后，根据之前的排序索引，将 Token 恢复到原始顺序，得到最终输出。
3.  **最终同步**:
    *   `CrossRankSync`：确保所有 Rank 完成所有操作。

---

## 2. 同步机制解析

在分布式并行计算中，同步至关重要，用于协调不同计算节点（Rank）和不同计算单元（Core）之间的进度。

### 2.1 什么时候需要同步？

1.  **写后读（RAW）依赖**:
    *   当 Rank A 向 Rank B 的内存写入数据后，Rank B 需要读取该数据。
    *   Rank A 必须在写入完成后发出信号（Signal）。
    *   Rank B 必须等到信号（Wait）后才能读取，否则可能读到旧数据或无效数据。
2.  **读后写（WAR）依赖**:
    *   当 Rank B 正在读取某块缓冲区时，Rank A 不能覆盖该缓冲区。
    *   通常通过 Barrier 保证一轮通信彻底完成后，再进入下一轮。
3.  **全卡同步（Global Barrier）**:
    *   在进入下一个主要阶段（如从 Dispatch 切换到 GMM）之前，所有 Rank 必须完成当前阶段的任务。
    *   例如：只有所有 Rank 都把 Token 发送到了目的地，计算节点才能开始 GMM 计算，否则数据不全。

### 2.2 `CrossRankSync` 与 `CrossRankSyncV1` 解析

在 `include/sync_util.h` 中定义的这两个函数是手动实现的 **全卡全核同步（Global Device Barrier）**。

#### `CrossRankSync` 实现原理
它实现了一个 **All-to-All** 的握手同步：
1.  **增加计数器**: 每个 Rank 维护一个本地的 `count`，表示当前是第几次同步。
2.  **发送信号 (Signal All)**:
    *   每个 Rank 遍历所有其他 Rank (`rankSize`)。
    *   使用 `shmem_ptr` 获取远端 Rank 的同步标志位地址 `sync_remote`。
    *   将自己的 `count` 写入所有远端的 `sync_remote`。
    *   调用 `gm_dcci` 刷新缓存，确保写入对远端可见。
3.  **等待信号 (Wait All)**:
    *   每个 Rank 轮询本地的一个标志位数组 `sync_check`。
    *   直到数组中 **所有** Rank 对应的位置都更新为当前的 `count`。
    *   这意味着：**我知道所有人都到了**。
4.  **核间同步**: `AscendC::SyncAll<true>()` 确保 Device 内所有 Core 都完成了上述过程。

**`CrossRankSyncV1`** 的逻辑基本一致，只是在同步前多加了一次 `SyncAll`，并微调了基地址偏移。

**本质**: 这是一个软件实现的 Dissemination Barrier (All-to-All 模式)，复杂度为 O(N)，其中 N 是 Rank 数。

---

## 3. 使用 `shmem_barrier` 替代的可行性方案

### 3.1 结论
**可以替代**。`CrossRankSync` 的功能与 SHMEM 库提供的标准 `shmem_barrier` 完全对应。且 SHMEM 库的实现通常经过了更深度的硬件适配和算法优化。

### 3.2 替代方案

#### 对应关系
*   **`CrossRankSync`** (手动实现) $\rightarrow$ **`shmem_barrier(SHMEM_TEAM_WORLD)`** (标准接口)
*   **`CrossRankSyncV1`** $\rightarrow$ **`shmem_barrier(SHMEM_TEAM_WORLD)`**

#### 为什么推荐替换？
1.  **性能更优**: SHMEM 库中的 `shmemi_barrier` 默认采用 **V3 (Centralized/Pull Mode)** 算法（见 `shmemi_barrier_npu_v3_analysis.md`），在小规模集群下复杂度为 O(N/K)，优于 `CrossRankSync` 的 O(N) 轮询。
2.  **硬件加速**: SHMEM 库可能利用硬件特性（如原子操作优化、硬件 Barrier 卸载）来减少 CPU/NPU 开销。
3.  **代码简洁**: 减少了用户侧维护复杂同步逻辑的负担，降低了死锁和缓存一致性错误的风险。
4.  **可移植性**: 标准接口屏蔽了底层硬件差异。

#### 替换步骤

1.  **引入头文件**:
    确保包含 SHMEM API 头文件：
    ```cpp
    #include "shmem_api.h"
    ```

2.  **替换调用**:
    将代码中所有的：
    ```cpp
    CrossRankSync(params.symmetricPtr, params.rank, params.EP);
    // 或
    CrossRankSyncV1(params.symmetricPtr, params.rank, params.EP);
    ```
    替换为：
    ```cpp
    shmem_barrier(SHMEM_TEAM_WORLD);
    ```
    *注意：如果算子使用了自定义的通信域 (Team)，则应传入对应的 `team_id`。在 demo 中通常是 `SHMEM_TEAM_WORLD`。*

3.  **删除冗余代码**:
    可以删除 `include/sync_util.h` 中关于 `CrossRankSync` 的定义，以及 `dispatch_gmm_combine.h` 中传递 `symmetricPtr` 用于同步的部分逻辑（如果 `symmetricPtr` 仅用于同步的话）。

### 3.3 关于 `shmemx_barrier_vec`
*   **`shmemx_barrier_vec`** 是 **Vector Core 专用** 的 Barrier。
*   如果 `DispatchGmmCombineKernel` 仅在 AIV (Vector Core) 上运行（如 `template<> void operator()<AscendC::AIV>`），且需要与其他 Rank 的 AIV 同步，可以使用 `shmemx_barrier_vec`。
*   **但是**，MoE 场景通常涉及 AIC (Matmul) 和 AIV (Dispatch/Combine) 的协同。`CrossRankSync` 位于 Host/Global 层面，通常作为全设备同步点。
*   **建议**: 优先使用通用的 `shmem_barrier`，除非你明确只需要同步 Vector Core 而不关心 AI Core 的状态（例如在 overlap 流水线中）。对于 `dispatch_gmm_combine.h` 这种全流程同步，`shmem_barrier` 是最安全的直接替代品。

### 3.4 示例代码修改

**原代码 (`dispatch_gmm_combine.h`):**
```cpp
// Combine 结束处
AscendC::SyncAll<true>();
blockEpilogue.Finalize();
CrossRankSync(params.symmetricPtr, params.rank, params.EP); // <--- 待替换
MoeTokenUnpermuteTilingData tilingData;
```

**修改后:**
```cpp
// Combine 结束处
AscendC::SyncAll<true>();
blockEpilogue.Finalize();
shmem_barrier(SHMEM_TEAM_WORLD); // <--- 使用标准 Barrier
MoeTokenUnpermuteTilingData tilingData;
```
