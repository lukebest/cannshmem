# shmemi_barrier_npu_v3 函数深度解析

`shmemi_barrier_npu_v3` 是 SHMEM 库中实现的一种**集中式 Pull 模式**（Centralized Pull Mode）的设备间同步屏障（Barrier）算法。相比于传统的 Dissemination Barrier（分发式屏障），该算法在小规模集群（如 8 ranks）下具有更好的性能表现，其时间复杂度为 $O(N/K)$，空间复杂度为 $O(1)$。

## 1. 函数概览

**函数签名**：
```cpp
SHMEM_DEVICE void shmemi_barrier_npu_v3(shmemi_team_t *team)
```

**核心特征**：
*   **模式**：Pull Mode（拉取模式）。
*   **机制**：每个 Rank 主动查询（Read/Wait）所有其他 Rank 的状态，而不是等待别人发送信号。
*   **并行度**：利用 Device 侧的多个 Vector Core 并行检查其他 Rank 的状态。

## 2. 核心逻辑流程

该函数的主要执行流程如下：

1.  **初始化与上下文获取**：获取当前核的 ID、Block 数量、Team 信息以及同步对象指针。
2.  **计算目标屏障计数值**：读取当前的同步计数器并加 1，作为本次屏障的目标值。
3.  **更新本地状态**：将本地的同步信号（`sync_array`）更新为目标值，表明自己已到达屏障。
4.  **拉取（Pull）其他 Rank 状态**：遍历 Team 中的其他所有 Rank，轮询检查它们的同步信号是否也达到了目标值。
5.  **更新同步计数器**：所有检查通过后，更新本地的 `sync_counter`，完成屏障。

## 3. Pull Mode 机制详解

在 `shmemi_barrier_npu_v3` 中，"Pull Mode" 的具体实现体现在**循环检测**逻辑中。

### 3.1 任务分配
为了加速同步过程，代码将检查 $N$ 个 Rank 的任务分配给了当前 Device 上的多个 Vector Core。

```cpp
int vec_id = AscendC::GetBlockIdx();
// ...
int k = SHMEM_BARRIER_TG_DISSEM_KVAL; // 并行度步长
// ...
for (int i = vec_id; i < size; i += k) {
    // 循环遍历 Team 中的所有成员
}
```

### 3.2 逻辑分支
在循环中，根据 `i` 是否指向自己，分为两种操作：

*   **如果是自己 (Write Local)**：
    把自己在全局内存（GM）中的信号量 `sync_array` 更新为 `count`。这相当于对外“广播”自己已经到达了屏障。
    ```cpp
    if (i == my_pe_in_team) {
        // write local
        shmemi_signal_set((__gm__ int32_t *)sync_array, count);
    }
    ```

*   **如果是他人 (Read Remote / Pull)**：
    计算目标 Rank 的远程地址，并**主动等待**该地址的值变为 `count`。这就是 "Pull" 的核心——主动去读别人的状态，而不是被动等别人发消息。
    ```cpp
    else {
        // read remote
        int remote_pe = start + i * stride;
        // 核心等待逻辑
        shmemi_signal_wait_until_eq_for_barrier(
            (__gm__ int32_t *)shmemi_ptr(sync_array, remote_pe), // 获取远程 PE 的地址指针
            count // 目标值
        );
    }
    ```

## 4. 如何 Wait (等待机制)

Wait 操作是通过 `shmemi_signal_wait_until_eq_for_barrier` 函数实现的。该函数通过**自旋轮询（Spin-Polling）**的方式等待远程内存的值变化。

**代码位置**：`include/internal/device/sync/shmemi_device_p2p.h`

**实现细节**：
```cpp
SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        // 1. 刷新缓存行 (Invalidate Cache)
        // 确保读取的是 GM 中的最新值，而不是 L1/L2 Cache 中的陈旧值
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        // 2. 检查条件 (Compare)
        if (*sig_addr == cmp_val) {
            return *sig_addr; // 等到了目标值
        }

        // 3. 防死锁机制 (Forward Progress)
        // 如果对方已经进入了下一个屏障 (cmp_val + 1)，也视为本轮同步成功
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true); // 自旋循环

    return -1;
}
```

*   **`dcci_cacheline`**：这是关键指令。由于是跨设备访问，必须强制使 Cache 失效，从总线/内存拉取最新数据。
*   **自旋**：函数在一个 `do...while(true)` 循环中不断重试，直到条件满足。

## 5. 如何更新本地 sync_counter

`sync_counter` 是一个用于记录屏障代数（Generation Count）的持久化变量，用于区分不同轮次的 Barrier 调用。

### 5.1 读取与预计算
在函数入口处，先读取当前值并加 1，计算出本轮 Barrier 的目标 `count`。
```cpp
// 读取 sync_counter 并 +1
int32_t count = shmemi_load((__gm__ int32_t *)sync_counter) + 1;
```

### 5.2 更新对外信号
这个 `count` 值首先被写入到 `sync_array` 中，作为对外展示的状态：
```cpp
shmemi_signal_set((__gm__ int32_t *)sync_array, count);
```

### 5.3 最终提交
当所有 Vector Core 都完成了对他人的检查（Pull 结束）后，整个 Barrier 操作完成。最后一步是将新的 `count` 写回 `sync_counter`，为下一次 Barrier 做准备。
```cpp
// 屏障结束，保存新的计数值
shmemi_store((__gm__ int32_t *)sync_counter, count);
```
此处的 `shmemi_store` 是一个原子性或强序的写操作，确保计数器的更新在逻辑上是最后一步。
