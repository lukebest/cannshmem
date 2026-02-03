# SHMEM Barrier/Barrier_all 实现详细分析

## 架构概述

Barrier实现采用**三级层次结构**：

```
用户应用程序
       ↓
Device Barrier API
       ↓
Level 1: 核心级同步（设备内部）
       ↓
Level 2: NPU级同步（设备之间）
       ↓
Level 3: 跨主机同步（主机之间 - 待实现）
```

---

## Device端Barrier流程

**入口点：** `include/device/shmem_device_sync.h:13` 的 `shmem_barrier(team_id)`

### 分步执行流程：

1. **调用模板函数** (`shmemi_barrier<false>`)
   - 验证PE是否在团队中
   - 获取团队配置（start、stride、size）

2. **Level 1: 核心级同步**（第一次）
   - 使用 `shmemi_barrier_core<false>()`
   - 硬件：`AscendC::SyncAll<false>()` 用于所有核心
   - 回退：如果硬件不可用，使用软件传播barrier

3. **Level 2: NPU级同步**（仅当ASCEND_IS_AIV时）
   - 实现**集中式barrier（拉取模式）**
   - 每个PE递增共享计数器：`count = sync_counter + 1`
   - 对团队中的每个对等节点（K=8并行读取）：
     - **自己**：`shmemi_signal_set(sync_array, count)` - 写本地
     - **对等节点**：`shmemi_signal_wait_until_eq_for_barrier(peer_sync_array, count)` - 读远程
   - 存储新的计数器值
   - **复杂度**：O(N/K) 时间，O(1) 空间
   - 文件：`include/internal/device/sync/shmemi_device_barrier.h:254-291`

4. **Level 1: 核心级同步**（最后一次）
   - 与步骤2相同 - 确保所有核心完成NPU同步

---

## Host端Barrier流程

**入口点：** `src/host/sync/shmemi_sync.cpp:10` 的 `shmem_barrier(team_id)`

1. **包装函数**
   - 调用 `shmemi_barrier_on_stream(team_id, nullptr)`

2. **启动设备内核** (`src/device/shmemi_barrier.cpp:14-18`)
   ```cpp
   k_shmem_barrier<<<1, nullptr, stream>>>((int32_t)tid);
   ```

3. **同步流**
   - `aclrtSynchronizeStream(stream)` - 等待内核完成

---

## 控制网络Barrier（Host操作）

**用途：** 初始化、内存分配操作
**入口：** `src/host/init/shmem_init.cpp:245` 的 `shmemi_control_barrier_all()`

### 流程：

1. **所有Rank递增计数器**
   - `store_->Add(addKey, 1, val)` - 原子递增
   - `val` = 到达顺序（1到size）

2. **最后一个Rank发出完成信号**
   - 如果 `val == size`：`store_->Set(waitKey, SMEM_GROUP_SET_STR)`

3. **所有Rank等待**
   - `store_->Get(waitKey, getVal, timeout)` - 轮询直到完成
   - 如果barrier挂起则超时保护

4. **清理**（由第一个Rank）
   - 定期删除旧的barrier键

文件：`src/memfabric_hybrid/src/smem/csrc/net/smem_net_group_engine.cpp:75-125`

---

## 关键同步原语

### 1. Signal Set (`include/internal/device/sync/shmemi_device_p2p.h:14-20`)
```cpp
shmemi_signal_set(addr, val)
→ shmemi_store(addr, val)
→ dcci_cacheline(addr)  // 刷新缓存到GM
```

### 2. Signal Wait (`include/internal/device/sync/shmemi_device_p2p.h:56-65`)
```cpp
shmemi_signal_wait_until_eq_for_barrier(sig_addr, cmp_val)
→ do {
    dcci_cacheline(sig_addr)  // 从GM加载
    if (*sig_addr == cmp_val) return;  // 就绪
    if (*sig_addr == cmp_val + 1) return;  // 已在下一个barrier
  } while (true)
```

### 3. 缓存一致性 (`include/internal/device/shmemi_device_arch.h`)
- `dcci_cacheline(addr)`：清理并使单个缓存行失效
- `dsb_all()`：数据同步barrier
- `dcci_entire_cache()`：刷新整个缓存（在quiet中使用）

---

## 内存布局

**同步数组**（定义在 `include/internal/host_device/shmemi_types.h`）：
```cpp
#define SHMEM_BARRIER_TG_DISSEM_KVAL 8  // 并行读取因子
#define SYNC_LOG_MAX_RANKS 5            // 16384个rank的最大轮数
#define SYNC_ARRAY_SIZE (64 * 5 * 8)    // 每个团队
#define SYNC_COUNTER_SIZE 64            // 每个团队
```

**状态结构**：
```cpp
shmemi_device_host_state_t {
    uint64_t sync_pool;          // NPU级同步数组
    uint64_t sync_counter;       // NPU级计数器
    uint64_t core_sync_pool;     // 核心级同步数组
    uint64_t partial_barrier_pool;  // 部分barrier
}
```

---

## 部分Barrier

**API：** `include/internal/device/sync/shmemi_device_partial_barrier.h` 的 `shmemx_partial_barrier(team_id, pes, count)`

**机制**：
- 轮询式槽分配（每个团队64个槽）
- 仅同步 `pes` 数组中的PE
- 使用与完整barrier相同的三级流程
- 由vec_id 0管理槽（循环时重置）

---

## 跨主机Barrier

**实现：** `include/internal/device/sync/shmemi_device_handle.h:14-85` 的 `shmemi_barrier_cross_host()`

**机制**：
- 只有vec_id 0参与
- 使用**传播算法**（O(log N)时间，O(N)空间）
- `shift` 每轮翻倍：1, 2, 4, 8, ...
- 使用基于ROCE的信令：`shmemi_highlevel_signal_set()`

**注意：** Level 3 (`shmemi_barrier_sys`) 标记为"待实现"，用于host端跨主机同步。

---

## 关键设计决策

1. **集中式 vs 传播Barrier**
   - v3使用集中式（拉取模式） - 小规模（≤8个rank）性能更好
   - v1/v2使用传播 - 大规模性能更好

2. **硬件 vs 软件同步**
   - 首选：`AscendC::SyncAll<>()`（硬件）
   - 回退：VEC核心之间的软件传播

3. **缓存一致性**
   - 所有使用 `dcci_cacheline()` 确保可见性
   - 缓存行对齐的数据防止伪共享

4. **Barrier类型**
   - `shmem_barrier`：所有核心（包括标量单元）
   - `shmemx_barrier_vec`：仅VEC核心（计算-通信重叠）

---

## 相关文件

| 文件 | 描述 |
|------|-------------|
| `/home/luke/shmem/include/device/shmem_device_sync.h` | Device barrier API定义 |
| `/home/luke/shmem/include/internal/device/sync/shmemi_device_barrier.h` | 内部barrier实现（核心+NPU级别） |
| `/home/luke/shmem/include/internal/device/sync/shmemi_device_partial_barrier.h` | 部分barrier实现 |
| `/home/luke/shmem/include/internal/device/sync/shmemi_device_p2p.h` | Signal/wait同步原语 |
| `/home/luke/shmem/include/internal/device/sync/shmemi_device_handle.h` | 跨主机barrier处理 |
| `/home/luke/shmem/include/internal/device/sync/shmemi_device_quiet.h` | Quiet/fence操作 |
| `/home/luke/shmem/include/internal/device/shmemi_device_arch.h` | 缓存一致性指令（dcci_cacheline, dsb_all） |
| `/home/luke/shmem/include/internal/host_device/shmemi_types.h` | 数据结构和常量 |
| `/home/luke/shmem/include/host/shmem_host_sync.h` | Host barrier API定义 |
| `/home/luke/shmem/src/host/sync/shmemi_sync.cpp` | Host barrier实现 |
| `/home/luke/shmem/src/device/shmemi_barrier.cpp` | Barrier的设备内核启动 |
| `/home/luke/shmem/src/host/init/shmem_init.cpp` | 初始化的控制barrier |
| `/home/luke/shmem/src/memfabric_hybrid/src/smem/csrc/net/smem_net_group_engine.cpp` | 控制网络barrier实现 |

---

## 限制和注意事项

### API使用限制 (`include/device/shmem_device_sync.h:12-32`)

1. Barrier API只能在MIX内核中使用（编译器不能优化为VEC或CUBE）
2. Barrier API与SyncAll冲突 - 避免混合使用它们
3. 两种类型的barrier：
   - `shmem_barrier_xxx`：所有核心的barrier（包括标量单元）
   - `shmemx_barrier_xxx_vec`：仅VEC核心的barrier（适用于通信-计算重叠）
4. 立方体核心的标量单元不受 `shmem_barrier_xxx` 影响

### 时间复杂度

- **核心级同步**：O(1)（硬件）或 O(log N)（软件）
- **NPU级同步**：O(N/K) 时间，O(1) 空间（集中式barrier v3）
- **跨主机同步**：O(log N) 时间，O(N) 空间（传播barrier）

### 内存开销

- 每个团队：SYNC_ARRAY_SIZE + SYNC_COUNTER_SIZE = 64*5*8 + 64 = 320 + 64 = 384 字节
- 核心级同步池：SHMEM_CORE_SYNC_POOL_SIZE
- 部分barrier池：SHMEM_PARTIAL_BARRIER_POOL_SIZE