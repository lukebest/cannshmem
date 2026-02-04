# NPU级同步硬件卸载(Hardware Offload)修改方案

针对带有barrier原语硬件卸载功能的交换机场景，在全局共享内存池化下的P2P操作中，库中的barrier操作需要进行以下流程与代码修改。

## 1. 初始化流程的变化 (Initialization)

在初始化阶段，主要任务是探测硬件支持并配置硬件卸载所需的“魔术地址”和“状态字”。

*   **主机端 (Host-side):**
    *   **资源探测**：在 `shmem_init` 过程中，通过驱动接口（如 `rtGetC2cCtrlAddr`）查询当前集群是否支持交换机硬件屏障。
    *   **资源分配**：为全局团队（`SHMEM_TEAM_WORLD`）分配一个固定的硬件屏障槽位（Slot）或对应的全局共享内存地址（HW Barrier Address）。
    *   **配置传递**：将硬件屏障的基地址、槽位信息以及当前PE的身份信息存储在 `shmem_state` 中。在启动混合算子（MIX Kernel）前，通过 `shmemx_set_hardware_barrier_config` 将配置下发到NPU寄存器或特定内存区域。
*   **设备端 (Device-side):**
    *   在内核入口处调用 `shmemx_set_hardware_barrier_config`，使硬件同步单元（如FFTS）知晓屏障报文的目标地址。

## 2. Barrier 主操作的变化 (shmem_barrier)

原本的软件算法（如 $O(\log N)$ 的分发算法）将被替换为硬件触发模式：

*   **流程修改**：
    1.  **触发阶段**：不再进行多轮信号交换，而是直接调用 `shmemi_store(addr, val)`。这里的 `addr` 是交换机能够识别并拦截的特定全局池化地址，`val` 通常是一个递增的序号（Counter），用于区分不同的屏障调用。
    2.  **汇聚阶段**：交换机识别到所有参与NPU都发出了针对该地址的报文后，在内部完成计数汇聚。
    3.  **等待阶段**：调用硬件同步原语（如 `AscendC::HardwareBarrier`），使NPU指令流进入等待状态。
*   **硬件交互逻辑**：
    *   交换机收齐所有PE的barrier等待信号后，向所有参与的NPU发回一个“同步完成”的通知报文。
    *   NPU硬件接收到通知后，解除阻塞，继续执行后续指令。

## 3. 部分 Barrier (Partial Barrier) 的变化

对于子团队（Team）或部分PE参与的同步，变化在于对“槽位”和“预期计数”的管理：

*   **槽位分配**：硬件通常提供多个并发屏障槽位。每个 `shmem_team` 在创建时应关联一个唯一的硬件槽位地址。
*   **代码修改 (`shmemi_partial_barrier`)**：
    *   **地址偏移**：`shmemi_store` 的目标地址由 `base_addr + team_slot_offset` 组成，以区分不同的团队。
    *   **计数参数**：在某些硬件实现中，`shmemi_store` 的 `val` 可能需要包含参与同步的 PE 数量（`count`），以便交换机知道何时完成汇聚。
    *   **回退机制**：如果硬件槽位耗尽或参与的 PE 集合不符合硬件拓扑限制，代码应自动回退到传统的软件 Dissemination 实现。

## 4. 关键代码变更示例

### 设备端同步头文件修改
```cpp
// include/internal/device/sync/shmemi_device_barrier.h

template<bool is_aiv_only = true>
SHMEM_DEVICE void shmemi_barrier(shmem_team_t tid) {
    if (shmemi_is_hardware_barrier_enabled()) {
        // 1. 硬件卸载路径：由交换机识别 shmemi_store 报文并完成汇聚
        // 内部封装了对硬件地址的操作与等待逻辑
        AscendC::HardwareBarrier(tid); 
    } else {
        // 2. 软件路径：传统的软件分发算法
        shmemi_barrier_core<is_aiv_only>();
        shmemi_barrier_npu_v3(team); 
        shmemi_barrier_core<is_aiv_only>();
    }
}
```

### 存储原语增强
```cpp
// include/internal/device/shmemi_device_common.h

template<typename T>
SHMEM_DEVICE void shmemi_store_hardware_offload(__gm__ T *addr, T val) {
    // 交换机识别此 store 报文
    *((__gm__ T *)addr) = val;
    
    if (shmemi_is_hardware_barrier_enabled()) {
        // 额外触发硬件信号，辅助交换机进行报文汇聚识别
        AscendC::SignalHardwareForStoreOperation();
    }
}
```

## 5. 总结
引入硬件卸载后，**同步流程从“软件控制的多步点对点交互”转变为“硬件驱动的单步触发与被动等待”**。库的主要工作从“实现同步算法”转向“管理硬件同步资源（地址/槽位）”。这种修改能显著降低大集群下的同步延迟，将复杂度从 $O(\log N)$ 降低到接近 $O(1)$ 的网络往返时间。
