# Switch Barrier Accelerator Implementation

## Overview
This document describes the implementation of a switch-based barrier accelerator offload for the SHMEM library. The accelerator uses an In-Network Computing capable switch to aggregate barrier signals and broadcast release signals, reducing synchronization latency in large-scale clusters.

## 1. Data Structure Changes

### `shmemi_team_t` (include/internal/host_device/shmemi_types.h)
The team structure is extended to support switch-specific configuration:

```cpp
typedef struct {
    // ... existing fields ...
    // Switch Barrier Accelerator Fields
    int use_switch_barrier;        // Enable flag
    uint32_t switch_group_id;      // Hardware Group ID on switch
    uint64_t switch_trigger_addr;  // Switch MMIO/Doorbell address for trigger
    uint32_t switch_rkey;          // Remote Key for switch memory
} shmemi_team_t;
```

## 2. Control Plane Implementation

### Switch Configuration (`src/host/team/shmem_team.cpp`)
A mock `SwitchBarrierController` is implemented to simulate interaction with the switch control plane.

*   **Initialization**: When a team is created (World or Split), the `setup_switch_barrier` function is called.
*   **Logic**:
    1.  Checks `SHMEM_ENABLE_SWITCH_BARRIER` environment variable.
    2.  Allocates a `switch_group_id`.
    3.  Calculates `flag_vaddr` (Sync Counter address) for each member rank.
    4.  Configures the switch via `SwitchBarrierController::initialize_switch_group`.
    5.  Populates `shmemi_team_t` fields.
    6.  **IOMMU Configuration**: A placeholder is added to map the `switch_trigger_addr` (physical/bus address) into the Device's virtual address space (e.g., via `aclrtMapMem`), ensuring the NPU can access it.

## 3. Device Data Plane Implementation

### Low-Level Interface
The implementation uses `shmemi_store` directly to trigger the switch. This relies on the `switch_trigger_addr` being mapped to the NPU's global memory address space (e.g., via MMIO or PCIe BAR).

### Barrier Algorithm v4 (`include/internal/device/sync/shmemi_device_barrier.h`)
A new barrier algorithm `shmemi_barrier_npu_v4` is implemented:

1.  **Check**: If `use_switch_barrier` is false, fallback to v3 (Software Barrier).
2.  **Trigger**:
    *   Rank 0 (of the block/vec) constructs a payload: `[Group ID | Sequence Count]`.
    *   Writes payload directly to `team->switch_trigger_addr` using `shmemi_store`.
    *   Flushes the cache line to ensure visibility.
3.  **Wait**:
    *   Polls the local `sync_counter` (Flag).
    *   The switch accelerator, upon receiving triggers from all group members, broadcasts a Remote Store to update this flag.

## 4. Usage
To enable the switch barrier:
```bash
export SHMEM_ENABLE_SWITCH_BARRIER=1
./your_shmem_application
```
