#ifndef SHMEM_SWITCH_DRIVER_H
#define SHMEM_SWITCH_DRIVER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t SwitchHandle;

class SwitchDriver {
public:
    /**
     * @brief Create a new multicast/barrier object in the switch fabric.
     * 
     * @param group_size Number of members participating in this barrier group.
     * @param handle Output: Handle to the created switch object.
     * @return 0 on success, error code otherwise.
     */
    static int BarrierObjectCreate(uint32_t group_size, SwitchHandle* handle);

    /**
     * @brief Register a device to the barrier object.
     * 
     * @param handle Handle to the barrier object.
     * @param dev_id Physical device ID (e.g., logical ID or PCIe BDF).
     * @param rank The rank of this device within the barrier group [0, group_size-1].
     * @return 0 on success, error code otherwise.
     */
    static int BarrierObjectAddDevice(SwitchHandle handle, uint32_t dev_id, uint32_t rank);

    /**
     * @brief Bind physical memory to the barrier object for callback signaling.
     * 
     * When the barrier is reached, the switch will write to this address.
     * 
     * @param handle Handle to the barrier object.
     * @param dev_id The device ID associated with the memory.
     * @param phys_addr Physical address (or IOVA) of the sync_counter on the device.
     * @param size Size of the memory region to bind.
     * @return 0 on success, error code otherwise.
     */
    static int BarrierObjectBindMem(SwitchHandle handle, uint32_t dev_id, uint64_t phys_addr, size_t size);

    /**
     * @brief Get a virtual address for triggering the barrier.
     * 
     * The device triggers the barrier by writing to this address.
     * 
     * @param handle Handle to the barrier object.
     * @param trigger_va Output: Virtual address mapped to the switch trigger mechanism.
     * @return 0 on success, error code otherwise.
     */
    static int BarrierObjectGetTriggerAddr(SwitchHandle handle, uint64_t* trigger_va);
    
    /**
     * @brief Destroy the barrier object and release resources.
     * 
     * @param handle Handle to the barrier object to destroy.
     * @return 0 on success, error code otherwise.
     */
    static int BarrierObjectDestroy(SwitchHandle handle);
};

#ifdef __cplusplus
}
#endif

#endif // SHMEM_SWITCH_DRIVER_H
