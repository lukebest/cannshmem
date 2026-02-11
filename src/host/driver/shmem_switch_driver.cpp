#include "internal/host_device/shmem_switch_driver.h"
#include <iostream>
#include <vector>
#include <map>
#include <mutex>

struct SwitchObject {
    uint32_t group_id;
    uint32_t group_size;
    std::vector<uint32_t> device_ids;
    std::map<uint32_t, uint64_t> bound_memory; 
    uint64_t trigger_va;
};

static std::map<SwitchHandle, SwitchObject*> g_objects;
static std::mutex g_mutex;
static uint32_t g_next_group_id = 100;

int SwitchDriver::BarrierObjectCreate(uint32_t group_size, SwitchHandle* handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    SwitchObject* obj = new SwitchObject();
    obj->group_size = group_size;
    
    obj->group_id = g_next_group_id++;
    
    *handle = (SwitchHandle)obj;
    
    g_objects[*handle] = obj;
    
    return 0;
}

int SwitchDriver::BarrierObjectAddDevice(SwitchHandle handle, uint32_t dev_id, uint32_t rank) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_objects.find(handle) == g_objects.end()) return -1;
    
    SwitchObject* obj = g_objects[handle];
    if (rank >= obj->group_size) return -1;
    
    return 0;
}

int SwitchDriver::BarrierObjectBindMem(SwitchHandle handle, uint32_t dev_id, uint64_t phys_addr, size_t size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_objects.find(handle) == g_objects.end()) return -1;
    
    SwitchObject* obj = g_objects[handle];
    obj->bound_memory[dev_id] = phys_addr;
    
    return 0;
}

int SwitchDriver::BarrierObjectGetTriggerAddr(SwitchHandle handle, uint64_t* trigger_va) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_objects.find(handle) == g_objects.end()) return -1;
    
    SwitchObject* obj = g_objects[handle];
    
    if (obj->trigger_va == 0) {
        obj->trigger_va = 0xBA00000000000000 + (uint64_t)obj->group_id * 0x1000;
    }
    
    *trigger_va = obj->trigger_va;
    return 0;
}

int SwitchDriver::BarrierObjectDestroy(SwitchHandle handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_objects.find(handle) == g_objects.end()) return -1;
    
    delete g_objects[handle];
    g_objects.erase(handle);
    return 0;
}
