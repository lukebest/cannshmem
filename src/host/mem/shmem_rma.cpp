/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "host/shmem_host_rma.h"
#include "shmemi_device_rma.h"
#include "host_device/shmem_types.h"

using namespace std;
// shmem_ptr Symmetric?
void* shmem_ptr(void *ptr, int32_t pe)
{
    uint64_t lower_bound = (uint64_t)shm::g_state.p2p_heap_base[shmem_my_pe()];
    uint64_t upper_bound = lower_bound + shm::g_state.heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        SHM_LOG_ERROR("PE: " << shmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }
    void *mype_ptr = shm::g_state.p2p_heap_base[shmem_my_pe()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mype_ptr);
    if (shm::g_state.heap_base != nullptr) {
        return (void *)((uint64_t)shm::g_state.heap_base + shm::g_state.heap_size * pe + offset);
    }
    else {
        return nullptr;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    shm::g_state.mte_config.shmem_ub = offset;
    shm::g_state.mte_config.ub_size = ub_size;
    shm::g_state.mte_config.event_id = event_id;
    SHMEM_CHECK_RET(shm::update_device_state());
    return SHMEM_SUCCESS;
}

#define SHMEM_TYPE_PUT(NAME, TYPE)                                                                                              \
    /**                                                                                                                         \
    * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.                \
    *                                                                                                                           \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                                       \
    * @param source             [in] Pointer on local device of the source data.                                                \
    * @param nelems             [in] Number of elements in the destination and source arrays.                                   \
    * @param pe                 [in] PE number of the remote PE.                                                                \
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_put_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe) {                               \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem", SHMEMI_OP_PUT, NO_NBI,                   \
                                      (uint8_t *)dest, (uint8_t *)source, nelems, sizeof(TYPE), pe,               \
                                      1, 1,                                                                       \
                                      shm::g_state_host.default_stream,                                           \
                                      shm::g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                            \
            SHM_LOG_ERROR("device calling transfer failed");                                                      \
        }                                                                                                         \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT)
#undef SHMEM_TYPE_PUT

#define SHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                                          \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.                \
    *                                                                                                                           \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                                       \
    * @param source             [in] Pointer on local device of the source data.                                                \
    * @param nelems             [in] Number of elements in the destination and source arrays.                                   \
    * @param pe                 [in] PE number of the remote PE.                                                                \
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe) {             \
        int ret = shmemi_prepare_and_post_rma("shmem_put_" #NAME "_mem_nbi", SHMEMI_OP_PUT, NBI,                            \
                                      (uint8_t *)dest, (uint8_t *)source, nelems, sizeof(TYPE), pe,               \
                                      1, 1,                                                                       \
                                      shm::g_state_host.default_stream,                                           \
                                      shm::g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                            \
            SHM_LOG_ERROR("device calling transfer failed");                                                      \
        }                                                                                                         \
}

SHMEM_TYPE_FUNC(SHMEM_TYPE_PUT_NBI)
#undef SHMEM_TYPE_PUT_NBI

#define SHMEM_TYPE_GET(NAME, TYPE)                                                                                              \
    /**                                                                                                                         \
    * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.  \
    *                                                                                                                           \
    * @param dest               [in] Pointer on local device of the destination data.                                           \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                            \
    * @param nelems             [in] Number of elements in the destination and source arrays.                                   \
    * @param pe                 [in] PE number of the remote PE.                                                                \
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_get_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe) {                               \
        int ret = shmemi_prepare_and_post_rma("shmem_get_" #NAME "_mem", SHMEMI_OP_GET, NO_NBI,                             \
                                      (uint8_t *)dest, (uint8_t *)source, nelems, sizeof(TYPE), pe,               \
                                      1, 1,                                                                       \
                                      shm::g_state_host.default_stream,                                           \
                                      shm::g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                            \
            SHM_LOG_ERROR("device calling transfer failed");                                                      \
        }                                                                                                         \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET)
#undef SHMEM_TYPE_GET

#define SHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                                          \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE. \
    *                                                                                                                           \
    * @param dest               [in] Pointer on local device of the destination data.                                           \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                            \
    * @param nelems             [in] Number of elements in the destination and source arrays.                                   \
    * @param pe                 [in] PE number of the remote PE.                                                                \
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_get_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe) {             \
        int ret = shmemi_prepare_and_post_rma("shmem_get_" #NAME "_mem_nbi", SHMEMI_OP_GET, NBI,                            \
                                      (uint8_t *)dest, (uint8_t *)source, nelems, sizeof(TYPE), pe,               \
                                      1, 1,                                                                       \
                                      shm::g_state_host.default_stream,                                           \
                                      shm::g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                            \
            SHM_LOG_ERROR("device calling transfer failed");                                                      \
        }                                                                                                         \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET_NBI)
#undef SHMEM_TYPE_GET_NBI

#define SHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                                          \
    /**
    * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param sig_addr          [in] Symmetric address of the signal word to be updated.
    * @param signal            [in] The value used to update sig_addr.
    * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations: SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
    * @param pe                [in] PE number of the remote PE.
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_put_##NAME##_mem_signal(TYPE* dst, TYPE* src, size_t elem_size,                                   \
                                                      uint8_t* sig_addr, int32_t signal, int sig_op, int pe){                   \
        int ret = shmem_putmem_signal_host((uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe,                              \
                                      sig_addr,  signal, sig_op,                                                                \
                                      1, 1,                                                                                     \
                                      shm::g_state_host.default_stream,                                                         \
                                      shm::g_state_host.default_block_num);                                                     \
        if (ret < 0) {                                                                                                          \
            SHM_LOG_ERROR("device calling transfer failed");                                                                    \
        }                                                                                                                       \
    }                                                                                                                           \

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_SIGNAL)
#undef SHMEM_PUT_TYPENAME_MEM_SIGNAL