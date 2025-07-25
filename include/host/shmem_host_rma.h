/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#include "shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NO_NBI = 0,
    NBI,
};

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *        Firstly, check whether the input address is legal on local PE. Then translate it into remote address 
 *        on specified PE. Otherwise, returns a null pointer.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return If the input address is legal, returns a remote symmetric address on the specified PE that can be 
 *         accessed using memory loads and stores. Otherwise, a null pointer is returned.
 */
SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe);

/**
 * @brief Set necessary parameters for put\\get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size                [in] The Size of Temp UB Buffer.
 * @param event_id               [in] Sync ID for put\\get interfaces.
 * @return Returns 0 on success or an error code on failure.
 */
SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id);

#define SHMEM_TYPE_PUT(NAME, TYPE)                                                                                              \
    /**                                                                                                                         \
    * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.                \
    *                                                                                                                           \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                                       \
    * @param source             [in] Pointer on local device of the source data.                                                \
    * @param nelems             [in] Number of elements in the destination and source arrays.                                   \
    * @param pe                 [in] PE number of the remote PE.                                                                \
    */                                                                                                                          \
    SHMEM_HOST_API void shmem_put_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe);

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
    SHMEM_HOST_API void shmem_put_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe);

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
    SHMEM_HOST_API void shmem_get_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe);

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
    SHMEM_HOST_API void shmem_get_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe);

SHMEM_TYPE_FUNC(SHMEM_TYPE_GET_NBI)
#undef SHMEM_TYPE_GET_NBI

#ifdef __cplusplus
}
#endif

#endif