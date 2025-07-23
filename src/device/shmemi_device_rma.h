/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_RMA_H
#define SHMEMI_DEVICE_RMA_H

#include <stdint.h>
#include <stddef.h>  // for ptrdiff_t, size_t
#include <acl/acl.h>
#include "shmem_api.h"
#include "host_device/shmem_types.h"

// internal kernels calling
int32_t shmemi_prepare_and_post_rma(const char *api_name, shmemi_op_t desc, bool is_nbi,
                                    uint8_t *lptr, uint8_t *rptr,
                                    size_t n_elems, size_t elem_bytes, int pe,
                                    ptrdiff_t lstride = 1, ptrdiff_t rstride = 1,
                                    aclrtStream acl_strm = nullptr, size_t block_size = 1);

#endif