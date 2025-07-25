/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_device_rma.h"
#include "kernel_operator.h"

using namespace std;

// kernels
SHMEM_GLOBAL void shmemi_putmem_nbi(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe) {
    shmem_put_uint8_mem_nbi(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_putmem(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe) {
    shmem_put_uint8_mem(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_getmem_nbi(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe) {
    shmem_get_uint8_mem_nbi(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_getmem(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size, int32_t pe) {
    shmem_get_uint8_mem(lptr, rptr, elem_size, pe);
}

SHMEM_GLOBAL void shmemi_putmem_signal(GM_ADDR lptr, GM_ADDR rptr, uint32_t elem_size,
                                       GM_ADDR sig_addr, int32_t signal, int sig_op, int pe) {
    __gm__ int32_t *sig_addr_int32 = reinterpret_cast<__gm__ int32_t *>(sig_addr);
    shmem_put_uint8_mem_signal(lptr, rptr, elem_size, sig_addr_int32, signal, sig_op, pe);
}

// kernel function calling entrance
int32_t shmemi_prepare_and_post_rma(const char *api_name, shmemi_op_t desc, bool is_nbi,
                                    uint8_t *lptr, uint8_t *rptr,
                                    size_t n_elems, size_t elem_bytes, int pe,
                                    ptrdiff_t lstride, ptrdiff_t rstride,
                                    aclrtStream acl_strm, size_t block_size)
{
    if ((lstride > 1) || (rstride > 1)) {
        return -1;
    }

    if (is_nbi) {
        switch (desc) {
            case SHMEMI_OP_PUT:
                shmemi_putmem_nbi<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_GET:
                shmemi_getmem_nbi<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
        }
    }
    else {
        switch (desc) {
            case SHMEMI_OP_PUT:
                shmemi_putmem<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
            case SHMEMI_OP_GET:
                shmemi_getmem<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, pe);
                break;
        }
    }
    return 0;
}

int32_t shmem_putmem_signal_host(uint8_t *lptr, uint8_t *rptr,
                            size_t n_elems, size_t elem_bytes, int pe,
                            uint8_t *sig_addr, int32_t signal, int sig_op,
                            ptrdiff_t lstride, ptrdiff_t rstride,
                            aclrtStream acl_strm, size_t block_size){
    if ((lstride > 1) || (rstride > 1)) {
      return -1;
    }

    shmemi_putmem_signal<<<block_size, 0, acl_strm>>>(lptr, rptr, n_elems * elem_bytes, sig_addr, signal, sig_op, pe);
    return 0;
}
