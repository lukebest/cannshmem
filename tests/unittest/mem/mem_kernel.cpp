/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

#include "mem_kernel.h"


void test_put_int32(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        put_mem_test_nbi<int><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        put_mem_test<int><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_put_float(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        put_mem_test_nbi<float><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        put_mem_test<float><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_put_void(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        put_mem_test_nbi<void><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        put_mem_test<void><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_put_char(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        put_mem_test_nbi<char><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        put_mem_test<char><<<block_dim, nullptr, stream>>>(gva, dev);
}


void test_get_int32(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        get_mem_test_nbi<int><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        get_mem_test<int><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_get_float(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{   
    if (is_nbi)
        get_mem_test_nbi<float><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        get_mem_test<float><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_get_void(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{
    if (is_nbi)
        get_mem_test_nbi<void><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        get_mem_test<void><<<block_dim, nullptr, stream>>>(gva, dev);
}

void test_get_char(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, bool is_nbi = true)
{   
    if (is_nbi)
        get_mem_test_nbi<char><<<block_dim, nullptr, stream>>>(gva, dev);
    else
        get_mem_test_nbi<char><<<block_dim, nullptr, stream>>>(gva, dev);
}
