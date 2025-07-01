#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

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
