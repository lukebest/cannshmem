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

#include "shmem_api.h"

class kernel_put_num {
public:
    __aicore__ inline kernel_put_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_put_float_mem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
private:
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void put_num_test(GM_ADDR gva, GM_ADDR dev)
{
    kernel_put_num op;
    op.Init(gva, dev);
    op.Process();
}

void test_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    put_num_test<<<block_dim, nullptr, stream>>>(gva, dev);
}

class kernel_get_num {
public:
    __aicore__ inline kernel_get_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(buf_queue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> buf_tensor = buf_queue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)buf_tensor.address_.bufferAddr;

        for (int i = 0; i < rank_size; i++) {
            shmem_mte_get_mem_nbi(dev_gm + 16 * i, gva_gm, buf, (uint32_t)256, 16, i % rank_size, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void get_num_test(GM_ADDR gva, GM_ADDR dev)
{
    kernel_get_num op;
    op.Init(gva, dev);
    op.Process();
}

void test_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    get_num_test<<<block_dim, nullptr, stream>>>(gva, dev);
}

class KernelPutmemSignal{
public:
    __aicore__ inline KernelPutmemSignal() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev, GM_ADDR sig_addr, int32_t signal_ , int sig_op_)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;
        sig_addr_gm = (__gm__ int32_t *)sig_addr;
        signal = signal_;
        rank = smem_shm_get_global_rank();
        sig_op = sig_op_;
    }

    __aicore__ inline void Process()
    {
        shmem_put_float_mem_signal(gva_gm, dev_gm, 8, sig_addr_gm, signal, sig_op, rank);
    }
private:
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;
    __gm__ int32_t *sig_addr_gm;

    int32_t signal;
    int64_t rank;
    int sig_op;
};

extern "C" __global__ __aicore__ void put_mem_signal_test(GM_ADDR gva, GM_ADDR dev,
                                                          GM_ADDR sig_addr, int32_t signal , int sig_op)
{
    KernelPutmemSignal op;
    op.Init(gva, dev, sig_addr, signal, sig_op);
    op.Process();
}

void test_put_mem_signal(uint32_t block_dim, void* stream, float* gva, float* dev, uint8_t *sig_addr, int32_t signal, int sig_op)
{
    put_mem_signal_test<<<block_dim, nullptr, stream>>>((uint8_t *)gva, (uint8_t *)dev, sig_addr, signal, sig_op);
}

class ShmemTest {
public:
    __aicore__ inline ShmemTest() {}
    __aicore__ inline void Init(GM_ADDR gva,  int cmp_, float cmp_value_)
    {
        gva_gm = (__gm__ float *)gva;

        cmp = cmp_;
        cmp_value = cmp_value_;
    }
    __aicore__ inline void Process()
    {
        result = shmem_float_test(gva_gm, cmp, cmp_value);
    }

    int result;
private:
    __gm__ float *gva_gm;

    int cmp;
    float cmp_value;
};

extern "C" __global__ __aicore__ void shemem_test_test(GM_ADDR gva, GM_ADDR result, int cmp, float cmp_value)
{
    ShmemTest op;
    op.Init(gva, cmp, cmp_value);
    op.Process();
    *(__gm__ int *)result = op.result;
}

void test_shmem_test(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* result, int cmp, float cmp_value)
{
    shemem_test_test<<<block_dim, nullptr, stream>>>(gva, result, cmp, cmp_value);
}
