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
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void test_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);
extern void test_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);
extern void test_put_mem_signal(uint32_t block_dim, void* stream, float* gva, float* dev_ptr,
                                uint8_t *sig_addr, int32_t signal, int sig_op);
extern void test_shmem_test(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* result, int cmp, float cmp_value);
extern void test_put_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);
extern void test_get_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);

static void test_put_get(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)
{
    int total_size = 16 * (int)rank_size;
    size_t input_size = total_size * sizeof(float);
    
    std::vector<float> input(total_size, 0);
    for (int i = 0; i < 16; i++) {
        input[i] = (rank_id + 10);
    }
    
    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(1024);
    test_put(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";
    std::cout << p_name;
    for (int i = 0; i < total_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    test_get(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << p_name;
    for (int i = 0; i < total_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    // for gtest
    int32_t flag = 0;
    for (int i = 0; i < total_size; i++){
        int stage = i / 16;
        if (input[i] != (stage + 10)) flag = 1;
    }
    ASSERT_EQ(flag, 0);
}

void test_shmem_mem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    test_put_get(stream, (uint8_t *)shm::g_state.heap_base, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void test_put_get_sync(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)
{
    int total_size = 16 * (int)rank_size;
    size_t input_size = total_size * sizeof(float);

    std::vector<float> input(total_size, 0);
    for (int i = 0; i < 16; i++) {
        input[i] = (rank_id + 10);
    }

    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(1024);
    test_put_sync(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";
    std::cout << p_name;
    for (int i = 0; i < total_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    test_get_sync(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << p_name;
    for (int i = 0; i < total_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    // for gtest
    int32_t flag = 0;
    for (int i = 0; i < total_size; i++){
        int stage = i / 16;
        if (input[i] != (stage + 10)) flag = 1;
    }
    ASSERT_EQ(flag, 0);
}

void test_shmem_mem_sync(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    test_put_get_sync(stream, (uint8_t *)shm::g_state.heap_base, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void test_mem_signal(aclrtStream stream, uint32_t rank_id, uint32_t rank_size, int sig_op)
{
    size_t input_size = 16 * sizeof(float);

    std::string p_name = std::to_string(rank_id) + "] ";
    std::vector<float> input(16, rank_id + 10);
    std::cout <<"[begin "  <<p_name;
    for (int i = 0; i < 16; i++) {
      std::cout << input[i] << " ";
    }
    std::vector<float> output(16, 0);
    std::vector<int32_t> output_signal(1, 0);
    void *sig_addr;
    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    ASSERT_EQ(aclrtMalloc(&sig_addr, 32, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(1024);
    auto signal = 6;
    test_put_mem_signal(block_dim, stream, (float *)ptr, (float *)dev_ptr, (uint8_t *)sig_addr, signal, sig_op);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(output.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(aclrtMemcpy(output_signal.data(), 1, sig_addr, 1, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    int32_t flag = 0;
    for (int i = 0; i < 8; i++){
        if (output[i] != input[i]) flag = 1;
    }
    std::cout <<"[ end " <<p_name;
    for (int i = 0; i < 16; i++) {
      std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    ASSERT_EQ(flag, 0);
    ASSERT_EQ(output_signal[0], 6);
}

void test_shmem_mem_signal(int rank_id, int n_ranks, uint64_t local_mem_size, int sig_op) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    test_mem_signal(stream, rank_id, n_ranks, sig_op);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void test_mem_cmp(aclrtStream stream) {
    void *dev_ptr;
    void *result;
    std::vector<int> cmp_result_out(1,2);
    ASSERT_EQ(aclrtMalloc(&result, 1, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    std::vector<float> input(16, 1);
    ASSERT_EQ(aclrtMalloc(&dev_ptr, 16, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemcpy(dev_ptr, 16, input.data(), 16, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    uint32_t block_dim = 1;
    test_shmem_test(block_dim, stream, (uint8_t *)dev_ptr, (uint8_t *)result,SHMEM_CMP_NE, 1);
    ASSERT_EQ(aclrtMemcpy(cmp_result_out.data(), 1, result, 1, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(cmp_result_out[0], 0);
}

void test_shmem_cmp(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int cmp = SHMEM_CMP_EQ;
    test_mem_cmp(stream);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemMem)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_mem, local_mem_size, process_count);
}

TEST(TestMemApi, TestShmemMemSync)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_mem_sync, local_mem_size, process_count);
}

TEST(TestMemApi, TestShmemMemSignal)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task([](int rank_id, int n_rank, uint64_t local_memsize){
      test_shmem_mem_signal(rank_id, n_rank, local_memsize, SHMEM_SIGNAL_SET);
    }, local_mem_size, process_count);
}

TEST(TestMemApi, TestShmemTest){
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_cmp, local_mem_size, process_count);
}