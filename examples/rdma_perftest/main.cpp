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
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>

#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *test_type;

extern void rdma_highlevel_put_pingpong_latency_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length);
extern void rdma_postsend_cost_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length);
extern void rdma_highlevel_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length);
extern void rdma_mte_put_bw_do(uint32_t block_dim, void* stream, uint64_t fftsConfig, uint8_t* gva, int message_length);

int test_shmem_rdma_highlevel_put_pingpong_latency(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    shmem_set_conf_store_tls(false, nullptr, 0);
    status = shmem_init_attr(attributes);

    uint64_t fftsConfig = shmemx_get_ffts_config();
    uint8_t* gva = (uint8_t*)shmem_malloc(1024 * 1024 * 6);

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost((void **)(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + 10;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gva + n_ranks * message_length + 32 * (rank_id + 1), 32, xHost, 32, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_highlevel_put_pingpong_latency_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_ranks, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA highlevel put pingpong latency test. Message length = " << message_length << " Byte; latency = " << xHost[0] / 50.0 << " us." << std::endl;
    }

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int test_shmem_rdma_postsend_cost(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    uint32_t iteration = 1;
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    shmem_set_conf_store_tls(false, nullptr, 0);
    status = shmem_init_attr(attributes);

    uint64_t fftsConfig = shmemx_get_ffts_config();
    uint8_t* gva = (uint8_t*)shmem_malloc(1024 * 1024 * 6);

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost((void **)(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + 10;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    for (uint32_t i = 0; i < iteration; i++) {
        rdma_postsend_cost_do(1, stream, fftsConfig, gva, message_length);
    }
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(uint32_t), gva + message_length * n_ranks, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA postsend cost test. Message length = " << message_length << " Byte; postsend cost = " << xHost[0] / (50.0 * 500) << " us." << std::endl;
    }

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int test_shmem_rdma_highlevel_put_bw(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    shmem_set_conf_store_tls(false, nullptr, 0);
    status = shmem_init_attr(attributes);

    uint64_t fftsConfig = shmemx_get_ffts_config();
    uint8_t* gva = (uint8_t*)shmem_malloc(1024 * 1024 * 6);

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost((void **)(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + 10;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    rdma_highlevel_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, sizeof(int64_t), gva + message_length * n_ranks, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA high level put bandwidth test. Message length = " << message_length << " Byte; time = " << xHost[0] / (50.0) << " us." << std::endl;
    }

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int test_shmem_rdma_mte_put_bw(int rank_id, int n_ranks, uint64_t local_mem_size, int message_length)
{
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    shmem_set_conf_store_tls(false, nullptr, 0);
    status = shmem_init_attr(attributes);
    shmem_mte_set_ub_params(0, 128 * 1024, 0);

    uint64_t fftsConfig = shmemx_get_ffts_config();
    uint8_t* gva = (uint8_t*)shmem_malloc(1024 * 1024 * 32);

    int64_t *xHost;
    size_t totalSize = message_length * n_ranks;

    aclrtMallocHost((void **)(&xHost), totalSize);
    for (uint32_t i = 0; i < message_length / sizeof(int64_t); i++) {
        xHost[i] = rank_id + 10;
    }
    aclrtMemcpy(gva + rank_id * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gva + (rank_id + n_ranks) * message_length, message_length, xHost, message_length, ACL_MEMCPY_HOST_TO_DEVICE);

    rdma_mte_put_bw_do(1, stream, fftsConfig, gva, message_length);
    aclrtSynchronizeStream(stream);
    if (rank_id == 0) {
        aclrtMemcpy(xHost, 64, gva + message_length * n_ranks * 2, 64, ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; RDMA time = " << xHost[0] / (50.0) << " us." << std::endl;
        std::cout << "RDMA rdma mte test. Message length = " << message_length << " Byte; MTE time = " << xHost[6] / (50.0) << " us." << std::endl;
    }

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 9) {
        std::cout << "[ERROR] Paramater number mismatch." << std::endl;
        std::cout << "[USAGE] ./rdma_perftest <n_ranks> <rank_id> <ipport> <g_npus> <f_rank> <f_npu> <test_type> <msg_len>. See README for more details." << std::endl;
    }
    int status = 0;
    int n_ranks = atoi(argv[1]);
    if (n_ranks != 2) {
        std::cout << "[ERROR] Error number of ranks! Only support 2 ranks!" << std::endl;
    }
    int rank_id = atoi(argv[2]);
    if (rank_id >= 2) {
        std::cout << "[ERROR] Error rank ID! Only support 2 ranks!" << std::endl;
    }
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    test_type = argv[7];
    int msg_len = atoi(argv[8]);
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    if (std::string(test_type) == "highlevel_put_pingpong_latency") {
        test_shmem_rdma_highlevel_put_pingpong_latency(rank_id, n_ranks, local_mem_size, msg_len);
    } else if (std::string(test_type) == "postsend_cost") {
        test_shmem_rdma_postsend_cost(rank_id, n_ranks, local_mem_size, msg_len);
    } else if (std::string(test_type) == "highlevel_put_bw") {
        test_shmem_rdma_highlevel_put_bw(rank_id, n_ranks, local_mem_size, msg_len);
    }else if (std::string(test_type) == "rdma_mte_bw") {
        test_shmem_rdma_mte_put_bw(rank_id, n_ranks, local_mem_size, msg_len);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}