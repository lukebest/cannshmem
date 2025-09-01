#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void test_rdma_put_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_get_low_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_put_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void test_rdma_get_high_level(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
extern void shmem_rdma_get_qpinfo_test_do(void* stream, uint8_t* gva, uint32_t rankId, uint64_t config);
extern void test_rdma_poll_cq_do(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

static void test_rdma_poll_cq(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint64_t heap_size)
{
    size_t messageSize = 128;
    uint64_t *xHost;
    size_t totalSize = 120;
    
    ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        xHost[i] = rank_id + 10;
    }
    ASSERT_EQ(aclrtMemcpy(gva + (rank_id + 1) * messageSize, messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    test_rdma_poll_cq_do(block_dim, stream, (uint8_t *)gva, shmemx_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";
    std::cout << p_name;
    ASSERT_EQ(aclrtMemcpy(xHost, totalSize, gva + 2048, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < totalSize / sizeof(uint64_t); i++) {
        printf("PollCQ index = %d, value = %lu\n", i, xHost[i]);
    }
}

static void test_rdma_put_get(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t *xHost;
    size_t totalSize = messageSize * rank_size;
    
    ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        xHost[i] = rank_id + 10;
    }
    ASSERT_EQ(aclrtMemcpy(gva + rank_id * messageSize, messageSize, xHost, messageSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    // test_rdma_put_low_level(block_dim, stream, (uint8_t *)gva, shmemx_get_ffts_config());
    // test_rdma_get_low_level(block_dim, stream, (uint8_t *)gva, shmemx_get_ffts_config());
    test_rdma_put_high_level(block_dim, stream, (uint8_t *)gva, shmemx_get_ffts_config());
    // test_rdma_get_high_level(block_dim, stream, (uint8_t *)gva, shmemx_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";
    std::cout << p_name;
    ASSERT_EQ(aclrtMemcpy(xHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(xHost[i * messageSize / sizeof(uint32_t)], i + 10);
    }
}

static void test_rdma_get_info(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize) {
    uint64_t *xHost;
    size_t totalSize = 120;
    ASSERT_EQ(aclrtMallocHost((void **)(&xHost), totalSize), 0);
    memset(xHost, 0xEE, totalSize);
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, xHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    for (uint32_t curRank = 0; curRank < rankSize; curRank++) {
        if (curRank == rankId) {
            continue;
        }
        shmem_rdma_get_qpinfo_test_do(stream, gva, curRank, shmemx_get_ffts_config());
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        sleep(1);

        ASSERT_EQ(aclrtMemcpy(xHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
        for (uint32_t i = 0; i < totalSize / sizeof(uint64_t); i++) {
            printf("GetQPInfo srcRank = %d, destRank = %d, index = %d, value = %lu\n", rankId, curRank, i, xHost[i]);
        }
    }

    ASSERT_EQ(aclrtFreeHost(xHost), 0);
}

void test_shmem_rdma_mem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    void* ptr = shmem_malloc(1024);
    // test_rdma_poll_cq(stream, (uint8_t *)ptr, rank_id, n_ranks);
    test_rdma_put_get(stream, (uint8_t *)ptr, rank_id, n_ranks);
    // test_rdma_get_info(stream, (uint8_t *)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemRDMAMem)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_shmem_rdma_mem, local_mem_size, processCount);
}