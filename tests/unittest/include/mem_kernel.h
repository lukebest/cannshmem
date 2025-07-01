#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

#include "shmem_api.h"

template<typename T>
class KernelPutTest {
public:
    __aicore__ inline KernelPutTest() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ T *)gva;
        dev_gm = (__gm__ T *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process_nbi()
    {
        if constexpr (std::is_same<T, int>::value) {
            shmem_put_int32_mem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, float>::value) {
            shmem_put_float_mem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, void>::value) {
            shmem_putmem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, char>::value) {
            shmem_put_char_mem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else {
            assert(false, "Unsupported type");
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (std::is_same<T, int>::value) {
            shmem_put_int32_mem(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, float>::value) {
            shmem_put_float_mem(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, void>::value) {
            shmem_putmem(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else if constexpr (std::is_same<T, char>::value) {
            shmem_put_char_mem(gva_gm, dev_gm, rank_size * 16, rank);
        }
        else {
            assert(false, "Unsupported type");
        }
    }
private:
    __gm__ T *gva_gm;
    __gm__ T *dev_gm;

    int64_t rank;
    int64_t rank_size;
};


template<typename T>
class KernelGetTest {
public:
    __aicore__ inline KernelGetTest() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ T *)gva;
        dev_gm = (__gm__ T *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(buf_queue, 1, 512);
    }
    __aicore__ inline void Process_nbi()
    {
        AscendC::LocalTensor<float> buf_tensor = buf_queue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)buf_tensor.address_.bufferAddr;

        for (int i = 0; i < rank_size; i++) {
            // shmem_mte_get_mem_nbi(dev_gm + 16 * i, gva_gm, buf, (uint32_t)256, 16, i % rank_size, EVENT_ID0);
            if constexpr (std::is_same<T, int>::value) {
                shmem_get_int32_mem_nbi(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, float>::value) {
                shmem_get_float_mem_nbi(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, void>::value) {
                shmem_getmem_nbi(reinterpret_cast<__gm__ void*>(reinterpret_cast<__gm__ char*>(dev_gm) + 16 * i), gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, char>::value) {
                shmem_get_char_mem_nbi(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else {
                assert(false, "Unsupported type");
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        buf_queue.FreeTensor(buf_tensor);
    }
    __aicore__ inline void Process()
    {
        for (int i = 0; i < rank_size; i++) {
            if constexpr (std::is_same<T, int>::value) {
                shmem_get_int32_mem(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, float>::value) {
                shmem_get_float_mem(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, void>::value) {
                shmem_getmem(reinterpret_cast<__gm__ void*>(reinterpret_cast<__gm__ char*>(dev_gm) + 16 * i), gva_gm, 16, i % rank_size);
            }
            else if constexpr (std::is_same<T, char>::value) {
                shmem_get_char_mem(dev_gm + 16 * i, gva_gm, 16, i % rank_size);
            }
            else {
                assert(false, "Unsupported type");
            }
        }
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;
    __gm__ T *gva_gm;
    __gm__ T *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

template<typename T>
__global__ __aicore__ void put_mem_test(GM_ADDR gva, GM_ADDR dev)
{
    KernelPutTest<T> op;
    op.Init(gva, dev);
    op.Process();
}

template<typename T>
__global__ __aicore__ void put_mem_test_nbi(GM_ADDR gva, GM_ADDR dev)
{
    KernelPutTest<T> op;
    op.Init(gva, dev);
    op.Process_nbi();
}

template<typename T>
__global__ __aicore__ void get_mem_test(GM_ADDR gva, GM_ADDR dev)
{
    KernelGetTest<T> op;
    op.Init(gva, dev);
    op.Process();
}

template<typename T>
__global__ __aicore__ void get_mem_test_nbi(GM_ADDR gva, GM_ADDR dev)
{
    KernelGetTest<T> op;
    op.Init(gva, dev);
    op.Process_nbi();
}
