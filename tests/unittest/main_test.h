#include <gtest/gtest.h>
#include <iostream>
#include "acl/acl.h"
#include "shmem_api.h"

extern int test_global_ranks;
extern int test_gnpu_num;
extern const char* test_global_ipport;
extern int test_first_rank;
extern int test_first_npu;

void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);

void test_finalize(aclrtStream stream, int device_id);

void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);

template<typename F, typename... Args>
void test_mutil_task_extra(F&& func, uint64_t local_mem_size, int process_count, Args&&... args){
    pid_t pids[process_count];
    int status[process_count];
    for (int i = 0; i < process_count; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            std::forward<F>(func)(i + test_first_rank, test_global_ranks, local_mem_size, std::forward<Args>(args)...);
            exit(0);
        }
    }
    for (int i = 0; i < process_count; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFEXITED(status[i]) && WEXITSTATUS(status[i]) != 0) {
            FAIL();
        }
    }
}

// template<typename... Args>
// void test_mutil_task_extra(std::function<void(int, int, uint64_t, Args...)> func, uint64_t local_mem_size, int process_count, Args&&... args){
//     pid_t pids[process_count];
//     int status[process_count];
//     for (int i = 0; i < process_count; ++i) {
//         pids[i] = fork();
//         if (pids[i] < 0) {
//             std::cout << "fork failed ! " << pids[i] << std::endl;
//         } else if (pids[i] == 0) {
//             func(i + test_first_rank, test_global_ranks, local_mem_size, std::forward<Args>(args)...);
//             exit(0);
//         }
//     }
//     for (int i = 0; i < process_count; ++i) {
//         waitpid(pids[i], &status[i], 0);
//         if (WIFEXITED(status[i]) && WEXITSTATUS(status[i]) != 0) {
//             FAIL();
//         }
//     }
// }