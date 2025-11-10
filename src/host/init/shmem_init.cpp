/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <random>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "internal/host/shmemi_host_def.h"

using namespace std;

namespace shm {
#define MIN_PORT 1024
#define MAX_PORT 65536
#define MAX_ATTEMPTS 500
#define MAX_IFCONFIG_LENGTH 23
#define MAX_IP 48
constexpr int DEFAULT_MY_PE = -1;
constexpr int DEFAULT_N_PES = -1;

constexpr int DEFAULT_FLAG = 0;
constexpr int DEFAULT_ID = 0;
constexpr int DEFAULT_TIMEOUT = 120;
constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

// initializer
#define SHMEM_DEVICE_HOST_STATE_INITIALIZER                                              \
    {                                                                                    \
        (1 << 16) + sizeof(shmemi_device_host_state_t), /* version */                    \
            (DEFAULT_MY_PE),                            /* mype */                       \
            (DEFAULT_N_PES),                            /* npes */                       \
            NULL,                                       /* heap_base */                  \
            {NULL},                                     /* p2p_heap_base */              \
            {NULL},                                     /* sdma_heap_base */             \
            {},                                         /* topo_list */                  \
            SIZE_MAX,                                   /* heap_size */                  \
            {NULL},                                     /* team_pools */                 \
            0,                                          /* sync_pool */                  \
            0,                                          /* sync_counter */               \
            0,                                          /* core_sync_pool */             \
            0,                                          /* core_sync_counter */          \
            false,                                      /* shmem_is_shmem_initialized */ \
            false,                                      /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                          /* shmem_mte_config */           \
    }

shmemi_device_host_state_t g_state = SHMEM_DEVICE_HOST_STATE_INITIALIZER;
shmemi_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};
shmem_init_attr_t g_attr;
static smem_shm_t g_smem_handle = nullptr;
static bool g_attr_init = false;
static char g_ipport[SHMEM_MAX_IP_PORT_LEN] = {0};

int32_t version_compatible()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_options_init()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_state_init_attr(shmem_init_attr_t *attributes)
{
    int32_t status = SHMEM_SUCCESS;
    g_state.mype = attributes->my_rank;
    g_state.npes = attributes->n_ranks;
    g_state.heap_size = attributes->local_mem_size + SHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    SHMEM_CHECK_RET(aclrtCreateStream(&stream));
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

void shmemi_reach_info_init(void *&gva)
{
    uint32_t reach_info = 0;
    int32_t status = SHMEM_SUCCESS;
    for (int32_t i = 0; i < g_state.npes; i++) {
        status = smem_shm_topology_can_reach(g_smem_handle, i, &reach_info);
        g_state.p2p_heap_base[i] = (void *)((uintptr_t)gva + g_state.heap_size * static_cast<uint32_t>(i));
        if (reach_info & SMEMS_DATA_OP_MTE) {
            g_state.topo_list[i] |= SHMEM_TRANSPORT_MTE;
        }
        if (reach_info & SMEMS_DATA_OP_SDMA) {
            g_state.sdma_heap_base[i] = (void *)((uintptr_t)gva + g_state.heap_size * static_cast<uint32_t>(i));
        } else {
            g_state.sdma_heap_base[i] = NULL;
        }
        if (reach_info & SMEMS_DATA_OP_RDMA) {
            g_state.topo_list[i] |= SHMEM_TRANSPORT_ROCE;
        }
    }
}

int32_t shmemi_heap_init(shmem_init_attr_t *attributes)
{
    void *gva = nullptr;
    int32_t status = SHMEM_SUCCESS;
    int32_t device_id;
    SHMEM_CHECK_RET(aclrtGetDevice(&device_id));

    status = smem_init(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    smem_shm_config_t config;
    status = smem_shm_config_init(&config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_shm_config_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    status = smem_shm_init(attributes->ip_port, attributes->n_ranks, attributes->my_rank, device_id, &config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_shm_init Failed");
        return SHMEM_SMEM_ERROR;
    }

    config.shmInitTimeout = attributes->option_attr.shm_init_timeout;
    config.shmCreateTimeout = attributes->option_attr.shm_create_timeout;
    config.controlOperationTimeout = attributes->option_attr.control_operation_timeout;

    g_smem_handle = smem_shm_create(DEFAULT_ID, attributes->n_ranks, attributes->my_rank, g_state.heap_size,
                                    static_cast<smem_shm_data_op_type>(attributes->option_attr.data_op_engine_type),
                                    DEFAULT_FLAG, &gva);
    if (g_smem_handle == nullptr || gva == nullptr) {
        SHM_LOG_ERROR("smem_shm_create Failed");
        return SHMEM_SMEM_ERROR;
    }
    g_state.heap_base = (void *)((uintptr_t)gva + g_state.heap_size * static_cast<uint32_t>(attributes->my_rank));
    shmemi_reach_info_init(gva);
    if (shm::g_ipport[0] != '\0') {
        g_ipport[0] = '\0';
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    } else {
        SHM_LOG_WARN("my_rank:" << attributes->my_rank << " shm::g_ipport is released in advance!");
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    }
    g_state.is_shmem_created = true;
    return status;
}

int32_t shmemi_control_barrier_all()
{
    SHM_ASSERT_RETURN(g_smem_handle != nullptr, SHMEM_INVALID_PARAM);
    return smem_shm_control_barrier(g_smem_handle);
}

int32_t update_device_state()
{
    if (!g_state.is_shmem_created) {
        return SHMEM_NOT_INITED;
    }
    return smem_shm_set_extra_context(g_smem_handle, (void *)&g_state, sizeof(shmemi_device_host_state_t));
}

int32_t check_attr(shmem_init_attr_t *attributes)
{
    if ((attributes->my_rank < 0) || (attributes->n_ranks <= 0)) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_rank << " and n_ranks: " << attributes->n_ranks
                                 << " cannot be less 0 , n_ranks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->n_ranks > SHMEM_MAX_RANKS) {
        SHM_LOG_ERROR("n_ranks: " << attributes->n_ranks << " cannot be more than " << SHMEM_MAX_RANKS);
        return SHMEM_INVALID_VALUE;
    } else if (attributes->my_rank >= attributes->n_ranks) {
        SHM_LOG_ERROR("n_ranks:" << attributes->n_ranks << " cannot be less than my_rank:" << attributes->my_rank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
        return SHMEM_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

}  // namespace shm

int32_t shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.data_op_engine_type = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.shm_init_timeout = value;
    attributes->option_attr.shm_create_timeout = value;
    attributes->option_attr.control_operation_timeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    SHM_ASSERT_RETURN(local_mem_size <= SHMEM_MAX_LOCAL_SIZE, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_ranks <= SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_rank <= SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    *attributes = &shm::g_attr;
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(shm::g_ipport) - 1);

        std::copy_n(ip_port, ip_len, shm::g_ipport);
        shm::g_ipport[ip_len] = '\0';
        std::copy_n(shm::g_ipport, ip_len, shm::g_attr.ip_port);
        if (shm::g_ipport[0] == '\0') {
            SHM_LOG_ERROR("my_rank:" << my_rank << " shm::g_ipport is nullptr!");
            return SHMEM_INVALID_VALUE;
        }
    }

    int attr_version = static_cast<int>((1 << 16) + sizeof(shmem_init_attr_t));
    shm::g_attr.my_rank = my_rank;
    shm::g_attr.n_ranks = n_ranks;
    shm::g_attr.ip_port[ip_len] = '\0';
    shm::g_attr.local_mem_size = local_mem_size;
    shm::g_attr.option_attr = {attr_version, SHMEM_DATA_OP_MTE, shm::DEFAULT_TIMEOUT,
                               shm::DEFAULT_TIMEOUT, shm::DEFAULT_TIMEOUT};
    shm::g_attr_init = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_get_uid_magic(shmem_uniqueid_inner_t *innerUId)
{
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        SHM_LOG_ERROR("open random failed");
        return SHMEM_INNER_ERROR;
    }

    urandom.read(reinterpret_cast<char *>(&innerUId->magic), sizeof(innerUId->magic));
    if (urandom.fail()) {
        SHM_LOG_ERROR("read random failed.");
        return SHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("init magic id to " << innerUId->magic);
    return SHMEM_SUCCESS;
}

uint16_t shmem_get_port_magic()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    const int min_port = MIN_PORT;
    const int max_port = MAX_PORT;
    const int max_attempts = MAX_ATTEMPTS;
    std::uniform_int_distribution<> dis(min_port, max_port);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = dis(gen);
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            continue;
        }

        int on = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) == -1) {
            close(sockfd);
            continue;
        }

        sockaddr_in bind_address{};
        bind_address.sin_family = AF_INET;
        bind_address.sin_port = htons(port);
        bind_address.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd, reinterpret_cast<sockaddr *>(&bind_address), sizeof(bind_address)) == 0) {
            close(sockfd);
            return port;
        }

        close(sockfd);
    }
    SHM_LOG_ERROR("Not find a available tcp port");
    return 0;
}

int32_t ParseInterfaceWithType(const char *ipInfo, char *IP, sa_family_t &sockType)
{
    const char *delim = ":";
    const char *sep = strchr(ipInfo, delim[0]);
    if (sep != nullptr) {
        size_t leftLen = sep - ipInfo;
        if (leftLen >= MAX_IFCONFIG_LENGTH - 1 || leftLen == 0) {
            return SHMEM_INVALID_VALUE;
        }
        strncpy(IP, ipInfo, leftLen);
        IP[leftLen] = '\0';
        sockType = (strcmp(sep + 1, "inet6") != 0) ? AF_INET : AF_INET6;
    }
    return SHMEM_SUCCESS;
}

int32_t shmem_get_ip_from_ifa(char *local, sa_family_t &sockType, const char *ipInfo)
{
    struct ifaddrs *ifaddr;
    char masterIf[MAX_IFCONFIG_LENGTH];
    sockType = AF_INET;
    if (ipInfo == nullptr) {
        strncpy(masterIf, "eth", sizeof(masterIf));
        masterIf[sizeof(masterIf) - 1] = '\0';
        sockType = AF_INET;
    } else if (ParseInterfaceWithType(ipInfo, masterIf, sockType) != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("IP size set in SHMEM_CONF_STORE_MASTER_IF format has wrong length");
        return SHMEM_INVALID_PARAM;
    }

    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno << ": " << strerror(errno));
        return SHMEM_INVALID_PARAM;
    }
    int32_t result = SHMEM_SUCCESS;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == nullptr) || (ifa->ifa_addr->sa_family != sockType) || (ifa->ifa_netmask == nullptr) ||
            (strcmp(ifa->ifa_name, masterIf) != 0)) {
            continue;
        }
        if (sockType == AF_INET) {
            auto localIp = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr)->sin_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv4 to string failed. ");
                result = SHMEM_INVALID_PARAM;
            }
        } else {
            auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv6 to string failed. ");
                result = SHMEM_INVALID_PARAM;
            }
        }
        break;
    }
    return result;
}

int32_t shmem_get_ip_from_env(char *ip, uint16_t &port, sa_family_t &sockType, const char *ipPort)
{
    if (ipPort != nullptr) {
        SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value:" << ipPort);
        std::string ipPortStr = ipPort;

        if (ipPort[0] == '[') {
            sockType = AF_INET6;
            size_t found = ipPortStr.find_last_of(']');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return SHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(1, found);
            std::string portStr = ipPortStr.substr(found + 1);

            std::strncpy(ip, ipStr.c_str(), MAX_IP);

            port = std::stoi(portStr);
        } else {
            sockType = AF_INET;
            size_t found = ipPortStr.find_last_of(':');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return SHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(0, found);
            std::string portStr = ipPortStr.substr(found + 1);

            std::strncpy(ip, ipStr.c_str(), MAX_IP);

            port = std::stoi(portStr);
        }
        return SHMEM_SUCCESS;
    }
    return SHMEM_INVALID_PARAM;
}

int32_t shmem_set_ip_info(shmem_uniqueid_t *uid, sa_family_t &sockType, char *pta_env_ip, uint16_t pta_env_port,
                          bool is_from_ifa)
{
    // init default uid
    *uid = SHMEM_UNIQUEID_INITIALIZER;
    shmem_uniqueid_inner_t *innerUID = reinterpret_cast<shmem_uniqueid_inner_t *>(uid);
    SHMEM_CHECK_RET(shmem_get_uid_magic(innerUID));

    // fill ip port as part of uid
    uint16_t port = 0;
    if (is_from_ifa) {
        port = shmem_get_port_magic();
    } else {
        port = pta_env_port;
    }
    if (port == 0) {
        SHM_LOG_ERROR("get available port failed.");
        return SHMEM_INVALID_PARAM;
    }
    if (sockType == AF_INET) {
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            perror("inet_pton IPv4 failed");
            return SHMEM_NOT_INITED;
        }
        innerUID->addr.addr.addr4.sin_port = htons(port);
        innerUID->addr.type = ADDR_IPv4;
    } else if (sockType == AF_INET6) {
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            perror("inet_pton IPv6 failed");
            return SHMEM_NOT_INITED;
        }
        innerUID->addr.addr.addr6.sin6_port = htons(port);
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return SHMEM_INVALID_PARAM;
    }
    SHM_LOG_INFO("gen unique id success.");
    return SHMEM_SUCCESS;
}

int32_t shmem_get_uniqueid(shmem_uniqueid_t *uid)
{
    char pta_env_ip[MAX_IP];
    uint16_t pta_env_port;
    sa_family_t sockType;
    const char *ipPort = std::getenv("SHMEM_UID_SESSION_ID");
    const char *ipInfo = std::getenv("SHMEM_UID_SOCK_IFNAM");
    bool is_from_ifa = false;
    if (ipPort != nullptr) {
        if (shmem_get_ip_from_env(pta_env_ip, pta_env_port, sockType, ipPort) != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get pta master addr.");
            return SHMEM_INVALID_PARAM;
        }
    } else {
        is_from_ifa = true;
        if (shmem_get_ip_from_ifa(pta_env_ip, sockType, ipInfo) != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get pta master addr.");
            return SHMEM_INVALID_PARAM;
        }
    }
    SHM_LOG_INFO("get master IP value:" << pta_env_ip);
    return shmem_set_ip_info(uid, sockType, pta_env_ip, pta_env_port, is_from_ifa);
}

int32_t shmem_set_attr_uniqueid_args(int rank_id, int nranks, const shmem_uniqueid_t *uid, shmem_init_attr_t *attr)
{
    if (shmem_set_log_level(shm::ERROR_LEVEL) != 0) {
        return SHMEM_INNER_ERROR;
    }

    if (attr == nullptr || uid == nullptr) {
        SHM_LOG_ERROR("set unique id attr/uid is null");
        return SHMEM_INVALID_PARAM;
    }

    if (rank_id != shm::g_attr.my_rank || nranks != shm::g_attr.n_ranks) {
        SHM_LOG_ERROR("rankid/nranks invalid, maybe call shmem_set_attr firstly.");
        return SHMEM_INVALID_PARAM;
    }

    if (uid->version != SHMEM_UNIQUEID_VERSION) {
        SHM_LOG_ERROR("uid version invalid, init unique id with shmem_get_uniqueid firstly.");
        return SHMEM_INVALID_PARAM;
    }

    // extract ip port from inner unique id
    shmem_uniqueid_inner_t *innerUID = reinterpret_cast<shmem_uniqueid_inner_t *>(const_cast<shmem_uniqueid_t *>(uid));

    // compatibility with shmem_init_attr, init ip_port from unique id
    std::string ipPort;
    if (innerUID->addr.type == ADDR_IPv6) {
        char ipStr[INET6_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET6, &(innerUID->addr.addr.addr6.sin6_addr), ipStr, sizeof(ipStr));
        uint16_t port = ntohs(innerUID->addr.addr.addr6.sin6_port);
        ipPort = "tcp6://[" + std::string(ipStr) + "]:" + std::to_string(port);
    } else {
        char ipStr[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &(innerUID->addr.addr.addr4.sin_addr), ipStr, sizeof(ipStr));
        uint16_t port = ntohs(innerUID->addr.addr.addr4.sin_port);
        ipPort = "tcp://" + std::string(ipStr) + ":" + std::to_string(port);
    }
    std::copy(ipPort.begin(), ipPort.end(), shm::g_ipport);
    std::copy(ipPort.begin(), ipPort.end(), shm::g_attr.ip_port);
    std::copy(ipPort.begin(), ipPort.end(), attr->ip_port);
    shm::g_ipport[ipPort.size()] = '\0';
    shm::g_attr.ip_port[ipPort.size()] = '\0';
    attr->ip_port[ipPort.size()] = '\0';
    SHM_LOG_INFO("extract ip port:" << ipPort);

    return shmem_init_attr(attr);
}

int32_t shmem_init_status(void)
{
    if (!shm::g_state.is_shmem_created)
        return SHMEM_STATUS_NOT_INITIALIZED;
    else if (!shm::g_state.is_shmem_initialized)
        return SHMEM_STATUS_SHM_CREATED;
    else if (shm::g_state.is_shmem_initialized)
        return SHMEM_STATUS_IS_INITIALIZED;
    else
        return SHMEM_STATUS_INVALID;
}

void shmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("shmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t shmem_init_attr(shmem_init_attr_t *attributes)
{
    int32_t ret;

    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    SHMEM_CHECK_RET(shmem_set_log_level(shm::ERROR_LEVEL));
    SHMEM_CHECK_RET(shm::check_attr(attributes));
    SHMEM_CHECK_RET(shm::version_compatible());
    SHMEM_CHECK_RET(shm::shmemi_options_init());

    SHMEM_CHECK_RET(shm::shmemi_state_init_attr(attributes));
    SHMEM_CHECK_RET(shm::shmemi_heap_init(attributes));
    SHMEM_CHECK_RET(shm::update_device_state());

    SHMEM_CHECK_RET(shm::memory_manager_initialize(shm::g_state.heap_base, shm::g_state.heap_size));
    SHMEM_CHECK_RET(shm::shmemi_team_init(shm::g_state.mype, shm::g_state.npes));
    SHMEM_CHECK_RET(shm::update_device_state());
    SHMEM_CHECK_RET(shm::shmemi_sync_init());
    SHMEM_CHECK_RET(smem_shm_register_exit(shm::g_smem_handle, &shmem_rank_exit));
    shm::g_state.is_shmem_initialized = true;
    SHMEM_CHECK_RET(shm::shmemi_control_barrier_all());
    return SHMEM_SUCCESS;
}

int32_t shmem_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const shmem_decrypt_handler decrypt_handler)
{
    return smem_set_config_store_tls_key(tls_pk, tls_pk_len, tls_pk_pw, tls_pk_pw_len, decrypt_handler);
}

int32_t shmem_set_extern_logger(void (*func)(int level, const char *msg))
{
    SHM_ASSERT_RETURN(func != nullptr, SHMEM_INVALID_PARAM);
    shm::shm_out_logger::Instance().set_extern_log_func(func, true);
    return smem_set_extern_logger(func);
}

int32_t shmem_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char *in_level = std::getenv("SHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = shm::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = shm::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = shm::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = shm::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = shm::FATAL_LEVEL;
        }
    }
    shm::shm_out_logger::Instance().set_log_level(static_cast<shm::log_level>(level));
    return smem_set_log_level(level);
}

int32_t shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    return smem_set_conf_store_tls(enable, tls_info, tls_info_len);
}

int32_t shmem_finalize(void)
{
    SHMEM_CHECK_RET(shm::shmemi_team_finalize());
    if (shm::g_smem_handle != nullptr) {
        int32_t status = smem_shm_destroy(shm::g_smem_handle, 0);
        if (status != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("smem_shm_destroy Failed");
            return SHMEM_SMEM_ERROR;
        }
        shm::g_smem_handle = nullptr;
    }
    smem_shm_uninit(0);
    smem_uninit();
    return SHMEM_SUCCESS;
}

void shmem_info_get_version(int *major, int *minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = SHMEM_MAJOR_VERSION;
    *minor = SHMEM_MINOR_VERSION;
}

void shmem_info_get_name(char *name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "SHMEM v" << SHMEM_VENDOR_MAJOR_VER << "." << SHMEM_VENDOR_MINOR_VER << "." << SHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < SHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

void shmem_global_exit(int status)
{
    smem_shm_global_exit(shm::g_smem_handle, status);
}
