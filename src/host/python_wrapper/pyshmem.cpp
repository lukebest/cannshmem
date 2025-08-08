/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>
#include <iostream>

#include "shmem_api.h"

namespace py = pybind11;

namespace shm {
namespace {
inline std::string get_connect_url()
{
    auto address = std::getenv("SHMEM_MASTER_ADDR");
    auto port = std::getenv("SHMEM_MASTER_PORT");
    if (address != nullptr && port != nullptr) {
        return std::string("tcp://").append(address).append(":").append(port);
    }

    address = std::getenv("MASTER_ADDR");
    port = std::getenv("MASTER_PORT");
    if (address == nullptr || port == nullptr) {
        return "";
    }

    char *endptr;
    auto port_long = std::strtol(port, &endptr, 10);
    // master port + 11 as non-master port.
    if (endptr == port || *endptr != '\0' || port_long <= 0 || port_long > 65535 - 11) {
        // SHM_LOG_ERROR is not available in this file, use std::cerr
        std::cerr << "[ERROR] Invalid MASTER_PORT value from environment: " << port << std::endl;
        return "";
    }
    auto port_int = port_long + 11;
    return std::string("tcp://").append(address).append(":").append(std::to_string(port_int));
}

int shmem_initialize(int rank, int world_size, int64_t mem_size)
{
    shmem_init_attr_t attribute{
        rank, world_size, "", static_cast<uint64_t>(mem_size), {0, SHMEM_DATA_OP_MTE, 120, 120, 120}};
    auto url = get_connect_url();
    if (url.empty()) {
        std::cerr << "cannot get store connect URL(" << url << ") from ENV." << std::endl;
        return -1;
    }

    attribute.ip_port = url.c_str();
    auto ret = shmem_init_attr(&attribute);
    if (ret != 0) {
        std::cerr << "initialize with mype: " << rank << ", npes: " << world_size << " failed: " << ret;
        return ret;
    }

    return 0;
}

int32_t shmem_set_op_engine_type(shmem_init_attr_t &attributes, data_op_engine_type_t value)
{
    int ret = shmem_set_data_op_engine_type(&attributes, value);
    if (ret != 0) {
        throw std::runtime_error("set data operation engine type failed");
    }
    return ret;
}

int32_t set_timeout(shmem_init_attr_t &attributes, uint32_t value)
{
    int ret = shmem_set_timeout(&attributes, value);
    if (ret != 0) {
        throw std::runtime_error("set time out failed");
    }
    return ret;
}

int32_t shmem_set_attributes(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                            shmem_init_attr_t &attributes)
{
    shmem_init_attr_t *attr_ptr = &attributes;
    int ret = shmem_set_attr(my_rank, n_ranks, local_mem_size, ip_port, &attr_ptr);
    if (ret != 0) {
        throw std::runtime_error("set shmem attributes failed");
    }
    return ret;
}
}
}

void DefineShmemAttr(py::module_ &m)
{
    py::enum_<data_op_engine_type_t>(m, "OpEngineType")
        .value("MTE", SHMEM_DATA_OP_MTE, "copy data from local space to global space");

    py::class_<shmem_init_optional_attr_t>(m, "OptionalAttr")
        .def(py::init([]() {
                auto optional_attr = new (std::nothrow)shmem_init_optional_attr_t;
                return optional_attr;
            }))
        .def_readwrite("version", &shmem_init_optional_attr_t::version)
        .def_readwrite("data_op_engine_type", &shmem_init_optional_attr_t::data_op_engine_type)
        .def_readwrite("shm_init_timeout", &shmem_init_optional_attr_t::shm_init_timeout)
        .def_readwrite("shm_create_timeout", &shmem_init_optional_attr_t::shm_create_timeout)
        .def_readwrite("control_operation_timeout", &shmem_init_optional_attr_t::control_operation_timeout);

    py::class_<shmem_init_attr_t>(m, "InitAttr")
        .def(py::init([]() {
                 auto init_attr = new (std::nothrow)shmem_init_attr_t;
                 return init_attr;
             }))
        .def_readwrite("my_rank", &shmem_init_attr_t::my_rank)
        .def_readwrite("n_ranks", &shmem_init_attr_t::n_ranks)
        .def_readwrite("ip_port", &shmem_init_attr_t::ip_port)
        .def_readwrite("local_mem_size", &shmem_init_attr_t::local_mem_size)
        .def_readwrite("option_attr", &shmem_init_attr_t::option_attr);

    py::class_<shmem_team_config_t>(m, "TeamConfig")
        .def(py::init<>())
        .def(py::init<int>())
        .def_readwrite("num_contexts", &shmem_team_config_t::num_contexts);
}

void DefineShmemInitStatus(py::module_ &m)
{
    py::enum_<shmem_init_status_t>(m, "InitStatus")
        .value("NOT_INITIALIZED", SHMEM_STATUS_NOT_INITIALIZED)
        .value("SHM_CREATED", SHMEM_STATUS_SHM_CREATED)
        .value("INITIALIZED", SHMEM_STATUS_IS_INITIALIZED)
        .value("INVALID", SHMEM_STATUS_INVALID);
}

PYBIND11_MODULE(_pyshmem, m)
{
    DefineShmemAttr(m);
    DefineShmemInitStatus(m);

    m.def("shmem_init", &shm::shmem_initialize, py::call_guard<py::gil_scoped_release>(), py::arg("mype"),
          py::arg("npes"), py::arg("mem_size"), R"(
Initialize share memory module.

Arguments:
    mype(int): local processing element index, range in [0, npes).
    npes(int): total count of processing elements.
    mem_size(int): memory size for each processing element in bytes.
Returns:
    returns zero on success. On error, -1 is returned.
    )");

    m.def("shmem_finialize", &shmem_finalize, py::call_guard<py::gil_scoped_release>(),
          R"(
Finalize share memory module.
    )");

    m.def("shmem_set_attributes", &shm::shmem_set_attributes, py::call_guard<py::gil_scoped_release>(),
        py::arg("my_rank"), py::arg("n_ranks"), py::arg("local_mem_size"), py::arg("ip_port"),
        py::arg("attributes"), R"(
Set the default attributes
Arguments:
    my_rank(int): Current rank.
    n_ranks(int): Total number of ranks.
    local_mem_size(int): The size of shared memory currently occupied by current rank.
    ip_port(str): The ip and port number of the sever, e.g. tcp://ip:port.
    attributes(InitAttr): Attributes set.
Returns:
    On success, returns 0. On error, error code on failure.
)");

    m.def("shmem_init_status", []() {
        int32_t ret = shmem_init_status();
        return static_cast<shmem_init_status_t>(ret);
    }, py::call_guard<py::gil_scoped_release>(), R"(
Query the current initialization status of shared memory module.

Returns:
    Returns initialization status. Returning SHMEM_STATUS_IS_INITIALIZED indicates that initialization is complete.
    All return types can be found in shmem_init_status_t.
    )");

    m.def("shmem_set_data_op_engine_type", &shm::shmem_set_op_engine_type, py::call_guard<py::gil_scoped_release>(),
          py::arg("attributes"), py::arg("vaue"), R"(
Modify the data operation engine type in the attributes that will be used for initialization.
Arguments:
    attributes(InitAttr): Attributes set.
    value(int): Value of data operation engine type.
Returns:
    On success, returns 0. On error, error code on failure.
    )");

    m.def("shmem_set_timeout", &shm::set_timeout, py::call_guard<py::gil_scoped_release>(),
          py::arg("attributes"), py::arg("vaue"), R"(
Modify the timeout in the attributes that will be used for initialization.
Arguments:
    attributes(InitAttr): Attributes set.
    value(int): Value of data operation engine type.
Returns:
    On success, returns 0. On error, error code on failure.
    )");

    m.def(
        "shmem_malloc",
        [](size_t size) {
            auto ptr = shmem_malloc(size);
            if (ptr == nullptr) {
                throw std::runtime_error("shmem_malloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("size"),
        R"(
Allocates size bytes and returns a pointer to the allocated memory. The memory is not initialized. If size is 0, then
shmem_malloc() returns NULL.
    )");

    m.def(
        "shmem_calloc",
        [](size_t nmemb, size_t size) {
            auto ptr = shmem_calloc(nmemb, size);
            if (ptr == nullptr) {
                throw std::runtime_error("shmem_calloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("nmemb"),
        py::arg("size"),
        R"(
Allocates memory for an array of nmemb elements of size bytes each and returns a pointer to the allocated memory.
The memory is set to zero. If nmemb or size is 0, then returns NULL.

Arguments:
    nmemb(int): number of elements
    size(int): size of each element in bytes
Returns:
    pointer to the allocated memory
    )");

    m.def(
        "shmem_align",
        [](size_t alignment, size_t size) {
            auto ptr = shmem_align(alignment, size);
            if (ptr == nullptr) {
                throw std::runtime_error("shmem_align failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("alignment"),
        py::arg("size"),
        R"(
Allocates size bytes of memory with specified alignment and returns a pointer to the allocated memory.
The memory address will be a multiple of the given alignment value (must be a power of two).

Arguments:
    alignment(int): required memory alignment (must be power of two)
    size(int): number of bytes to allocate
Returns:
    Pointer to the allocated memory, or NULL if allocation failed
    )");

    m.def(
        "shmem_free",
        [](intptr_t ptr) {
            auto mem = (void *)ptr;
            shmem_free(mem);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"),
        R"(
Frees the memory space pointed to by ptr, which must have been returned by a previous call to shmem_malloc.
    )");

    m.def(
        "shmem_ptr", [](intptr_t ptr, int pe) { return (intptr_t)shmem_ptr((void *)ptr, pe); },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"), py::arg("peer"), R"(
Get address that may be used to directly reference dest on the specified PE.

Arguments:
    ptr(int): The symmetric address of the remotely accessible data.
    pe(int): PE number
    )");

    m.def("my_pe", &shmem_my_pe, py::call_guard<py::gil_scoped_release>(), R"(Get my PE number.)");

    m.def("pe_count", &shmem_n_pes, py::call_guard<py::gil_scoped_release>(), R"(Get number of PEs.)");

    m.def("mte_set_ub_params", &shmem_mte_set_ub_params, py::call_guard<py::gil_scoped_release>(), py::arg("offset"),
          py::arg("size"), py::arg("event"), R"(
Set the params of UB used for MTE operation initiated by NPU.

Arguments:
    offset(int): start offset of UB
    size(int): size of UB
    event(int): event_id used for sync
    )");

    m.def(
        "team_split_strided",
        [](int parent, int start, int stride, int size) {
            shmem_team_t new_team;
            auto ret = shmem_team_split_strided(parent, start, stride, size, &new_team);
            if (ret != 0) {
                std::cerr << "split parent team(" << parent << ") failed: " << ret << std::endl;
                return ret;
            }
            return new_team;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("parent"), py::arg("start"), py::arg("stride"),
        py::arg("size"), R"(
Split team from an existing parent team, this is a collective operation.

Arguments:
    parent(int): parent team id
    start(int): the lowest PE number of the subset of PEs from parent team that will form the new team
    stride(int): the stride between team PE numbers in the parent team
    size(int): the number of PEs from the parent team
Returns:
    On success, returns new team id. On error, -1 is returned.
    )");

    m.def(
        "shmem_team_split_2d",
        [](int parent, int x_range) {
            shmem_team_t new_x_team;
            shmem_team_t new_y_team;
            auto ret = shmem_team_split_2d(parent, x_range, &new_x_team, &new_y_team);
            if (ret != 0) {
                std::cerr << "split parent team(" << parent << ") failed: " << ret << std::endl;
                return std::make_pair(ret, ret);
            }
            return std::make_pair(new_x_team, new_y_team);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("parent"), py::arg("x_range"),  R"(
Collective Interface. Split team from an existing parent team based on a 2D Cartsian Space

Arguments:
    parent_team       [in] A team handle.
    x_range           [in] represents the number of elements in the first dimensions
    x_team            [in] A new x-axis team after team split.
    y_team            [in] A new y-axis team after team split.
Returns:
    On success, returns new x team id and new y team id. On error, (-1, -1) is returned.
    )");

    m.def(
        "shmem_team_get_config",
        [](int team, shmem_team_config_t team_config){
            return shmem_team_get_config(team, &team_config);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("team"), py::arg("team_config"), R"(
Return team config which pass in as team created

Arguments:
    team(int)                 [in] team id
    team_config(TeamConfig)   [out] the config of team
Returns:
    0 if success, -1 if fail.
    )");

    m.def(
        "shmem_putmem",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void*)dst;
            auto src_addr = (void*)src;
            shmem_putmem(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"),
        py::arg("elem_size"), py::arg("pe"), R"(
Synchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE

Arguments:
    dst                [in] Pointer on Symmetric addr of local PE.
    src                [in] Pointer on local memory of the source data.
    elem_size          [in] size of elements in the destination and source addr.
    pe                 [in] PE number of the remote PE.
    )");

    m.def(
        "shmem_getmem",
        [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
            auto dst_addr = (void*)dst;
            auto src_addr = (void*)src;
            shmem_getmem(dst_addr, src_addr, elem_size, pe);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"),
        py::arg("elem_size"), py::arg("pe"),  R"(
Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE

Arguments:
    dst                [in] Pointer on Symmetric addr of local PE.
    src                [in] Pointer on local memory of the source data.
    elem_size          [in] size of elements in the destination and source addr.
    pe                 [in] PE number of the remote PE.
    )");

    
    m.def(
        "shmem_info_get_version",
        []() {
            int major = 0;
            int minor = 0;
            shmem_info_get_version(&major, &minor);
            return major, minor;
        },
        py::call_guard<py::gil_scoped_release>(), R"(
Returns the major and minor version.

Arguments:
    None
Returns:
    major(int)      [out]major version
    minor(int)      [out]minor version
    )");

    m.def(
        "shmem_info_get_name",
        []() {
            std::string name;
            name.resize(SHMEM_MAX_NAME_LEN);
            shmem_info_get_name(name.c_str());
            return name;
        },
        py::call_guard<py::gil_scoped_release>(), R"(
returns the vendor defined name string.

Arguments:
    None
Returns:
    name(str)      [out]defined name
    )");

#define PYBIND_SHMEM_TYPENAME_P(NAME, TYPE)                                                 \
    {                                                                                       \
        std::string funcName = "shmem_" #NAME "_p";                                         \
        m.def(                                                                              \
            funcName.c_str(),                                                               \
            [](intptr_t dst, const TYPE value, int pe) {                                    \
                auto dst_addr = (TYPE*)dst;                                                 \
                shmem_##NAME##_p(dst_addr, value, pe);                                      \
            },                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("value"),     \
            py::arg("pe"), R"(                                                              \
    Provide a low latency put capability for single element of most basic types             \
                                                                                            \
    Arguments:                                                                              \
        dst               [in] Symmetric address of the destination data on local PE.       \
        value             [in] The element to be put.                                       \
        pe                [in] The number of the remote PE.                                 \
        )");                                                                                \
    }


SHMEM_TYPE_FUNC(PYBIND_SHMEM_TYPENAME_P)
#undef PYBIND_SHMEM_TYPENAME_P

#define PYBIND_SHMEM_TYPENAME_G(NAME, TYPE)                                                 \
    {                                                                                       \
        std::string funcName = "shmem_" #NAME "_g";                                         \
        m.def(                                                                              \
            funcName.c_str(),                                                               \
            [](intptr_t src, int pe) {                                    \
                auto src_addr = (TYPE*)src;                                                 \
                return shmem_##NAME##_g(src_addr, pe);                                      \
            },                                                                              \
            py::call_guard<py::gil_scoped_release>(), py::arg("src"),     \
            py::arg("pe"), R"(                                                              \
    Provide a low latency get single element of most basic types.             \
                                                                                            \
    Arguments:                                                                              \
        src               [in] Symmetric address of the destination data on local PE.  \
        pe                [in] The number of the remote PE.                            \
        A single element of type specified in the input pointer.                             \
        )");                                                                                \
    }


SHMEM_TYPE_FUNC(PYBIND_SHMEM_TYPENAME_G)
#undef PYBIND_SHMEM_TYPENAME_G

    m.def("team_translate_pe", &shmem_team_translate_pe, py::call_guard<py::gil_scoped_release>(),
          py::arg("src_team"), py::arg("src_pe"), py::arg("dest_team"), R"(
Translate a given PE number in one team into the corresponding PE number in another team

Arguments:
    src_team(int): source team id
    src_pe(int): source PE number
    dest_team(int): destination team id
Returns:
    On success, returns the specified PE's number in the dest_team. On error, -1 is returned.
    )");

    m.def("team_destroy", &shmem_team_destroy, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Destroy a team with team id

Arguments:
    team(int): team id to be destroyed
    )");

    m.def("get_ffts_config", &shmemx_get_ffts_config, py::call_guard<py::gil_scoped_release>(), R"(
Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel using shmemx_set_ffts.
    )");

    m.def(
    "shmem_putmem_nbi",
    [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
    auto dst_addr = (void*)dst;
    auto src_addr = (void*)src;
    shmem_putmem_nbi(dst_addr, src_addr, elem_size, pe);
    },
    py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"),
            py::arg("elem_size"), py::arg("pe"), R"(
    Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.

    Arguments:
        dst                [in] Pointer on Symmetric addr of local PE.
        src                [in] Pointer on local memory of the source data.
        elem_size          [in] size of elements in the destination and source addr.
        pe                 [in] PE number of the remote PE.
        )");

    m.def(
    "shmem_getmem_nbi",
    [](intptr_t dst, intptr_t src, size_t elem_size, int pe) {
    auto dst_addr = (void*)dst;
    auto src_addr = (void*)src;
    shmem_getmem_nbi(dst_addr, src_addr, elem_size, pe);
    },
    py::call_guard<py::gil_scoped_release>(), py::arg("dst"), py::arg("src"),
            py::arg("elem_size"), py::arg("pe"),  R"(
    Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE

    Arguments:
        dst                [in] Pointer on Symmetric addr of local PE.
        src                [in] Pointer on local memory of the source data.
        elem_size          [in] size of elements in the destination and source addr.
        pe                 [in] PE number of the remote PE.
        )");

    m.def("my_pe", &shmem_team_my_pe, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get my PE number within a team, i.e. index of the PE

Arguments:
    team(int): team id
Returns:
    On success, returns the PE's number in the specified team. On error, -1 is returned.
    )");

    m.def("pe_count", &shmem_team_n_pes, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get number of PEs with in a team, i.e. how many PEs in the team.

Arguments:
    team(int): team id
Returns:
    On success, returns total number of PEs in the specified team. On error, -1 is returned.
    )");
}
