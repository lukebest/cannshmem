# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include_directories(
        ${PROJECT_SMEM_SRC_BASE}/include/host
        ${PROJECT_SMEM_SRC_BASE}/csrc/common
        ${PROJECT_SMEM_SRC_BASE}/csrc/config_store
        ${PROJECT_SMEM_SRC_BASE}/csrc/net
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem_shm
        ${PROJECT_HYBM_SRC_BASE}/include
        ${PROJECT_ACCLINKS_SRC_BASE}/include
)

file(GLOB_RECURSE PLATFORM_SRC
        ${PROJECT_SMEM_SRC_BASE}/csrc/common/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/config_store/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/net/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem_shm/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem_bm/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem_trans/*.cpp
        ${PROJECT_SMEM_SRC_BASE}/csrc/smem.cpp
)

## build the static and shared library
add_library(smem_objects OBJECT ${PLATFORM_SRC})
target_link_libraries(smem_objects PRIVATE hybmm_objects)

target_link_libraries(smem_objects PUBLIC smem)
target_compile_options(smem_objects PRIVATE -fPIC)

add_library(smem_static STATIC $<TARGET_OBJECTS:smem_objects>)
set_target_properties(smem_static PROPERTIES OUTPUT_NAME "mf_smem")
target_link_directories(smem_static PUBLIC ${CMAKE_BINARY_DIR}/memfabric_hybrid/acc_links)
target_link_libraries(smem_static PUBLIC pthread acc_tcp_net_static hybmm_static)

add_library(smem_shared SHARED $<TARGET_OBJECTS:smem_objects>)
set_target_properties(smem_shared PROPERTIES OUTPUT_NAME "mf_smem"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/memfabric_hybrid/smem")
target_link_directories(smem_shared PUBLIC ${CMAKE_BINARY_DIR}/memfabric_hybrid/acc_links)
target_link_libraries(smem_shared PUBLIC pthread acc_tcp_net_static hybmm_shared)
set_target_properties(smem_shared PROPERTIES LINK_LIBRARIES "")

if(BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(python_wrapper)
endif()