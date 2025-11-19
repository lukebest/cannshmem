# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

option(ENABLE_TRACE_LOG "Enable trace log" OFF)
option(ENABLE_ASAN "Enable ASAN" OFF)
option(BUILD_TESTS "build test or not" OFF)
option(BUILD_OPEN_ABI "build open _GLIBCXX_USE_CXX11_ABI" ON)
option(ENABLE_CPU_MONOTONIC "enable monotonic time using cpu hard instruction" OFF)
option(BUILD_GIT_COMMIT "build library version with commit" ON)
option(BUILD_PYTHON "build python file" ON)

message(STATUS "BUILD_TESTS = ${BUILD_TESTS}")
message(STATUS "BUILD_USE_CXX11_ABI = ${BUILD_OPEN_ABI}")
message(STATUS "ENABLE_CPU_MONOTONIC = ${ENABLE_CPU_MONOTONIC}")
message(STATUS "BUILD_GIT_COMMIT = ${BUILD_GIT_COMMIT}")
message(STATUS "BUILD_PYTHON = ${BUILD_PYTHON}")
message(STATUS "CMAKE_COMPILER = ${CMAKE_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# set(CMAKE_VERBOSE_MAKEFILE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif ()

add_compile_options(-Wunused-variable -Wunused-value -Wcast-align)
add_compile_options(-fpermissive -Wno-error)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wfloat-equal -Wextra)

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-fPIE -fPIC -ftrapv)
    add_compile_options(-D_FORTIFY_SOURCE=2 -O2)
    add_link_options(-Wl,-z,noexecstack,-z,relro,-z,now)
    add_link_options(-s)
    set(CMAKE_SKIP_RPATH TRUE)
    message(STATUS "build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic -fPIC)
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "ASAN")
    # 开启多种 sanitizer   "address,undefined,pointer-compare,pointer-subtract,alignment"
    set(SANITIZERS "address,undefined,pointer-compare,pointer-subtract,alignment")
    # 编译选项
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=${SANITIZERS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${SANITIZERS}")
    # 链接选项
    add_link_options(-fsanitize=${SANITIZERS})
endif ()

if (NOT BUILD_OPEN_ABI STREQUAL "ON")
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
endif ()

if (ENABLE_CPU_MONOTONIC STREQUAL "ON")
    add_compile_options(-DENABLE_CPU_MONOTONIC)
endif ()

# set log commit into compile definition
if (BUILD_GIT_COMMIT STREQUAL "ON")
    find_program(GIT_EXECUTABLE NAMES git)
    if (EXISTS ${GIT_EXECUTABLE})
        # get commit id from file
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            RESULT_VARIABLE GIT_COMMIT_RESULT
            OUTPUT_VARIABLE GIT_LAST_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (${GIT_COMMIT_RESULT} EQUAL 1)
            set(GIT_LAST_COMMIT "empty")
        endif ()
        add_definitions(-DGIT_LAST_COMMIT=${GIT_LAST_COMMIT})
        message(STATUS "add definition for last commit: ${GIT_LAST_COMMIT}")
    else ()
        add_definitions(-DGIT_LAST_COMMIT=empty)
        message(STATUS "Failed to find git command, not GIT_LAST_COMMIT will be set")
    endif ()
else ()
    add_definitions(-DGIT_LAST_COMMIT=empty)
    message(STATUS "add definition for last commit: empty")
endif ()

if (ENABLE_TRACE_L0G)
    add_definitions(-DLOG_TRACE_INFO_ENABLED)
endif ()

add_compile_options(
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-const-variable
    -Wno-c99-designator
    -Wno-implicit-exception-spec-mismatch
    -Wno-string-compare
)

set(BUILD_PATH ${PROJECT_BINARY_DIR}/)
set(PROJECT_SMEM_SRC_BASE ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem)
set(PROJECT_HYBM_SRC_BASE ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/hybm)
set(PROJECT_ACCLINKS_SRC_BASE ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/acc_links)

include_directories(
        ${PROJECT_SMEM_SRC_BASE}/include/host
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/hybm/include
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/utils
)

add_subdirectory(src)