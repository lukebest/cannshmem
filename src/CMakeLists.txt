# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_subdirectory(memfabric_hybrid)
if (BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(host/python_wrapper)
endif ()

# shmem_device OBJECT库的建立
file(GLOB_RECURSE SHMEM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp)
add_library(shmem_device OBJECT ${SHMEM_KERNEL_FILES})
target_compile_options(shmem_device PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(shmem_device
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/device/
)

# shmem_host OBJECT库的建立
file(GLOB_RECURSE SHMEM_HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
list(FILTER SHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
add_library(shmem_host OBJECT ${SHMEM_HOST_FILES})
target_compile_options(shmem_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(shmem_host
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/src/memfabric_hybrid/src/smem/include/host/
        ${PROJECT_SOURCE_DIR}/src/host
        ${PROJECT_SOURCE_DIR}/src/device
)

add_library(shmem SHARED $<TARGET_OBJECTS:shmem_device> $<TARGET_OBJECTS:shmem_host>)
target_link_options(shmem PRIVATE --cce-fatobj-link)

target_link_libraries(shmem PUBLIC smem_shared hybmm_shared)

# 安装配置
install(TARGETS shmem
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)
