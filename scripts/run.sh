#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd ${PROJECT_ROOT}

set -e
RANK_SIZE="8"
IPPORT="tcp://127.0.0.1:8666"
GNPU_NUM="8"
FIRST_NPU="0"
FIRST_RANK="0"
if [ -z "${GTEST_FILTER}" ]; then
    TEST_FILTER="*.*"
else
    TEST_FILTER="${GTEST_FILTER}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -ranks requires a numeric value."
                    exit 1
                fi
                RANK_SIZE="$2"
                if [ "$GNPU_NUM" -gt "$RANK_SIZE" ]; then
                    GNPU_NUM="$RANK_SIZE"
                    echo "Because GNPU_NUM is greater than RANK_SIZE, GNPU_NUM is assigned the value of RANK_SIZE=${RANK_SIZE}."
                fi
                shift 2
            else
                echo "Error: -ranks requires a value."
                exit 1
            fi
            ;;
        -frank)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -frank requires a numeric value."
                    exit 1
                fi
                FIRST_RANK="$2"
                shift 2
            else
                echo "Error: -frank requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                if [[ "$2" =~ ^[a-zA-z0-9.:/_-]+$ ]]; then
                    IPPORT="$2"
                    shift 2
                else
                    echo "Error: Invalid -ipport format, only alphanumeric and :/_- allowed"
                    exit 1
                fi
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -gnpus requires a numeric value."
                    exit 1
                fi
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: -fnpu requires a numeric value."
                    exit 1
                fi
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -test_filter)
            if [ -n "$2" ]; then
                FILTERED_VALUE=$(echo "$2" | sed 's/[;&|]*//g')
                if [ -z "$FILTERED_VALUE" ]; then
                    echo "Invalid test_filter value"
                    exit 1
                fi
                TEST_FILTER="*$2*"
                shift 2
            else
                echo "Error: -test_filter requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/install/memfabric_hybrid/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
./build/bin/shmem_unittest "$RANK_SIZE" "$IPPORT" "$GNPU_NUM" "$FIRST_RANK" "$FIRST_NPU"  --gtest_output=xml:test_detail.xml --gtest_filter=${TEST_FILTER}

cd ${CURRENT_DIR}