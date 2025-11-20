#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/src/memfabric_hybrid/output/smem/lib64/:${PROJECT_ROOT}/src/memfabric_hybrid/output/hybm/lib64/:$LD_LIBRARY_PATH
./build/bin/rdma_demo 2 0 tcp://127.0.0.1:8765 2 0 0 & # rank 0
./build/bin/rdma_demo 2 1 tcp://127.0.0.1:8765 2 0 0 & # rank 1