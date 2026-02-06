## USER 🧑‍💻

This is the Gemini CLI. We are setting up the context for our chat.
Today's date is Friday, February 6, 2026 (formatted according to the user's locale).
My operating system is: linux
The project's temporary directory is: /home/luke/.gemini/tmp/6746d22a74d214e02f1f496f67c4f1780af27eda11f779fb053f5ac0d22290ee
I'm currently working in the directory: /home/luke/shmem
Here is the folder structure of the current working directories:

Showing up to 200 items (files + folders). Folders or files indicated with ... contain more items not shown, were ignored, or the display limit (200 items) was reached.

/home/luke/shmem/
├───.gitignore
├───barrier_analysis_report.md
├───barrier_analysis_report2.md
├───barrier_comparison.md
├───barrier_details.md
├───barrier_flow_analysis.md
├───CMakeLists.txt
├───hardware_barrier_offload_design.md
├───hardware_barrier.md
├───issues.md
├───LICENSE
├───OAT.xml
├───OWNERS
├───README.md
├───SECURITY.md
├───shmem_init.md
├───shmemi_barrier_core_soft_analysis.md
├───shmemi_barrier_npu_v3_analysis.md
├───Third_Party_Open_Source_Software_Notice
├───version.info
├───.gemini/...
├───.git/...
├───docs/
│   ├───api_demo.md
│   ├───code_organization.md
│   ├───conf.py
│   ├───CONTRIBUTING.md
│   ├───def.rst
│   ├───device_api.rst
│   ├───example.md
│   ├───host_api.rst
│   ├───index.rst
│   ├───principles.md
│   ├───pythonAPI.md
│   ├───quickstart.md
│   ├───related_scripts.md
│   ├───Troubleshooting_FAQs.md
│   ├───contributors/
│   │   ├───code-of-conduct.md
│   │   ├───gitee-workflow.md
│   │   ├───infra-command.md
│   │   ├───infra-faqs.md
│   │   └───issue-submit.md
│   ├───doxygen/
│   │   └───custom.css
│   ├───images/
│   │   ├───contri-flow.png
│   │   ├───gitee-clone.png
│   │   ├───gitee-conflict.png
│   │   ├───gitee-fork.png
│   │   ├───gitee-ssh.png
│   │   ├───issue-banner.png
│   │   ├───issue-details.png
│   │   ├───iusse_select.png
│   │   └───pr-list.png
│   └───picture/
│       ├───1.png
│       ├───2.png
│       ├───3.png
│       ├───4.png
│       ├───5.png
│       ├───6.png
│       ├───7.png
│       ├───8.png
│       └───malloc.png
├───examples/
│   ├───CMakeLists.txt
│   ├───allgather/
│   │   ├───allgather_kernel.cpp
│   │   ├───allgather_kernel.h
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   ├───run.sh
│   │   └───scripts/
│   ├───allgather_matmul/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   └───scripts/
│   ├───allgather_matmul_padding/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   └───scripts/
│   ├───allgather_matmul_with_gather_result/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   └───scripts/
│   ├───dispatch_gmm_combine/
│   │   ├───CMakeLists.txt
│   │   ├───dispatch_gmm_combine_analysis.md
│   │   ├───main.cpp
│   │   ├───README.md
│   │   ├───include/
│   │   ├───scripts/
│   │   └───utils/
│   ├───dynamic_tiling/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   ├───impl/
│   │   ├───include/
│   │   ├───scripts/
│   │   ├───tiling/
│   │   └───utils/
│   ├───kv_shuffle/
│   │   ├───CMakeLists.txt
│   │   ├───kv_shuffle_kernel.cpp
│   │   ├───kv_shuffle_kernel.h
│   │   ├───main.cpp
│   │   └───scripts/
│   ├───matmul_allreduce/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   ├───epilogue/
│   │   └───scripts/
│   ├───matmul_reduce_scatter/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   └───scripts/
│   ├───matmul_reduce_scatter_padding/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───README.md
│   │   └───scripts/
│   ├───python_extension/
│   │   ├───CMakeLists.txt
│   │   ├───run.sh
│   │   ├───src/
│   │   └───test/
│   ├───rdma_demo/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───rdma_demo_kernel.cpp
│   │   ├───README.md
│   │   └───run.sh
│   ├───rdma_handlewait_test/
│   │   ├───unuse_handlewait/
│   │   └───use_handlewait/
│   ├───rdma_perftest/
│   │   ├───CMakeLists.txt
│   │   ├───main.cpp
│   │   ├───rdma_perftest_kernel.cpp
│   │   ├───README.md
│   │   └───run.sh
│   ├───shared_lib/
│   │   ├───CMakeLists.txt
│   │   ├───include/
│   │   └───src/
│   ├───templates/
│   │   ├───docs/
│   │   └───include/
│   └───utils/
│       ├───gen_data.py
│       ├───param.h
│       ├───utils.h
│       ├───utils.py
│       └───verify_result.py
├───include/
│   ├───debug.h
│   ├───shmem_api.h
│   ├───device/
│   │   ├───shmem_device_atomic.h
│   │   ├───shmem_device_def.h
│   │   ├───shmem_device_rma.h
│   │   ├───shmem_device_sync.h
│   │   ├───shmem_device_team.h
│   │   ├───shmemx_device_rma.h
│   │   └───low_level/
│   ├───host/
│   │   ├───shmem_host_def.h
│   │   ├───shmem_host_heap.h
│   │   ├───shmem_host_init.h
│   │   └───shmem_host_rma.h
│   │   └───...
│   ├───host_device/
│   └───internal/
├───scripts/
│   ├───build.sh
│   ├───help.info
│   ├───install.sh
│   ├───release.sh
│   ├───run_examples.sh
│   ├───run_fuzz.sh
│   ├───run.sh
│   ├───set_env.sh
│   └───uninstall.sh
├───src/
│   ├───CMakeLists.txt
│   ├───device/
│   ├───host/
│   ├───memfabric_hybrid/
│   ├───python/
│   └───transport/
└───tests/
    ├───examples/
    ├───fuzz/
    └───unittest/

Reminder: Do not return an empty response when a tool call is required.

My setup is complete. I will provide my first command in the next turn.

---

## USER 🧑‍💻

examples/dispatch_gmm_combine中，Dispatch算子有卡间同步吗？在哪里体现？只看到了Combine中调用了一次CrossRankSync，请解释并生成markdown

---

## MODEL ✨

I will begin by listing the files in `examples/dispatch_gmm_combine` to locate the relevant source code for the `Dispatch` operator.
**Tool Command**:
```json
{
  "name": "list_directory",
  "args": {
    "dir_path": "examples/dispatch_gmm_combine"
  }
}
```

---

## USER 🧑‍💻

**Tool Response**:
```json
{
  "id": "list_directory-1770367449248-ba6d4ddd6fb128",
  "name": "list_directory",
  "response": {
    "output": "Directory listing for /home/luke/shmem/examples/dispatch_gmm_combine:\n[DIR] include\n[DIR] scripts\n[DIR] utils\nCMakeLists.txt\ndispatch_gmm_combine_analysis.md\nmain.cpp\nREADME.md"
  }
}
```

---

## MODEL ✨

I will check the `include` directory for kernel implementations.
**Tool Command**:
```json
{
  "name": "list_directory",
  "args": {
    "dir_path": "examples/dispatch_gmm_combine/include"
  }
}
```

---

## USER 🧑‍💻

**Tool Response**:
```json
{
  "id": "list_directory-1770367452696-0642d2f229af9",
  "name": "list_directory",
  "response": {
    "output": "Directory listing for /home/luke/shmem/examples/dispatch_gmm_combine/include:\n[DIR] moe_init_routing_quant_v2\n[DIR] tiling\nblock_epilogue_pertoken_row.h\nblock_epilogue_pertoken_swiglu.h\nblock_mmad_preload_async_fixpipe_quant.h\nblock_mmad_preload_fixpipe.h\ncopy_gm_to_l1_custom.h\ncopy_l0c_to_gm_custom.h\ndispatch_gmm_combine.h\ndispatch_policy_custom.h\nmoe_token_unpermute_tiling.h\nmoe_token_unpermute.h\nselect_helper.h\nsync_util.h"
  }
}
```

---

## MODEL ✨

I will examine `examples/dispatch_gmm_combine/include/dispatch_gmm_combine.h` to understand the implementation of the `Dispatch` operator.
**Tool Command**:
```json
{
  "name": "read_file",
  "args": {
    "file_path": "examples/dispatch_gmm_combine/include/dispatch_gmm_combine.h"
  }
}
```

---

## USER 🧑‍💻

**Tool Response**:
```json
{
  "id": "read_file-1770367455253-44d263682b1a18",
  "name": "read_file",
  "response": {
    "output": "/*\n * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.\n * This program is free software, you can redistribute it and/or modify it under the terms and conditions of\n * CANN Open Software License Agreement Version 2.0 (the \"License\").\n * Please refer to the License for details. You may not use this file except in compliance with the License.\n * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n * See LICENSE in the root of the software repository for the full text of the License.\n */\n\n#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_DEQUANT_FIXPIPE_H\n#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_DEQUANT_FIXPIPE_H\n\n#include \"catlass/catlass.hpp\"\n#include \"catlass/arch/cross_core_sync.hpp\"\n#include \"catlass/arch/resource.hpp\"\n#include \"catlass/coord.hpp\"\n#include \"catlass/detail/callback.hpp\"\n#include \"catlass/gemm_coord.hpp\"\n#include \"catlass/matrix_coord.hpp\"\n#include \"catlass/epilogue/tile/tile_copy.hpp\"\n#include \"shmem_api.h\"\n\n#include \"block_mmad_preload_async_fixpipe_quant.h\"\n#include \"block_mmad_preload_fixpipe.h\"\n#include \"copy_gm_to_l1_custom.h\"\n#include \"copy_l0c_to_gm_custom.h\"\n#include \"block_epilogue_pertoken_row.h\"\n#include \"block_epilogue_pertoken_swiglu.h\"\n#include \"sync_util.h\"\n#include \"kernel_operator.h\"\n\n#include \"catlass/gemm/helper.hpp\"\n#include \"kernel_operator.h\"\n#include \"tiling/moe_init_routing_quant_v2_tiling.h\"\n\n#include \"moe_init_routing_quant_v2/moe_init_routing_quant_v2.h\"\n#include \"moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h\"\n\n#include \"moe_token_unpermute.h\"\n\nusing namespace AscendC;\ninline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};\n\nnamespace Catlass::Gemm::Kernel {\nconstexpr uint32_t EXPERT_OFFSET = 8;\nconstexpr uint32_t FLAG_VALUE = 2;\nconstexpr uint32_t LOOP_DENOMINATOR = 8;\n\ntemplate<\n    class BlockMmad_,\n    class BlockScheduler_,\n    class ElementGroupList_,\n    class BlockEpilogue1_,\n    class BlockEpilogue2_\n>\nclass DispatchGmmCombineKernel {\npublic:\n    using BlockMmad = BlockMmad_;\n    using ArchTag = typename BlockMmad::ArchTag;\n    using L1TileShape = typename BlockMmad::L1TileShape;\n    using ElementA = typename BlockMmad::ElementA;\n    using LayoutA = typename BlockMmad::LayoutA;\n    using ElementB = typename BlockMmad::ElementB;\n    using LayoutB = typename BlockMmad::LayoutB;\n    using ElementC = typename BlockMmad::ElementC;\n    using LayoutC = typename BlockMmad::LayoutC;\n    using ElementAccumulator = typename BlockMmad::ElementAccumulator;\n    using ElementScale = uint64_t;\n    using LayoutScale = typename layout::VectorLayout;\n    using ElementPerTokenScale = float;\n    using LayoutPerTokenScale = typename layout::VectorLayout;\n    using BlockScheduler = BlockScheduler_;\n    using BlockEpilogue1 = BlockEpilogue1_;\n    using BlockEpilogue2 = BlockEpilogue2_;\n    using ElementD1 = typename BlockEpilogue1::ElementD;\n    using LayoutD1 = typename BlockEpilogue1::LayoutD;\n    using ElementD2 = typename BlockEpilogue2::ElementD;\n    using LayoutD2 = typename BlockEpilogue2::LayoutD;\n\n    /// Parameters structure\n    struct Params {\n        // Data members\n        GemmCoord problemShape;\n        __gm__ ElementA *ptrA;\n        LayoutA layoutA;\n        LayoutA layoutA2;\n        __gm__ ElementB *ptrB1;\n        LayoutB layoutB1;\n        __gm__ ElementB *ptrB2;\n        LayoutB layoutB2;\n        __gm__ ElementScale *ptrScale1;\n        LayoutScale layoutScale1;\n        __gm__ ElementScale *ptrScale2;\n        LayoutScale layoutScale2;\n        __gm__ ElementD2 *ptrOutput;\n        LayoutD1 layoutD1;\n        LayoutD2 layoutD2;\n        GM_ADDR ptrWorkspace;\n        int32_t EP;\n        int32_t expertPerRank;\n        uint32_t maxOutputSize;\n        uint32_t rank;\n        uint32_t rankSize;\n        GM_ADDR symmetricPtr;\n        int32_t ubMoveNum;\n        GM_ADDR expertIdx;\n        GM_ADDR moeInitRoutingQuantV2Scale;\n        GM_ADDR moeInitRoutingQuantV2Offset;\n        GM_ADDR expandedX;\n        GM_ADDR expandedRowIdx;\n        GM_ADDR expertTokensCountOrCumsum;\n        GM_ADDR expertTokensBeforeCapacity;\n        GM_ADDR dynamicQuantScale;\n        GM_ADDR probs;\n        int64_t activeNum;\n        int64_t expertCapacity;\n        int64_t expertNum;\n        int64_t dropPadMode;\n        int64_t expertTokensCountOrCumsumFlag;\n        bool expertTokensBeforeCapacityFlag;\n        int64_t quantMode;\n        int64_t topK;\n        uint64_t initRoutingQuantTilingKey;\n        uint32_t epilogueCoreNum;\n        uint32_t epilogueGranularity;\n\n        optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;\n\n        // Methods\n        CATLASS_HOST_DEVICE\n        Params()\n        {}\n\n        CATLASS_HOST_DEVICE\n        Params(\n                GemmCoord problemShape_,\n                uint32_t EP_, uint32_t expertPerRank_, uint32_t maxOutputSize_,\n                uint32_t rank_, uint32_t rankSize_,\n                int64_t activeNum_, int64_t expertCapacity_, int64_t expertNum_,\n                int64_t dropPadMode_, int64_t expertTokensCountOrCumsumFlag_,\n                bool expertTokensBeforeCapacityFlag_, int64_t quantMode_, int64_t topK_,\n                uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_, uint32_t epilogueGranularity_,\n                GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_,\n                GM_ADDR ptrB1_, LayoutB layoutB1_,\n                GM_ADDR ptrB2_, LayoutB layoutB2_,\n                GM_ADDR ptrScale1_, LayoutScale layoutScale1_,\n                GM_ADDR ptrScale2_, LayoutScale layoutScale2_,\n                GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_,\n                GM_ADDR expertIdx_, GM_ADDR moeInitRoutingQuantV2Scale_,\n                GM_ADDR moeInitRoutingQuantV2Offset_,\n                GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_,\n                GM_ADDR ptrWorkspace_, GM_ADDR symmetricPtr_, int32_t ubMoveNum_,\n                optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_\n        ) : problemShape(problemShape_),\n            EP(EP_), expertPerRank(expertPerRank_), maxOutputSize(maxOutputSize_),\n            rank(rank_), rankSize(rankSize_),\n            activeNum(activeNum_), expertCapacity(expertCapacity_), expertNum(expertNum_),\n            dropPadMode(dropPadMode_), expertTokensCountOrCumsumFlag(expertTokensCountOrCumsumFlag_),\n            expertTokensBeforeCapacityFlag(expertTokensBeforeCapacityFlag_), quantMode(quantMode_), topK(topK_),\n            initRoutingQuantTilingKey(initRoutingQuantTilingKey_),\n            epilogueCoreNum(epilogueCoreNum_), epilogueGranularity(epilogueGranularity_),\n            ptrA(reinterpret_cast<__gm__ ElementA * > (ptrA_)), layoutA(layoutA_), layoutA2(layoutA2_),\n            ptrB1(reinterpret_cast<__gm__ ElementB * > (ptrB1_)), layoutB1(layoutB1_),\n            ptrB2(reinterpret_cast<__gm__ ElementB * > (ptrB2_)), layoutB2(layoutB2_),\n            ptrScale1(reinterpret_cast<__gm__ ElementScale * > (ptrScale1_)),\n            layoutScale1(layoutScale1_),\n            ptrScale2(reinterpret_cast<__gm__ ElementScale * > (ptrScale2_)),\n            layoutScale2(layoutScale2_),\n            ptrOutput(reinterpret_cast<__gm__ ElementD2 * > (ptrOutput_)),\n            layoutD1(layoutD1_), layoutD2(layoutD2_),\n            expertIdx(expertIdx_),\n            moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),\n            moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),\n            expertTokensBeforeCapacity(expertTokensBeforeCapacity_),\n            probs(probs_),\n            ptrWorkspace(ptrWorkspace_), symmetricPtr(symmetricPtr_),\n            ubMoveNum(ubMoveNum_),\n            moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_)\n        {\n            moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingData_.vbsComputeParamsOp;\n            moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp = moeInitRoutingQuantV2TilingData_\n                                                                    .vmsMiddleComputeParamsOp;\n            moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp = moeInitRoutingQuantV2TilingData_\n                                                                    .sortOutComputeParamsOp;\n            moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp = moeInitRoutingQuantV2TilingData_\n                                                                    .srcToDstComputeParamsOp;\n            moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp = moeInitRoutingQuantV2TilingData_\n                                                                    .srcToDstCapacityComputeParamsOp;\n            moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp = moeInitRoutingQuantV2TilingData_\n                                                                    .gatherOutComputeParamsOp;\n        }\n    };\n\n    // Methods\n    CATLASS_DEVICE\n    DispatchGmmCombineKernel(Params const &params)\n    {\n        if ASCEND_IS_AIC {\n            coreIdx = AscendC::GetBlockIdx();\n            coreNum = AscendC::GetBlockNum();\n        }\n\n        if ASCEND_IS_AIV {\n            coreIdx = get_block_idx() + get_subblockid() * get_block_num();\n            coreNum = get_block_num() * get_subblockdim();\n        }\n\n        initBuffer(params);\n\n        if ASCEND_IS_AIV {\n        }\n    }\n\n    CATLASS_DEVICE\n    ~DispatchGmmCombineKernel()\n    {\n        if ASCEND_IS_AIV {\n        }\n    }\n\n    template<int32_t CORE_TYPE = g_coreType>\n    CATLASS_DEVICE\n    void operator()(Params const &params);\n\n    template<>\n    CATLASS_DEVICE\n    void operator()<AscendC::AIC>(Params const &params)\n    {\n        GMM1(params);\n        AscendC::CrossCoreWaitFlag<0x2>(FLAG_VALUE);\n        GMM2(params);\n    }\n\n    template<>\n    CATLASS_DEVICE\n    void operator()<AscendC::AIV>(Params const &params)\n    {\n        Dispatch(params);\n        AscendC::SyncAll<true>();\n        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_VALUE);\n        Combine(params);\n    }\n\nprivate:\n    CATLASS_DEVICE void initBuffer(Params const &params)\n    {\n        workspaceInfo = WorkspaceInfo(params);\n        peermemInfo = PeermemInfo(params);\n\n        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));\n        cumsumSend.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumSend));\n\n        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(workspaceInfo.ptrA));\n        gmS.SetGlobalBuffer(params.ptrScale1);\n        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC));\n\n        gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1*>(workspaceInfo.ptrPermutedToken));\n        gmS2.SetGlobalBuffer(params.ptrScale2);\n        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrC2));\n\n        gmPerTokenScale1.SetGlobalBuffer(\n                reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale));\n        gmPerTokenScale2.SetGlobalBuffer(\n                reinterpret_cast<__gm__ ElementPerTokenScale*>(workspaceInfo.ptrPerTokenScale2));\n\n        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(peermemInfo.ptrPeerTokenPerExpert));\n    }\n\n    template<typename T>\n    CATLASS_DEVICE void CopyGMToGM(\n        AscendC::GlobalTensor<T> dst,\n        AscendC::GlobalTensor<T> src,\n        int32_t elemNum,\n        int32_t ubMoveNum\n    )\n    {\n        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);\n        using TType = Gemm::GemmType<T, layout::RowMajor>;\n        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;\n        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;\n        CopyGmToUb copyGmToUb;\n        CopyUbToGm copyUbToGm;\n        constexpr int32_t BufferNum = 2;\n        int tmpBufferSize = 32 * 1024 / sizeof(T);   // 32 KB\n        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);\n        tmpBuffer1.SetSize(tmpBufferSize);\n        int tmpBufferOffset = 96 * 1024; // half of UB\n        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);\n        tmpBuffer2.SetSize(tmpBufferSize);\n\n        // [ReduceScatter] 2. Pre Interface Sync\n        int pingpongId = 0;\n        auto processCount = CeilDiv(elemNum, ubMoveNum);\n        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {\n            uint32_t curProcessNum = (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) :\n                                      ubMoveNum;\n            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;\n            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;\n            auto processOffset = processIndex * ubMoveNum;\n\n            auto inputOffset = processOffset;\n            auto outputOffset = processOffset;\n            // [ReduceScatter] 2. Pre Interface Sync\n            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);\n            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi\n            copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});\n            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);\n            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);\n            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});\n\n            // [ReduceScatter] 4. Post Interface Sync\n            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);\n            pingpongId = (pingpongId + 1) % BufferNum;\n        }\n        // [ReduceScatter] 4. Post Interface Sync\n\n        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);\n    }\n\n    CATLASS_DEVICE\n    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> &tokenPerExpert, AscendC::GlobalTensor<int32_t> &result,\n                           uint32_t expertPerRank, uint32_t rankId, uint32_t EP)\n    {\n        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;\n        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);\n\n        AscendC::DataCopyPad(\n            tmpBuffer1,\n            tokenPerExpert[rankId * expertPerRank],\n            {static_cast<uint16_t>(EP),\n             static_cast<uint16_t>(expertPerRank * sizeof(int32_t)),\n             static_cast<uint16_t>(((EP - 1) * expertPerRank + 8) * sizeof(int32_t)), 0},\n            {});\n        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);\n        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);\n\n        for (uint32_t i = 1; i < EP; ++i) {\n            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],\n                         tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);\n            AscendC::PipeBarrier<PIPE_V>();\n        }\n\n        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);\n        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);\n\n        AscendC::DataCopyPad(\n            result,\n            tmpBuffer1,\n            {static_cast<uint16_t>(EP), static_cast<uint16_t>((expertPerRank) * sizeof(int32_t)), 0, 0}\n        );\n    }\n\n    CATLASS_DEVICE\n    void GetCumsumForSendRecv(AscendC::GlobalTensor<int32_t> &tokenPerExpert,\n                              AscendC::GlobalTensor<int32_t> &resultSend,\n                              uint32_t expertPerRank, uint32_t rankId, uint32_t EP)\n    {\n        layout::RowMajor commMatLayout{EP, EP * expertPerRank};\n        for (uint32_t i = coreIdx; i < EP; i += coreNum) {\n            int32_t sum = 0;\n            for (uint32_t j = 0; j < rankId; ++j) {\n                int32_t srcIdx = commMatLayout.GetOffset(MakeCoord(i, j * expertPerRank));\n                for (uint32_t k = 0; k < expertPerRank; ++k) {\n                    sum += tokenPerExpert(srcIdx + k);\n                }\n            }\n            resultSend(i) = sum;\n        }\n    }\n\n    CATLASS_DEVICE\n    void weightMatrixPreLoad(Params const &params, int32_t preLoadExperts)\n    {\n        BlockScheduler blockScheduler;\n        AscendC::GlobalTensor<ElementB> gmB1;\n        gmB1.SetGlobalBuffer(params.ptrB1);\n        int64_t gmGroupOffsetB = 0;\n        uint32_t startCoreIdx = 0;\n        uint32_t l1BOffset = 0;\n        for (uint32_t i = 0; i < L1_STAGES; i++) {\n            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);\n        }\n        using LayoutBInL1 = typename helper::L1BTypeSelector<Gemm::GemmType<int8_t, layout::zN>>::L1BType::Layout;\n        static constexpr auto L1B_LAYOUT = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);\n\n        for (uint32_t groupIdx = 0; groupIdx < preLoadExperts; ++groupIdx) {\n            GemmCoord inGroupProblemShape{params.problemShape.k(), params.problemShape.n(), 0};\n            LayoutB layoutB = params.layoutB1;\n            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::K, L1TileShape::N));\n            uint32_t coreLoops = blockScheduler.GetCoreLoops();\n\n            // Determine the starting loopIdx of the current core under the current groupIdx\n            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;\n            // Loop through the matmul of each groupIdx\n            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {\n                // Compute block location\n                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);\n                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);\n\n                // Compute initial location in logical coordinates\n\n                MatrixCoord offsetB{blockCoord.m() * L1TileShape::K, blockCoord.n() * L1TileShape::N};\n                int64_t gmOffsetB = layoutB.GetOffset(offsetB);\n\n                auto gmBlockB = gmB1[gmGroupOffsetB + gmOffsetB];\n                // Load first matrix B tile from GM to L1\n                auto layoutTileB = layoutB.GetTileLayout(actualBlockShape.GetCoordMN());\n                copyGmToL1B(l1BTensorList[l1ListId], gmBlockB, L1B_LAYOUT, layoutTileB);\n                l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;\n            }\n            gmGroupOffsetB += inGroupProblemShape.m() * inGroupProblemShape.n();\n            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;\n        }\n    }\n\n    CATLASS_DEVICE\n    void GMM1(Params const &params)\n    {\n        BlockScheduler blockScheduler;\n        BlockMmad blockMmad(resource);\n\n        int64_t gmGroupOffsetA = 0;\n        int64_t gmGroupOffsetB = 0;\n        int64_t gmGroupOffsetC = 0;\n        uint32_t startCoreIdx = 0;\n        uint32_t syncGroupIdx = 0;\n\n        AscendC::CrossCoreWaitFlag<0x2>(0); // 等待aiv计算cumsumformm\n\n        int64_t preCurrentmSum = 0;\n        int32_t syncLoopIdx = -1;\n        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {\n            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);\n            if (preCurrentmSum >= params.maxOutputSize) {\n                currentM = 0;\n            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {\n                currentM = params.maxOutputSize - preCurrentmSum;\n            }\n            AscendC::GlobalTensor<ElementB> gmB1;\n            gmB1.SetGlobalBuffer(params.ptrB1);\n            if (currentM <= L1TileShape::M) {\n                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);\n            }\n            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};\n            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());\n            LayoutB layoutB1 = params.layoutB1;\n            LayoutScale layoutScale = params.layoutScale1;\n            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());\n            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));\n            uint32_t coreLoops = blockScheduler.GetCoreLoops();\n            // Determine the starting loopIdx of the current core under the current groupIdx\n            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;\n            // Loop through the matmul of each groupIdx\n            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {\n                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {\n                    AscendC::CrossCoreWaitFlag<0x2>(0);\n                }\n                // Compute block location\n                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);\n                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);\n                // Compute initial location in logical coordinates\n                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};\n                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};\n                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};\n                int64_t gmOffsetA = layoutA.GetOffset(offsetA);\n                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);\n                int64_t gmOffsetC = layoutC.GetOffset(offsetC);\n                int64_t gmOffsetS =\n                        groupIdx * params.problemShape.n() + blockCoord.n() * L1TileShape::N;   // 每个expert一组scale\n                if (currentM > 0) {\n                    blockMmad(\n                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,\n                        gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,\n                        gmC[gmGroupOffsetC + gmOffsetC], layoutC,\n                        gmS[gmOffsetS], layoutScale,\n                        actualBlockShape);\n                }\n            }\n\n            if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {\n                syncLoopIdx++;\n                if constexpr(BlockMmad::DispatchPolicy::ASYNC) {\n                    blockMmad.SynchronizeBlock();\n                }\n                blockMmad.Finalize(syncLoopIdx, 1);\n            }\n\n            preCurrentmSum += currentM;\n            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();\n            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();\n            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();\n            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;\n        }\n\n        if constexpr(BlockMmad::DispatchPolicy::ASYNC) {\n            blockMmad.SynchronizeBlock();\n        }\n        blockMmad.Finalize(syncLoopIdx + 1, 1);\n    }\n\n    CATLASS_DEVICE\n    void GMM2(Params const &params)\n    {\n        BlockScheduler blockScheduler;\n        BlockMmad blockMmad(resource);\n\n        uint32_t n2 = params.problemShape.k();\n        uint32_t k2 = params.problemShape.n() / 2;\n\n        int64_t gmGroupOffsetA = 0;\n        int64_t gmGroupOffsetB = 0;\n        int64_t gmGroupOffsetC = 0;\n\n        uint32_t startCoreIdx = 0;\n        constexpr uint32_t OPERATION_IDENTIFIER = 3;\n        constexpr uint32_t OPERATION_IDENTIFIER_2 = 2;\n\n        AscendC::PipeBarrier<PIPE_ALL>();\n\n        int64_t preCurrentmSum = 0;\n        int32_t syncLoopIdx = -1;\n        uint32_t lastDequantExpertNum = params.expertPerRank;\n\n        if (params.epilogueGranularity < params.expertPerRank) {\n            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;\n        }\n\n        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {\n            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);\n            if (preCurrentmSum >= params.maxOutputSize) {\n                currentM = 0;\n            } else if (preCurrentmSum + currentM > params.maxOutputSize) {\n                currentM = params.maxOutputSize - preCurrentmSum;\n            }\n            AscendC::GlobalTensor<ElementB> gmB2;\n            gmB2.SetGlobalBuffer(params.ptrB2);\n            if (currentM <= L1TileShape::M) {\n                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);\n            }\n            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K\n\n            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());\n            LayoutB layoutB2 = params.layoutB2;\n            LayoutScale layoutScale = params.layoutScale2;\n            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());\n\n            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));\n            uint32_t coreLoops = blockScheduler.GetCoreLoops();\n\n            // Determine the starting loopIdx of the current core under the current groupIdx\n            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;\n            if (params.expertPerRank > lastDequantExpertNum &&\n                groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {\n                AscendC::CrossCoreWaitFlag<0x2>(OPERATION_IDENTIFIER_2);\n            }\n\n            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {\n                if (loopIdx + coreNum >= coreLoops) {\n                    syncLoopIdx = groupIdx;\n                }\n                // Compute block location\n                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);\n                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);\n\n                // Compute initial location in logical coordinates\n                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};\n                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};\n                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};\n\n                int64_t gmOffsetA = layoutA.GetOffset(offsetA);\n                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);\n                int64_t gmOffsetC = layoutC.GetOffset(offsetC);\n                int64_t gmOffsetS = groupIdx * n2 + blockCoord.n() * L1TileShape::N;   // 每个expert一组scale\n                if (currentM > 0) {\n                    blockMmad(\n                        gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,\n                        gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,\n                        gmC2[gmGroupOffsetC + gmOffsetC], layoutC,\n                        gmS2[gmOffsetS], layoutScale,\n                        actualBlockShape, syncLoopIdx, OPERATION_IDENTIFIER);\n                }\n            }\n            preCurrentmSum += currentM;\n            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();\n            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();\n            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();\n\n            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;\n        }\n\n        if constexpr(BlockMmad::DispatchPolicy::ASYNC) {\n            blockMmad.SynchronizeBlock();\n        }\n        blockMmad.Finalize(params.expertPerRank - 1, OPERATION_IDENTIFIER);\n    }\n\n    CATLASS_DEVICE\n    void Dispatch(Params const &params)\n    {\n        __gm__ void *srcPeermemPtr = shmem_ptr(params.symmetricPtr, params.rank);\n        GM_ADDR localTokenPerExpert = peermemInfo.ptrPeerTokenPerExpert +\n                                      params.rank * (params.EP * params.expertPerRank + 8) *\n                                      sizeof(int32_t);\n\n        moe_init_routing_quant_v2<float16_t>(reinterpret_cast<GM_ADDR> (params.ptrA), params.expertIdx,\n                                             params.moeInitRoutingQuantV2Scale, params.moeInitRoutingQuantV2Offset,\n                                             peermemInfo.ptrA,\n                                             workspaceInfo.expandedRowIdx, localTokenPerExpert,\n                                             params.expertTokensBeforeCapacity,\n                                             peermemInfo.ptrPeerPerTokenScale, params.ptrWorkspace,\n                                             &params.moeInitRoutingQuantV2TilingData, params.initRoutingQuantTilingKey);\n        AscendC::SyncAll<true>();\n\n        // 本卡peermem->其他卡peermem，远端写\n        __gm__ int32_t *sync_base = (__gm__ int32_t *)params.symmetricPtr + FLAG_OFFSET + 1024;\n        int count = gm_load(sync_base) + 1;\n        if (coreIdx < params.EP && coreIdx != params.rank) {\n            AscendC::GlobalTensor<int32_t> srcAddress;\n            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));\n            AscendC::GlobalTensor<int32_t> dstAddress;\n            __gm__ void *dstPeermemPtr = shmem_ptr(localTokenPerExpert, coreIdx);\n            dstAddress.SetGlobalBuffer((__gm__ int32_t *)dstPeermemPtr);\n\n            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;\n            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;\n            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;\n            CopyGmToUb copyGmToUb;\n            CopyUbToGm copyUbToGm;\n            AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);\n            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n            uint32_t tmp = params.EP * params.expertPerRank;\n            copyGmToUb(tmpBuffer, srcAddress[0],\n                       layout::RowMajor{1, tmp},\n                       layout::RowMajor{1, tmp});\n\n            tmpBuffer.SetValue(params.EP * params.expertPerRank, count);\n            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);\n            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);\n            copyUbToGm(dstAddress[0], tmpBuffer,\n                       layout::RowMajor{1, tmp + 1},\n                       layout::RowMajor{1, tmp + 1});\n            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);\n\n            __gm__ int32_t *sync_check = reinterpret_cast<__gm__ int32_t * > (peermemInfo.ptrPeerTokenPerExpert) +\n                coreIdx * (params.EP * params.expertPerRank + 8) + params.EP * params.expertPerRank;\n            gm_signal_wait_until_eq_for_barrier(sync_check, count);\n        }\n        AscendC::SyncAll<true>();\n        gm_store(sync_base, count);\n\n        if (coreIdx == 0) {\n            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.rank, params.EP);\n        }\n\n        AscendC::SyncAll<true>();\n        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);\n        uint32_t curGroupOffset = 0;\n        int32_t prevSumBeforeRank = 0;\n        int32_t groupIdxDeq = 0;\n        constexpr int32_t TOKEN_PER_EXPERT_OFFSET = 8;\n        if (coreIdx < params.EP) {\n            for (int32_t i = 0; i < params.rank * params.expertPerRank; i++) {\n                prevSumBeforeRank += tokenPerExpert(coreIdx * (params.EP * params.expertPerRank +\n                                                    TOKEN_PER_EXPERT_OFFSET) + i);\n            }\n            m_prevSumBeforeRank[coreIdx] = prevSumBeforeRank;\n        }\n        int prevSum = prevSumBeforeRank;\n        uint32_t prevGroupSum1 = 0;\n        uint32_t dequantSum = 0;\n        int32_t syncLoopIdx = -1;\n        constexpr int32_t GROUP_INDEX_OFFSET = 2;\n        BlockEpilogue1 blockEpilogue(resource);\n        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {\n            // 第i个core从第i个rank的peermem读数据\n            groupIdxDeq = groupIdx - GROUP_INDEX_OFFSET;\n            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {\n                uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) +\n                                    prevGroupSum1;\n                if (rowStart < params.maxOutputSize) {\n                    uint32_t rows = tokenPerExpert(\n                        dstEpIdx * (params.EP * params.expertPerRank + 8) +\n                        params.rank * params.expertPerRank + groupIdx);\n                    if (rowStart + rows > params.maxOutputSize) {\n                        rows = params.maxOutputSize - rowStart;\n                    }\n                    uint32_t rowSrc = prevSum;\n                    prevSum += rows;\n                    __gm__ void *peermemPtr = shmem_ptr(peermemInfo.ptrA, dstEpIdx);\n                    AscendC::GlobalTensor<ElementA> gmRemoteA;\n                    gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA * > (peermemPtr));\n                    AscendC::GlobalTensor<ElementPerTokenScale> gmRemotePerTokenScale;\n                    gmRemotePerTokenScale.SetGlobalBuffer(\n                        reinterpret_cast<__gm__ ElementPerTokenScale * >(reinterpret_cast<GM_ADDR>(peermemPtr) +\n                            params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA)));\n                    MatrixCoord offsetA{rowStart, 0};\n                    MatrixCoord shapeA{rows, params.problemShape.k()};\n                    MatrixCoord offsetPeer{rowSrc, 0};\n                    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);\n                    int64_t gmOffsetPeer = params.layoutA.GetOffset(offsetPeer);\n                    // 通信Data\n                    CopyGMToGM(gmA[gmOffsetA], gmRemoteA[gmOffsetPeer],\n                               rows * params.problemShape.k(), params.ubMoveNum);\n                    // 通信scale\n                    CopyGMToGM(gmPerTokenScale1[rowStart], gmRemotePerTokenScale[rowSrc], rows, rows);\n                }\n            }\n            if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0) &&\n                groupIdx == params.expertPerRank - 1) {\n                syncLoopIdx++;\n                AscendC::CrossCoreWaitFlag<0x2>(syncLoopIdx / LOOP_DENOMINATOR + 1);\n            }\n            AscendC::SyncAll<true>();\n            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);   // V通知C当前轮的通信已完成\n\n            if (groupIdx + 1 == params.epilogueGranularity) {\n                dequantSum = 0;\n            }\n\n            if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0) &&\n                groupIdx == params.expertPerRank - 1 && prevGroupSum1 > 0) {\n                uint32_t rowStartThisCore = 0;\n                MatrixCoord offsetC{0U, 0};\n                uint32_t dequantLen = prevGroupSum1 - dequantSum;\n                if (dequantLen >= params.maxOutputSize) {\n                    dequantLen = dequantLen - params.maxOutputSize;\n                }\n\n                MatrixCoord shapeC{dequantLen, params.problemShape.n()};\n                LayoutC layoutC{dequantLen, params.problemShape.n()};\n                int64_t gmOffsetC = layoutC.GetOffset(offsetC);\n                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);\n                blockEpilogue(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD],\n                              gmPerTokenScale2[rowStartThisCore], params.epilogueCoreNum);\n            }\n\n            prevGroupSum1 += cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);\n            dequantSum += cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);\n        }\n        syncLoopIdx++;\n        AscendC::CrossCoreWaitFlag<0x2>(syncLoopIdx / LOOP_DENOMINATOR + 1);\n        AscendC::SyncAll<true>();\n        uint32_t lastDequantExpertNum = params.expertPerRank;\n        if (params.epilogueGranularity < params.expertPerRank) {\n            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;\n        }\n        if (lastDequantExpertNum < params.expertPerRank) {\n            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_VALUE);\n        }\n        if (prevGroupSum1 - dequantSum < params.maxOutputSize) {\n            uint32_t rowStartThisCore = prevGroupSum1 - dequantSum;\n            MatrixCoord offsetC{rowStartThisCore, 0};\n            uint32_t dequantLen = dequantSum;\n            if (prevGroupSum1 >= params.maxOutputSize) {\n                dequantLen = dequantSum - (prevGroupSum1 - params.maxOutputSize);\n            }\n            MatrixCoord shapeC{dequantLen, params.problemShape.n()};\n            LayoutC layoutC{dequantLen, params.problemShape.n()};\n            int64_t gmOffsetC = layoutC.GetOffset(offsetC);\n            int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);\n            blockEpilogue(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD],\n                          gmPerTokenScale2[rowStartThisCore], coreNum);\n        }\n        blockEpilogue.Finalize();\n    }\n\n    CATLASS_DEVICE\n    void Combine(Params const &params)\n    {\n        int32_t prevSumBeforeRank = 0;\n        if (coreIdx < params.EP) {\n            prevSumBeforeRank = m_prevSumBeforeRank[coreIdx];\n        }\n\n        int prevSum = prevSumBeforeRank;\n        uint32_t n2 = params.problemShape.k();\n        uint32_t k2 = params.problemShape.n() / 2;\n\n        typename BlockEpilogue2::Params epilogueParams {\n                static_cast<int32_t>(params.EP),\n                static_cast<int32_t>(params.expertPerRank),\n                reinterpret_cast<__gm__ int32_t * > (params.ptrWorkspace)\n        };\n        BlockEpilogue2 blockEpilogue(resource, epilogueParams);\n        int32_t prevGroupSum2 = 0;\n        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {\n            AscendC::CrossCoreWaitFlag<0x2>(groupIdx / 8 + 3);\n            AscendC::SyncAll<true>();\n\n            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {\n                __gm__ void *dstPeermemPtr = shmem_ptr(peermemInfo.ptrD, dstEpIdx);\n                AscendC::GlobalTensor<ElementD2> gmRemotePeer;\n                gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2 * > (dstPeermemPtr));\n                uint32_t srcRowOffset =\n                    (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum2;\n                if (srcRowOffset < params.maxOutputSize) {\n                    uint32_t dataRows = tokenPerExpert(dstEpIdx * (params.EP * params.expertPerRank + 8) + groupIdx +\n                                                       params.rank * params.expertPerRank);\n                    if (srcRowOffset + dataRows > params.maxOutputSize) {\n                        dataRows = params.maxOutputSize - srcRowOffset;\n                    }\n                    uint32_t dstRowOffset = prevSum;\n                    prevSum += dataRows;\n                    MatrixCoord offsetC{srcRowOffset, 0};\n                    MatrixCoord offsetPeer{dstRowOffset, 0};\n                    MatrixCoord shapeC{dataRows, n2};\n                    int64_t gmOffsetC = params.layoutD2.GetOffset(offsetC);\n                    int64_t gmOffsetPeer = params.layoutD2.GetOffset(offsetPeer);\n                    if constexpr(std::is_same_v < ElementA, int8_t >) {\n                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmPerTokenScale2[srcRowOffset],\n                                      gmRemotePeer[gmOffsetPeer]);\n                    } else {\n                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmRemotePeer[gmOffsetPeer]);\n                    }\n                }\n            }\n            prevGroupSum2 += cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);\n        }\n        AscendC::SyncAll<true>();\n        blockEpilogue.Finalize();\n        CrossRankSync(params.symmetricPtr, params.rank, params.EP);\n        MoeTokenUnpermuteTilingData tilingData;\n        MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum);\n        KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;\n\n        kernelMoeTokenUnpermuteOp.Init(peermemInfo.ptrD, workspaceInfo.expandedRowIdx, params.probs,\n                                       reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);\n        kernelMoeTokenUnpermuteOp.Process();\n    }\n\nprivate:\n    struct WorkspaceInfo {\n        GM_ADDR ptrA;\n        GM_ADDR ptrPerTokenScale;\n        GM_ADDR ptrcumsumMM;\n        GM_ADDR ptrcumsumSend;\n        GM_ADDR ptrC;\n        GM_ADDR ptrC2;\n        GM_ADDR ptrPermutedToken;\n        GM_ADDR ptrPerTokenScale2;\n        GM_ADDR expandedRowIdx;\n        GM_ADDR ptrTokenPerExpert;\n\n        CATLASS_DEVICE\n        WorkspaceInfo()\n        {}\n\n        CATLASS_DEVICE\n        WorkspaceInfo(const Params &params)\n        {\n            uint32_t k2 = params.problemShape.n() / 2;\n            uint32_t n2 = params.problemShape.k();\n            int64_t workspaceOffset = 0;\n            expandedRowIdx = params.ptrWorkspace;\n\n            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);\n            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;\n\n            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);\n            ptrcumsumSend = params.ptrWorkspace + workspaceOffset;\n\n            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);\n            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;\n\n            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);\n            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;\n\n            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);\n            ptrTokenPerExpert = params.ptrWorkspace + workspaceOffset;\n\n            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);\n            ptrC = params.ptrWorkspace + workspaceOffset;\n            ptrC2 = ptrC;\n\n            workspaceOffset += max(params.maxOutputSize * params.problemShape.n() * sizeof(ElementC),\n                                   params.maxOutputSize * n2 * sizeof(ElementC));\n            ptrA = params.ptrWorkspace + workspaceOffset;\n            ptrPermutedToken = ptrA;\n\n            workspaceOffset += max(params.maxOutputSize * params.problemShape.k() * sizeof(ElementA),\n                                   params.maxOutputSize * k2 * sizeof(ElementA));\n        }\n    };\n\n    struct PeermemInfo {\n        GM_ADDR ptrA;\n        GM_ADDR ptrPeerPerTokenScale;\n        GM_ADDR ptrPeerTokenPerExpert;\n        GM_ADDR ptrD;\n\n        CATLASS_DEVICE\n        PeermemInfo()\n        {}\n\n        CATLASS_DEVICE\n        PeermemInfo(const Params &params)\n        {\n            ptrPeerTokenPerExpert = params.symmetricPtr;\n            ptrA = ptrPeerTokenPerExpert + (params.EP * (params.EP * params.expertPerRank +\n                   EXPERT_OFFSET)) * sizeof(int32_t);\n            ptrPeerPerTokenScale =\n                    ptrA + params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA);\n            ptrD = ptrPeerPerTokenScale + params.problemShape.m() * params.topK * sizeof(ElementPerTokenScale);\n        }\n    };\n\n    Arch::Resource <ArchTag> resource;\n\n    uint32_t coreIdx;\n    uint32_t coreNum;\n\n    Params params;\n    WorkspaceInfo workspaceInfo;\n    PeermemInfo peermemInfo;\n\n    int64_t m_prevSumBeforeRank[32];\n\n    AscendC::GlobalTensor<int32_t> cumsumMM;\n    AscendC::GlobalTensor<int32_t> cumsumSend;\n\n    AscendC::GlobalTensor<ElementA> gmA;\n    AscendC::GlobalTensor<ElementC> gmC;\n    AscendC::GlobalTensor<ElementScale> gmS;\n\n    AscendC::GlobalTensor<ElementD1> gmPermutedToken;\n    AscendC::GlobalTensor<ElementScale> gmS2;\n    AscendC::GlobalTensor<ElementC> gmC2;\n\n    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;\n    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;\n\n    AscendC::GlobalTensor<int32_t> tokenPerExpert;\n\n    // preload B matrix\n    using CopyGmToL1B = Gemm::Tile::CopyGmToL1 <ArchTag, Gemm::GemmType<int8_t, layout::zN>>;\n    CopyGmToL1B copyGmToL1B;\n    static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementB);\n    static constexpr uint32_t L1_STAGES = 2;\n    AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];\n    uint32_t l1ListId{0};\n};\n} // namespace Catlass::Gemm::Kernel\n\n#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_PER_TOKEN_DEQUANT_MULTISTAGE_WORKSPACE_HPP\n"
  }
}
```

---

## MODEL ✨

I will examine `examples/dispatch_gmm_combine/include/sync_util.h` to see if it contains the definitions for `gm_signal_wait_until_eq_for_barrier` or other synchronization primitives.
**Tool Command**:
```json
{
  "name": "read_file",
  "args": {
    "file_path": "examples/dispatch_gmm_combine/include/sync_util.h"
  }
}
```

---

## USER 🧑‍💻

**Tool Response**:
```json
{
  "id": "read_file-1770367462764-f1a318ddfc763",
  "name": "read_file",
  "response": {
    "output": "/*\r\n * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.\r\n * This program is free software, you can redistribute it and/or modify it under the terms and conditions of\r\n * CANN Open Software License Agreement Version 2.0 (the \"License\").\r\n * Please refer to the License for details. You may not use this file except in compliance with the License.\r\n * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\r\n * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\r\n * See LICENSE in the root of the software repository for the full text of the License.\r\n */\r\n#ifndef SYNC_UTIL_H\r\n#define SYNC_UTIL_H\r\n#include \"kernel_operator.h\"\r\nusing namespace AscendC;\r\n\r\n#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__\r\nconstexpr int32_t BUFF_SIZE = 500 * 1024 * 1024;\r\nconstexpr int32_t FLAG_OFFSET = 500 * 1024 * 1024 / sizeof(int32_t);\r\n\r\n\r\ntemplate<typename T>\r\nFORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val) {\r\n    *((__gm__ T *)addr) = val;\r\n}\r\n\r\ntemplate<typename T>\r\nFORCE_INLINE_AICORE T gm_load(__gm__ T *cache) {\r\n    return *((__gm__ T *)cache);\r\n}\r\n\r\nFORCE_INLINE_AICORE void gm_dcci(__gm__ uint8_t * addr) {\r\n    using namespace AscendC;\r\n    GlobalTensor<uint8_t> global;\r\n    global.SetGlobalBuffer(addr);\r\n\r\n    // Important: add hint to avoid dcci being optimized by compiler\r\n    __asm__ __volatile__(\"\");\r\n    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);\r\n    __asm__ __volatile__(\"\");\r\n}\r\n\r\nFORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {\r\n    do {\r\n        gm_dcci((__gm__ uint8_t *)sig_addr);\r\n\r\n        if (*sig_addr == cmp_val) {\r\n            return *sig_addr;\r\n        }\r\n\r\n        // in case when peer pe enters next barrier\r\n        if (*sig_addr == cmp_val + 1) {\r\n            return *sig_addr;\r\n        }\r\n    } while (true);\r\n\r\n    return -1;\r\n}\r\n \r\nFORCE_INLINE_AICORE void CrossRankSync(GM_ADDR symmetricPtr, int32_t rank, int32_t rankSize)\r\n{\r\n    // 全核同步\r\n    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET;\r\n    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET + 1024;\r\n    int count = gm_load(sync_base) + 1;\r\n    int vec_id = AscendC::GetBlockIdx();\r\n    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();\r\n    for (int i = vec_id; i < rankSize; i += vec_size) {\r\n        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(shmem_ptr(symmetricPtr, i)) + FLAG_OFFSET + rank * 16;\r\n        gm_store(sync_remote, count);\r\n        gm_dcci((__gm__ uint8_t*)sync_remote);\r\n        auto sync_check = sync_counter + i * 16;\r\n        gm_signal_wait_until_eq_for_barrier(sync_check, count);\r\n    }\r\n\r\n    AscendC::SyncAll<true>();\r\n    gm_store(sync_base, count);\r\n}\r\n\r\n// 全卡同步接口\r\nFORCE_INLINE_AICORE void CrossRankSyncV1(GM_ADDR symmetricPtr, int32_t rank, int32_t rankSize)\r\n{\r\n    // 全核同步\r\n    AscendC::SyncAll<true>();\r\n    __gm__ int32_t* sync_counter = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET;\r\n    __gm__ int32_t* sync_base = (__gm__ int32_t*)symmetricPtr + FLAG_OFFSET + 32;\r\n    int count = gm_load(sync_base) + 1;\r\n    int vec_id = AscendC::GetBlockIdx();\r\n    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();\r\n    for (int i = vec_id; i < rankSize; i += vec_size) {\r\n        __gm__ int32_t* sync_remote = (__gm__ int32_t*)(shmem_ptr(symmetricPtr, i)) + FLAG_OFFSET + rank * 16;\r\n        gm_store(sync_remote, count);\r\n        gm_dcci((__gm__ uint8_t*)sync_remote);\r\n        auto sync_check = sync_counter + i * 16;\r\n        gm_signal_wait_until_eq_for_barrier(sync_check, count);\r\n    }\r\n\r\n    AscendC::SyncAll<true>();\r\n    gm_store(sync_base, count);\r\n}\r\n#endif"
  }
}
```

---

## MODEL ✨

在 `examples/dispatch_gmm_combine` 示例中，`Dispatch` 算子确实包含**卡间同步**逻辑。虽然它没有直接调用 `CrossRankSync` 这种显式的全局屏障函数，但它通过 **SHMEM 远程写**和**信号量轮询**实现了一种点对点（P2P）的卡间同步。

以下是详细的分析和解释：

### Dispatch 算子中的卡间同步分析

在 `examples/dispatch_gmm_combine/include/dispatch_gmm_combine.h` 的 `Dispatch` 函数中，卡间同步体现在以下几个阶段：

#### 1. 路由元数据同步 (Metadata Sync)
在执行真正的 Data Movement（数据搬运）之前，各卡需要知道其他卡上的 Expert 路由信息（即 `tokenPerExpert`）。

```cpp
// 位于 Dispatch 函数中
// 1. 获取本地计算的路由信息
__gm__ void *srcPeermemPtr = shmem_ptr(params.symmetricPtr, params.rank);
GM_ADDR localTokenPerExpert = peermemInfo.ptrPeerTokenPerExpert + ...;

// 2. 将本地的 tokenPerExpert 通过远程写发送到其他所有卡
if (coreIdx < params.EP && coreIdx != params.rank) {
    __gm__ void *dstPeermemPtr = shmem_ptr(localTokenPerExpert, coreIdx);
    dstAddress.SetGlobalBuffer((__gm__ int32_t *)dstPeermemPtr);
    
    // ... 准备数据 ...
    tmpBuffer.SetValue(params.EP * params.expertPerRank, count); // 设置同步标记(flag)
    
    // 远程写：将元数据 + flag 写入远端卡的 symmetricPtr 对应位置
    copyUbToGm(dstAddress[0], tmpBuffer, layout::RowMajor{1, tmp + 1}, ...);

    // 3. 等待远端卡也将它们的元数据写到本卡的内存中
    __gm__ int32_t *sync_check = reinterpret_cast<__gm__ int32_t * > (peermemInfo.ptrPeerTokenPerExpert) + ...;
    gm_signal_wait_until_eq_for_barrier(sync_check, count); // 轮询等待 flag 更新
}
```

*   **体现位置**：上述代码段使用了 `shmem_ptr` 获取远端地址，并通过 `copyUbToGm` 进行跨卡写入。
*   **同步机制**：通过 `gm_signal_wait_until_eq_for_barrier` 轮询内存位置。只有当远端卡完成了它的路由计算并把数据（含 flag）推送到本卡后，本卡才会继续执行。这确保了后续数据搬运时，所有的路由偏移量都是准确且就绪的。

#### 2. 数据搬运阶段 (Data Dispatch)
在元数据同步完成后，各卡开始从远端卡拉取（Pull）数据。

```cpp
for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
    for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
        // ... 计算偏移 ...
        __gm__ void *peermemPtr = shmem_ptr(peermemInfo.ptrA, dstEpIdx); // 获取远端数据指针
        // ...
        // 跨卡数据拉取
        CopyGMToGM(gmA[gmOffsetA], gmRemoteA[gmOffsetPeer], rows * ..., ...);
    }
    // ...
    AscendC::SyncAll<true>(); // 卡内多核同步
}
```

*   **隐含同步**：虽然 `CopyGMToGM` 本身是数据传输，但由于之前的元数据同步（第1步）已经保证了远端卡至少已经完成了 `moe_init_routing_quant_v2`（即数据已经准备好存放在 `ptrA` 中），因此这里的拉取操作是安全的。

### 总结与对比

| 算子 | 同步形式 | 体现位置 | 目的 |
| :--- | :--- | :--- | :--- |
| **Dispatch** | **P2P 隐式同步** | `Dispatch` 函数中的 `gm_signal_wait_until_eq_for_barrier` | 同步 `tokenPerExpert` 元数据，确保 All-to-All 搬运前的路由信息就绪。 |
| **Combine** | **Global Barrier** | `Combine` 末尾调用的 `CrossRankSync` | 确保所有卡都完成了最后的计算和结果回写，保证全局一致性。 |

### 为什么 Dispatch 不使用 CrossRankSync？
`Dispatch` 阶段的同步需求是“生产-消费”型的：我需要知道你打算发给我多少数据，我才能开始接。这种需求通过在传输元数据末尾附带一个 `flag`（即代码中的 `count`）并进行轮询（`gm_signal_wait_until_eq_for_barrier`）效率更高，且能与数据传输深度耦合。而 `Combine` 后的 `CrossRankSync` 更多是为了确保整个计算流水线的完整结束。