# Complete CUTLASS Code Examples for SAM 3.1

## 1. ViT QKV Projection — Fused Triple GEMM

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

using namespace cute;

// SAM 3.1 ViT QKV: (seq, 1024) @ (1024, 3072) -> (seq, 3072)
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentA,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, CollectiveMainloop,
    cutlass::epilogue::collective::DefaultEpilogue<
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 8, ElementAccumulator, ElementAccumulator>>>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## 2. ViT MLP FC1 with Fused Bias + GELU (addmm_act replacement)

```cpp
// EVT-based fusion: GEMM -> bias -> GELU
using EVT_BiasGelu = cutlass::epilogue::Sm90EVT<
    cutlass::epilogue::Sm90Compute<
        cutlass::epilogue::thread::GELU, float, ElementC>,
    cutlass::epilogue::Sm90EVT<
        cutlass::epilogue::Sm90Compute<cutlass::plus, float, float>,
        cutlass::epilogue::Sm90AccFetch,
        cutlass::epilogue::Sm90RowBroadcast<0, float, TileShape>>>;
```

## 3. RTX 5060 Optimized Tile Configs

```cpp
struct Sam31KernelConfig {
    struct ViT_QKV {
        using TileShape = Shape<_128, _256, _64>;
        using ClusterShape = Shape<_1, _1, _1>;
        static constexpr int Stages = 4;
    };
    struct ViT_Attention {
        using TileShape = Shape<_64, _64, _32>;
        using ClusterShape = Shape<_1, _2, _1>;
        static constexpr int Stages = 3;
    };
    struct ViT_MLP_FC1 {
        using TileShape = Shape<_128, _256, _64>;
        static constexpr int Stages = 4;
    };
    struct ViT_MLP_FC2 {
        using TileShape = Shape<_128, _128, _64>;
        static constexpr int Stages = 5;
    };
    struct DETR_Small {
        using TileShape = Shape<_32, _128, _32>;
        using ClusterShape = Shape<_2, _1, _1>;
        static constexpr int Stages = 3;
    };
};
```

## 4. Custom RoPE EVT Node

```cpp
struct RotaryEmbedCompute {
    template <typename Fragment>
    CUTLASS_DEVICE Fragment operator()(
        Fragment const& acc, int row, int col, Params const& p) const {
        Fragment result;
        float angle = (1.0f / powf(p.theta, 2.0f*(col/2)/64.0f)) * row;
        #pragma unroll
        for (int i = 0; i < Fragment::kElements; i++) {
            if ((col+i) % 2 == 0) result[i] = float(acc[i]) * cosf(angle);
            else result[i] = float(acc[i]) * sinf(angle);
        }
        return result;
    }
};
```
