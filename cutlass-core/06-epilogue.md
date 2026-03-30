# CUTLASS Epilogue: Deep Technical Analysis & Fusion Opportunities

## Executive Summary

The epilogue is the final stage of a CUTLASS GEMM kernel — post-processing after the matrix multiply accumulator completes. For SAM 3.1's `addmm_act` bottleneck (GEMM + bias + activation), the epilogue is where activation fusion lives. This document covers CUTLASS's epilogue architecture across all three generations (thread-level, collective, and Hopper/TMA-based), with focus on activation function fusion.

---

## 1. What the Epilogue Does

After the mainloop computes `C += A * B` in registers (the accumulator fragment), the epilogue handles everything between "accumulator is ready" and "output is written to global memory":

```
GEMM Mainloop:                     Epilogue:
┌─────────────────┐    ┌──────────────────────────────────┐
│ Load A from     │    │ 1. Convert accumulator type      │
│ Load B from     │───>│ 2. Scale: alpha * acc            │
│ MMA accumulate  │    │ 3. Optional: load source C       │
│ C += A * B      │    │ 4. Optional: beta * C (residual) │
│ (in registers)  │    │ 5. Optional: add bias            │
└─────────────────┘    │ 6. Optional: apply activation    │
                       │ 7. Convert to output type        │
                       │ 8. Store to global memory         │
                       └──────────────────────────────────┘
```

The core formula:
```
D = activation(alpha * accumulator + beta * source_C + bias)
```

Key constraints:
- The accumulator is in **register fragments** (typically `float` for sm90)
- Output must be in the target type (`f16`, `bf16`, `e4m3`, `f32`)
- The epilogue must handle **tile-based output** — writing a subtile of the output matrix
- Shared memory is used as an intermediary between registers and global memory (except sm90 TMA paths)

---

## 2. Thread-Level Epilogue vs Collective Epilogue

### Thread-Level Epilogue (Pre-CUTLASS 3.x / sm70-sm80)

Located in `include/cutlass/epilogue/thread/`. These operate **per-thread** — each CUDA thread processes its own fragment independently.

**Key files**:
- `linear_combination.h` — Basic `D = alpha * acc + beta * C`
- `linear_combination_generic.h` — Template for `D = activation(alpha * acc + beta * C)`
- `linear_combination_relu.h` — `D = ReLU(alpha * acc + beta * C)`
- `linear_combination_gelu.h` — `D = GELU(alpha * acc + beta * C)`
- `linear_combination_silu.h` — `D = SiLU(alpha * acc + beta * C)`
- `linear_combination_bias_relu.h` — `D = ReLU(alpha * acc + beta * C + bias)` with ReLU conditional storage
- `activation.h` — All activation functor definitions

**How it works** (from `linear_combination_generic.h`):

```cpp
template <template<typename T> class ActivationFunctor, ...>
class LinearCombinationGeneric {
public:
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source) const {

    // Convert to compute type
    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Linear combination
    FragmentCompute intermediate;
    intermediate = mul_add_source(params_.beta, converted_source);      // beta * C
    intermediate = mul_add_accumulator(params_.alpha,
                                       converted_accumulator,
                                       intermediate);                    // alpha * acc + beta * C

    // Apply activation
    ActivationFunctor<FragmentCompute> activation;
    intermediate = skip_elementwise_ ? intermediate : activation(intermediate);

    // Convert to output type
    return destination_converter(intermediate);
  }
};
```

The template parameter `ActivationFunctor` is a policy class. For example:
- `LinearCombinationGELU = LinearCombinationGeneric<GELU, ...>`
- `LinearCombinationSilu = LinearCombinationGeneric<SiLu, ...>`

**Fragment size**: Typically `Count = 128 / sizeof_bits<ElementOutput>`. For f16: 8 elements per thread. For f32: 4 elements. This determines the vectorization width.

### Collective Epilogue (CUTLASS 3.x / sm90+)

Located in `include/cutlass/epilogue/collective/`. These coordinate **across the thread block** using shared memory pipelines and TMA (Tensor Memory Accelerator) on sm90+.

**Key files**:
- `collective_epilogue.hpp` — Main template with dispatch policy specializations
- `sm90_epilogue_tma_warpspecialized.hpp` — Hopper TMA-based epilogue
- `sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp` — Hopper with bias + elementwise
- `sm70_epilogue_vectorized.hpp` — Pre-Hopper vectorized epilogue

**Architecture**: The collective epilogue is parameterized by a `DispatchPolicy` that controls:
- Number of pipeline stages for C (source) and D (output)
- Fragment size
- Whether to reuse shared memory between C and D
- Whether to delay TMA store

The `Sm90TmaWarpSpecialized` dispatch policy (from `sm90_epilogue_tma_warpspecialized.hpp`):

```cpp
template <
  int StagesC_,           // Pipeline stages for loading C
  int StagesD_,           // Pipeline stages for storing D
  int FragmentSize_,      // Elements per thread
  bool ReuseSmemC_,       // Can D reuse C's shared memory?
  bool DelayTmaStore_,    // Delay TMA store to overlap with compute?
  ...
>
class CollectiveEpilogue<
    Sm90TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>,
    ...
> {
  // Uses TMA for both loading C and storing D
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using StorePipeline = cutlass::PipelineTmaStore<StagesD>;
};
```

---

## 3. Supported Epilogue Operations

### From `activation.h` — Activation Function Library

| Activation | `kIsHeavy` | Compute Cost | Implementation |
|------------|-----------|--------------|----------------|
| `Identity` | false | 0 | Pass-through |
| `ReLU` | false | 1 | `max(value, 0)` — NaN-propagating |
| `LeakyReLU` | false | 2 | `value > 0 ? value : value * leaky_alpha` |
| `ThresholdReLU` | false | 2 | `value <= threshold ? 0 : min(value, upper_bound)` |
| `HardSwish` | false | 5 | `x * min(max(x+3, 0), 6) / 6` |
| `GELU` | true | ~8 | Fast tanh approximation |
| `SiLU` | true | ~7 | `x * sigmoid(x)` |
| `Sigmoid` | true | ~5 | `1 / (1 + exp(-x))` or tanh-based |
| `Tanh` | true | ~3 | Hardware `fast_tanh` on sm80+ |
| `Clamp` | false | 2 | `min(max(value, lower), upper)` |

### From `linear_combination_*.h` — Combined Operations

| File | Operation |
|------|-----------|
| `linear_combination.h` | `D = alpha * acc + beta * C` |
| `linear_combination_relu.h` | `D = ReLU(alpha * acc + beta * C)` |
| `linear_combination_gelu.h` | `D = GELU(alpha * acc + beta * C)` |
| `linear_combination_silu.h` | `D = SiLU(alpha * acc + beta * C)` |
| `linear_combination_hardswish.h` | `D = HardSwish(alpha * acc + beta * C)` |
| `linear_combination_leaky_relu.h` | `D = LeakyReLU(alpha * acc + beta * C)` |
| `linear_combination_sigmoid.h` | `D = Sigmoid(alpha * acc + beta * C)` |
| `linear_combination_bias_relu.h` | `D = ReLU(alpha * acc + beta * C + bias)` |
| `linear_combination_dgelu.h` | `D = dGELU(alpha * acc + beta * C)` (for backward) |
| `linear_combination_drelu.h` | `D = dReLU(alpha * acc + beta * C)` (for backward) |
| `linear_combination_clamp.h` | `D = Clamp(alpha * acc + beta * C)` |
| `linear_combination_residual_block.h` | Residual block pattern |
| `linear_combination_generic.h` | `D = ActivationFunctor(alpha * acc + beta * C)` |

### From `fusion/operations.hpp` — CUTLASS 3.x Fusion Operations

| Fusion Operation | Formula |
|-----------------|---------|
| `ScaledAcc` | `D = alpha * acc` |
| `LinearCombination` | `D = alpha * acc + beta * C` |
| `LinCombEltAct` | `D = activation(alpha * acc + beta * C)` |
| `LinCombPerRowBias` | `D = alpha * acc + beta * C + per_row_bias` |
| `LinCombPerColBias` | `D = alpha * acc + beta * C + per_col_bias` |
| `LinCombPerRowBiasEltAct` | `D = activation(alpha * acc + beta * C + per_row_bias)` |
| `LinCombPerColBiasEltAct` | `D = activation(alpha * acc + beta * C + per_col_bias)` |
| `LinCombPerRowBiasEltActAux` | `D = activation(...), aux = pre_activation_value` |
| `ScaledLinCombPerRowBiasEltAct` | `D = scale_d * activation(scale_a * scale_b * alpha * acc + ...)` |
| `PerRowLinCombPerRowBiasEltAct` | Per-row alpha/beta + per-row bias + activation |
| `LinCombTopKSoftmaxCol` | `D = softmax(top_k(alpha * acc + beta * C))` |

---

## 4. Epilogue Fusion: How to Fuse Activation + Bias into GEMM

### The Thread-Level Approach (Pre-CUTLASS 3.x)

Fusion at the thread level is straightforward: extend `LinearCombinationGeneric` with an additional bias parameter.

From `linear_combination_bias_relu.h`, the fused Bias+ReLU epilogue:

```cpp
// D = ReLU(alpha * acc + beta * C + bias)
// Additionally stores ReLU conditionals as a bitvector for backward pass
template <typename ElementOutput_, int Count, ...>
class LinearCombinationBiasReLU {
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source,
    FragmentOutput const &bias) const {

    // 1. Convert types
    FragmentCompute acc = accumulator_converter(accumulator);
    FragmentCompute src = source_converter(source);
    FragmentCompute bias_val = source_converter(bias);

    // 2. Linear combination: alpha * acc + beta * C
    FragmentCompute intermediate;
    intermediate = mul_add_source(params_.beta, src);
    intermediate = mul_add_accumulator(params_.alpha, acc, intermediate);

    // 3. Add bias
    intermediate = add(intermediate, bias_val);

    // 4. Apply ReLU with conditional storage
    ArrayMaximum<ElementCompute, kCount> maximum_op;
    intermediate = maximum_op(intermediate, ElementCompute(0));

    // 5. Store ReLU mask for backward pass
    ReluConditional<ElementCompute, kCount> relu_cond;
    bool conditional[kCount];
    relu_cond(conditional, intermediate, ElementCompute(0));

    return destination_converter(intermediate);
  }
};
```

**Key insight**: The bias is loaded from global memory as a separate fragment, then added element-wise. The ReLU conditional (which elements were zeroed) can be stored for the backward pass — a critical optimization for training.

### The CUTLASS 3.x EVT Approach

The CUTLASS 3.x system uses a composable tree structure (Epilogue Visitor Tree). Each node in the tree is either:

1. **Data source**: `Sm90AccFetch`, `Sm90SrcFetch`, `Sm90ColBroadcast`, `Sm90RowBroadcast`
2. **Compute node**: `Sm90Compute<Op, ElementOutput, ElementCompute>`
3. **Store node**: `Sm90AuxStore`, `Sm90TmaStore`

For `D = GELU(alpha * acc + beta * C + per_row_bias)`:

```cpp
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    // Root: apply GELU activation
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::GELU,    // Activation functor
        float,                                // ElementOutput
        float                                 // ElementCompute
    >,
    // Child: add per-row bias to linear combination
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus,                    // Binary add
            float,
            float
        >,
        // Left: alpha * acc + beta * C
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiply_add,        // FMA: a*b + c
                float,
                float
            >,
            cutlass::epilogue::fusion::Sm90AccFetch,   // Fetch accumulator
            cutlass::epilogue::fusion::Sm90SrcFetch    // Fetch source C
        >,
        // Right: per-row bias vector
        cutlass::epilogue::fusion::Sm90RowBroadcast<
            cutlass::layout::RowMajor,
            float
        >
    >
>;
```

**Critical difference from thread-level**: The EVT system handles data movement (TMA loads for bias, shared memory staging) automatically based on the tree structure. The user only specifies the computation.

---

## 5. The EVT (Epilogue Visitor Tree) System

### Design Philosophy

The EVT system was introduced in CUTLASS 3.x to solve the combinatorial explosion of thread-level epilogue variants. Instead of writing N! combinations of (bias, activation, scaling, residual, quantization), users compose a tree of primitive operations.

### Node Types

From `fusion/sm90_callbacks_tma_warpspecialized.hpp` and `fusion/operations.hpp`:

**Data Sources**:
- `Sm90AccFetch` — Fetch accumulator from register file
- `Sm90SrcFetch` — Fetch source C from shared memory (loaded via TMA)
- `Sm90ColBroadcast` — Broadcast per-column vector (bias, scale)
- `Sm90RowBroadcast` — Broadcast per-row vector (bias, scale)
- `Sm90ScalarBroadcast` — Broadcast scalar (alpha, beta)
- `Sm90AuxLoad` — Load auxiliary tensor (for residual connections)

**Compute Nodes**:
- `Sm90Compute<Op, ElementOutput, ElementCompute>` — Apply a unary or binary compute operation
- The `Op` can be any functor: `cutlass::plus`, `cutlass::multiply_add`, `GELU`, `SiLU`, `ReLU`, etc.

**Store Nodes**:
- `Sm90TmaStore` — Store result via TMA
- `Sm90AuxStore` — Store auxiliary output (pre-activation values, ReLU masks)

### How the Tree Is Evaluated

At compile time, the EVT is flattened into a sequence of operations:

1. **Load phase**: Fetch all data sources (accumulator, source C, bias, etc.)
2. **Compute phase**: Apply operations in tree order (bottom-up)
3. **Store phase**: Write results to global memory

The tree structure determines **data dependencies** and enables the compiler to optimize register allocation and instruction scheduling.

### The `FusionCallbacks` Template

From `fusion/callbacks.hpp`:

```cpp
template <
  class DispatchPolicy,    // e.g., Sm90TmaWarpSpecialized
  class Operation,         // e.g., LinCombPerRowBiasEltAct
  class CtaTile_MNK,       // Tile dimensions
  class EpilogueTile_MN,   // Epilogue subtile
  class... Args            // Implementation-specific args
>
struct FusionCallbacks {
  // Specialized per dispatch policy
  // Provides: SharedStorage, Arguments, Params
  // Handles: data loading, compute, storing
};
```

The `FusionCallbacksTraits` metafunction extracts metadata from a callbacks type:

```cpp
template <class T>
struct FusionCallbacksTraits {
  using Operation = FusionOperation;  // The fusion operation type
  using ElementCompute = void;        // Compute element type
};
```

---

## 6. Hopper/sm90 TMA-Based Epilogue

### The Shared Memory Bottleneck

In pre-Hopper architectures, the epilogue flow was:
```
Registers → Shared Memory → Global Memory
```
Shared memory was the bottleneck because:
1. Writing accumulator to shared memory requires explicit `st.shared` instructions
2. Reading from shared memory for type conversion requires `ld.shared` instructions
3. Writing from shared memory to global memory requires `st.global` instructions
4. All three compete for shared memory bandwidth

### Hopper's TMA Revolution

On sm90 (Hopper), the epilogue can use **TMA (Tensor Memory Accelerator)** for both loading source C and storing output D:

```
Registers → [Shared Memory] → TMA Store → Global Memory
     ↑
TMA Load → Shared Memory → Registers (for source C)
```

TMA advantages:
1. **Hardware-managed address generation**: TMA computes addresses in hardware, reducing instruction overhead
2. **Asynchronous operation**: TMA loads/stores happen asynchronously, overlapping with compute
3. **Multicast**: TMA can multicast loads to multiple SMs in a cluster
4. **Reduced register pressure**: Address computation doesn't consume registers

From `sm90_epilogue_tma_warpspecialized.hpp`:

```cpp
// TMA copy operations for loading C and storing D
using GmemTiledCopyC = CopyOpG2S_;  // e.g., SM90_TMA_LOAD
using GmemTiledCopyD = CopyOpS2G_;  // e.g., SM90_TMA_STORE

// TMA descriptors constructed from tensor metadata
typename Params::TMA_C tma_load_c = make_tma_copy_C_sm90(
    CopyOpG2S{},           // TMA load op
    tensor_c,              // Source tensor
    SmemLayoutC{},         // Shared memory layout
    EpilogueTile{}         // Tile shape
);
```

### Warp Specialization

The Hopper epilogue uses **warp specialization** — different warps handle different phases:

- **Load warp**: Issues TMA loads for source C
- **Compute warp**: Performs epilogue computation (scaling, activation, type conversion)
- **Store warp**: Issues TMA stores for output D

This is managed by the `Sm90TmaWarpSpecialized` dispatch policy:

```cpp
// Dispatch policy parameters
Sm90TmaWarpSpecialized<
  StagesC,       // Number of pipeline stages for C
  StagesD,       // Number of pipeline stages for D
  FragmentSize,  // Elements per thread per iteration
  ReuseSmemC,    // Whether D can reuse C's shared memory buffer
  DelayTmaStore  // Whether to delay TMA store for better overlap
>
```

### Shared Memory Reuse

When `ReuseSmemC = true`, the shared memory buffer used for loading C is reused for storing D (since C is consumed before D is produced). This saves shared memory, enabling larger tile sizes:

```cpp
union CollectiveStorageReuseC {
  alignas(MaxSmemAlignment)
    ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
  alignas(MaxSmemAlignment)
    ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
};
```

### Pipeline Architecture

The Hopper epilogue uses a multi-stage pipeline:

```cpp
// Load pipeline: TMA loads for source C
using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;

// Store pipeline: TMA stores for output D
using StorePipeline = cutlass::PipelineTmaStore<StagesD>;
```

The pipeline stages enable:
- Loading the next tile's source C while processing the current tile
- Storing the previous tile's output D while computing the current tile
- Triple-buffered overlap: load(N+1) → compute(N) → store(N-1)

---

## 7. Output Layout Transformations

### M-major vs N-major

The epilogue handles layout transformations between the accumulator layout (which is in register tile order) and the output layout (which is in matrix order).

From `sm90_epilogue_tma_warpspecialized.hpp`:

```cpp
constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();
```

The shared memory layout is constructed to match the output layout:

```cpp
using SmemLayoutD = decltype(tile_to_shape(
    SmemLayoutAtomD{},
    make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesD>{}),
    cute::conditional_t<is_m_major_D, Step<_2,_1,_3>, Step<_1,_2,_3>>{}
));
```

### IM2COL Support

The epilogue also supports IM2COL (image-to-column) layouts for convolution:

```cpp
constexpr static bool is_im2col_C = cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>;
constexpr static bool is_im2col_D = cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>;
```

This enables the epilogue to write outputs directly in the format expected by convolution operations, avoiding a separate layout transformation kernel.

---

## 8. Per-Channel and Per-Tensor Quantization

### Per-Tensor Quantization

The simplest form: a single scale factor applied to the entire output tensor.

```cpp
// D = scale * activation(alpha * acc + beta * C)
using Op = ScaledAcc<float, float, float>;
```

### Per-Channel Quantization

Each output channel (column) has its own scale factor:

```cpp
// D = per_col_scale * activation(alpha * acc + beta * C + per_col_bias)
using Op = ScaledLinCombPerColBiasEltAct<
    GELU,       // Activation
    float,      // ElementOutput
    float,      // ElementCompute
    float,      // ElementBias
    float,      // ElementSource
    float       // ElementScalar (per-column scale)
>;
```

### Block Scaling (FP8)

For FP8 quantization, CUTLASS 3.x supports **block scaling** — each block of N elements has its own scale factor:

```cpp
// From operations.hpp: ScaledLinCombPerRowBiasEltAct with block scaling
template <...>
struct ScaledLinCombPerRowBiasEltAct
    : LinCombPerRowBiasEltAct<...> {
  static constexpr bool IsScaleFactorSupported = true;
};
```

The block scale factor is loaded from global memory and applied element-wise within the epilogue.

### AMAX Tracking

For dynamic quantization, the epilogue can compute the absolute maximum value (AMAX) of the output:

```cpp
// From operations.hpp
using ElementAmax = void;
static constexpr bool IsAbsMaxSupported = false;
```

When `IsAbsMaxSupported = true`, the epilogue computes `max(|D|)` in addition to writing the output. This is used for dynamic per-tensor or per-channel quantization.

---

## 9. Relationship to SAM 3.1's addmm_act

### SAM 3.1's Operation

SAM 3.1's `addmm_act` performs:
```
D = activation(alpha * A @ B + per_row_bias + beta * C)
```

This is exactly the `LinCombPerRowBiasEltAct` fusion operation in CUTLASS 3.x:

```cpp
// CUTLASS 3.x fusion operation for SAM 3.1's addmm_act
using SamAddmmAct = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
    SiLU,           // Activation (SAM uses SiLU)
    cutlass::half_t, // ElementOutput
    float,           // ElementCompute (always f32 for accuracy)
    cutlass::half_t, // ElementBias
    cutlass::half_t  // ElementSource
>;
```

### EVT Tree for SAM 3.1

```cpp
using SamEpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    // Root: SiLU activation
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::SiLu,
        cutlass::half_t,
        float
    >,
    // Child: add per-row bias
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus,
            float,
            float
        >,
        // Left: alpha * acc + beta * C
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiply_add,
                float,
                float
            >,
            cutlass::epilogue::fusion::Sm90AccFetch,
            cutlass::epilogue::fusion::Sm90SrcFetch
        >,
        // Right: per-row bias vector
        cutlass::epilogue::fusion::Sm90RowBroadcast<
            cutlass::layout::RowMajor,
            cutlass::half_t
        >
    >
>;
```

### Performance Impact

Without epilogue fusion, SAM 3.1's `addmm_act` would be:
1. **Kernel 1**: GEMM → write `alpha * acc + beta * C` to global memory
2. **Kernel 2**: Read from global memory, add per-row bias, write to global memory
3. **Kernel 3**: Read from global memory, apply SiLU, write to global memory

With epilogue fusion:
1. **Kernel 1**: GEMM with fused epilogue → write `SiLU(alpha * acc + beta * C + bias)` directly

This eliminates **2 global memory round-trips**, which is critical because:
- Each global memory read/write costs ~200-400 cycles of latency
- SiLU is compute-heavy (`kIsHeavy = true`) but still faster than a memory round-trip
- The fused epilogue processes data while it's still in registers/L1 cache

---

## 10. Custom Epilogue Design Patterns

### Pattern 1: Custom Activation via Functor

Define a new activation function:

```cpp
template <typename T>
struct SwishBeta {
  static const bool kIsHeavy = true;

  struct Arguments {
    T beta = T(1.0);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const &value, T beta) const {
    Sigmoid<T> sigmoid;
    return value * sigmoid(beta * value);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value, Arguments const &args = Arguments()) const {
    return operator()(value, args.beta);
  }
};

template <typename T, int N>
struct SwishBeta<Array<T, N>> {
  static const bool kIsHeavy = true;
  using Arguments = typename SwishBeta<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value, T beta) const {
    Sigmoid<Array<T, N>> sigmoid_op;
    multiplies<Array<T, N>> mul;
    Array<T, N> scaled = mul(Array<T, N>(beta), value);
    return mul(value, sigmoid_op(scaled));
  }
};
```

Then compose with `LinearCombinationGeneric`:

```cpp
using MyEpilogue = LinearCombinationGeneric<SwishBeta, half_t, 8, float, float>;
```

### Pattern 2: Custom EVT Node

For CUTLASS 3.x, create a custom EVT compute node:

```cpp
// Custom compute operation for the EVT system
struct MyCustomOp {
  template <typename T>
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    // Custom computation: e.g., gated activation
    return a * sigmoid(b);
  }
};

// Use in EVT tree
using CustomEVT = Sm90EVT<
    Sm90Compute<MyCustomOp, float, float>,
    Sm90AccFetch,
    Sm90SrcFetch
>;
```

### Pattern 3: Multi-Output Epilogue

For cases where you need both the activated output AND the pre-activation value (for backward pass):

```cpp
// D = activation(pre_activation)
// aux = pre_activation (stored for backward)
using DualOutputEVT = cutlass::epilogue::fusion::LinCombPerRowBiasEltActAux<
    cutlass::layout::RowMajor,  // GmemLayoutTagAux
    SiLU,                        // Activation
    cutlass::half_t,             // ElementOutput
    float,                       // ElementCompute
    cutlass::half_t,             // ElementAux
    cutlass::half_t              // ElementBias
>;
```

### Pattern 4: Residual Block Epilogue

For ResNet-style residual blocks:

```cpp
// D = activation(alpha * acc + beta * C + bias) + residual
// The residual is added AFTER activation
using ResidualEVT = cutlass::epilogue::fusion::PerColResAddPerColBiasEltAct<
    ReLU,
    cutlass::half_t,
    float,
    cutlass::half_t,
    cutlass::half_t
>;
```

Note: `IsResidualSupported = true` means the source C is added AFTER activation, not before. This is the key distinction between:
- `LinCombPerRowBiasEltAct`: `D = activation(alpha*acc + beta*C + bias)` (residual before activation)
- `PerColResAddPerColBiasEltAct`: `D = activation(alpha*acc + bias) + beta*C` (residual after activation)

---

## Performance Considerations for SAM 3.1

### Activation Cost Comparison

| Activation | Instructions per element | Register pressure | `kIsHeavy` |
|------------|------------------------|-------------------|-----------|
| ReLU | 1 (FMAX) | Low | false |
| SiLU | 7+ (sigmoid + mul) | Medium | true |
| GELU | 8+ (tanh approx) | Medium | true |

### Recommendations

1. **Use CUTLASS 3.x EVT** for SAM 3.1's `addmm_act`. The `LinCombPerRowBiasEltAct<SiLU>` or equivalent EVT tree is the right approach.

2. **Per-row bias vs per-column bias**: If the bias is per-column, use the simpler `LinCombPerColBiasEltAct`. For per-row bias, use `LinCombPerRowBiasEltAct` or `PerRowLinCombPerRowBiasEltAct`.

3. **Compute type**: Always use `float` for the compute type (`ElementCompute`). The accumulator is `float`, and converting to a lower precision before activation loses accuracy.

4. **TMA optimization**: On Hopper, use `Sm90TmaWarpSpecialized` dispatch policy for maximum epilogue throughput. The TMA handles all memory operations asynchronously.

5. **Shared memory reuse**: Enable `ReuseSmemC = true` when possible to save shared memory for larger tiles.

6. **SiLU optimization**: Since SiLU is `kIsHeavy = true`, the scheduler will overlap its computation with memory operations. This makes the fused SiLU epilogue nearly "free" — the activation computation hides behind memory latency.

---

## Sources

- NVIDIA CUTLASS `include/cutlass/epilogue/` — All epilogue source files
- NVIDIA CUTLASS `include/cutlass/epilogue/thread/activation.h` — Activation function definitions
- NVIDIA CUTLASS `include/cutlass/epilogue/thread/linear_combination_generic.h` — Generic epilogue template
- NVIDIA CUTLASS `include/cutlass/epilogue/thread/linear_combination_bias_relu.h` — Fused bias+ReLU
- NVIDIA CUTLASS `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` — Hopper TMA epilogue
- NVIDIA CUTLASS `include/cutlass/epilogue/fusion/callbacks.hpp` — EVT callback system
- NVIDIA CUTLASS `include/cutlass/epilogue/fusion/operations.hpp` — Fusion operation definitions
- NVIDIA CUTLASS `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` — sm90 EVT implementation
