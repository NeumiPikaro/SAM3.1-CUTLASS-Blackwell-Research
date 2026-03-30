# CuTe & CuTe DSL — Layout Abstraction Layer

## 1. What is CuTe

CuTe (CUDA Tensor Templates) is CUTLASS's core layout abstraction library. Introduced in CUTLASS 3.0, it replaced the older layout system with a mathematically rigorous framework for describing how data is arranged in memory at every level of the hierarchy.

### Core Idea
A "layout" maps logical coordinates to physical addresses. CuTe represents this as:
```
Layout: (Shape, Stride)
  Shape  = (S0, S1, S2, ...) — dimensions
  Stride = (d0, d1, d2, ...) — address increment per dimension

Address(coord) = sum(coord[i] * stride[i] for i in range(ndim))
```

### Why This Matters
Traditional CUDA code manually computes offsets: `A[row * lda + col]`. CuTe encodes this as `Layout<Shape<M,K>, Stride<K,_1>>` and handles the math automatically. When you compose, partition, or tile layouts, CuTe computes all the nested indexing.

## 2. CuTe Key Concepts

### Layout
```cpp
using A_layout = Layout<Shape<_128,_64>, Stride<_64,_1>>; // Row-major 128×64
using B_layout = Layout<Shape<_64,_128>, Stride<_1,_64>>; // Column-major 64×128
```

### Tensor
```cpp
// A tensor packages: pointer + layout + memory space
auto tensor = make_tensor(ptr, layout, GenRowMajor{});
// Access: tensor(make_coord(3, 5)) → ptr + 3*64 + 5
```

### Composition & Tiling
```cpp
// Compose two layouts: (A_layout, B_layout) → flat layout
auto composed = composition(flat_layout, tile_shape);

// Tile a layout: divide into sub-tiles
auto tiled = tile(flat_layout, TileShape<_32,_32>);
```

### Swizzle
```cpp
// XOR-based swizzle to eliminate bank conflicts
using Swizzle = Swizzle<3, 3, 3>;  // 8×8 tile, XOR on bits [2:0]
auto swizzled = composition(base_layout, Swizzle{});
```

## 3. CuTe DSL (Python) — New in CUTLASS 4.x

CUTLASS 4.0 introduced **CuTe DSL** — Python-native kernel authoring. This is a paradigm shift:

### Before (C++ Templates)
```cpp
using GemmKernel = GemmUniversal<
  Shape<int,int,int>,
  CollectiveBuilder<Sm90, OpClassTensorOp, ...>::CollectiveOp,
  DefaultEpilogue<...>
>;
```
Compilation: 5-15 minutes per kernel variant.

### After (CuTe DSL)
```python
@cute.kernel
def gemm_kernel(gA, gB, gC, problem_shape):
    # Direct Python → CUDA kernel
    sA = cute.make_tensor(cute.smem_ptr, smem_layout_A)
    tAgA = local_partition(gA, tA, threadIdx.x)
    # ... same concepts, Python syntax
```
Compilation: seconds (JIT), instant iteration.

### Key CuTe DSL Features (4.4)
- **Fragment-free programming:** copy/dot APIs take memrefs directly
- **Automatic TMA descriptor generation**
- **AoT compilation:** export kernels as .cubin files
- **JAX integration:** use CuTeDSL with JAX workflows
- **PyTorch integration:** epilogue fusion via Python EFC functions
- **experimental module:** higher-level composable APIs

### Blackwell CuTe DSL Examples
- `dense_gemm_persistent.py` — basic Blackwell GEMM
- `dense_blockscaled_gemm_persistent.py` — FP8 block-scaled GEMM
- `fmha.py` — Flash Multi-Head Attention on Blackwell
- `fmha_bwd.py` — FMHA backward pass
- `mixed_input_fmha/` — mixed-precision FMHA (INT4 KV)
- `mla/` — Multi-Latent Attention
- `grouped_gemm.py` — grouped GEMM
- `grouped_blockscaled_gemm.py` — grouped block-scaled
- `blockwise_gemm/` — blockwise quantized GEMM
- `mamba2_ssd/` — State Space Decomposition
- `rmsnorm.py` — RMSNorm kernel
- `reduce.py` — reduction kernel
- `programmatic_dependent_launch.py` — PDL example

## 4. SAM 3.1 Implications

CuTe DSL is the **fastest path to prototyping SAM 3.1 Blackwell kernels**. Instead of spending days on C++ template debugging, we can:
1. Write attention kernels in Python with CuTe DSL
2. JIT-compile and benchmark in seconds
3. Iterate on tile shapes, swizzle patterns, pipeline stages interactively
4. Export production kernels via AoT compilation

For the RTX 5060 target, CuTe DSL can auto-tune tile shapes to the specific SM count and SMEM size.

