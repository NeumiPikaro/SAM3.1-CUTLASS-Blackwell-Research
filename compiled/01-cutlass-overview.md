# CUTLASS Overview & Architecture — Deep Technical Analysis

## 1. What is CUTLASS

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's official template library for writing high-performance matrix computation kernels on NVIDIA GPUs. As of March 2026, CUTLASS is at version 4.4.2 and supports architectures from Volta (SM70) through Blackwell (SM100/SM103).

### Core Philosophy
Unlike cuBLAS (a monolithic binary library), CUTLASS provides **composable C++ template abstractions** that let developers:
- Customize every level of the GEMM hierarchy
- Mix and match tile sizes, data types, and scheduling strategies
- Build domain-specific kernels (attention, MoE, SSD) on top of proven GEMM primitives
- Target specific architectures without overhead from generality

## 2. The GEMM Hierarchy — Four Levels of Decomposition

CUTLASS decomposes GEMM = alpha * A * B + beta * C into four levels:

### Level 1: Device (Grid) Level
- Maps the full (M, N, K) problem to a grid of threadblocks
- Handles non-tile-aligned problems via padding/residual
- Persistent kernel scheduling (Hopper+) launches max SMs and distributes work
- StreamK mode divides K-dimension for better load balancing

### Level 2: Threadblock Level  
- Each threadblock computes a (TileM, TileN, TileK) sub-matrix of D
- Loads tiles of A and B from global memory → shared memory
- Software pipelining: overlap load of stage N+1 with compute of stage N
- On Hopper/Blackwell: TMA hardware handles async global→SMEM copies
- Typical tile shapes: (128, 128, 64), (256, 128, 64), (128, 256, 64)

### Level 3: Warp Level
- Each warp (32 threads) handles a sub-tile of the threadblock's work
- Warp-level MMA (matrix multiply-accumulate) instructions
- On Hopper+: WGMMA (Warp Group Matrix Multiply-Accumulate) — 4 warps (128 threads) cooperate
- LDMATRIX/STMATRIX instructions for shared memory → register file movement

### Level 4: Instruction (Thread) Level
- Hardware Tensor Core instructions: mma.sync (Ampere), wgmma.mma_async (Hopper/Blackwell)
- These compute small matrix tiles: typically 16×8×16 for FP16/BF16, 16×8×32 for FP8
- Results accumulate in register file (typically FP32 accumulators)
- The actual silicon doing the math — everything above is about feeding these instructions

## 3. Template Architecture

### Key Template Parameters
```
GemmUniversal<
  ProblemShape,           // Shape<int,int,int> for static, Shape<int,int,int> for dynamic
  CollectiveMainloop,     // How to load and compute
  CollectiveEpilogue      // How to store and post-process
>
```

### CollectiveBuilder (The Easy Way)
```cpp
using CollectiveMainloop = CollectiveBuilder<
  Sm90, OpClassTensorOp,    // Architecture + operator class
  half_t, RowMajor, 8,      // A: type, layout, alignment
  half_t, ColumnMajor, 8,   // B: type, layout, alignment
  float,                    // Accumulator type
  Shape<_128,_128,_64>,     // Threadblock tile shape
  Shape<_1,_2,_1>,          // Cluster shape
  StageCountAuto,           // Auto-compute pipeline stages
  KernelScheduleAuto        // Auto-select kernel schedule
>::CollectiveOp;
```

### Manual Construction (Full Control)
```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
  CollectiveMmaArgs<
    Sm90, Shape<_128,_128,_64>, ...>,
  TileScheduler  // Static vs dynamic
>;
```

## 4. Data Type Support Matrix (CUTLASS 4.4)

| Input A | Input B | Accumulator | Output | Architecture |
|---------|---------|-------------|--------|-------------|
| FP64    | FP64    | FP64        | FP64   | Volta+      |
| FP32    | FP32    | FP32        | FP32   | All         |
| TF32    | TF32    | FP32        | TF32   | Ampere+     |
| FP16    | FP16    | FP32        | FP16   | Volta+      |
| BF16    | BF16    | FP32        | BF16   | Ampere+     |
| FP8 E4M3| FP8 E4M3| FP32      | FP8/BF16| Hopper+    |
| FP8 E5M2| FP8 E5M2| FP32      | FP8/BF16| Hopper+    |
| INT8    | INT8    | INT32       | INT8   | Turing+     |
| INT4    | INT4    | INT32       | INT4   | Ampere+     |
| NVFP4   | NVFP4   | FP32        | FP8    | Blackwell   |
| MXFP4   | MXFP4   | FP32        | FP8    | Blackwell   |
| MXFP8   | MXFP8   | FP32        | BF16   | Blackwell   |
| Binary  | Binary  | INT32       | Binary | Turing+     |

**SAM 3.1 uses BF16 inputs → FP32 accumulator → BF16 output.** This maps perfectly to CUTLASS's BF16 GEMM support on Ampere+.

## 5. Architecture-Specific Features

### Volta (SM70)
- First Tensor Core: mma.sync 8×8×4 FP16
- WMMA API
- cp.async not available (synchronous loads only)

### Turing (SM75)
- INT8/INT4 Tensor Cores
- LDMATRIX instruction (shared memory → register)
- Still uses mma.sync 16×8×8

### Ampere (SM80/SM86)
- cp.async for async shared memory loads
- mma.sync 16×8×16 FP16/BF16
- TF32 Tensor Cores
- 2-stage software pipelining with cp.async barriers

### Hopper (SM90) — Game Changer
- **TMA (Tensor Memory Accelerator):** dedicated hardware for global→SMEM copies
- **WGMMA:** Warp Group MMA — 4 warps (128 threads) cooperate, async pipeline
- **Thread Block Clusters:** multiple SMs sharing SMEM
- **Persistent kernels:** launch max SMs, device-side work distribution
- **Ping-pong/Cooperative scheduling:** warp specialization strategies

### Blackwell (SM100/SM103) — Latest
- **Enhanced WGMMA:** higher throughput than Hopper
- **TMEM (Tensor Memory):** new memory space for Tensor Core operands
- **FP4 native support:** NVFP4, MXFP4 on Tensor Cores
- **Block-scaled GEMM:** per-block scale factors for quantized types
- **SM103:** GB300 variant with additional features
- **Programmatic Dependent Launch (PDL):** kernels can trigger follow-up kernels
- **Cluster enhancements:** larger cluster sizes, improved multicast

## 6. CUTLASS vs cuBLAS vs Custom CUDA

| Aspect | cuBLASS | CUTLASS | Custom CUDA |
|--------|---------|---------|-------------|
| Flexibility | None (fixed API) | Full (template params) | Full (from scratch) |
| Performance | Optimal for GEMM | Near-optimal | Depends on skill |
| Epilogue fusion | Limited | Full (EVT) | Full |
| Custom ops | No | Yes (collective build) | Yes |
| Maintenance | NVIDIA maintains | NVIDIA maintains | You maintain |
| Learning curve | Low | Medium-high | Very high |
| Blackwell support | Day 1 | Day 1 | Months of work |

**For SAM 3.1:** CUTLASS is the sweet spot — we get near-optimal GEMM performance with the ability to fuse operations (RoPE, addmm_act, bias) into the epilogue.

