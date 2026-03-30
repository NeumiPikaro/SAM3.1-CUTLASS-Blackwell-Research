# PTX/SASS Level Optimization for Maximum GPU Performance

**Target:** NVIDIA Blackwell (sm_100) / SAM 3.1 Kernel Optimization
**Date:** 2026-03-30

## 1. PTX vs SASS: What They Are, When to Optimize Each

### PTX (Parallel Thread Execution)

PTX is NVIDIA's virtual ISA (Instruction Set Architecture) — an intermediate representation that sits between CUDA C++ source and the actual machine code the GPU executes. PTX is:

- **Stable across architectures.** A PTX kernel compiled for sm_70 can be JIT-compiled by the driver to sm_100 (Blackwell).
- **Human-readable** with register names (%r1, %f3), typed instructions (ld.global.f32), and structured control flow.
- **JIT-compiled** at runtime by the NVIDIA driver (ptxas) to produce SASS.
- **Higher-level than SASS.** PTX exposes registers, predicates, and memory spaces, but leaves scheduling and encoding to the driver's JIT compiler.

PTX optimization is useful when you want performance gains that survive driver upgrades. The driver's JIT compiler improves over time, so well-written PTX can benefit from future compiler improvements without recompilation.

### SASS (Streaming ASSembler)

SASS is the actual machine code executed by the GPU hardware. It is:

- **Architecture-specific.** SASS for sm_90 (Hopper) will not run on sm_100 (Blackwell). Each architecture has its own SASS encoding.
- **Stable within a generation.** A kernel compiled to sm_90 SASS runs on any sm_90 GPU without JIT overhead.
- **The ground truth.** All scheduling decisions, register allocation, and instruction encoding are final in SASS.
- **Not forward-compatible.** SASS compiled for sm_90 cannot be JIT-compiled for sm_100.

SASS optimization matters when you need the absolute last cycle of performance — hand-tuning instruction scheduling, NOP padding, dual-issue pairing, and cache configuration that the compiler may not get right.

### When to Optimize Which

| Scenario | Optimize PTX | Optimize SASS |
|---|---|---|
| You want driver-upgrade portability | ✅ | ❌ |
| You need cycle-exact control | ❌ | ✅ |
| You're writing CUTLASS-level templates | ✅ (ptxas flags) | ✅ (cuobjdump analysis) |
| You want to ship precompiled kernels | ❌ | ✅ (fatbin) |
| You're hand-writing critical math | Both | Both |

**Bottom line:** For SAM 3.1, focus on **PTX-level optimization** (compiler flags, inline asm, CUTLASS template parameters) and **validate against SASS** (cuobjdump analysis) to confirm the compiler is doing what you expect.


## 2. Key PTX Instructions for Blackwell (sm_100)

### 2.1 WGMMA — Warp Group Matrix Multiply-Accumulate

WGMMA is the core Tensor Core instruction introduced in Hopper and continued in Blackwell. It replaces the older `wmma` instructions with a more capable, higher-throughput variant.

```ptx
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
  {%f0, %f1, ..., %f31},   // accumulator (64x128 fp32)
  %descA,                   // shared memory descriptor for A
  %descB;                   // shared memory descriptor for B
```

Key characteristics:
- **64x128x32 tiles** per instruction (vs 16x16x16 for wmma).
- **Requires shared memory descriptors** — the A/B matrices are referenced through opaque descriptors, not raw pointers.
- **Accumulates in registers** — 64x128 fp32 values = 8192 bytes of register file per warp group.
- **Warp group = 4 warps.** A single wgmma instruction operates across 128 threads (4 warps x 32 threads).
- **Async execution.** WGMMA is inherently asynchronous — the GPU can overlap wgmma with memory loads.

For Blackwell, sm_100 adds `wgmma.mma_async` with wider types:
- `f32.e4m3.e4m3` — FP8 E4M3 inputs, FP32 accumulator
- `f32.e4m5.e4m5` — FP8 E5M2 inputs, FP32 accumulator
- `f32.f16.f16` — FP16 inputs
- `f32.bf16.bf16` — BF16 inputs

The CUTLASS library wraps these in its `KernelTmaWarpSpecialized` kernel templates. For SAM 3.1, ensure your CUTLASS version supports sm_100 wgmma.

### 2.2 TMA — Tensor Memory Accelerator

TMA is the hardware copy engine introduced in Hopper and enhanced in Blackwell. It performs multidimensional copies between global memory and shared memory with zero thread-level overhead.

```ptx
cp.async.bulk.tensor.shared.global.L2.mbarrier::complete_tx::bytes.cta_group::1
  [%shared_addr],    // destination in shared memory
  [%barrier],        // mbarrier for synchronization
  [%tensor_map],     // tensor map descriptor (opaque)
  [%coord0, %coord1, %coord2, %coord3, %coord4];  // tensor coordinates
```

TMA features:
- **Multidimensional addressing.** Up to 5D tensors can be loaded with a single instruction.
- **Automatic padding, swizzling, and boundary handling.** No manual out-of-bounds checks.
- **Hardware-managed async.** TMA operations are queued in a hardware FIFO and execute independently of the SM.
- **Bulk copies.** For 1D copies, `cp.async.bulk` handles arbitrary sizes.
- **mbarrier synchronization.** The `mbarrier` primitive counts arriving TMA operations and releases waiting threads when complete.

Blackwell TMA enhancements: improved throughput vs Hopper, support for `cta_group::2` enabling multi-CTA cooperative TMA loads.

### 2.3 FENCE — Memory Fences

Memory fences enforce ordering between memory operations:

```ptx
// Fence for shared memory writes visible to all threads in CTA
membar.cta

// Fence for global memory visibility across CTAs
membar.gl

// Fence for async operations (wgmma, TMA) before consuming results
fence.proxy.async
```

The `fence.proxy.async` instruction (Hopper+) ensures that all prior async operations (TMA loads, wgmma) have completed before any subsequent memory accesses depend on their results.

For Blackwell pipeline kernels:
- Use `fence.proxy.async` between wgmma completion and result consumption.
- Use `cp.async.commit_group` + `cp.async.wait_group` for staged pipeline control.
- Use `barrier.sync` (or mbarrier) for cross-warp/warp-group coordination.

### 2.4 Other Important Blackwell Instructions

| Instruction | Purpose |
|---|---|
| `ldmatrix` | Load 8x8 matrix tiles from shared memory into registers |
| `stmatrix` | Store register tiles back to shared memory |
| `cp.async` | Async copy from global to shared memory |
| `cp.async.bulk` | Bulk async copy (TMA-style for non-tensor data) |
| `mbarrier.arrive` | Signal completion of an async operation |
| `mbarrier.wait` | Block until mbarrier reaches expected count |
| `elect.sync` | Elect a leader thread for warp-level operations |
| `redux.sync` | Warp-level reduction (sum, min, max) |


## 3. Compiler Flags for Maximum Performance

### 3.1 Basic Optimization

```
nvcc -O3 --use_fast_math -arch=sm_100 kernel.cu
```

| Flag | Effect |
|---|---|
| `-O3` | Maximum host and device code optimization. |
| `--use_fast_math` | Replaces IEEE-compliant math with faster, less precise intrinsics. Uses `__sinf()`, `__cosf()`, `__expf()`, etc. Trades ~1 ULP accuracy for ~2-4x speed on transcendental math. |
| `-arch=sm_100` | Compile device code for Blackwell. Generates PTX/sm_100 SASS. |

`--use_fast_math` trade-offs for SAM 3.1:
- Vision models (ViT, SAM decoder) are typically tolerant of reduced precision.
- If computing attention scores that need numerical stability, `--use_fast_math` can cause issues (denormal flushing).
- Safe for: activation functions (GELU, SiLU), positional embeddings, normalization.
- Avoid for: loss computation, softmax numerics (unless verified).

### 3.2 Intermediate Compilation

```
nvcc -O3 -arch=sm_100 --split-compile=32 kernel.cu
```

`--split-compile=N` splits compilation of a single translation unit into N parallel compilations:
- Dramatically reduces compile time for large CUDA files.
- Can improve optimization because each chunk has fresh register pressure analysis.
- Default is N=0 (no splitting).

### 3.3 Debug vs Release

```
# Debug (slow, correct)
nvcc -G -g -O0 -arch=sm_100 kernel.cu

# Release (fast)
nvcc -O3 --use_fast_math -DNDEBUG -arch=sm_100 kernel.cu

# Profile (fast, with line info for nsight)
nvcc -O3 --use_fast_math -lineinfo -arch=sm_100 kernel.cu
```

`-lineinfo` adds source line mapping without affecting performance — essential for Nsight Compute profiling.

### 3.4 Forward Compatibility

```
nvcc -O3 -arch=sm_100   --generate-code arch=compute_100,code=sm_100   --generate-code arch=compute_100,code=compute_100   kernel.cu
```

This generates:
1. **sm_100 SASS** — precompiled binary for Blackwell.
2. **compute_100 PTX** — embeds PTX for JIT compilation on future architectures.

For production SAM 3.1 deployment, generate both SASS and PTX to cover current and future GPUs.


## 4. Advanced Compiler Flags

### 4.1 Register Pressure Control

```
nvcc -O3 -arch=sm_100 --maxrregcount=255 kernel.cu
```

| Flag | Effect |
|---|---|
| `--maxrregcount=N` | Limits device function register usage to N per thread. Default: unlimited (up to 255 on Blackwell). |
| `-Xptxas -v` | Reports per-kernel register, shared memory, and constant memory usage. |

Why register pressure matters:
- Blackwell sm_100 has **256 registers per thread** (up from 255 on Hopper).
- Each SM has **65536 registers** total.
- Occupancy = (65536 / registers_per_thread / threads_per_block) / max_concurrent_blocks.
- At 255 registers/thread and 128 threads/block: occupancy ~ 2 blocks per SM.
- At 128 registers/thread: occupancy ~ 4 blocks per SM.

For SAM 3.1 Tensor Core kernels:
- WGMMA kernels typically use 200-255 registers (large accumulator tiles).
- The compiler spills to local memory if registers are capped too low — **local memory spills are catastrophic** (100x slower than registers).
- **Strategy:** Use `-Xptxas -v` to see actual register usage, then cap at `actual + 8` for headroom.

```
# Check register usage
nvcc -O3 -arch=sm_100 -Xptxas -v kernel.cu 2>&1 | grep registers

# If kernel uses 187 registers, cap at 196
nvcc -O3 -arch=sm_100 --maxrregcount=196 kernel.cu
```

### 4.2 -Xptxas Options

These pass flags directly to the PTX-to-SASS compiler (ptxas):

```
nvcc -O3 -arch=sm_100   -Xptxas -v   -Xptxas -warn-spills   -Xptxas -warn-lmem-usage   -Xptxas -warn-double-usage   -Xptxas -dlcm=cg   kernel.cu
```

| `-Xptxas` Flag | Effect |
|---|---|
| `-v` | Verbose: report register, shared, constant memory usage per kernel |
| `-warn-spills` | Warn on register spills to local memory |
| `-warn-lmem-usage` | Warn on local memory usage (stack frames, arrays) |
| `-warn-double-usage` | Warn when doubles are used |
| `-dlcm=cg` | Default L1 cache policy: `cg` = cache global (L2 only), `ca` = cache all |
| `-O3` | Maximum ptxas optimization (default when nvcc -O3) |

L1 cache policy for SAM 3.1:
- Tensor Core kernels: use `-dlcm=cg` (skip L1, use L2) — Tensor Core operands go through shared memory.
- Memory-bound kernels (embedding lookups): use `-dlcm=ca` (use L1 for caching).

### 4.3 Shared Memory Configuration

```
nvcc -O3 -arch=sm_100 --shared-memory-size=164000 kernel.cu
```

Or at runtime:
```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
```

Blackwell supports up to **228KB shared memory per SM** (configurable between L1/shared).

### 4.4 Template Optimization

```
nvcc -O3 -arch=sm_100   --ftemplate-backtrace-limit=0   --ftemplate-depth=1024   kernel.cu
```

CUTLASS uses deeply nested templates. Increase template depth if you hit compiler limits.


## 5. Link Time Optimization (LTO) for CUDA

NVCC supports LTO for device code (CUDA 12.0+), enabling cross-file optimization of CUDA kernels.

### 5.1 How to Enable

```
# Step 1: Compile to LTO IR
nvcc -O3 -arch=sm_100 -dlto -c kernel1.cu -o kernel1.o
nvcc -O3 -arch=sm_100 -dlto -c kernel2.cu -o kernel2.o

# Step 2: Link with LTO
nvcc -O3 -arch=sm_100 -dlto kernel1.o kernel2.o -o my_app
```

### 5.2 What LTO Enables

- **Cross-file inlining.** A device function in kernel1.cu can be inlined into kernel2.cu.
- **Dead code elimination.** Unused template instantiations are removed.
- **Constant propagation.** Template parameters and constexpr values are propagated across translation units.
- **Register pressure optimization.** The compiler sees all callers and optimizes register allocation globally.

### 5.3 Performance Impact

For CUTLASS-based SAM 3.1 kernels:
- LTO can recover **2-5%** performance by enabling cross-file inlining.
- Particularly effective when CUTLASS template instantiations are split across multiple .cu files.
- **Compile time increases significantly** — LTO links are single-threaded and memory-intensive.

### 5.4 CUTLASS Integration

CUTLASS 3.x has built-in LTO support:
```
# CMakeLists.txt
set(CUTLASS_ENABLE_LTO ON CACHE BOOL "Enable LTO for CUTLASS")
```

When building CUTLASS for SAM 3.1:
```
cmake -DCUTLASS_ENABLE_LTO=ON       -DCUTLASS_NVCC_ARCHS="100a"       -DCMAKE_BUILD_TYPE=Release       ..
```

## 6. Inline PTX Assembly for Critical Paths

When the compiler doesn't generate optimal code, you can inline PTX/SASS directly into CUDA C++.

### 6.1 Basic Inline PTX

```cuda
__device__ __forceinline__ float fast_rsqrt(float x) {
    float result;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
```

### 6.2 Register Constraints

| Constraint | Meaning |
|---|---|
| `"r"` | 32-bit integer register (%r) |
| `"l"` | 64-bit integer register (%rd) |
| `"f"` | 32-bit float register (%f) |
| `"d"` | 64-bit float register (%fd) |
| `"h"` | 16-bit half register (%h) |

### 6.3 Useful Inline PTX for SAM 3.1

```cuda
// Fast approximate GELU (tanh-based)
__device__ __forceinline__ float fast_gelu(float x) {
    float c0 = 0.044715f;
    float c1 = 0.7978845608f;
    float half_x = x * 0.5f;
    float inner = x + x * x * x * c0;
    inner = inner * c1;
    float tanh_val;
    asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_val) : "f"(inner));
    return half_x * (1.0f + tanh_val);
}

// Global load with CG cache hint (skip L1)
__device__ __forceinline__ float ld_global_cg(float* addr) {
    float result;
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(result) : "l"(addr));
    return result;
}
```

### 6.4 When to Use Inline PTX

Good candidates:
- Transcendental functions not available as intrinsics (e.g., `tanh.approx`)
- Specific cache policies the compiler doesn't select (`ld.global.cg`)
- Barrier/fence instructions for custom synchronization
- TMA descriptor setup (complex, multi-instruction sequences)

Avoid:
- Simple arithmetic (the compiler handles this well)
- Instructions the compiler already generates correctly
- Anything that breaks register allocation


## 7. Instruction Scheduling and Dual-Issue Optimization

### 7.1 GPU Instruction Scheduling

Unlike CPUs, GPUs don't reorder instructions at runtime. The compiler schedules instructions, and the hardware executes them in order (within each warp). GPUs exploit latency hiding through warp interleaving — when one warp stalls on memory, the SM switches to another warp.

Key principle: **Minimize stall cycles per warp.** The compiler does this by:
1. Moving independent instructions between dependent ones (instruction-level parallelism).
2. Ensuring enough independent work to hide memory latency.

### 7.2 Dual Issue

Modern NVIDIA GPUs (including Blackwell) support dual-issue — issuing two independent instructions in the same clock cycle from the same warp. This effectively doubles throughput for certain instruction mixes.

Dual-issue pairs (Blackwell sm_100):
- Two independent ALU instructions (e.g., `add.f32` + `mul.f32`)
- One ALU + one memory instruction (e.g., `add.f32` + `ld.global.f32`)
- Two independent memory instructions (to different cache lines)

NOT dual-issuable:
- Instructions that read/write the same register
- WGMMA instructions (they have their own pipeline)
- TMA instructions (hardware-managed)

### 7.3 How to Encourage Dual Issue

In CUDA:
```cuda
// Good: independent operations that can dual-issue
float a = x[i];
float b = y[i];
float c = a + b;    // ALU
float d = a * b;    // ALU (independent of c)
z[i] = c + d;
```

For SAM 3.1 convolution kernels:
- Interleave TMA loads with register math (the compiler usually does this).
- Ensure WGMMA accumulator computation doesn't block on the WGMMA pipeline.

### 7.4 Checking SASS for Dual Issue

Use `cuobjdump` to verify the compiler generated dual-issue pairs:

```
cuobjdump -sass my_app | grep -A 20 "MyKernel"
```

Two instructions at the same offset indicates dual issue:
```
/*00a0*/  IADD3 R4, R4, 0x4, RZ ;       // ALU
/*00a0*/  LDG.E R8, [R12+0x10] ;        // Memory (same cycle = dual issue)
```

### 7.5 Compiler Hints for Better Scheduling

```cuda
// Unroll hint
#pragma unroll 4
for (int i = 0; i < N; i++) {
    acc += data[i] * weights[i];
}

// Force inline for better scheduling across function boundaries
__device__ __forceinline__ float compute(float a, float b) { ... }

// Restrict pointers for more aggressive scheduling
__global__ void kernel(const float* __restrict__ input,
                       float* __restrict__ output) { ... }
```

## 8. Memory Instruction Reordering for Coalescing

### 8.1 What is Memory Coalescing?

When a warp (32 threads) accesses global memory, the hardware coalesces accesses to consecutive addresses into as few memory transactions as possible:
- **128-byte cache line** — if all 32 threads access within a 128-byte window, it's a single transaction.
- **Ideal:** 32 threads x 4 bytes = 128 bytes = 1 coalesced access.
- **Worst case:** 32 threads x scattered addresses = 32 separate transactions.

### 8.2 Compiler-Generated Coalescing

The nvcc compiler automatically reorders memory instructions to improve coalescing when:
- Array accesses are contiguous: `data[threadIdx.x]`
- Stride-1 pattern: `data[base + threadIdx.x]`

Not coalesced (compiler cannot fix):
```cuda
// Strided access — each thread reads every 32nd element
float val = data[threadIdx.x * 32];  // stride = 32
```

Coalesced:
```cuda
float val = data[threadIdx.x];  // stride = 1
```

### 8.3 Shared Memory Bank Conflict Avoidance

Shared memory is divided into 32 banks. Accesses that hit the same bank from different threads cause bank conflicts (serialized access).

For Blackwell SAM 3.1:
- Shared memory is swizzled by TMA hardware (automatic bank conflict avoidance).
- CUTLASS handles swizzle patterns in its SharedStorage layouts.
- You typically don't need to manually pad shared memory when using CUTLASS TMA kernels.

For custom shared memory access:
```cuda
// Fix: add padding
__shared__ float smem[32][33];  // 33 instead of 32 to avoid bank conflicts
```


## 9. How Much Can PTX Optimization Improve Over CUTLASS Defaults?

### 9.1 CUTLASS Baseline Performance

CUTLASS is already heavily optimized. NVIDIA's engineers spend significant effort on:
- Optimal wgmma/tma instruction sequences
- Software pipelining (staged TMA loads overlapping with compute)
- Register allocation for large accumulator tiles
- Shared memory layout for bank-conflict-free access

For most workloads, CUTLASS 3.x with sm_100 targets achieves **90-95% of theoretical peak throughput**.

### 9.2 Where PTX/SASS Optimization Can Help

| Optimization | Typical Improvement | Effort |
|---|---|---|
| `--use_fast_math` | 5-15% for math-bound kernels | Low |
| Register cap tuning | 3-10% (improved occupancy) | Medium |
| `--maxrregcount` spill analysis | 5-20% (avoid spills or improve occ.) | Medium |
| Inline PTX for specific ops | 2-8% for targeted operations | High |
| SASS hand-scheduling | 5-15% for critical inner loops | Very High |
| LTO cross-file inlining | 2-5% | Low (just enable flag) |
| Dual-issue optimization | 3-10% for compute-bound | High |
| Shared memory layout tuning | 2-8% for memory-bound | Medium |

### 9.3 Realistic Expectations for SAM 3.1

For a well-tuned CUTLASS-based SAM 3.1 inference pipeline:
- **Compiler flags alone** (-O3, --use_fast_math, register tuning): **3-8%** improvement.
- **LTO + compiler flags**: **5-10%** improvement.
- **Hand-optimized PTX for critical kernels** (attention, conv): **10-20%** improvement over compiler defaults.
- **SASS-level hand-tuning**: **5-15%** additional (diminishing returns, very high effort).

**Total realistic improvement: 10-25%** over naive CUTLASS defaults with careful optimization.

**Important:** Most of the gain comes from getting the right CUTLASS kernel configs (tile sizes, pipeline stages, warp specialization strategy), not from low-level PTX hacking. Focus there first.

## 10. Practical: Compiler Flags for SAM 3.1

### 10.1 Recommended Build Configuration

```
nvcc \
  -O3 \
  --use_fast_math \
  -arch=sm_100 \
  -lineinfo \
  --generate-code arch=compute_100,code=sm_100 \
  --generate-code arch=compute_100,code=compute_100 \
  --split-compile=32 \
  -Xptxas -v \
  -Xptxas -warn-spills \
  -Xptxas -warn-lmem-usage \
  -Xptxas -dlcm=cg \
  --ftemplate-backtrace-limit=0 \
  --ftemplate-depth=1024 \
  -DNDEBUG \
  sam31_inference.cu \
  -lcublas -lcublasLt -lcudart \
  -o sam31_infer
```

### 10.2 With LTO (Production Builds)

```
# Compile
nvcc -O3 --use_fast_math -arch=sm_100 -dlto -lineinfo \
  --split-compile=32 \
  -Xptxas -v -Xptxas -warn-spills \
  -Xptxas -dlcm=cg \
  -c sam31_encoder.cu -o sam31_encoder.o

nvcc -O3 --use_fast_math -arch=sm_100 -dlto -lineinfo \
  --split-compile=32 \
  -Xptxas -v -Xptxas -warn-spills \
  -Xptxas -dlcm=cg \
  -c sam31_decoder.cu -o sam31_decoder.o

# Link
nvcc -O3 --use_fast_math -arch=sm_100 -dlto \
  sam31_encoder.o sam31_decoder.o \
  -lcublas -lcublasLt -lcudart \
  -o sam31_infer
```

### 10.3 CMake Integration

```cmake
# CMakeLists.txt
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo --split-compile=32")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v -Xptxas -warn-spills")

# For specific kernels
set_source_files_properties(sam31_attention.cu PROPERTIES
  COMPILE_FLAGS "--maxrregcount=224 -Xptxas -dlcm=cg"
)
```

### 10.4 Verification Checklist

After building, verify:

```
# 1. Check register usage
nvcc -O3 --use_fast_math -arch=sm_100 -Xptxas -v sam31_inference.cu 2>&1 | \
  grep -E "(registers|bytes smem|spill)"

# 2. Verify no local memory spills
nvcc -O3 --use_fast_math -arch=sm_100 -Xptxas -warn-spills sam31_inference.cu 2>&1 | \
  grep -i spill

# 3. Profile with Nsight Compute
ncu --set full --launch-skip 10 --launch-count 1 \
  -o sam31_profile \
  ./sam31_infer --input test_image.png

# 4. Check SASS for dual-issue
cuobjdump -sass sam31_infer | grep -A 5 "sam31_attention"

# 5. Verify FP8 / WGMMA usage
cuobjdump -sass sam31_infer | grep -c wgmma
```

## Summary

| Optimization Category | Key Actions | Expected Gain |
|---|---|---|
| **Compiler flags** | `-O3`, `--use_fast_math`, `--split-compile` | 3-8% |
| **Register tuning** | `-Xptxas -v` then `--maxrregcount` | 3-10% |
| **LTO** | `-dlto` on compile + link | 2-5% |
| **Cache policy** | `-Xptxas -dlcm=cg` for Tensor Core kernels | 1-3% |
| **Inline PTX** | Critical math, cache hints, barriers | 2-8% |
| **SASS validation** | `cuobjdump` analysis, dual-issue check | Prevents regressions |
| **CUTLASS config** | Tile sizes, pipeline stages, warp specialization | 10-30% (biggest lever) |

**Priority order for SAM 3.1:**
1. Get CUTLASS kernel configs right (tile sizes, pipeline depth)
2. Set compiler flags (the recommended set above)
3. Analyze register usage and tune --maxrregcount
4. Enable LTO for production builds
5. Profile with Nsight Compute and optimize hotspots
6. Only then consider inline PTX or SASS hand-tuning
