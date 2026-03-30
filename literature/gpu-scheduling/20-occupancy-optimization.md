# GPU Occupancy Optimization for Maximum Throughput

**Date:** 2026-03-30  
**Context:** SAM 3.1 / CUTLASS kernel optimization on RTX 5060 (Blackwell)  
**Audience:** Kernel developers targeting maximum sustained throughput

---

## 1. Occupancy Calculation: Threads/SM vs Max Threads

GPU occupancy is the ratio of active warps to the maximum number of warps an SM can support. It is **not** simply a measure of "how busy" the GPU is — it quantifies how well you hide latency through warp-level parallelism.

### The Math

```
Occupancy = Active Warps / Max Warps per SM
```

For a concrete example on an RTX 5060 (Blackwell SM, assumed 32 warps/SM = 1024 threads max):

```
Threads per block = 256
Warps per block   = 256 / 32 = 8

If kernel uses 40 registers and 48 KB SMEM:
  Register limit: 65536 registers / 40 regs = 1638 threads → 1638/256 = 6 blocks → 48 warps... wait
  Actually: max_threads_reg = reg_file_size / registers_per_thread
  On Ampere/Blackwell: 65536 registers per SM / 40 = 1638 threads → floor(1638/256) = 6 blocks → 6×8 = 48 warps
  
  But max warps = 48, and max is typically 48-64 depending on arch...
  SMEM limit: 100KB / 48KB = 2 blocks → 16 warps → Occupancy = 16/48 = 33%
```

### The Three Bottlenecks

Occupancy is bounded by the **minimum** of three limiters:

| Limiter | Formula | Key Variable |
|---------|---------|--------------|
| **Registers** | `⌊(RegFile / RegsPerThread) / ThreadsPerBlock⌋ × BlocksPerSM` | Registers per thread |
| **Shared Memory** | `⌊SMEM_Per_SM / SMEM_Per_Block⌋ × WarpsPerBlock` | SMEM allocation |
| **Thread Slots** | `MaxThreadsPerSM / ThreadsPerBlock × WarpsPerBlock` | Block size |

Plus a fourth: **block limit** (typically 16-32 blocks per SM depending on arch). Even if registers and SMEM allow more blocks, the hardware block limit can cap you.

### Practical Calculation

```cpp
// Quick occupancy estimation pseudocode
int warpsPerBlock = threadsPerBlock / warpSize;
int blocksByReg   = regPerSM / (registersPerThread * threadsPerBlock);
int blocksBySMEM  = smemPerSM / smemPerBlock;
int blocksByThread = maxThreadsPerSM / threadsPerBlock;
int blocksByHW    = min(maxBlocksPerSM, blocksByReg, blocksBySMEM, blocksByThread);
int activeWarps   = blocksByHW * warpsPerBlock;
float occupancy   = (float)activeWarps / maxWarpsPerSM;
```

**Critical insight:** The register file is shared across all warps on an SM. A kernel using 64 registers per thread with 256 threads per block consumes 64 × 256 = 16,384 registers per block. On an SM with 65,536 registers, that's only 4 blocks (1024 threads, 32 warps) before registers become the limiter.

---

## 2. Register Pressure vs Occupancy Tradeoff

Register usage is often the **dominant occupancy limiter** for compute-heavy kernels, and the relationship is inversely linear:

```
Max Threads (register-limited) = RegFile_Size / Registers_Per_Thread
```

### The Tradeoff Curve

| Registers/Thread | Max Threads/SM | Max Warps/SM | Occupancy (of 48) |
|-----------------|----------------|--------------|-------------------|
| 32              | 2048           | 64           | 100% (capped)     |
| 40              | 1638           | 51           | ~100%             |
| 48              | 1365           | 42           | 88%               |
| 64              | 1024           | 32           | 67%               |
| 80              | 819            | 25           | 53%               |
| 96              | 682            | 21           | 44%               |
| 128             | 512            | 16           | 33%               |

*(Assumes 64KB register file, no SMEM/thread limits binding)*

### Spilling: The Hidden Cost

When you exceed 256 registers per thread (the hardware maximum on most architectures), the compiler spills to local memory (L1/L2 cache or DRAM). Even **approaching** the register limit causes the compiler to be conservative:

- **Register spilling** adds LD/ST instructions → memory latency instead of compute
- **Register occupancy pressure** forces fewer warps → less latency hiding
- **The irony:** Spilling to hide registers often costs more than just accepting lower occupancy

### The Sweet Spot Heuristic

For CUTLASS GEMM kernels, the register allocation per thread is typically:

- **Warp-level MMA (WMMA/WGMMA):** 32-64 registers for the accumulator fragment
- **Operand staging:** 8-16 registers for A/B fragments
- **Index/pointer math:** 10-20 registers
- **Total typical:** 48-96 registers

CUTLASS deliberately targets **50-75% register utilization** per thread — enough registers for the critical accumulator path without forcing spilling, while maintaining 50-75% occupancy.

### Compiler Flags and Manual Control

```bash
# Limit registers per thread (forces more occupancy, may cause spilling)
nvcc -maxrregcount=64 kernel.cu

# Let compiler choose (default, usually optimal)
nvcc kernel.cu

# CUTLASS uses template parameters:
# Kernel template parameter `kStages` influences register allocation
```

**Warning:** `-maxrregcount` applies to **all** functions in the translation unit. CUTLASS uses `__launch_bounds__` and `__maxnreg__` attributes for per-kernel control instead.

---

## 3. Shared Memory Usage vs Occupancy Tradeoff

Shared memory is allocated **per block** and is another primary occupancy limiter. The tradeoff is more nuanced than registers because SMEM serves a deliberate purpose in tiling.

### The Numbers

On RTX 5060 (Blackwell SM):

```
Total SMEM per SM: ~100 KB (configurable as 0/8/16/.../100 KB)
Max blocks per SM: 32
Min SMEM per block: depends on kernel
```

If your CUTLASS kernel allocates 48 KB of SMEM per block:

```
Blocks by SMEM = floor(100 / 48) = 2 blocks per SM
```

That's 2 blocks × warps-per-block of active warps. If you're running 128-thread blocks (4 warps), that's only 8 warps = 25% occupancy on a 32-warp SM.

### CUTLASS SMEM Strategy

CUTLASS uses SMEM primarily for:

1. **Operand staging buffers (A and B tiles):** Typically 24-48 KB total
2. **Software pipelining (kStages):** Multiple buffer stages multiply SMEM usage
3. **Epilogue staging:** Usually small (< 4 KB)

The key parameter is `kStages` (pipeline depth):

```
SMEM usage ≈ (TileM × TileK + TileK × TileN) × sizeof(element) × kStages
```

For an FP16 GEMM with 128×128×32 tiles and 3 stages:

```
SMEM = (128×32 + 32×128) × 2 bytes × 3 = (4096 + 4096) × 6 = 49,152 bytes ≈ 48 KB
```

**Cutting stages from 3 to 2 reduces SMEM to ~32 KB**, potentially allowing 3 blocks per SM instead of 2 — a 50% occupancy boost, but with reduced pipelining (more stall risk on memory latency).

### SMEM Configuration on Blackwell

Blackwell added **dynamic SMEM partitioning** — the driver can reconfigure the L1/SMEM split at kernel launch:

```cpp
// CUDA API call
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
```

CUTLASS 3.x uses this to max out SMEM when needed (up to ~98 KB on Blackwell).

---

## 4. The Myth: Higher Occupancy = Better Performance

This is one of the most persistent misconceptions in CUDA programming. **Higher occupancy does not always mean better performance.** In fact, beyond a threshold, increasing occupancy can **hurt** performance.

### Why the Myth Exists

The reasoning goes: "More warps = more latency hiding = better throughput." This is true **up to the point where latency is fully hidden.** Beyond that:

1. **Register file contention:** More warps means fewer registers per warp. Each warp gets a smaller working set, potentially causing spilling.
2. **Reduced ILP:** With fewer registers, the compiler cannot keep as many values live, reducing instruction-level parallelism.
3. **Cache pressure:** More concurrent warps mean more cache lines in flight, potentially increasing eviction rates.
4. **No additional latency hiding:** If 50% occupancy already hides all memory latency, going to 100% adds nothing.

### NVIDIA's Own Guidance

NVIDIA's own documentation states:

> "100% occupancy does not guarantee maximum performance. The optimal occupancy is application-dependent."

From the CUDA C++ Best Practices Guide:

> "Some algorithms benefit from high occupancy... others see no benefit from occupancy above a certain threshold... For some compute-bound kernels, lower occupancy with higher register usage can outperform higher occupancy."

### Empirical Evidence

In CUTLASS benchmarks, the best-performing GEMM kernels frequently operate at **50-75% occupancy**, not 100%. The reason:

- 50% occupancy provides sufficient warp-level parallelism to hide the ~20 cycle pipeline latency
- The extra registers (64-96 per thread) enable larger accumulator fragments, reducing the number of iterations
- Larger tiles per thread reduce the ratio of address computation to computation

### The Real Metric: Warps Eligible to Issue

What matters isn't occupancy — it's **how often there's at least one warp ready to issue.** This is the "warp scheduler efficiency" metric. A kernel at 25% occupancy with all warps always ready beats a kernel at 100% occupancy where half the warps are stalled on memory.

---

## 5. CUTLASS Occupancy Sweet Spots

CUTLASS is designed to explore the occupancy-performance tradeoff systematically. The library exposes occupancy as a tunable through template parameters.

### Key CUTLASS Parameters Affecting Occupancy

| Parameter | Effect on Occupancy | Typical Values |
|-----------|-------------------|----------------|
| `ThreadblockShape` (M×N×K) | Larger tiles → more SMEM → lower occupancy | 64×64×32 to 256×128×64 |
| `kStages` (pipeline depth) | More stages → more SMEM → lower occupancy | 2-5 |
| `WarpShape` | Affects register usage | Typically 1/4 of ThreadblockShape |
| `InstructionShape` | Hardware MMA shape (fixed) | 16×8×16 (Ampere), 16×8×32 (Hopper+) |
| `kSplits` or cluster size | Multi-CTA strategies | 1-8 |

### CUTLASS 3.x Default Sweet Spots

For common data types on Ampere/Blackwell:

**FP16/BF16 GEMM:**
- Threadblock: 128×128×64
- Stages: 3-4
- SMEM: ~48-64 KB
- Achieved occupancy: 50-75%
- Register usage: ~64-80 per thread

**INT8 GEMM:**
- Threadblock: 128×256×64
- Stages: 3
- SMEM: ~48 KB
- Achieved occupancy: 50-67%

**FP8 (E4M3/E5M2) on Hopper/Blackwell:**
- Threadblock: 128×128×64
- Stages: 4-5 (more aggressive pipelining for HBM3)
- SMEM: ~80-96 KB
- Achieved occupancy: 25-50%
- These kernels are **deliberately** low-occupancy to maximize tile size

### CUTLASS Profiling-Driven Selection

CUTLASS 3.x includes a profiler that sweeps over configurations:

```bash
# CUTLASS profiler
./cutlass_profiler --operation=Gemm --m=4096 --n=4096 --k=4096 \
  --providers=cutlass --kernels=all
```

This empirically finds the best-performing configuration, which is often NOT the highest-occupancy one.

---

## 6. Tensor Core Utilization vs Occupancy

Tensor Cores have their own occupancy dynamics that are **orthogonal** to traditional CUDA core occupancy.

### Tensor Core Pipeline

A warp-level matrix multiply-accumulate (WMMA/WGMMA/MMA) instruction:

1. **Dispatches** a matrix operation to the Tensor Core
2. The Tensor Core **pipelines** the operation across multiple cycles
3. Results are **collected** in the accumulator register fragment

The warp can **NOT** issue another MMA instruction to the same Tensor Core until the previous one completes (data dependency). This creates a pipeline bubble if there's nothing else to do.

### Occupancy's Role in Tensor Core Utilization

Tensor Core utilization depends on keeping the Tensor Core pipeline fed:

```
Tensor Core Utilization = Useful MMA cycles / Total cycles
```

Higher occupancy helps Tensor Core utilization by:

1. **Providing alternative warps** to issue MMA instructions while others wait
2. **Interleaving MMA with memory operations** (different warps doing different stages)

But there are diminishing returns:

- On Ampere, a Tensor Core can complete an MMA in ~16 cycles. If you have 4+ warps ready, the pipeline stays full.
- Going from 8 warps to 16 warps adds no Tensor Core utilization benefit if the pipeline was already saturated.

### WGMMA (Hopper/Blackwell)

The Warp Group MMA (WGMMA) instruction introduced in Hopper operates across **4 warps simultaneously**:

```
WGMMA requires: 4 warps cooperating on one MMA
Minimum warps for 1 WGMMA in flight: 4 warps (128 threads)
Minimum for pipelined: 8 warps (overlap memory with compute)
```

This means **even 25% occupancy (12 warps on a 48-warp SM) can achieve near-peak Tensor Core throughput** if the warps are well-scheduled.

### CUTLASS WGMMA Configuration

CUTLASS 3.x for Hopper+ uses:

```cpp
// Warp Group level MMA
using MmaOp = cutlass::arch::OpClassWarpGroupSm90;

// Cluster shape: 2×1×1 means 2 CTAs cooperate
// This affects inter-CTA scheduling, not just intra-SM occupancy
```

---

## 7. When Low Occupancy with High Registers Wins

There are specific scenarios where deliberately accepting 25-50% occupancy in exchange for more registers produces superior performance.

### Case 1: Compute-Bound Kernels with Large Accumulators

GEMM kernels with large tile sizes accumulate many partial results:

```
128×128 tile = 16,384 output elements
In FP32 accumulator: 16,384 × 4 bytes = 64 KB just for accumulators
Per thread (256 threads): 64 elements × 4 bytes = 256 bytes = 64 registers
```

If you reduce tile size to increase occupancy, you:
- Increase the number of tiles to process
- Increase address computation overhead
- Decrease the reuse of loaded operands
- **Net effect: slower despite higher occupancy**

### Case 2: Kernels with High ILP

Some kernels have abundant instruction-level parallelism that the compiler can exploit only with sufficient registers:

```cpp
// Example: 4 independent MACs that can be dual-issued
float acc0, acc1, acc2, acc3;  // 4 registers
acc0 += a0 * b0;
acc1 += a1 * b1;  // independent, can dual-issue
acc2 += a2 * b2;
acc3 += a3 * b3;
```

With 32 registers, the compiler can keep all accumulators live and schedule dual-issue. With 16 registers (forced by high occupancy), the compiler spills and serializes.

### Case 3: Memory-Bound Kernels with Large Working Sets

Kernels that iterate over large data structures (e.g., attention mechanisms, convolutions with large receptive fields) benefit from:

- Larger per-thread buffers in registers (avoids SMEM/L1 round-trips)
- Fewer warps competing for cache lines
- **Lower occupancy actually improves cache hit rates**

### Case 4: SAM 3.1 Vision Encoder Kernels

The SAM 3.1 vision transformer contains kernels that benefit from this tradeoff:

- **LayerNorm:** Low compute intensity, benefits from high occupancy (memory-bound)
- **Attention QKV projection:** Moderate compute, 50% occupancy is fine
- **Window attention:** High ILP, benefits from 25-50% occupancy with more registers for Q/K/V fragments
- **MLP (FFN):** Compute-bound GEMM, CUTLASS sweet spot applies

### Decision Framework

```
Is kernel compute-bound?
  → Yes: Accept 25-50% occupancy, maximize registers for accumulator
  → No (memory-bound):

Does kernel have high ILP (>4 independent ops)?
  → Yes: 50% occupancy, enough registers for ILP
  → No:

Is kernel latency-bound (many small ops)?
  → Yes: Maximize occupancy for latency hiding (80-100%)
  → No: 50-75% is likely optimal
```

---

## 8. RTX 5060 Optimal Occupancy Targets

The RTX 5060 is based on the Blackwell architecture (GB206 die). Based on Blackwell SM specifications:

### RTX 5060 Architecture Assumptions

| Spec | Estimated Value |
|------|----------------|
| SMs | 30-36 |
| Max threads per SM | 1536 |
| Max warps per SM | 48 |
| Register file per SM | 65,536 (256 KB) |
| SMEM per SM | 100 KB (configurable) |
| L1 cache | 128 KB (shared with SMEM) |
| Tensor Cores | 5th gen (Blackwell) |
| Max SMEM per block | ~98 KB |
| Max blocks per SM | 32 |

### Recommended Occupancy by Kernel Type

| Kernel Type | Target Occupancy | Reasoning |
|------------|-----------------|-----------|
| **FP16/BF16 GEMM** | 50-67% | CUTLASS default sweet spot, sufficient for WGMMA pipeline |
| **FP8 GEMM** | 25-50% | Larger tiles, aggressive SMEM staging |
| **Attention (Flash Attention style)** | 50% | Balances SMEM for K/V tiles with register pressure |
| **Conv2D forward** | 50-75% | Moderate compute intensity |
| **Element-wise** | 75-100% | Memory-bound, maximize latency hiding |
| **Reduction** | 75-100% | Memory-bound, warp shuffle + high occupancy |
| **LayerNorm/RMSNorm** | 75-100% | Memory-bound |
| **SAM 3.1 Image Encoder GEMM** | 50% | CUTLASS default, Tensor Core saturation |
| **SAM 3.1 Prompt Encoder** | 50-75% | Smaller shapes, memory-bound |

### Why 25-50% for FP8/Tensor Core Kernels?

On Blackwell, the 5th-gen Tensor Cores are extremely throughput-capable. The bottleneck shifts from compute to memory:

1. FP8 MMA produces results faster than HBM can supply operands
2. The solution: larger tiles (more SMEM reuse), more pipeline stages
3. This uses more SMEM per block → fewer blocks per SM → lower occupancy
4. But the Tensor Core stays saturated because the pipeline is deeper

**Counter-intuitive result:** An FP8 GEMM at 25% occupancy can achieve higher TOPS than an FP16 GEMM at 75% occupancy, because the Tensor Core utilization is higher.

---

## 9. Measuring with Nsight Compute

Nsight Compute provides direct occupancy measurement and analysis.

### Key Metrics

```
# Occupancy metrics
sm__warps_active.avg.per_cycle_active     # Average active warps
sm__warps_active.avg.pct_of_peak_sustained_active  # Achieved occupancy %

# Occupancy limiters  
launch__occupancy_limit_registers         # Occupancy limited by registers?
launch__occupancy_limit_shared_mem        # Occupancy limited by SMEM?
launch__occupancy_limit_warps             # Occupancy limited by thread slots?
launch__occupancy_limit_blocks            # Occupancy limited by block count?

# Theoretical vs achieved
sm__maximum_warps_avg_per_active_cycle    # Theoretical max warps
sm__warps_active.avg.per_cycle_active     # Actual active warps
```

### Command-Line Measurement

```bash
# Basic occupancy analysis
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__occupancy_limit_registers,\
  launch__occupancy_limit_shared_mem,\
  launch__occupancy_limit_warps \
  ./my_kernel_benchmark

# Detailed occupancy breakdown
ncu --section Occupancy ./my_kernel_benchmark

# Full roofline + occupancy
ncu --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section Occupancy \
    ./my_kernel_benchmark
```

### Interpreting Results

**Scenario 1: Register-limited**
```
launch__occupancy_limit_registers: 1
sm__warps_active.avg.pct_of_peak_sustained_active: 42%
```
→ Kernel uses too many registers. Consider `-maxrregcount`, loop unrolling reduction, or algorithmic change.

**Scenario 2: SMEM-limited**
```
launch__occupancy_limit_shared_mem: 1
sm__warps_active.avg.pct_of_peak_sustained_active: 33%
```
→ Kernel uses too much SMEM. Consider reducing tile size or pipeline stages.

**Scenario 3: Good occupancy but poor performance**
```
sm__warps_active.avg.pct_of_peak_sustained_active: 75%
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active: 20%
```
→ High occupancy but low Tensor Core utilization. Problem is not occupancy — it's instruction mix, data layout, or synchronization.

### Nsight Compute GUI Workflow

1. Profile kernel → Occupancy section
2. Check "Theoretical Occupancy" vs "Achieved Occupancy" — gap indicates launch configuration issues
3. Check occupancy limiter breakdown (register / SMEM / warp / block)
4. Correlate with "Speed of Light" section — is the kernel compute or memory bound?
5. If compute-bound and register-limited → occupancy might be fine as-is
6. If memory-bound and occupancy < 50% → consider reducing register/SMEM usage

### Python Script for Occupancy Estimation (Pre-Profile)

```python
def estimate_occupancy(sm_arch, threads_per_block, regs_per_thread, smem_per_block):
    """Estimate occupancy before profiling."""
    sm_specs = {
        'sm_80': {'max_warps': 64, 'regs': 65536, 'smem': 164*1024, 'max_threads': 2048, 'max_blocks': 32},
        'sm_86': {'max_warps': 48, 'regs': 65536, 'smem': 100*1024, 'max_threads': 1536, 'max_blocks': 16},
        'sm_89': {'max_warps': 48, 'regs': 65536, 'smem': 100*1024, 'max_threads': 1536, 'max_blocks': 16},
        'sm_90': {'max_warps': 64, 'regs': 65536, 'smem': 228*1024, 'max_threads': 2048, 'max_blocks': 32},
        'sm_100':{'max_warps': 48, 'regs': 65536, 'smem': 100*1024, 'max_threads': 1536, 'max_blocks': 32},  # Blackwell
    }
    spec = sm_specs[sm_arch]
    warps_per_block = threads_per_block // 32
    
    blocks_by_reg = spec['regs'] // (regs_per_thread * threads_per_block)
    blocks_by_smem = spec['smem'] // smem_per_block if smem_per_block > 0 else spec['max_blocks']
    blocks_by_threads = spec['max_threads'] // threads_per_block
    blocks = min(blocks_by_reg, blocks_by_smem, blocks_by_threads, spec['max_blocks'])
    
    active_warps = blocks * warps_per_block
    occupancy = active_warps / spec['max_warps']
    
    return {
        'occupancy': occupancy,
        'active_warps': active_warps,
        'blocks': blocks,
        'limiter': 'registers' if blocks == blocks_by_reg else 
                   'smem' if blocks == blocks_by_smem else 
                   'threads' if blocks == blocks_by_threads else 'block_limit'
    }

# Example for RTX 5060 (sm_100)
result = estimate_occupancy('sm_100', threads_per_block=256, regs_per_thread=64, smem_per_block=48*1024)
print(f"Occupancy: {result['occupancy']:.0%}, Limiter: {result['limiter']}")
# Output: Occupancy: 50%, Limiter: smem
```

---

## 10. SAM 3.1 Kernel Occupancy Targets

SAM 3.1 (Segment Anything Model 3.1) combines a vision encoder, prompt encoder, and mask decoder. Each stage has different occupancy requirements.

### SAM 3.1 Architecture Kernels

#### Vision Encoder (ViT-based)

The vision encoder dominates compute (~90% of total inference time). It consists of:

| Layer | Kernel Type | Target Occupancy | Notes |
|-------|------------|-----------------|-------|
| Patch Embedding | Conv2D/Linear | 50-75% | Moderate compute intensity |
| QKV Projection | GEMM | 50% | CUTLASS FP16/BF16 default |
| Attention Score | Batched GEMM + Softmax | 50-75% | Softmax is memory-bound |
| Attention Output | GEMM | 50% | CUTLASS default |
| MLP Up/Down | GEMM | 50% | CUTLASS default |
| LayerNorm | Custom | 75-100% | Memory-bound reduction |
| SiLU/GELU | Element-wise | 100% | Pure memory-bound |

**Vision Encoder GEMM Strategy:**
- Use CUTLASS 3.x with `128×128×64` threadblock shape
- 3 pipeline stages
- Target 50% achieved occupancy
- This saturates 5th-gen Tensor Cores on Blackwell
- Register usage: ~64-80 per thread

#### Prompt Encoder

The prompt encoder handles point/box/text prompts — much smaller compute:

| Layer | Target Occupancy | Reasoning |
|-------|-----------------|-----------|
| Point/Box Embedding | 75-100% | Small GEMMs, memory-bound |
| Text Embedding (if used) | 50-75% | Moderate sequence lengths |
| Cross-attention to image | 50% | Standard attention pattern |

**Prompt Encoder Strategy:** Use CUTLASS with smaller tiles (64×64×32) to accommodate the smaller problem sizes. Occupancy can be higher (67-75%) because tiles are smaller → less SMEM per block.

#### Mask Decoder

The mask decoder uses cross-attention between prompt tokens and image embeddings:

| Layer | Target Occupancy | Notes |
|-------|-----------------|-------|
| Cross-attention QKV | 50% | Standard GEMM |
| Cross-attention Score | 50-75% | Attention kernels |
| MLP layers | 50% | CUTLASS default |
| Mask prediction head | 75-100% | Small, memory-bound |

### SAM 3.1 CUTLASS Configuration

Recommended CUTLASS template parameters for SAM 3.1 GEMM kernels:

```cpp
// Vision encoder main GEMMs (FP16 on RTX 5060)
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::RowMajor,                 // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OpClass
    cutlass::arch::Sm100,                      // Arch (Blackwell)
    cutlass::gemm::GemmShape<128, 128, 64>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,     // WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>,      // InstructionShape (Blackwell MMA)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 1, float, float>,     // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    3                                          // kStages
>;
// Expected: ~50% occupancy, ~64-80 registers/thread
```

```cpp
// Smaller prompt encoder GEMMs
using GemmSmall = cutlass::gemm::device::Gemm<
    // ... same types ...
    cutlass::gemm::GemmShape<64, 64, 32>,     // Smaller ThreadblockShape
    cutlass::gemm::GemmShape<32, 32, 32>,     // Smaller WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>,      // InstructionShape
    // ... epilogue ...
    3                                          // kStages
>;
// Expected: ~67-75% occupancy
```

### Occupancy Optimization Workflow for SAM 3.1

1. **Profile first:** Run `ncu --section Occupancy` on each kernel type
2. **Identify limiter:** Is it registers, SMEM, or thread slots?
3. **Adjust tiles:** If SMEM-limited, reduce tile size or stages. If register-limited, consider reducing accumulator fragments.
4. **Validate Tensor Core utilization:** Ensure `sm__pipe_tensor_cycles_active` is > 60% regardless of occupancy
5. **Benchmark end-to-end:** Individual kernel occupancy doesn't matter if the overall pipeline has bubbles

### Expected SAM 3.1 Performance Profile

| Component | % of Time | Occupancy Target | Optimizer Focus |
|-----------|-----------|-----------------|----------------|
| Vision Encoder GEMMs | 70% | 50% | CUTLASS default, Tensor Core saturation |
| Attention (QK^T, softmax) | 15% | 50-75% | Fused attention kernel |
| LayerNorm/SiLU | 8% | 75-100% | Fused element-wise + reduction |
| Mask Decoder | 5% | 50-75% | Standard CUTLASS |
| Prompt Encoder | 2% | 67-75% | Smaller tiles |

**Target: >50% average occupancy across all kernels, >60% Tensor Core utilization on GEMM kernels.**

---

## Summary: Key Takeaways

1. **Occupancy is a tool, not a goal.** Measure what matters: throughput, latency, Tensor Core utilization.

2. **Register pressure is the silent killer.** A kernel with 80 registers/thread at 50% occupancy often beats a kernel with 40 registers/thread at 100%.

3. **SMEM vs occupancy is a tuning knob** for pipeline depth. CUTLASS `kStages` directly controls this tradeoff.

4. **The "higher occupancy = better" myth** is debunked by CUTLASS benchmarks showing 50-75% as the GEMM sweet spot.

5. **Tensor Core utilization is orthogonal** to CUDA core occupancy. 25% occupancy can achieve >80% Tensor Core utilization with proper pipelining.

6. **For RTX 5060 (Blackwell),** target 50% occupancy for GEMM, 75-100% for memory-bound kernels. FP8 kernels can go as low as 25%.

7. **Nsight Compute is essential.** Profile, don't theorize. The occupancy section immediately shows your limiter.

8. **SAM 3.1's vision encoder** should target CUTLASS defaults (50% occupancy, 3-stage pipeline) for maximum Tensor Core throughput on Blackwell.

---

*References: NVIDIA CUDA C++ Best Practices Guide, NVIDIA Occupancy Calculator, CUTLASS 3.x documentation and source code, NVIDIA Nsight Compute user guide, "Dissecting the NVIDIA GPU Architectures" (various microarchitecture analyses).*
