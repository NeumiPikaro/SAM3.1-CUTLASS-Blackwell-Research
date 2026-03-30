# ThunderKittens: Tile Primitives for Speedy Kernels

**Source:** Hazy Research Lab, Stanford University
**Repository:** https://github.com/HazyResearch/ThunderKittens
**Key References:**
- "GPUs Go Brrr" blog post (HazyResearch, May 2024)
- ThunderKittens 2.0 release (Jan 2026) — Blackwell support
- Together AI adoption blog post
- Cursor engineering blog on custom kernels
- Source code: kernels/attention/ directory (mha_h100, mha_h100_lcf, bf16_b300_mha_causal, bf16_b300_mha_noncausal)

## 1. Overview and Philosophy

ThunderKittens is an embedded domain-specific language (DSL) for writing high-performance deep learning kernels in CUDA, developed by Stanford's Hazy Research Lab — the same group behind FlashAttention, FlashAttention-2, and FlashFFTConv. It emerged from the lab's realization that despite the elegance of algorithms like FlashAttention, the gap from "beautiful pictures to actual GPU go brr CUDA is still too damn high."

The framework is built around three principles:
1. **Simplicity** — Kernels should be stupidly simple to write
2. **Extensibility** — Embedded in raw CUDA; if TK can't do it, you can extend it yourself
3. **Speed** — Kernels should be at least as fast as hand-written CUDA; their FlashAttention-3 implementation is the proof

ThunderKittens is used in production by Together AI, Jump Trading, and Cursor for large-scale training and inference. It has an AMD port called HipKittens.

## 2. Hardware-Software Co-Design Philosophy

ThunderKittens is explicitly built "from the hardware up" — it does "what the silicon tells us." The core insight is that modern NVIDIA GPUs are **not** 1000×1000 matrix multiply machines; they are manycore processors where each core can efficiently run ~16×16 matrix multiplies. The tensor cores constitute 94% of the H100's compute capability (989 TFLOPs of half-precision matmul vs ~60 TFLOPs of everything else). Therefore:

> % utilization of H100 = % tensor cores active cycles ± 6%

This drives every design decision. The framework identifies four critical hardware requirements for high utilization:

### WGMMA Instructions (Hopper)
The H100's `wgmma.mma_async` instructions allow 128 threads across all 4 SM quadrants to collaboratively launch asynchronous matrix multiplies directly from shared memory. Without WGMMA, the GPU tops out at ~63% peak utilization — losing 37% of potential performance. However, WGMMA memory layouts are notoriously complicated: unswizzled layouts suffer poor coalescing requiring L2 bandwidth, and swizzled layouts were found to be "flat-out incorrectly documented" by NVIDIA. The HazyResearch team spent weeks reverse-engineering the actual swizzled WGMMA layouts, finding NVIDIA's documentation "extraordinarily misleading."

### TCGEN05 Instructions (Blackwell)
On B200/B300 GPUs, ThunderKittens 2.0 uses TCGEN05 tensor core instructions instead of WGMMA. The B300 non-causal attention kernel demonstrates this with `mm2_ABt`, `mm2_AB`, and `mma2_AB` calls — the Blackwell-generation matrix multiply operations — along with `detail::tcgen05::commit` for barrier management.

### TMA (Tensor Memory Accelerator)
The Hopper-introduced TMA is considered "completely indispensable" — possibly more important than WGMMA. TMA handles asynchronous address generation, multi-dimensional tensor fetching, and barrier tripping. It saves register resources and instruction dispatches, and enables asynchronous global memory reductions crucial for backward kernels.

### Shared Memory Bank Conflict Avoidance
Shared memory has ~30 cycle latency — enough time for tensor cores to complete nearly two 32×32 matrix multiplies. ThunderKittens automatically handles swizzling patterns to avoid 32-bank conflicts, something that historically consumed weeks of developer time in hand-written kernels.

## 3. Tile-Based Abstraction Layer

ThunderKittens provides exactly four templated data types:

| Type | Scope | Description |
|------|-------|-------------|
| `rt` (Register Tile) | Warp | 2D tensor on register file, e.g., `rt_bf<32,16>` |
| `rv` (Register Vector) | Warp | 1D tensor on register file, with naive/aligned/orthogonal layouts |
| `st` (Shared Tile) | Block | 2D tensor in shared memory, e.g., `st_bf<64,64>` |
| `sv` (Shared Vector) | Block | 1D tensor in shared memory |

Tiles are parameterized by height, width, and layout. The fundamental tile size matches what fits into tensor cores (~16×16). Operations follow a RISC-like assembly pattern: destination is the first operand, sources follow.

The API mirrors PyTorch's tensor operations for familiarity:
```cuda
kittens::mul(c, a, b);           // element-wise multiply
kittens::store(s, c);            // register tile → shared tile
warpgroup::mma_AB(accum, a, b);  // tensor core matrix multiply
warp::exp2(x);                   // element-wise exp2
warp::sum<axis::COL>(x);         // column-wise reduction
```

Compared to CUTLASS, ThunderKittens is:
- **More specialized** — focused purely on DL kernel patterns, not general GEMM
- **Simpler** — 4 types vs CUTLASS's extensive template hierarchy
- **More transparent** — you can "compile TK in your head" if you know CUDA
- **Gracefully degrading** — when TK doesn't provide something, you drop to raw CUDA seamlessly (it's an embedded DSL, not an abstraction barrier)

## 4. Fused Attention Kernels

### FlashAttention in ~100 Lines

ThunderKittens' flagship demonstration is a FlashAttention-2 forward pass for H100 in ~100 lines of CUDA that **outperforms FlashAttention-2 by about 30%**. The load-compute-finish (LCF) pipeline template is used:

```cuda
template<int D> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12;
    static constexpr int NUM_WORKERS = NUM_CONSUMER_WARPS / 4;  // 3 warpgroups
    static constexpr int INPUT_PIPE_STAGES = 2;
    // ...
};
```

The kernel uses persistent grid (132 blocks = number of SMs) and the LCF template to overlap I/O with computation.

### Online Softmax Implementation

The online softmax is the algorithmic core of FlashAttention, implemented directly in TK register operations. From the `mha_h100_lcf` kernel source:

```cuda
// A = Q @ K^T
warpgroup::mm<transpose::N, transpose::T>(att_block, q_smem, k_smem);
max_vec_last_scaled = max_vec * TEMPERATURE_SCALE;
warpgroup::mma_async_wait();

// Online softmax — maintains running max and normalization
warp::right_fill(att_block, att_block, remaining_rows, neg_infty);
max_vec = warp::max<axis::COL>(att_block, max_vec);  // running max
max_vec_scaled = max_vec * TEMPERATURE_SCALE;
att_block = warp::exp2((att_block * TEMPERATURE_SCALE) - max_vec_scaled);
max_vec_last_scaled = warp::exp2(max_vec_last_scaled - max_vec_scaled);
norm_vec *= max_vec_last_scaled;  // rescale previous normalization
norm_vec = warp::sum<axis::COL>(att_block, norm_vec);  // running sum
o_reg *= max_vec_last_scaled;  // rescale previous output

// O += A @ V
att_block_mma = att_block;  // convert to bf16 for MMA
warpgroup::mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_smem);
```

Key implementation details:
- Uses `exp2` with `TEMPERATURE_SCALE = 1/sqrt(d) * log2(e)` for fused temperature scaling
- The `max_vec` and `norm_vec` are column vectors (`col_vec<rt_fl<16, kv_tile::rows>>`) that accumulate across KV tile iterations
- On each new KV block, the old output is rescaled by `exp2(old_max - new_max)` to account for the updated running statistics
- `right_fill` handles non-multiple-of-tile-size sequence lengths by padding with `-inf`
- The attention block is cast from fp32 to bf16 before the PV matmul

### Non-Causal Attention (Blackwell B300)

The B300 non-causal attention kernel (`bf16_b300_mha_noncausal`) is dramatically more sophisticated. It uses:

- **Cluster-level parallelism**: `CLUSTER_SIZE = 2` with distributed shared memory
- **Separate pipeline stages as dedicated warpgroups**:
  - **Producer** (1 warpgroup): TMA loads for Q/K/V with 3-stage pipelining
  - **MMA warpgroup** (1 warpgroup, single warp): TCGEN05 matrix multiplies for QK^T and PV
  - **Softmaxxers** (2 warpgroups): Dedicated online softmax computation
  - **Corrector** (1 warpgroup): Handles output rescaling when running statistics change
- **Tensor Memory (TMEM)** allocation for inter-warpgroup communication of score and output tiles
- **Semaphore-based synchronization** between pipeline stages

The non-causal kernel processes Q tiles of 128×192 and KV tiles of 64×192 (K) and 128×64 (V), with the V tile split into quarters for fine-grained pipeline overlap.

## 5. Performance Numbers

### Reported Benchmarks

| Configuration | Hardware | TFLOPS | % of Peak | Notes |
|---------------|----------|--------|-----------|-------|
| Matmul kernel | H100 SXM | **855** | **86%** | <100 lines, bf16 |
| FlashAttention-2 forward | H100 SXM | ~730* | ~74% | ~100 lines, **30% faster than FA2 PyTorch** |
| FA2 on 4090/A100 | 4090/A100 | matches FA2 | — | Few lines of code |
| Based linear attention | H100 | 215 (300+ w/ recompute) | — | Historically hard to make efficient |
| FA3 implementation | H100 | competitive with hand-written | — | Core proof of TK's speed claim |

*Estimated from "30% faster than FA2" claim, where FA2 PyTorch typically achieves ~550-600 TFLOPS on H100.

### Comparison with Baselines

**vs FlashAttention-2 (PyTorch)**: TK's FA2 implementation outperforms by ~30% on H100 across a wide range of configurations. On 4090 and A100, TK matches FA2 performance.

**vs Triton**: Some TK kernels (Based, hedgehog) could not achieve similar performance in Triton. TK's advantage comes from direct access to WGMMA/TCGEN05 instructions and precise control over shared memory swizzling — things Triton abstracts away.

**vs cuDNN**: Not directly compared in published materials, but TK's 86% peak utilization on matmul and 30% speedup over FA2 suggest competitive or superior performance for attention workloads.

**vs hand-written CUDA**: TK achieves equivalent or better performance with dramatically less code (~100 lines vs ~1200 for FA2), because TK "can do things the right way under the hood."

## 6. The Fused Attention-MLP Kernel Concept

While ThunderKittens does not ship a pre-built fused attention+MLP kernel, its architecture is explicitly designed to enable such fusion. The key enablers are:

### Pipeline Templates
The LCF (Load-Compute-Finish) template used in attention kernels is the same template used for matmul, rotary embeddings, Mamba, and FFTConv. This means the framework already supports chaining multiple compute stages within a single kernel launch.

### Register Persistence
After attention completes, the output tiles remain in registers. The `finish` step normalizes by dividing by `norm_vec`:
```cuda
o_reg /= norm_vec;
```
At this point, the register-resident output could feed directly into an MLP weight multiplication — the W_up and W_gate projections in a ViT block — without round-tripping through global memory.

### Warpgroup Specialization
The B300 kernel demonstrates that TK can assign different warpgroups to different pipeline stages. For a fused attention+MLP kernel:
- Warpgroups 0-2: Attention computation (QK^T → softmax → PV)
- Warpgroups 3-5: MLP computation (GELU(x·W_up) ⊙ x·W_gate)·W_down
- Producer warpgroup: Prefetches attention weights, then MLP weights

### Persistent Grid
TK kernels use persistent grids (typically 132 for H100, 148 for B300) matching the SM count. Each block processes multiple work items sequentially, which is ideal for fusing attention+MLP because:
1. The block stays resident, avoiding kernel launch overhead
2. L2 cache may retain MLP weights if attention blocks are processed sequentially
3. Register state can be reused between stages

**Relevance to SAM 3.1 ViT Blocks**: SAM 3.1's ViT blocks perform attention followed by an MLP in each transformer layer. Currently, these are separate kernel launches, each incurring:
- ~5-15μs kernel launch overhead
- Register file save/restore
- L2 cache pollution from intermediate results
- Global memory round-trip for the attention output

A ThunderKittens-based fused kernel could keep attention outputs in registers, immediately apply MLP weights (loaded via TMA during attention's final iterations), and write the final result directly. This eliminates at least one full global memory round-trip per transformer block.

## 7. Blackwell Support (ThunderKittens 2.0)

Released January 11, 2026, ThunderKittens 2.0 brings:

- **Full Blackwell GPU support** (B200/B300)
- **TCGEN05 instructions** replacing WGMMA for Blackwell tensor cores
- **MXFP8 and NVFP4 precision** support
- **Cluster-level operations** with distributed shared memory (`tma::cluster::load_async`, `everyone::tma::cluster::sync`)
- **Tensor Memory (TMEM)** allocation for inter-warpgroup communication
- **No more Ampere support** — the project has moved forward

The B300 attention kernels use cluster size 2 with dedicated warpgroups for producer/MMA/softmax/corrector stages, 3-stage KV pipelining, and fine-grained V-split computation (quarter-tile V matmuls for tighter pipeline overlap).

## 8. How ThunderKittens Could Eliminate Kernel Launch Overhead in SAM 3.1

### Current SAM 3.1 Architecture Problem
In a standard ViT block, each transformer layer executes:
1. LayerNorm → QKV projection → Attention → Output projection → Residual add
2. LayerNorm → MLP (up_proj → activation → down_proj) → Residual add

Each `→` potentially represents a separate kernel launch, totaling 8-12 kernel launches per transformer block. For SAM 3.1's 32-layer ViT-Huge, that's 256-384 kernel launches per forward pass.

### ThunderKittens Fusion Strategy

**Level 1 — Attention Fusion**: Fuse QKV projection + attention + output projection into a single TK kernel using the LCF template. During the attention loop's final iterations, prefetch output projection weights via TMA.

**Level 2 — Block Fusion**: Extend Level 1 to include the first LayerNorm and residual add. The attention output in registers feeds directly into LayerNorm (TK supports `warp::layernorm` operations), then into the MLP computation. The second LayerNorm can be fused similarly.

**Level 3 — Full Block Fusion**: A single kernel per transformer block that handles:
- LayerNorm₁ → Attention → Output Proj → Residual₁
- LayerNorm₂ → MLP → Output Proj → Residual₂

This reduces 8-12 kernel launches to 1, eliminating:
- **Kernel launch overhead**: ~5-15μs × 8 = 40-120μs per block
- **Global memory round-trips**: Attention output stays in registers/shared memory
- **Register file thrashing**: Persistent kernel avoids save/restore cycles

### Estimated Impact
For SAM 3.1 ViT-Huge processing a 1024×1024 image (256 tokens) on H100:
- Current: ~30-50ms total (attention + MLP + overhead)
- With TK fusion: ~20-35ms (15-30% reduction), primarily from:
  - Eliminated global memory round-trip: saves ~2-4ms
  - Eliminated kernel launch overhead: saves ~1-3ms
  - Better tensor core utilization: saves ~3-8ms

## 9. Implementation Considerations for SAM 3.1

### Advantages
1. **Non-causal attention support**: SAM 3.1 uses bidirectional attention (no causal mask), which TK explicitly supports (bf16_b300_mha_noncausal kernel)
2. **Variable sequence lengths**: TK handles non-multiple-of-tile sequences via `right_fill`
3. **Header-only library**: Just `#include "kittens.cuh"` — no build system changes needed
4. **PyTorch integration**: TK has `pyutils/torchutils.cuh` for PyTorch tensor interop
5. **Production-proven**: Used by Together AI for inference at scale

### Challenges
1. **Hopper/Blackwell only**: No Ampere support — requires H100/B200/B300
2. **CUDA 12.8+**: Aggressive version requirements
3. **C++20 required**: Uses concepts extensively
4. **Custom development**: Fused attention+MLP doesn't exist off-the-shelf; must be written
5. **ViT-specific considerations**: SAM 3.1 uses relative position embeddings and window attention in some layers, requiring kernel variants

## 10. Summary

ThunderKittens represents a paradigm shift in GPU kernel development: instead of fighting hardware complexity through abstraction (Triton) or accepting it through verbosity (raw CUDA/CUTLASS), TK provides a thin, hardware-aligned layer that makes the right thing easy. Its 86% peak utilization with ~100 lines of code, 30% speedup over FA2, and explicit Blackwell support make it the most promising framework for implementing fused attention-MLP kernels for SAM 3.1's ViT architecture. The key opportunity is eliminating the global memory round-trip between attention and MLP stages, which could yield 15-30% end-to-end speedup for ViT inference.
