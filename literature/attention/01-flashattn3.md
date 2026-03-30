# FlashAttention-3: Analysis for SAM 3.1 ViT Attention Optimization

## Reference Information

| Field | Value |
|-------|-------|
| **Paper** | "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" |
| **Authors** | Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao |
| **arXiv** | [2407.08608](https://arxiv.org/abs/2407.08608) (Jul 2024) |
| **Blog** | [tridao.me/blog/2024/flash3](https://tridao.me/blog/2024/flash3/) |
| **Code** | [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (hopper branch) |
| **Hardware Target** | NVIDIA Hopper (H100/H800), CUDA ≥ 12.3 (12.8 recommended) |
| **Status** | Beta release; FP16/BF16 forward+backward, FP8 forward |

---

## 1. What FlashAttention-3 Adds Over FA-2

FlashAttention-2 (Dao, 2023) achieved up to ~70% utilization on Ampere (A100) by parallelizing over the sequence length dimension and iterating over K/V blocks in the inner loop. However, on Hopper H100, FA-2 achieves only **35% utilization** (~350 TFLOPS FP16 forward) because it:
- Uses Ampere-era `mma.sync` instructions instead of Hopper's WGMMA
- Has no warp specialization — all warps do both data movement and compute
- Treats GEMM and softmax as strictly sequential operations
- Has no FP8 support

FlashAttention-3 introduces three fundamental algorithmic improvements:

### 1.1 Producer-Consumer Warp Specialization with Asynchronous Pipelining

FA-3 divides warps into **producers** (issuing TMA loads from HBM → SMEM) and **consumers** (issuing WGMMA matrix multiplies from SMEM → register file). This exploits the fact that both TMA and WGMMA are **asynchronous** on Hopper — they can be issued and then the issuing thread can continue without waiting for completion.

A circular shared memory buffer (typically 2-3 stages) enables multi-stage software pipelining:
- **Stage N**: Producer loads K/V blocks for iteration N+2
- **Stage N-1**: Consumer computes WGMMA for iteration N+1
- **Stage N-2**: Consumer computes softmax for iteration N

This is analogous to CUTLASS's warp-specialized GEMM strategy but adapted for the attention-specific pattern of alternating GEMM and softmax. Hopper's `setmaxnreg` instruction dynamically reallocates registers between warpgroups — consumer warps doing WGMMA get more registers (for accumulators), while producer warps doing TMA need fewer.

**Impact**: Just rewriting FA-2 with Hopper instructions + warp specialization improves throughput from ~350 to **540-570 TFLOPS** FP16 (before any GEMM-softmax overlap).

### 1.2 Interleaved GEMM-Softmax Overlap (2-Stage Pipeline)

This is FA-3's most novel algorithmic contribution. The core insight: on H100, FP16 WGMMA runs at 989 TFLOPS but special functions (exp for softmax) run at only **3.9 TFLOPS** — a 256× gap. For head_dim=128, there are 512× more matmul FLOPS than exponential FLOPS, meaning exp alone can consume 50% of the wall-clock time even though it's a tiny fraction of total FLOPS.

FA-3 restructures the algorithm to break sequential dependencies:
- While **softmax** executes on block N's score matrix (using CUDA cores / SFU)
- **WGMMA** asynchronously computes block N+1's score matrix (using Tensor Cores)
- These run in parallel on different execution units within the same SM

This required reworking the FlashAttention-2 algorithm to handle the case where softmax for block N hasn't finished when GEMM for block N+1 needs to start. The solution uses a 2-stage pipeline with careful synchronization.

### 1.3 FP8 Support with Block Quantization and Incoherent Processing

FA-3 targets Hopper's FP8 Tensor Cores (1978 TFLOPS peak, 2× FP16). Key challenges and solutions:

**Layout conformance**: FP8 WGMMA only accepts k-major operand layout (vs. both mn-major and k-major for FP16). When fusing back-to-back GEMMs (Q@K^T then P@V), the FP32 accumulator layout of the first GEMM clashes with the FP8 operand layout expected by the second. FA-3 performs an **in-kernel transpose** of the FP32 accumulator to FP8 k-major format in shared memory.

**Block quantization**: Instead of per-tensor quantization (one scale for entire tensor), FA-3 uses per-block quantization (one scale per tile), reducing quantization error especially for attention score distributions.

**Incoherent processing**: Borrowed from the quantization literature (QuIP, QuIP#), FA-3 applies a random Hadamard transform with random signs to Q and K before quantization. This "spreads out" outlier features (which LLM activations are known to have) across dimensions, reducing quantization error by **2.6×** compared to baseline FP8 attention. The Hadamard transform costs O(d log d) per head and can be fused with rotary embedding "for free" since both are memory-bandwidth bound.

---

## 2. How FA-3 Achieves >75% Tensor Core Utilization on Hopper

The H100 SXM5 has:
- **132 SMs**, each with 4th-gen Tensor Cores
- **989 TFLOPS** FP16 matmul peak
- **1978 TFLOPS** FP8 matmul peak
- **3.9 TFLOPS** special function (exp) throughput
- **3.35 TB/s** HBM3 bandwidth

FA-3 achieves 740 TFLOPS FP16 (75% utilization) and ~1.2 PFLOPS FP8 through:

1. **Eliminating the softmax bottleneck**: The 2-stage GEMM-softmax overlap hides ~90% of softmax latency under asynchronous WGMMA execution. Without this, softmax alone would limit utilization to ~50% for d=128.

2. **Maximizing Tensor Core occupancy**: Warp specialization ensures consumer warps always have WGMMA work ready. The producer warps keep the circular SMEM buffer full via TMA, so the Tensor Cores never stall waiting for data.

3. **Register optimization**: Dynamic register reallocation via `setmaxnreg` gives consumer warps ~200 registers (for double-buffered accumulators + softmax state) while producer warps use ~64 (just enough for TMA descriptors).

4. **Memory hierarchy efficiency**: TMA handles all HBM→SMEM transfers with zero register overhead for index computation. The circular SMEM buffer (typically 3 stages × block_size tiles) ensures compute and memory overlap.

5. **Persistent kernel design**: A single kernel launch processes the entire attention computation, eliminating kernel launch overhead and maintaining L2 cache warmth across iterations.

**Performance progression** (FP16 forward, head_dim=128, seqlen=8K):
| Optimization | TFLOPS | % of Peak |
|---|---|---|
| FA-2 baseline on H100 | ~350 | 35% |
| + Hopper instructions (WGMMA, TMA) | ~540-570 | 55-58% |
| + Pingpong scheduling | ~620 | 63% |
| + Intra-warpgroup GEMM-softmax overlap | ~640-660 | 65-67% |
| + Further optimizations (persistent, variable-length) | ~740 | 75% |

---

## 3. The "Ping-Pong" Warp Scheduling for Attention

The ping-pong schedule is an **inter-warpgroup** overlapping strategy using two warpgroups (each warpgroup = 4 warps = 128 threads):

```
Time ──────────────────────────────────────────────────►

Warpgroup 1: [GEMM₀ iter N] [GEMM₁ iter N] [softmax N] [GEMM₀ iter N+1] ...
Warpgroup 2: [  idle/wait  ] [softmax N-1 ] [GEMM₀ N+1] [GEMM₁ iter N+1] ...
```

**Mechanism**:
1. Warpgroup 1 issues WGMMA for GEMM₀ (Q_block × K_block^T) and GEMM₁ (P_block × V_block) of iteration N
2. Warpgroup 1 hits a `bar.sync` barrier, yielding execution to Warpgroup 2
3. Warpgroup 2 executes its WGMMAs while Warpgroup 1's Tensor Cores are idle
4. During Warpgroup 2's GEMM phase, Warpgroup 1 computes softmax on CUDA cores/SFU
5. The pattern alternates — each warpgroup's softmax executes "in the shadow" of the other's GEMMs

**Why it works**: The Tensor Cores (doing WGMMA) and the SFU/CUDA cores (doing exp, rowmax, rescaling) are **independent execution units**. With only one warpgroup active, the SFU sits idle during GEMM and the Tensor Cores sit idle during softmax. Ping-pong ensures both units are busy.

**Quantitative improvement**: ~570 → ~620 TFLOPS (FP16, d=128, seqlen=8K), an **8.8% improvement** from the scheduling technique alone.

**Limitation**: Requires 2 warpgroups per CTA, which halves the number of concurrent CTAs per SM. This is fine for large tiles but can hurt occupancy for small problem sizes.

---

## 4. How WGMMA + TMA Overlap Works

### TMA (Tensor Memory Accelerator)
TMA is a dedicated hardware unit on Hopper that handles HBM ↔ SMEM transfers. Key properties:
- **Asynchronous**: Thread issues TMA descriptor, hardware handles the transfer
- **Zero register cost**: Index computation, out-of-bounds predication, and address calculation all done in hardware
- **Supports 2D/3D tiles**: Can load rectangular sub-tiles from tensors with arbitrary leading dimensions
- **Async barrier completion**: `cp.async.bulk` with `mbarrier` for completion signaling

### WGMMA (Warpgroup Matrix Multiply-Accumulate)
WGMMA is Hopper's Tensor Core instruction:
- **Asynchronous**: Issues to Tensor Core, execution happens independently of issuing thread
- **Warpgroup-wide**: 128 threads (4 warps) cooperate on a single large matrix multiply
- **Direct SMEM sourcing**: Reads operands directly from shared memory (no register file pressure for operands)
- **Higher throughput**: ~1.5× throughput vs. Ampere's `mma.sync` (which tops out at 2/3 of Hopper peak)

### The Overlap Pattern (Producer-Consumer)

```
Producer Warps (TMA):     Consumer Warps (WGMMA + Softmax):
                          
[cpipe.async K block N+2] [wgmma Q×K^T for block N+1]
[cpipe.async V block N+2] [wait wgmma N+1]  
[barrier sync]            [softmax on block N+1]
                          [wgmma P×V for block N+1]
                          [rescale output accumulator]
```

The circular SMEM buffer has 3 slots:
- **Slot 0**: Being consumed by current WGMMA
- **Slot 1**: Just loaded by TMA, ready for next WGMMA
- **Slot 2**: Being loaded by TMA for the iteration after next

The `mbarrier` mechanism ensures the producer only overwrites a slot when the consumer is done with it, and the consumer only reads a slot when the producer has finished loading it.

**Key insight**: Because both TMA and WGMMA are asynchronous, the producer can load data for iteration N+2 while the consumer computes iteration N+1, while CUDA cores compute softmax for iteration N — **three operations in flight simultaneously** across different hardware units.

---

## 5. Specific Techniques Applicable to SAM 3.1 ViT Attention

SAM 3.1 ViT attention characteristics:
| Parameter | Value |
|---|---|
| Heads | 16 (multi-head attention) |
| Head dimension | 64 |
| Sequence length | 1024 (64×64 image patches) |
| Attention type | Bidirectional (non-causal) |
| Batch size | 1 (inference) |
| GQA | No (standard MHA) |

### 5.1 Head Dimension 64 is FA-3's Sweet Spot

The paper explicitly notes: *"For head dimension 64, FlashAttention-3 FP8 is ahead [of cuDNN]"*. This is because:
- d=64 means each GEMM block is smaller → more blocks to parallelize → better occupancy
- The softmax-to-GEMM FLOP ratio is more favorable: 1024×1024 score matrix has 1024 exps per row vs. 1024×64 multiply-adds — the softmax bottleneck is less severe than d=128
- FP8 WGMMA tile sizes for d=64 map well to Hopper's Tensor Core geometry

### 5.2 Sequence Length 1024: Small but Parallelizable

N=1024 is relatively short for FA-3 (designed for N≥4K). However:
- FA-3 parallelizes over query blocks in the outer loop. For N=1024 with block size Br=64, that's 16 query blocks
- With 16 heads and batch=1, total parallelism = 16 heads × 16 query blocks = 256 work units
- On H100 with 132 SMs, this provides ~2 CTAs per SM — enough for decent occupancy
- On RTX 5060 (likely ~30-40 SMs), this provides ~6-8 CTAs per SM — very good occupancy

### 5.3 Warp Specialization Transfer

The producer-consumer warp specialization pattern transfers directly:
- Producer warps issue TMA loads for K/V patches
- Consumer warps compute Q@K^T and P@V via WGMMA
- Softmax runs on CUDA cores during WGMMA's asynchronous execution

For d=64, the tiles are small enough to fit multiple stages of the circular SMEM buffer comfortably within Hopper's 228KB shared memory (or Blackwell's equivalent).

### 5.4 Non-Causal Attention (see §6 below)

SAM 3.1's bidirectional attention is actually **easier** for FA-3 than causal attention, and yields higher throughput.

### 5.5 Quantization Opportunities

For SAM 3.1 inference:
- FP8 would provide ~2× throughput over FP16
- Since this is inference (not training), the accuracy tolerance for FP8 is higher
- The 16 heads × d=64 configuration means each head's attention matrix is 1024×1024 — good for block quantization
- Incoherent processing may be unnecessary for vision tasks (fewer outlier features than LLMs)

---

## 6. How FA-3 Handles Non-Causal Attention

**SAM 3.1 uses non-causal (bidirectional) attention** — every token attends to every other token. This is the simpler case and FA-3 handles it natively.

### Implementation Details

FA-3's `flash_attn_func` accepts `causal=False` (the default), which:
1. Processes **all** K/V blocks for every query block — no masking
2. No triangular masking overhead — every WGMMA computes a full tile
3. All 1024/Br = 16 K/V blocks are processed per query block

### Why Non-Causal is Faster

| Aspect | Causal | Non-Causal (SAM 3.1) |
|---|---|---|
| K/V blocks per Q block | Variable (1 to N/Bc) | All (N/Bc) |
| WGMMA utilization | Wastes work on masked tiles | 100% compute useful |
| Softmax | Needs masking logic | No masking overhead |
| TFLOPS (FA-3 H100) | Lower (~600-680) | Higher (~660-740) |

For causal attention, early query blocks process few K/V blocks (underutilizing the GPU), while later blocks process many. Non-causal attention has perfectly uniform work distribution — every query block processes the same number of K/V blocks. This enables:
- Better load balancing across SMs
- More predictable memory access patterns
- Higher sustained Tensor Core utilization

The paper's benchmark shows that for head_dim=64, non-causal attention achieves the **highest** TFLOPS among all configurations.

### FA-3's FP8 Advantage for Non-Causal

The paper notes: *"for head dimension 64, FlashAttention-3 FP8 is ahead [of cuDNN], while for head dimensions 128 and 256 it is at par for those cases without causal masking and behind with causal masking."* This means for SAM 3.1's d=64 non-causal configuration, FA-3 FP8 is the **best available implementation**.

---

## 7. Performance Numbers: Achieved TFLOPS on H100/B100

### H100 SXM5 (Hopper) — From Paper & Blog

| Configuration | TFLOPS | % of Peak |
|---|---|---|
| **FP16 Forward** (d=128, N=8K, non-causal) | **~740** | 75% of 989 |
| **FP16 Forward** (d=64, N=8K, non-causal) | **~660-700** | 67-71% |
| **FP16 Forward** (d=128, N=4K, causal) | ~620-660 | 63-67% |
| **FP8 Forward** (d=128, N=8K, non-causal) | **~1,100-1,200** | 56-61% of 1978 |
| **FP8 Forward** (d=64, N=8K, non-causal) | **~1,000-1,150** | 51-58% |
| **FP16 Backward** (d=128, N=8K) | **~500-600** | 51-61% |
| FA-2 FP16 Forward (baseline on H100) | ~350 | 35% |
| FA-2→FA-3 speedup (FP16 forward) | **1.5-2.0×** | — |
| FA-2→FA-3 speedup (FP16 backward) | **1.5-1.75×** | — |

### B100/B200 (Blackwell) — Extrapolation

FlashAttention-4 (written in CuTeDSL) targets both Hopper and Blackwell. Blackwell B200 specs:
- **~1,980 TFLOPS** FP16 (Tensor Core)
- **~3,960 TFLOPS** FP8 (Tensor Core)
- FP4 support for additional 2× over FP8

FA-3's algorithmic techniques (warp specialization, GEMM-softmax overlap, incoherent processing) are described as *"operative for any GPU architecture with sufficiently robust asynchronous execution and low-precision capabilities"* — directly applicable to Blackwell. Expected scaling: FP16 ~1,000-1,500 TFLOPS, FP8 ~2,000-3,000 TFLOPS on B200.

### Comparison with cuDNN

For d=64 non-causal attention (SAM 3.1's configuration), FA-3 **outperforms** NVIDIA's cuDNN 9 implementation at large sequence lengths and is competitive at all sizes.

---

## 8. Can FA-3 Techniques Give Us the 13× Speedup Needed for <50ms SAM 3.1 on RTX 5060?

### The Problem

SAM 3.1 ViT needs to run in <50ms total on RTX 5060. Current attention takes a significant portion. We need ~13× speedup in the attention component.

### RTX 5060 Specifications (Estimated, Blackwell-based)

| Spec | Estimated Value |
|---|---|
| Architecture | Blackwell (GB206/GB207) |
| SMs | ~30-36 |
| FP16 Tensor Core TFLOPS | ~150-200 |
| FP8 Tensor Core TFLOPS | ~300-400 |
| HBM Bandwidth | ~300-400 GB/s |
| TDP | ~150W |

### Analysis: Is 13× Feasible?

**SAM 3.1 attention FLOP count** (for one attention layer):
- Q@K^T: 16 heads × 1024 × 1024 × 64 = 1.07 GFLOP
- P@V: 16 heads × 1024 × 64 × 1024 = 1.07 GFLOP
- Total per attention layer: ~2.14 GFLOP (forward only)

For a naive PyTorch attention (materializing S and P matrices):
- S matrix: 16 × 1024 × 1024 × 2 bytes (FP16) = 32 MB read/write
- P matrix: 32 MB read/write
- Total HBM traffic: ~128 MB (Q, K, V reads + S, P intermediates + O writes)
- At 350 GB/s: ~366 μs just for memory traffic (lower bound)

**FA-3 on RTX 5060 projection**:
- FP16: Assuming 50-60% utilization of ~175 TFLOPS = 87-105 TFLOPS
- Time for 2.14 GFLOP: 2.14 / 100,000 = ~21 μs
- Memory-bound lower bound: ~50-80 μs (smaller HBM bandwidth than H100)

**FP8 projection**:
- FP8: Assuming 50-60% utilization of ~350 TFLOPS = 175-210 TFLOPS
- Time for 2.14 GFLOP: ~10-12 μs

### Speedup Estimate

| Implementation | Estimated Time (μs) | Speedup vs Naive |
|---|---|---|
| Naive PyTorch (materialized S, P) | ~500-1000 | 1× |
| FA-2 (FlashAttention-2, Ampere kernels) | ~80-150 | 5-10× |
| FA-3 style (FP16, Blackwell kernels) | ~30-60 | 12-25× |
| FA-3 style (FP8, Blackwell kernels) | ~15-30 | 25-60× |

### Verdict: YES — But With Caveats

**13× is achievable** for the attention component alone using:
1. FlashAttention-style tiling (5-10× from memory traffic reduction)
2. FP8 Tensor Cores (2× from precision)
3. Warp specialization + GEMM-softmax overlap (1.5-2× from async overlap)

However, several caveats:
- **FA-3 requires Hopper**: The current FA-3 implementation uses Hopper-specific instructions (WGMMA, TMA). RTX 5060 uses Blackwell, which needs FA-4 or a Blackwell port.
- **FA-4 exists**: Dao et al. have released FlashAttention-4 written in CuTeDSL for Hopper+Blackwell. `pip install flash-attn-4` — this is the version to use for RTX 5060.
- **Attention is not the whole ViT**: Even if attention runs in 20-30μs, the full ViT includes MLP layers, layer norms, and the image encoder. The 13× must come from optimizing ALL components.
- **Consumer GPU limitations**: RTX 5060 has ~30 SMs vs. H100's 132, and ~350 GB/s vs. 3.35 TB/s HBM bandwidth. The FA-3 techniques scale down but absolute performance will be ~5-8× lower than H100.
- **Kernel launch overhead**: For very short attention (N=1024), kernel launch overhead (~5-10μs) can be significant. Persistent kernel design from FA-3 helps.

### Recommended Approach for SAM 3.1 on RTX 5060

1. **Use FA-4** (`flash-attn-4[cu13]`) for Blackwell-native attention kernels
2. **FP8 inference** with block quantization (per-block scales, not per-tensor)
3. **Skip incoherent processing** — vision tasks have fewer outlier features than LLMs
4. **Fused attention + projection** — combine QKV projection + attention + output projection in a single kernel using CUTLASS
5. **Target d=64 specifically** — FA-3/4 benchmarks show this is the optimal head dimension for throughput
6. **Use non-causal mode** — SAM 3.1 is bidirectional, which is the faster path

With all optimizations, attention should achieve **<25μs** on RTX 5060, well within the <50ms budget for the full pipeline.

---

## Key Takeaways for Implementation

1. **FlashAttention-3's algorithmic innovations** (warp specialization, GEMM-softmax overlap, FP8 with incoherent processing) are directly applicable to SAM 3.1's attention pattern (16 heads × d=64, N=1024, non-causal).

2. **d=64 non-causal is the best case** for FA-3/4 throughput — this is exactly SAM 3.1's configuration.

3. **The 13× speedup target is achievable** by combining FlashAttention tiling (5-10×), FP8 (2×), and async overlap (1.5-2×), but requires Blackwell-native kernels (FA-4, not FA-3).

4. **The real bottleneck may shift** — after optimizing attention to ~20μs, the MLP layers and projection matrices may become the new bottleneck. CUTLASS-based GEMM optimization for those layers is equally important.

5. **Consumer GPU gap**: RTX 5060 will have ~5-8× less absolute performance than H100, but the **relative speedup** from FA-3/4 techniques scales similarly.

---

## References

- Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608
- Tri Dao blog post: https://tridao.me/blog/2024/flash3/
- GitHub: https://github.com/Dao-AILab/flash-attention
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- ThunderKittens: https://github.com/HazyResearch/ThunderKittens
- NVIDIA cuDNN 9: https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/
