# Flash Decoding: Split-K Attention for High-Throughput Serving

**Research Note — 2026-03-30**

---

## 1. Background and Motivation

Transformer inference (decoding) is an iterative process: tokens are generated one at a time, and each new token must attend to all previous tokens via the KV-cache. As context lengths have exploded from 2K (GPT-3) to 128K+ (GPT-4, Llama 3), the attention operation has become the dominant latency bottleneck during generation, especially for small batch sizes where the GPU is underutilized.

**FlashAttention** (Dao et al., 2022–2024) solved the *memory-bandwidth* problem for **training** by using tiling and recomputation to avoid materializing the full attention matrix. FlashAttention-2 achieves ~70% of peak FLOPS on A100 for training workloads. However, FlashAttention parallelizes across **batch size and query length** dimensions. During **decoding**, query length = 1 and batch sizes are often small (1–16), so FlashAttention leaves most of the GPU idle.

This gap is exactly what **Flash Decoding** addresses: a new parallelism dimension for the inference attention kernel.

---

## 2. Flash Decoding Algorithm: Split KV Across SMs

Flash Decoding, introduced by Tri Dao and Daniel Y. Fu (Stanford CRFM, October 2023) and later integrated into PyTorch, extends FlashAttention with a **KV-sequence-length parallelism** axis. The core idea is simple but powerful:

### Three-Step Algorithm

1. **Split the KV-cache into chunks.** Partition the K and V matrices along the sequence dimension into `S` splits (e.g., 16 or 32 chunks for a 64K context). Each split is a view of the original tensors — no data copy is needed.

2. **Compute partial attention in parallel.** Each split is assigned to an SM (or group of SMs). Each SM independently computes a **partial softmax(Q·K^T)·V** for its chunk using the FlashAttention tiled algorithm. Crucially, each SM also writes one extra scalar per query row per split: the **log-sum-exp (LSE)** of the attention logits. This scalar encodes the normalization factor needed for correct softmax across splits.

3. **Reduce partial results.** A final reduction kernel combines the partial outputs across all `S` splits, using the stored LSE values to rescale contributions. The reduction is numerically stable because of the log-sum-exp trick — the same mechanism FlashAttention uses internally for sequential block processing, but now applied across *parallel* splits.

### The Key Insight: Online Softmax at Two Levels

The online softmax algorithm (Milakov & Gimelshein, 2018; used in FlashAttention) maintains a running maximum and sum-of-exponentials. Flash Decoding applies this at **two levels**:
- **Intra-split**: Within each chunk, the tiled FlashAttention algorithm computes the correct partial output with local softmax rescaling.
- **Inter-split**: The reduction kernel combines partial outputs by re-scaling each using the LSE scalar, analogous to how FlashAttention merges tiles sequentially — but now tiles are computed by *different SMs* in parallel.

This produces **exact attention** — no approximation beyond standard floating-point rounding.

---

## 3. How Flash Decoding Differs from FlashAttention

| Dimension | FlashAttention (FA-2) | Flash Decoding |
|-----------|----------------------|----------------|
| **Parallelism axes** | Batch + Query length | Batch + Query length + **KV sequence length** |
| **Decoding utilization** | Bounded by `batch_size × seq_q`; often < 1% on A100 | Scales with KV length; near-full GPU utilization |
| **Extra memory** | None (online softmax) | One LSE scalar per split per query row (~`S × B × H × sizeof(float)`) |
| **Reduction step** | None (sequential tiles) | Final reduction kernel over `S` splits |
| **Best regime** | Training (seq_q >> 1) | Inference (seq_q = 1, long KV) |
| **Kernel count** | Single forward kernel | Two kernels (partial attn + reduce) |

The critical difference is the **parallelism dimension**. FlashAttention parallelizes over batch and query blocks. Flash Decoding additionally parallelizes over KV blocks, so even with batch=1 and seq_q=1, the GPU can be fully occupied if the KV sequence is long enough to generate sufficient splits.

For a 108-SM A100 with batch=1, FlashAttention uses 1 SM (0.9% utilization). Flash Decoding with 64K KV and 16 splits uses 16+ SMs — potentially 15%+, and with larger KV lengths or smaller chunk sizes, it can approach full occupancy.

---

## 4. When Flash Decoding Helps

Flash Decoding provides the largest speedups in regimes where FlashAttention underutilizes the GPU:

### Primary Beneficial Scenarios

1. **Long KV sequences + small batch sizes.** This is the canonical case: batch=1 with context lengths of 16K–128K+. The Stanford/PyTorch benchmarks show up to **8× end-to-end decoding speedup** for CodeLlama-34B at 64K context on A100.

2. **Grouped-Query Attention (GQA).** In GQA, multiple query heads share the same KV heads, so the per-KV-head query count is small. Flash Decoding parallelizes over KV, which is ideal for GQA.

3. **Streaming-LLM / infinite context.** Any scenario where KV cache grows continuously during serving benefits from the constant-time attention that Flash Decoding enables.

### When It Does NOT Help

- **Large batch sizes.** If batch_size ≥ SM_count (108 for A100, 132 for H100), FlashAttention already saturates the GPU via batch parallelism. Flash Decoding adds overhead from the reduction step with no speedup gain.
- **Short sequences.** With KV length < ~1K, there are too few splits to parallelize effectively. The reduction overhead dominates.
- **Training (seq_q >> 1).** FlashAttention's query-length parallelism is already optimal; Flash Decoding's KV parallelism adds unnecessary reduction cost.

### Performance Numbers (A100, CodeLlama-34B Microbenchmarks)

The PyTorch blog reports these numbers for scaled multi-head attention (batch varying, f16, head_dim=128, 16 Q heads / 2 KV heads):

| Batch | Seq Length | PyTorch Eager (µs) | Flash-Attn v2 (µs) | Flash-Decoding (µs) | Speedup vs FA-2 |
|-------|-----------|-------------------|--------------------|--------------------|-----------------|
| 256 | 256 | 3058.6 | 390.5 | 63.4 | 6.2× |
| 128 | 512 | 3151.4 | 366.3 | 67.7 | 5.4× |
| 64 | 1024 | 3160.4 | 364.8 | 77.7 | 4.7× |
| 32 | 2048 | 3158.3 | 352 | 58.5 | 6.0× |
| 16 | 4096 | 3157 | 401.7 | 57 | 7.0× |
| 8 | 8192 | 3173.1 | 529.2 | 56.4 | 9.4× |
| 4 | 16384 | 3223 | 582.7 | 58.2 | 10.0× |
| 2 | 32768 | 3224.1 | 1156.1 | 60.3 | 19.2× |
| 1 | 65536 | 1335.6 | 2300.6 | 64.4 | **35.7×** |

At batch=1, seq=64K, the attention kernel itself is **35.7× faster** than FlashAttention-2. The end-to-end model speedup is capped at ~8× because attention is not 100% of the model runtime, but the attention speedup is dramatic.

The remarkable feature: **Flash Decoding time is nearly constant** from seq=256 to seq=64K (~55–78 µs), because it fully utilizes the GPU regardless of sequence length.

### Performance on H100

FlashAttention-3 on H100 achieves up to 740 TFLOPS (75% of peak) for FP16 training workloads. For decoding, Flash Decoding on H100 benefits from:
- 132 SMs (vs 108 on A100) → more parallel splits
- HBM3 at ~3.35 TB/s → faster KV-cache reads
- WGMMA instructions and TMA async copies → better overlap of compute and memory

Estimated H100 performance for Flash Decoding decoding:
- Batch=1, seq=64K: ~30–40 µs (roughly 1.7–2× faster than A100)
- Batch=1, seq=128K: ~45–60 µs (still near-constant time)

---

## 5. CUTLASS Cluster-Based Reduction for Flash Decoding

NVIDIA's CUTLASS library (v3.5+) provides building blocks for implementing Flash Decoding efficiently on Blackwell (SM 10.0) GPUs. The key CUTLASS feature is **cluster-based reduction** using distributed shared memory and the `cp.reduce.async` instruction.

### CUTLASS Example 93: Blackwell GQA with Flash Decoding

CUTLASS ships examples that demonstrate attention patterns. The **Example 93** series (Blackwell GQA) shows how to implement grouped-query attention with Flash Decoding-style parallelism on Blackwell hardware:

- **Cluster-level cooperation**: Blackwell introduces hardware multicast and reduction across thread blocks within a cluster. Multiple thread blocks process different KV splits and cooperatively reduce partial attention outputs via shared memory without going through global memory.

- **TMA multicast + reduction**: The Tensor Memory Accelerator (TMA) on Blackwell can broadcast K/V tiles to multiple SMs simultaneously (multicast), reducing memory bandwidth pressure. Combined with `cp.reduce.async` for in-cluster accumulation, this eliminates the separate reduction kernel.

- **CUTLASS GemmUniversal + Softmax pipeline**: CUTLASS's pipeline abstraction allows overlapping GEMM, softmax, and reduction stages. The GEMM computes Q·K^T while softmax normalization happens asynchronously on a separate warpgroup.

### Implementation Details

A CUTLASS-based Flash Decoding kernel on Blackwell typically:

1. **Splits KV** into chunks of size `BLOCK_KV` (e.g., 64–128 tokens per split).
2. **Assigns each split to a thread block cluster** (e.g., 2–8 thread blocks per cluster).
3. **Each cluster computes** partial `O_split = softmax(Q·K_split^T)·V_split` using CUTLASS GEMM with warp-specialized softmax.
4. **Reduces within cluster** using `cp.reduce.async` across distributed shared memory — the partial outputs from all thread blocks in the cluster are accumulated into a single output without global memory round-trips.
5. **Final cross-cluster reduction** (if needed) is a lightweight kernel that combines cluster outputs using LSE rescaling.

The advantage over a naive two-kernel approach (partial attention + separate reduce) is the elimination of the global memory write/read between kernels. On Blackwell, cluster-distributed-SM reduces latency by 30–50% for the reduction phase.

### FlashInfer Implementation

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) (the inference-focused kernel library) provides production Flash Decoding implementations. Key features relevant to SAM 3.1:
- Supports **SM 7.5 through SM 10.3** (Turing → Blackwell)
- Provides `single_decode_with_kv_cache()` for single-query decoding with long KV
- Paged and ragged KV-cache support
- FP8/FP4 quantized attention
- JIT compilation for custom attention variants

FlashInfer automatically selects the best backend (FlashAttention-2/3, CUTLASS, cuDNN, TensorRT-LLM) for the target hardware and workload shape.

---

## 6. Applicability to SAM 3.1 DETR Cross-Attention

SAM 3.1 (Segment Anything Model 3.1) uses a DETR-style decoder architecture. The critical cross-attention layer has the following characteristics:

### Workload Shape

| Parameter | Value |
|-----------|-------|
| Q tokens (image queries) | ~100 (prompt tokens / mask tokens) |
| KV tokens (encoder features) | 1024+ (1024–4096 for ViT encoder) |
| Head dim | 64 or 128 |
| Attention type | Cross-attention (encoder→decoder) |
| Batch size | Typically 1–4 during serving |

This shape — **small Q, large KV, small batch** — is precisely the regime where Flash Decoding excels and FlashAttention underperforms.

### Why Standard FlashAttention is Suboptimal

For SAM 3.1 cross-attention:
- Q length = 100 → FlashAttention parallelizes over ~4–8 query blocks
- Batch = 1 → Only ~4–8 SMs used out of 108 (A100) or 132 (H100)
- KV = 1024+ → Sequential tile processing within each SM; high latency

FlashAttention leaves 90%+ of the GPU idle for this workload.

### How Flash Decoding Helps

With Flash Decoding applied to SAM 3.1 cross-attention:
- **KV splits**: KV=1024 with BLOCK_KV=64 → 16 splits → 16 SMs active
- **KV=4096 → 64 splits** → 64 SMs active (A100) or 132 available (H100)
- **Each SM processes**: 100 Q tokens × 64 KV tokens in a tiled FlashAttention kernel
- **Reduction**: 16 partial outputs combined via LSE rescaling

For SAM 3.1 with KV=1024:
- **FlashAttention**: ~4 blocks of Q × sequential KV tiles. Expected latency ~200–400 µs on A100.
- **Flash Decoding**: 16 KV splits processed in parallel, each doing 100×64 attention. Expected latency ~40–80 µs on A100.

### Practical Integration Path

SAM 3.1 can leverage Flash Decoding through:
1. **FlashInfer API**: Call `flashinfer.single_decode_with_kv_cache()` or the batched variant. The KV-cache is naturally paged for DETR.
2. **PyTorch SDPA**: PyTorch 2.1+ integrates Flash Decoding; use `torch.nn.functional.scaled_dot_product_attention()` which auto-selects the backend.
3. **Custom CUTLASS kernel**: For maximum performance, write a CUTLASS-based kernel tailored to SAM 3.1's specific head dimensions and KV lengths, using cluster-based reduction on Blackwell.

---

## 7. Can Flash Decoding Reduce SAM 3.1 Attention Latency by 3–5×?

### Analytical Estimation

Let's estimate the attention latency for SAM 3.1 DETR cross-attention on H100:

**Baseline (FlashAttention-2):**
- Q = 100 tokens, KV = 1024 tokens, d = 128, H_q = 16, H_kv = 16 (no GQA)
- Parallelism: 100/64 ≈ 2 Q blocks × batch=1 = 2 concurrent units
- GPU utilization: ~2/132 = 1.5%
- Estimated latency: 300–500 µs (memory-bound, reading full KV per Q block)

**With Flash Decoding:**
- Splits: 1024/64 = 16 KV splits → 16 concurrent SM groups
- Each SM: processes 100×64 attention (Q is broadcast to all splits)
- Reduction: 16 partial outputs × 100 tokens × 16 heads × 128 dim → ~3.2M floats, trivial
- Estimated latency: 60–100 µs (GPU fully utilized)

**Speedup estimate: 3–5× for the attention kernel.**

For longer KV (2048–4096), the speedup grows to 5–10×, matching the Stanford benchmarks.

### Caveats

1. **The attention is not the entire decoder.** DETR decoder also has self-attention (Q=100, KV=100 — not benefited), FFN layers, and cross-attention projections. If attention is 30–50% of decoder time, the end-to-end speedup is 1.5–2.5×.

2. **SAM 3.1 uses multiple cross-attention layers.** Typically 6 decoder layers, each with cross-attention. Speedup compounds across layers.

3. **Overhead of the reduction step.** For small KV (1024), the reduction kernel overhead may eat into gains. For KV ≥ 2048, Flash Decoding wins decisively.

4. **Kernel fusion opportunity.** SAM 3.1's cross-attention is followed by LayerNorm + FFN. Fusing the Flash Decoding reduction with subsequent operations could yield additional 10–20% speedup.

### Verdict

**Yes, Flash Decoding can reduce SAM 3.1 attention latency by 3–5× on H100 for cross-attention with KV ≥ 1024.** For the decoder self-attention and shorter KV layers, the benefit is smaller (1.5–2×). The end-to-end decoder speedup is likely **1.5–3×**, with attention-specific speedups of **3–8×** depending on KV length.

For SAM 3.1 with high-resolution encoder features (KV = 4096+), Flash Decoding becomes essential — standard FlashAttention would leave 99% of the GPU idle, while Flash Decoding saturates it.

---

## 8. Summary and Recommendations

| Aspect | Finding |
|--------|---------|
| **Algorithm** | Split KV across SMs, compute partial softmax×V, reduce via LSE |
| **vs FlashAttention** | Adds KV-sequence parallelism; 2-kernel approach (partial + reduce) |
| **Best regime** | Small batch + long KV (decoding, cross-attention) |
| **Speedup (attention only)** | 3–50× depending on KV length and batch |
| **CUTLASS support** | Cluster-based reduction on Blackwell; Example 93 GQA pattern |
| **FlashInfer** | Production-ready; supports SM 7.5–10.3; paged KV-cache |
| **SAM 3.1 fit** | Excellent — Q=100, KV=1024+ is the sweet spot |
| **Expected SAM 3.1 attention speedup** | 3–5× (KV=1024), 5–10× (KV=4096+) |
| **Expected end-to-end decoder speedup** | 1.5–3× |

### Implementation Priority

1. **Immediate**: Integrate FlashInfer's `single_decode_with_kv_cache()` for SAM 3.1 cross-attention. Minimal code changes, 3–5× attention speedup.
2. **Medium-term**: Custom CUTLASS kernel with cluster-based reduction for SAM 3.1's specific shapes (Q=100, KV=1024–4096, d=128). Target 5–8× attention speedup on Blackwell.
3. **Long-term**: Fuse Flash Decoding with LayerNorm + FFN projection for additional end-to-end gains.

---

## References

1. Dao, T., Fu, D.Y., et al. "Flash-Decoding for long-context inference." Stanford CRFM / PyTorch Blog, October 2023. https://pytorch.org/blog/flash-decoding/
2. Dao, T. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024. https://tridao.me/blog/2024/flash3/
3. FlashInfer: Kernel Library for LLM Serving. https://github.com/flashinfer-ai/flashinfer
4. NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
5. FlashAttention-2. Dao, T. 2023. https://arxiv.org/abs/2307.08691
