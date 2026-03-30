# Attention Fusion Strategies for SAM 3.1 — QKV→SDPA→Projection

## 1. The Attention Fusion Opportunity

Standard multi-head attention in SAM 3.1's ViT executes as 5-7 separate operations:

```
1. Q = X @ Wq + bq          // GEMM + bias
2. K = X @ Wk + bk          // GEMM + bias
3. V = X @ Wv + bv          // GEMM + bias
4. Attn = softmax(Q @ K^T / √d)  // GEMM + softmax
5. O = Attn @ V             // GEMM
6. Out = O @ Wo + bo        // GEMM + bias
```

Each step writes to global memory and reads back — 6 memory round-trips for what is fundamentally one linear attention operation. Fusion eliminates intermediate memory traffic by keeping data in shared memory or registers across operations.

---

## 2. Fusion Level 1: QKV Projection Fusion

### Three GEMMs → One
Q, K, V projections share the same input X. By concatenating Wq, Wk, Wv into a single weight matrix Wqkv of shape (d_model, 3×d_model), we compute all three projections in one GEMM:

```cpp
// Before: 3 separate GEMMs
Q = X @ Wq    // (seq, d_model) × (d_model, d_head) → (seq, d_head)
K = X @ Wk    // same
V = X @ Wv    // same

// After: 1 GEMM + split
QKV = X @ Wqkv  // (seq, d_model) × (d_model, 3×d_head) → (seq, 3×d_head)
// Split QKV into Q, K, V in epilogue or via layout tricks
```

**CUTLASS implementation:** Use a single GEMM with M=seq_len, N=3×d_head, K=d_model. The output is naturally laid out as [Q|K|V] contiguous in memory. The epilogue can optionally split into separate buffers using `Sm90RowBroadcast` with different strides, or downstream ops access slices.

**SAM 3.1 impact:** Reduces 3 kernel launches to 1. On ViT-L/14 with 256 GEMMs across 32 blocks, this saves 64 GEMM launches (32 blocks × 2 extra Q/K/V GEMMs avoided).

### Performance estimate
- 3× reduction in kernel launch overhead: ~15-30 μs saved
- 3×→1× reduction in input X reads: saves d_model × seq_len × sizeof(fp16) × 2 reads per block
- For seq_len=5376, d_model=1024: saves ~21MB of global memory traffic per block

---

## 3. Fusion Level 2: SDPA (Scaled Dot-Product Attention) Fusion

### Fusing Q@K^T → Softmax → Attn@V
The core attention computation is:
```
S = Q @ K^T / √d_k     // (seq, seq) matrix
P = softmax(S)          // row-wise softmax
O = P @ V               // (seq, d_v)
```

**Flash Attention** (and CUTLASS FMHA) fuses this into a single kernel by using **tiling and online softmax**:

1. Process Q, K, V in tiles (e.g., 128×64)
2. For each Q tile, iterate over K/V tiles
3. Maintain running max and sum for numerically stable softmax
4. Accumulate output incrementally: O_i = Σ_j softmax(S_ij) × V_j

This eliminates the (seq × seq) attention score matrix entirely — the biggest memory savings in transformer attention.

### CUTLASS FMHA Implementation
CUTLASS 3.x provides `collective_fmha` for Hopper/Blackwell:

```cpp
using ProblemShape = Shape<int, int, int, int>;  // (batch, heads, seq, dim)
using TileShape = Shape<128, 64, 64>;             // (Q_tile, K_tile, V_tile)

// The FMHA mainloop fuses Q@K^T → online softmax → P@V
using Mainloop = cutlass::fmha::collective::Sm90FMHAFlashAttention<
    TileShape, PipelineStages, HeadSpec>;
```

**SAM 3.1 impact:** Eliminates the (seq × seq) attention matrix. For 1024×1024 input with 16 heads, this saves 16 × 5376 × 5376 × 2 bytes = ~924 MB of memory traffic.

---

## 4. Fusion Level 3: QKV→SDPA→Projection Mega-Fusion

### The Ultimate Fusion: One Kernel for All Attention
The holy grail is fusing the entire attention block into a single kernel:

```
Input: X (seq, d_model)
→ QKV projection (X @ Wqkv)
→ Split into Q, K, V
→ SDPA (Q @ K^T → softmax → @ V)  
→ Output projection (O @ Wo)
→ Output: (seq, d_model)
```

**Challenges:**
1. QKV projection produces (seq, 3×d_head) — SDPA needs separate Q, K, V
2. SDPA operates on (seq, seq) tiles — needs careful memory management
3. Output projection needs concatenated multi-head output

**CUTLASS approach:**
1. Use EVT to fuse QKV projection + output projection bias into the GEMM epilogues
2. Use FMHA mainloop for the SDPA core
3. Chain via shared memory — QKV output stays in SMEM, feeds directly into FMHA

**Practical limitation on SM100 (RTX 5060):**
- SM100 has 100KB SMEM per SM
- QKV tile: 128 × 3 × 64 × 2 bytes = 48KB (FP16)
- FMHA working set: ~32KB (K tile + V tile + accumulator)
- Total: ~80KB — barely fits, leaving no room for double-buffering

**Recommendation for SAM 3.1:** Partial fusion — fuse QKV projection with FMHA, keep output projection separate. This captures ~70% of the memory savings with much simpler implementation.

---

## 5. Fusion Level 4: Block-Level Attention Fusion

### Across ViT Blocks
SAM 3.1's ViT-L/14 has 32 blocks, each with: Attn → residual → LayerNorm → MLP → residual. Between blocks, the output of block N feeds into block N+1.

**CUDA Graph capture** captures the entire attention+MLP sequence for a block and replays it 32 times with different weights:
```
Graph: [Attn_kernel → residual_add → LayerNorm → MLP_kernel → residual_add]
Replay: 32 times, swapping weight pointers between iterations
```

This eliminates kernel launch overhead for all 64+ kernels per ViT forward pass.

---

## 6. ViT Attention Head Configuration in SAM 3.1

### SAM 3.1 ViT-L/14 Parameters
| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| num_heads | 16 |
| d_head (d_k = d_v) | 64 |
| Wqkv shape | (1024, 3×1024) = (1024, 3072) |
| Wo shape | (1024, 1024) |
| Sequence length (1024² input) | 5376 (16×16 patches + special tokens) |
| Attention score matrix | 5376 × 5376 = 28.9M elements |

### GEMM Sizes per Block
| Operation | M | N | K | FLOPs |
|-----------|---|---|---|-------|
| QKV projection | 5376 | 3072 | 1024 | 33.8 GFLOP |
| Q @ K^T (per head) | 5376 | 5376 | 64 | 3.7 GFLOP |
| Attn @ V (per head) | 5376 | 64 | 5376 | 3.7 GFLOP |
| Output projection | 5376 | 1024 | 1024 | 11.3 GFLOP |
| **Total attention per block** | | | | **~52.5 GFLOP** |
| **Total attention (32 blocks)** | | | | **~1680 GFLOP** |

---

## 7. CUTLASS Fusion Implementation Patterns

### Pattern A: EVT-Fused QKV Projection
```cpp
// Single GEMM: QKV = X @ Wqkv + bqkv
// Epilogue splits output into Q, K, V via stride tricks
using EpilogueQKV = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::plus, half_t, float>,
    cutlass::epilogue::fusion::Sm90AccFetch,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        cutlass::layout::RowMajor, half_t>  // per-row bias
>;
```

### Pattern B: FMHA Mainloop with TMA
```cpp
// Flash Attention using CUTLASS FMHA collective
using FMHA = cutlass::fmha::kernel::Sm90FlashAttention<
    ProblemShape, QKVLayout, TileShape,
    CollectiveMainloop, CollectiveEpilogue>;
// Processes all 16 heads in one kernel via grouped/batched dispatch
```

### Pattern C: Chained Attention via CUDA Graph
```cpp
// Capture entire block as graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
qkv_gemm(stream);       // QKV projection
fmha(stream);           // Flash Attention
out_gemm(stream);       // Output projection
residual_add(stream);   // Residual connection
layernorm(stream);      // LayerNorm
mlp_gemm1(stream);      // MLP FC1
activation(stream);     // GELU
mlp_gemm2(stream);      // MLP FC2
residual_add(stream);   // Residual
cudaStreamEndCapture(stream, &graph);
// Replay 32 times for 32 ViT blocks
```

---

## 8. Expected Performance Impact

| Fusion Level | Kernels Eliminated | Memory Saved | Latency Saved |
|-------------|-------------------|--------------|---------------|
| QKV fusion | 2 per block (64 total) | 2× input reads | ~30 μs |
| SDPA fusion (Flash) | 2 per block (64 total) | seq² attention matrix | ~200 μs |
| QKV+SDPA partial | 4 per block (128 total) | Above combined | ~220 μs |
| Full attention fusion | 5 per block (160 total) | All intermediates | ~250 μs |
| Block-level (CUDA Graph) | Launch overhead | 0 | ~640 μs (20μs × 32) |

**Total attention fusion savings: ~500-900 μs on RTX 5060**, representing 15-25% of total inference time.

---

## 9. Implementation Priority for SAM 3.1

1. **QKV projection fusion** — Easy, high impact. Concatenate Wq/Wk/Wv weights.
2. **Flash Attention (FMHA)** — Medium difficulty, highest single-kernel impact. Use CUTLASS 3.x FMHA.
3. **CUDA Graph capture** — Easy, captures launch overhead. Use `cudaStreamBeginCapture`.
4. **QKV+SDPA mega-fusion** — Hard, diminishing returns. Defer to Phase 2.
5. **Full block fusion** — Very hard, requires custom CUDA. Long-term goal.

---

## References

- `include/cutlass/fmha/` — CUTLASS FMHA collective implementation
- `examples/64_blackwell_fmha/` — Blackwell FMHA example
- `examples/52_hopper_gemm_with_collective_builder/` — Collective builder patterns
- Dao et al., "FlashAttention-3" — Fast attention with Hopper/Blackwell async
- `include/cutlass/epilogue/fusion/` — EVT fusion node definitions
