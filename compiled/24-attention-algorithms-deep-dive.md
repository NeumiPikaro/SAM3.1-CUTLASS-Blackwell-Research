# Attention Algorithms — Mathematical Deep Dive

## 1. Standard Scaled Dot-Product Attention

### Mathematical Foundation
Given Q ∈ ℝ^(n×d), K ∈ ℝ^(m×d), V ∈ ℝ^(m×d):

```
S = Q × K^T / √d        # Attention scores, shape (n, m)
P = softmax(S)           # Attention weights, shape (n, m)  
O = P × V                # Output, shape (n, d)
```

Where softmax is applied row-wise:
```
P_ij = exp(S_ij) / Σ_k exp(S_ik)
```

### Numerical Stability: Online Softmax
Raw softmax is numerically unstable for large scores. The log-sum-exp trick:
```
m_i = max_j(S_ij)                          # Row maximum
P_ij = exp(S_ij - m_i) / Σ_k exp(S_ik - m_i)  # Stable softmax
```

### Complexity Analysis
- Compute: O(n × m × d) for Q×K^T, O(n × m × d) for P×V → O(nmd)
- Memory: O(n × m) for attention matrix S
- For SAM 3.1 ViT: n=m=1024, d=64 → 67M FLOPs per head, 2MB memory per head
- 16 heads = 1.07 GFLOPs, 32MB memory

## 2. Flash Attention Algorithm (Dao et al., 2022)

### Core Insight
Instead of materializing the full S matrix (n×m), compute attention in blocks using online softmax state (m_i, l_i, O_i):

```
Algorithm: FlashAttention-2
Input: Q, K, V ∈ ℝ^(n×d), block sizes B_r, B_c
Output: O ∈ ℝ^(n×d)

1. Initialize: O = 0, l = 0, m = -∞
2. Divide Q into B_r-sized blocks: Q_1, Q_2, ..., Q_{⌈n/B_r⌉}
3. Divide K, V into B_c-sized blocks: K_1, ..., V_{⌈m/B_c⌉}
4. For each Q_i block:
   a. Load Q_i into SRAM
   b. For each K_j, V_j block:
      i.   Load K_j, V_j into SRAM
      ii.  S_ij = Q_i × K_j^T / √d          # On-chip matmul
      iii. m_new = max(m_old, rowmax(S_ij))
      iv.  P_ij = exp(S_ij - m_new)
      v.   l_new = exp(m_old - m_new) × l_old + rowsum(P_ij)
      vi.  O_i = diag(exp(m_old - m_new)) × O_i + P_ij × V_j
      vii. m_old = m_new, l_old = l_new
   c. O_i = O_i / l_i
   d. Write O_i to HBM
```

### Memory Complexity
- Standard: O(n × m) — full attention matrix in HBM
- Flash Attention: O(n + m + B_r × B_c) — only blocks in SRAM
- For n=m=1024, B_r=B_c=64: 2048 + 4096 = 6KB vs 2MB → 341× reduction

### Arithmetic Intensity Improvement
- Standard: AI = 4nm / (2(n+m)d + nm) ≈ 4 for large n
- Flash: AI = 4nmd / (2(n+m)d × n/B_r) ≈ 2B_r = 128 for B_r=64
- 32× higher arithmetic intensity → better Tensor Core utilization

## 3. Flash Attention-2 Improvements

### Parallelization Over Q Blocks
FA-1 parallelizes over batch and heads. FA-2 also parallelizes over Q blocks:
```
Grid: (batch, heads, ⌈n/B_r⌉)
Each block processes one Q_i block against all K_j blocks
```
Better SM utilization for small batch sizes (SAM 3.1 has batch=1).

### Reduced Non-Matmul Operations
FA-2 moves rescaling outside the inner loop:
```
Instead of rescaling O after each K_j block:
  Accumulate (O_i, m_i, l_i) without rescaling
  Rescale once at the end
```
Saves ~15% non-matmul FLOPs.

### Backward Pass
FA-2 backward recomputes attention from SRAM (no HBM round-trip):
```
Forward: store (O, l, m) — O(n) memory
Backward: recompute S from Q,K in SRAM — no O(nm) storage
```
Enables training with O(n) memory instead of O(nm).

## 4. Flash Decoding (Split-K Attention)

For serving with long KV sequences and small batch:
```
KV sequence: [===K_1===|===K_2===|===K_3===|===K_4===]
CTA 0:       processes K_1 segment
CTA 1:       processes K_2 segment
CTA 2:       processes K_3 segment
CTA 3:       processes K_4 segment

Each CTA: partial (O_i, l_i, m_i)
Reduction: combine partial results across CTAs
```

### CUTLASS Implementation (Example 93)
```
Each cluster processes 1 KV head across multiple CTAs:
- DMA_Q warp: loads Q
- DMA_KV warp: loads K, V for assigned segment
- MMA warp: computes attention scores
- EPILOG warps: softmax + cluster reduction

Cluster reduction via distributed SMEM:
- T0,T32,T64,T96 store partial fmax/fsum to destination CTA's DSMEM
- Reduction CTAs combine and produce final output
```

## 5. Grouped Query Attention (GQA)

GQA groups multiple query heads to share one KV head:
```
Q heads: 64 (4 groups × 16 queries)
KV heads: 8 (1 per group)
Each Q head in group attends to same K, V
```

### CUTLASS Support
Example 93 supports GQA natively:
```cpp
--kvH 8      // 8 KV heads
--qH 64      // 64 Q heads (= 8 groups × 8 queries per group)
--dH 64      // head dimension
```

### SAM 3.1 Relevance
SAM 3.1 uses standard MHA (16 heads, all independent). No GQA.
But if Meta updates to GQA in future, CUTLASS is ready.

## 6. Multi-Latent Attention (MLA)

MLA (DeepSeek) compresses KV into latent representations:
```
Standard: K ∈ ℝ^(n×d), V ∈ ℝ^(n×d) — 2nd memory
MLA: K = W_k × c, V = W_v × c where c ∈ ℝ^(n×d_c), d_c << d
```
CUTLASS has MLA examples in CuTe DSL (mla/ directory).

### SAM 3.1 Relevance
Not used currently. Future optimization opportunity.

## 7. Attention Sink Tokens

Some attention heads allocate disproportionate weight to special tokens:
```
Attention to sink token: 30-50% of total attention weight
Effect: useful information gets diluted
```

### CUTLASS Support
Example 93 GQA supports attention sinks:
```
--attention_sink  # Enable sink token handling
```

## 8. Sliding Window Attention

For long sequences, limit attention to local window:
```
Standard: attend to all m positions — O(nm)
Windowed: attend to w positions — O(nw)
```

### SAM 3.1 Relevance
SAM 3.1 attends to all patches (no windowing). But DETR cross-attention attends to all encoder features — could benefit from windowing if resolution is very high.

## 9. Causal Masking

For autoregressive models (GPT, not SAM 3.1):
```
S_ij = -∞ for j > i (can't attend to future tokens)
```

### Implementation
CUTLASS implements causal masking by skipping upper-triangular tiles:
```
For tile (i, j) where j > i: skip entirely
Saves ~50% compute for causal attention
```

### SAM 3.1 Relevance
SAM 3.1 is non-autoregressive — no causal masking needed.

## 10. RoPE in Attention

Rotary Position Embeddings modify Q, K before attention:
```
Q'_i = RoPE(Q_i, position=i)
K'_j = RoPE(K_j, position=j)
S_ij = Q'_i × K'_j^T / √d
```

### Fusion Strategy
Option 1: Separate RoPE kernel before attention (simple, 1 extra kernel)
Option 2: Fuse RoPE into QKV projection epilogue (efficient, CUTLASS EVT)
Option 3: Fuse RoPE into attention mainloop (most complex, highest payoff)

### SAM 3.1 Specific
SAM 3.1 uses 2D RoPE (complex tensors):
```
For patch at (row, col):
  Q_rotated = Q ⊙ exp(i × (θ_row + θ_col))
```
Requires converting complex RoPE to real-only for CUTLASS compatibility.

