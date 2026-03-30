# Flash Attention in CUTLASS — Scaling Attention Efficiently

## 1. The Attention Bottleneck

Standard attention: O(N²) memory, O(N²) compute
```
S = Q @ K^T          // (N, N) matrix — N² memory
P = softmax(S)       // (N, N) matrix — N² memory
O = P @ V            // (N, d) matrix
```
For SAM 3.1 ViT with N=1024 patches: S matrix = 1024×1024 × 2 bytes = 2 MB per head.
16 heads = 32 MB just for attention scores — won't fit in SMEM.

## 2. Flash Attention Algorithm

Key insight: compute attention in tiles, using online softmax.

### Online Softmax
```
For each block of Q rows:
  For each block of K,V columns:
    S_block = Q_block @ K_block^T
    m_new = max(m_old, rowmax(S_block))
    P_block = exp(S_block - m_new)
    l_new = exp(m_old - m_new) * l_old + rowsum(P_block)
    O_block = exp(m_old - m_new) * O_old + P_block @ V_block
  O_final = O_block / l_new
```

### Memory Complexity
- Standard: O(N²) — store full attention matrix
- Flash Attention: O(N·d) — only store running (O, l, m) per query row
- For N=1024, d=64: 32 KB vs 2 MB — **64× reduction**

## 3. CUTLASS Flash Attention Implementation

### Blackwell FMHA (Example: fmha.py)
CUTLASS provides a full FMHA (Flash Multi-Head Attention) kernel for Blackwell:

```python
@cute.kernel
def fmha_kernel(Q, K, V, O, ...):
    # Q, K, V in shared memory
    # Tile-based attention with online softmax
    
    # Allocate SMEM tiles
    sQ = cute.make_tensor(smem_Q, layout_Q)
    sK = cute.make_tensor(smem_K, layout_K)
    sV = cute.make_tensor(smem_V, layout_V)
    
    # Main loop: iterate over K,V blocks
    for kv_block in range(num_kv_blocks):
        # Load K, V block from global → SMEM
        copy(tK_g2s, tK_g, tK_s)
        copy(tV_g2s, tV_g, tV_s)
        cp_async_wait()
        
        # Compute S = Q @ K^T
        gemm(sQ, sK, sS)
        
        # Online softmax
        m_new = max(m_old, rowmax(sS))
        sP = exp(sS - m_new)
        l_new = exp(m_old - m_new) * l_old + rowsum(sP)
        
        # Accumulate O
        O_new = exp(m_old - m_new) * O_old + sP @ sV
        
        m_old, l_old, O_old = m_new, l_new, O_new
    
    # Final normalization
    O_out = O_old / l_old
    copy(O_out, gO)
```

### Hopper FMHA Variants
- `fmha.py` — forward pass
- `fmha_bwd.py` — backward pass (for training)
- `mixed_input_fmha/` — INT4/INT8 KV cache support

### CUTLASS C++ FMHA
Example 93: Blackwell GQA with Flash Decoding:
- Splits KV across cluster CTAs
- Each CTA computes partial attention
- Cluster reduction combines results
- Supports dH=64, 128

## 4. Flash Decoding (Variable-Length)

Standard Flash Attention: one KV sequence per batch element.
Flash Decoding: splits KV across multiple CTAs:

```
KV Sequence: [===========================================]
CTA 0:       [==========]
CTA 1:                 [==========]
CTA 2:                           [==========]
CTA 3:                                     [==========]

Each CTA computes partial attention for its KV segment
Final: cluster reduction combines partial results
```

### SAM 3.1 Application
ViT attention: N=1024, all same length → standard Flash Attention
DETR cross-attention: variable HW per image → Flash Decoding beneficial

## 5. Attention Sink Support

Some attention heads have "sink" tokens that receive disproportionate attention.
CUTLASS GQA example supports this:
```
Attention output = Σ(softmax(score_i) * V_i) + sink_weight * sink_token
```

## 6. SAM 3.1 Attention Optimization Plan

### ViT Self-Attention (16 heads × 64-dim, seq=1024)
```
Q shape: (16, 1024, 64) — 16 heads
K shape: (16, 1024, 64)
V shape: (16, 1024, 64)
Standard attention: 16 × 1024² × 64 × 2 ops = 2.1 GFLOP
Flash Attention tile: (64, 64) for Q and KV blocks
SMEM per head: Q(64×64×2B) + K(64×64×2B) + V(64×64×2B) + S(64×64×4B) = 40KB
```

### CLIP Text Attention (16 heads × 64-dim, seq=77)
```
Tiny sequence → attention is not the bottleneck
Standard matmul is fine, no need for Flash Attention
```

### DETR Self-Attention (8 heads × 32-dim, seq=100)
```
Extremely small → overhead dominates
Consider: SIMT (non-Tensor Core) or even cuBLAS
```

### DETR Cross-Attention (8 heads × 32-dim, Q=100, KV=HW)
```
Short Q, long KV (e.g., 1024) → Flash Decoding helps
Tile Q in blocks of 16-32, iterate over KV
```

