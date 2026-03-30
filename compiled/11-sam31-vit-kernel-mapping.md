# SAM 3.1 ViT-L/14 — Complete Kernel Mapping

## 1. ViT-L/14 Architecture Recap

SAM 3.1's image encoder is a ViT-L/14:
- **Patch size:** 14×14 pixels
- **Embedding dimension:** 1024
- **Layers:** 32 transformer blocks
- **Heads:** 16 (64-dim each)
- **MLP ratio:** 4× (hidden dim 4096)
- **RoPE:** 2D rotary position embeddings (complex tensors)
- **Total parameters:** ~307M (ViT trunk) + ~24M (FPN neck)

## 2. Per-Block Operator Breakdown

Each of the 32 ViT blocks executes:

### Attention Path
```
1. LayerNorm: x → (x - mean) / sqrt(var + eps) * gamma + beta
2. Q projection: Q = X @ Wq — GEMM (seq, 1024) @ (1024, 1024) = (seq, 1024)
3. K projection: K = X @ Wk — same
4. V projection: V = X @ Wv — same
5. Reshape: (seq, 1024) → (16, seq, 64)
6. RoPE: Q, K = apply_rotary(Q, K, positions) — complex tensor multiply
7. Attention: S = Q @ K^T / sqrt(64) — GEMM (16, seq, seq)
8. Softmax: P = softmax(S, dim=-1)
9. Context: C = P @ V — GEMM (16, seq, 64)
10. Reshape: (16, seq, 64) → (seq, 1024)
11. Output projection: O = C @ Wo — GEMM (seq, 1024) @ (1024, 1024)
12. Residual: x = x + O
```

### MLP Path
```
13. LayerNorm: x → norm(x)
14. FC1: h = X @ W1 — GEMM (seq, 1024) @ (1024, 4096) = (seq, 4096)
15. addmm_act: h = fused_bias_gelu(h, bias1) — META'S CUSTOM KERNEL
16. FC2: o = h @ W2 — GEMM (seq, 4096) @ (4096, 1024) = (seq, 1024)
17. Residual: x = x + o
```

## 3. GEMM Inventory (Per Block)

| # | Operation | M | N | K | A Type | B Type | FLOPs |
|---|-----------|---|---|---|--------|--------|-------|
| 1 | Q projection | seq | 1024 | 1024 | BF16 | BF16 | 2×seq×1024² |
| 2 | K projection | seq | 1024 | 1024 | BF16 | BF16 | 2×seq×1024² |
| 3 | V projection | seq | 1024 | 1024 | BF16 | BF16 | 2×seq×1024² |
| 4 | Q@K^T | 16 | seq | 64 | BF16 | BF16 | 2×16×seq²×64 |
| 5 | P@V | 16 | 64 | seq | BF16 | BF16 | 2×16×seq×64×seq |
| 6 | Output proj | seq | 1024 | 1024 | BF16 | BF16 | 2×seq×1024² |
| 7 | FC1 | seq | 4096 | 1024 | BF16 | BF16 | 2×seq×4096×1024 |
| 8 | FC2 | seq | 1024 | 4096 | BF16 | BF16 | 2×seq×1024×4096 |

**Total GEMMs per block:** 8
**Total per ViT:** 8 × 32 = **256 GEMMs** (128 QKV/O + 64 attention + 64 MLP)

### FLOPs per Block (seq=1024)
- QKV projection: 3 × 2 × 1024 × 1024² = 6.4 GFLOP
- Attention (Q@K^T + P@V): 2 × 16 × 1024² × 64 = 2.1 GFLOP
- Output projection: 2 × 1024 × 1024² = 2.1 GFLOP
- MLP: 2 × 1024 × 4096 × 1024 × 2 = 8.6 GFLOP
- **Total per block: ~19.2 GFLOP**
- **Total per ViT: 32 × 19.2 = 614 GFLOP**

## 4. CUTLASS Kernel Assignments

### QKV Projection — Fused Triple GEMM
```
Option A: Single GEMM with N=3072
  Input: X (seq, 1024), W_qkv (1024, 3072)
  Output: Q||K||V (seq, 3072)
  Split in epilogue: 3 × (seq, 1024)
  
  CUTLASS: Standard GemmUniversal, tile (128, 128, 64)
  Epilogue: Custom EVT that splits output into 3 tensors

Option B: Three separate GEMMs
  Simpler but 3× kernel launch overhead
  CUTLASS: 3 × GemmUniversal
```
**Recommendation:** Option A — saves 2 kernel launches per block = 64 launches saved.

### Attention Q@K^T — GEMM Batch (16 heads)
```
Batch GEMM: 16 independent (seq × 64) @ (64 × seq) = (seq × seq)
Or: Grouped GEMM with 16 groups, each (seq, seq, 64)

CUTLASS: GemmGrouped with 16 problems
Tile: (64, 64, 32) — small for per-head computation
Alternative: Treat as (seq, 16*seq, 64) with head stride
```

### Attention P@V — GEMM Batch (16 heads)
```
Same structure as Q@K^T
Batch GEMM: 16 independent (seq × seq) @ (seq × 64) = (seq × 64)
```

### Output Projection
```
Standard GEMM: (seq, 1024) @ (1024, 1024)
Tile: (128, 128, 64)
Epilogue: fusion with residual connection (+ x_residual)
```

### MLP FC1 + addmm_act — THE BIG BOTTLENECK
```
GEMM: (seq, 1024) @ (1024, 4096) = (seq, 4096)
Then: bias + GELU (Meta's addmm_act fuses these)

CUTLASS approach:
  GemmUniversal with EVT: bias broadcast + GELU in epilogue
  
  Using EVT:
    Sm90EVT<Sm90Compute<gelu, float>,       // GELU
      Sm90EVT<Sm90Compute<plus, float>,     // + bias
        Sm90AccFetch,                         // GEMM result
        Sm90RowBroadcast<float>               // bias vector (4096,)
      >
    >
  
  This replicates addmm_act in CUTLASS!
```

### MLP FC2
```
GEMM: (seq, 4096) @ (4096, 1024) = (seq, 1024)
Tile: (128, 256, 64) — wider N benefits from larger tile
Epilogue: residual connection
```

## 5. Non-GEMM Operations

### RoPE (Rotary Position Embeddings)
```
Element-wise: Q[i] *= exp(i * θ * pos)
Complex: Q_complex *= rotary_complex(pos)
Not a GEMM — custom CUDA kernel or fused into QKV epilogue
```

**Fusion strategy:** Apply RoPE in the QKV projection epilogue:
```
GEMM(QKV) → EVT: split Q, K, V → apply RoPE to Q, K → store
```
CUTLASS EVT supports custom compute — we'd add a `RotaryEmbed` compute node.

### LayerNorm
```
Row-wise: mean, var per row, then normalize
Not a GEMM — custom CUDA kernel
CUTLASS has: rmsnorm.py example (CuTe DSL)
```

### Softmax
```
Row-wise: exp(x - max) / sum(exp(x - max))
In Flash Attention: fused into the attention mainloop
```

