# SAM 3.1 DETR & Geometry Encoder — Kernel Mapping

## 1. DETR Encoder (Fusion Module)

The DETR encoder fuses image features with text features via cross-attention.

### Architecture
- **6 layers** of transformer encoder
- **d_model:** 256
- **Heads:** 8 (32-dim each)
- **FFN hidden:** 2048 (8× expansion)
- **Cross-attention:** queries=image, keys/values=text

### Per-Layer Operations
```
1. Self-attention on image features:
   Q = img @ Wq — GEMM (HW, 256) @ (256, 256)
   K = img @ Wk — same
   V = img @ Wv — same
   Attn = softmax(Q @ K^T) @ V
   Out = Attn @ Wo — GEMM (HW, 256) @ (256, 256)

2. Cross-attention (image attends to text):
   Q = img @ Wq — GEMM (HW, 256) @ (256, 256)
   K = text @ Wk — GEMM (text_len, 256) @ (256, 256)
   V = text @ Wv — same
   Attn = softmax(Q @ K^T) @ V — GEMM (HW, text_len) @ (text_len, 256)

3. FFN:
   FC1: (HW, 256) @ (256, 2048)
   GELU activation
   FC2: (HW, 2048) @ (2048, 256)
```

### GEMM Inventory (6 layers)
| Operation | M | N | K | Layers | Total GEMMs |
|-----------|---|---|---|--------|-------------|
| Self-attn QKV | HW | 768 | 256 | 6 | 6 (fused) |
| Self-attn Q@K^T | HW | HW | 256 | 6 | 6 |
| Self-attn P@V | HW | 256 | HW | 6 | 6 |
| Self-attn output | HW | 256 | 256 | 6 | 6 |
| Cross-attn Q | HW | 256 | 256 | 6 | 6 |
| Cross-attn KV | text | 512 | 256 | 6 | 6 |
| Cross-attn Q@K^T | HW | text | 256 | 6 | 6 |
| Cross-attn P@V | HW | 256 | text | 6 | 6 |
| FFN FC1 | HW | 2048 | 256 | 6 | 6 |
| FFN FC2 | HW | 256 | 2048 | 6 | 6 |
| **Total** | | | | | **60 GEMMs** |

### CUTLASS Strategy
- Small d_model (256) → small tiles (64, 64, 32)
- HW varies per image resolution → need dynamic problem sizes
- Cross-attention: Q=HW (1024-4096), KV=text_len (77) → **asymmetric tiles**
- FFN: wider N=2048 → use (128, 128, 64) tiles

## 2. DETR Decoder

### Architecture
- **6 layers** of transformer decoder
- **d_model:** 256
- **Heads:** 8 (32-dim each)
- **Queries:** learnable object queries (N=100 by default)
- **Self-attention:** queries attend to each other
- **Cross-attention:** queries attend to encoder output
- **Causal masking:** none (all queries attend to all)

### GEMM Inventory (6 layers)
| Operation | M | N | K | Notes |
|-----------|---|---|---|-------|
| Self-attn QKV | 100 | 768 | 256 | Tiny — may use SIMT |
| Self-attn Q@K^T | 100 | 100 | 256 | Very small |
| Self-attn P@V | 100 | 256 | 100 | Very small |
| Cross-attn Q | 100 | 256 | 256 | Small Q |
| Cross-attn KV | HW | 512 | 256 | Long KV |
| Cross-attn Q@K^T | 100 | HW | 256 | Short Q, long K |
| Cross-attn P@V | 100 | 256 | HW | — |
| FFN FC1 | 100 | 2048 | 256 | Small M |
| FFN FC2 | 100 | 256 | 2048 | — |

**Total: ~54 GEMMs across 6 decoder layers**

### Key Challenge: M=100 is Tiny
With M=100, each GEMM has very few output rows. Options:
1. **StreamK:** Divide K across SMs for better utilization
2. **Small tiles:** (32, 64, 32) — even if suboptimal per-SM, more parallelism
3. **Batch queries:** Treat 100 queries as 10 batches of 10 — more parallelism
4. **cuBLAS:** For tiny GEMMs, cuBLAS may be faster than CUTLASS (less template overhead)

**Recommendation:** Hybrid approach — CUTLASS for encoder (large GEMMs), cuBLAS or custom small-kernel for decoder (tiny GEMMs).

## 3. Geometry Encoder

The geometry encoder encodes point/box prompts into 256-dim vectors.

### Architecture (10 Sub-Modules)
```
1. points_direct_project: Linear(2 → 256) — 3.2 KB
2. boxes_direct_project: Linear(4 → 256) — 5.2 KB
3. points_pool_project: Linear(256 → 256) — 257 KB
4. boxes_pool_project: Conv2d(256 → 256, 7×7) — 12.3 MB
5. points_pos_enc_project: Linear(256 → 256) — 257 KB
6. boxes_pos_enc_project: Linear(256 → 256) — 259 KB
7. final_proj: Linear(256 → 256) — 257 KB
8. encoder_transformer: 3-layer cross-attn transformer — 18.1 MB
9. label_embed: Embedding(2, 256) — 2.2 KB
10. cls_embed: Embedding(1, 256) — 1.2 KB
```

### CUTLASS Mapping
- Linear projections: Standard GEMM, tiny (1-4 rows × 256 cols)
- Conv2d: Implicit GEMM — CUTLASS conv2d support
- Transformer: Same as DETR encoder but smaller (3 layers, 256-dim)
- Embeddings: Lookup table — not a GEMM

### Challenge
Geometry encoder processes few prompts (1-10 points/boxes). GEMMs are tiny (M=1-10).
**Strategy:** Batch all prompts into single GEMM, or use vectorized element-wise ops.

## 4. FPN Neck

The FPN (Feature Pyramid Network) converts ViT output to multi-scale features.

### Architecture
```
ViT output: (H/14, W/14, 1024) — single scale
↓ Conv2d(1024 → 256, 1×1) + Upsample 2×
Level 1: (H/7, W/7, 256)
↓ Conv2d(256 → 256, 3×3) + Upsample 2×
Level 2: (H/3.5, W/3.5, 256)  
↓ Conv2d(256 → 256, 3×3)
Level 3: (H/3.5, W/3.5, 256)
```

### CUTLASS Mapping
- 1×1 Conv2d: GEMM (implicit or explicit)
- 3×3 Conv2d: Implicit GEMM with CUTLASS conv2d support
- Upsample: Bilinear interpolation — custom CUDA kernel

## 5. Text Encoder (CLIP ViT-L/14)

### Architecture
- 24 transformer layers
- d_model: 1024
- 16 heads × 64-dim
- MLP: 1024 → 4096 → 1024
- Sequence length: 77 (fixed for CLIP)

### Kernel Mapping
Same as ViT layers but with seq=77 (tiny):
- QKV: (77, 3072) — very small
- Attention: (77, 77) — tiny
- MLP: (77, 4096) — small M

**Strategy:** Batch text encoder with ViT, or use small-tile GEMM.

## 6. Total SAM 3.1 GEMM Count

| Component | GEMMs per Layer | Layers | Total |
|-----------|----------------|--------|-------|
| ViT Image Encoder | 8 | 32 | 256 |
| CLIP Text Encoder | 8 | 24 | 192 |
| DETR Encoder | 10 | 6 | 60 |
| DETR Decoder | 9 | 6 | 54 |
| Geometry Encoder | ~5 | 3 | ~15 |
| FPN Neck | ~3 | 3 | ~9 |
| **Total** | | | **~586 GEMMs** |

