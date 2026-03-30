# Quantization Strategy for SAM 3.1 on Blackwell

## 1. Current State: BF16

SAM 3.1 natively uses BF16:
- ViT weights: BF16 (848M params × 2 bytes = 1.7 GB)
- Text encoder: BF16 (CLIP ViT-L, ~600M params × 2 = 1.2 GB)
- DETR encoder/decoder: BF16 (~30M params × 2 = 60 MB)

Total model: ~3.4 GB in BF16. Fits in RTX 5060's 8 GB VRAM.

## 2. FP8 Quantization

### E4M3 vs E5M2
- **E4M3:** 4-bit exponent, 3-bit mantissa — better precision, range ±448
- **E5M2:** 5-bit exponent, 2-bit mantissa — wider range, less precision
- **Recommendation for SAM 3.1:** E4M3 for weights, E5M2 for activations

### CUTLASS FP8 GEMM
```cpp
using CollectiveMainloop = CollectiveBuilder<
  Sm90, OpClassTensorOp,
  float_e4m3_t, RowMajor, 16,    // A: FP8 E4M3
  float_e4m3_t, ColumnMajor, 16, // B: FP8 E4M3
  float,                          // Accumulator: FP32
  Shape<_128,_128,_128>,         // K dimension doubled for FP8
  ...
>::CollectiveOp;
```

### Performance Impact
- FP8 Tensor Core throughput: 2× BF16 on Blackwell
- ViT attention: 2897ms BF16 → **~1500ms FP8** (theoretical)
- Memory: 3.4 GB → **1.7 GB** (half the VRAM)

### Accuracy Impact (Estimated)
- Per-channel quantization: <0.5% mIoU loss on segmentation
- Per-tensor quantization: 1-2% mIoU loss
- SmoothQuant-style: <0.3% mIoU loss

## 3. FP4 (NVFP4/MXFP4) — Blackwell Exclusive

### NVFP4 Format
- 1 sign + 2 exponent + 1 mantissa = 4 bits
- NVIDIA proprietary, optimized for Tensor Cores
- Requires per-block scale factors (16 elements per block)

### CUTLASS FP4 Support (CUTLASS 4.4)
```cpp
// Blackwell SM100 block-scaled FP4 GEMM
using CollectiveMainloop = CollectiveBuilder<
  Sm100, OpClassTensorOp,
  nv_float4_t, RowMajor, 32,     // NVFP4
  nv_float4_t, ColumnMajor, 32,
  float,
  Shape<_128,_128,_128>,
  ...
>::CollectiveOp;
```

### Performance Impact
- FP4 throughput: 4× BF16 on Blackwell
- SAM 3.1 in FP4: **~700ms end-to-end** (theoretical, 4× faster than BF16)
- Memory: 3.4 GB → **0.85 GB**

### Accuracy Impact
- FP4 is aggressive — expect 2-5% mIoU loss without fine-tuning
- With QAT (quantization-aware training): <1% mIoU loss
- **Recommendation:** FP8 for production, FP4 only if speed is critical

## 4. Mixed Precision Pipeline

### Recommended Recipe for RTX 5060
```
Stage 1: ViT Image Encoder
  - Attention GEMM: BF16 inputs → FP32 acc → BF16 output
  - MLP GEMM: BF16 inputs → FP32 acc → BF16 output
  - Future: FP8 weights, BF16 activations

Stage 2: CLIP Text Encoder  
  - Same as ViT

Stage 3: Fusion (Transformer Encoder)
  - Cross-attention: BF16
  - Self-attention: BF16

Stage 4: DETR Decoder
  - Cross-attention: BF16 (small, latency-critical)
  - Self-attention: BF16

Stage 5: Segmentation Head
  - Conv layers: BF16
  - Final output: FP32 (for quality)
```

### CUTLASS Mixed-Dtype GEMM
CUTLASS 4.4 Example 55 shows mixed-type GEMM:
```
A: BF16, B: INT8 → accumulate FP32 → output BF16
A: FP8, B: INT4 → accumulate FP32 → output BF16
```
The narrower type goes through register file, upcast to wider type.

## 5. Blackwell Block-Scaled GEMM

CUTLASS 4.4 introduces block-scaled GEMM:
```
Each block of 16 elements has its own scale factor
D = (A_block * scale_A) @ (B_block * scale_B)
```
This is the quantization scheme used by:
- MX (Microscaling) formats: MXFP4, MXFP8
- NVFP4: NVIDIA's proprietary 4-bit format
- OCP standard formats

### CUTLASS Implementation
```python
# CuTe DSL Blackwell block-scaled GEMM
@cute.kernel
def blockscaled_gemm(gA, gB, gC, scale_A, scale_B):
    # Load scale factors alongside data
    # Hardware applies scale during WGMMA
```
Example: `dense_blockscaled_gemm_persistent.py`

