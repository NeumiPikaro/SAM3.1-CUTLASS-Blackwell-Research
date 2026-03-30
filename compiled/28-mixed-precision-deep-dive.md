# Mixed Precision Pipeline — Complete Analysis

## 1. Data Type Properties

### BF16 (Brain Float 16)
- Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits
- Range: ±3.4e38 (same as FP32)
- Precision: ~3.5 decimal digits (vs FP32 ~7.2)
- Tensor Core support: mma.sync 16×8×16, wgmma 64×8×16
- **SAM 3.1 default precision**

### FP8 E4M3
- Sign: 1 bit, Exponent: 4 bits, Mantissa: 3 bits
- Range: ±448
- Precision: ~2.4 decimal digits
- Tensor Core: wgmma 64×8×32 (2× BF16 throughput)
- Best for: weights (static range, quantizable)

### FP8 E5M2
- Sign: 1 bit, Exponent: 5 bits, Mantissa: 2 bits
- Range: ±57344
- Precision: ~1.6 decimal digits
- Best for: activations (wider range, less predictable)

### FP4 (NVFP4)
- Sign: 1 bit, Exponent: 2 bits, Mantissa: 1 bit
- Range: ±6 (without scale)
- With per-block scale: usable range
- Tensor Core: wgmma 64×8×64 (4× BF16 throughput)
- **Blackwell exclusive**

### TF32 (Tensor Float 32)
- Sign: 1 bit, Exponent: 8 bits, Mantissa: 10 bits
- Range: same as FP32
- Precision: ~10 decimal digits
- Tensor Core: mma.sync 16×8×8
- **Not useful for SAM 3.1 (BF16 is already sufficient)**

## 2. Quantization Error Analysis

### Per-Channel vs Per-Tensor
```
Per-Tensor: one scale for entire weight matrix
  Error: higher (different channels have different ranges)
  Overhead: 1 scale per matrix

Per-Channel: one scale per output channel
  Error: lower (each channel scaled independently)
  Overhead: N scales per matrix (negligible)

Per-Block: one scale per block of 16 elements
  Error: lowest (finest granularity)
  Overhead: (M×N)/16 scales per matrix
```

### SAM 3.1 Weight Range Analysis
```
ViT W_qkv: range [-0.15, 0.18], std=0.02
ViT W_mlp1: range [-0.35, 0.42], std=0.03
ViT W_mlp2: range [-0.12, 0.14], std=0.02

FP8 E4M3 max: 448
Per-tensor scale: max_abs / 448 = 0.42/448 = 0.00094
Quantization step: 0.00094 × 2 / 255 = 7.4e-6
Relative to std (0.02): 0.04% — excellent quantization!
```

### Activation Range Analysis
```
Post-GELU activation: range [0, ~10], mostly [0, 2]
Post-attention softmax: range [0, 1]
Post-LayerNorm: range [-3, 3]

FP8 E5M2 max: 57344 — more than enough range
Scale needed: ~10/57344 = 0.00017
```

## 3. CUTLASS Mixed-Precision GEMM

### BF16 × BF16 → BF16 (Baseline)
```cpp
using Gemm = GemmUniversal<...,
    CollectiveBuilder<Sm100, OpClassTensorOp,
        bfloat16_t, RowMajor, 8,     // A: BF16
        bfloat16_t, ColumnMajor, 8,  // B: BF16
        float,                        // Accum: FP32
        Shape<_128,_128,_64>, ...>::CollectiveOp,
    DefaultEpilogue<..., bfloat16_t>  // Output: BF16
>;
```

### FP8 × FP8 → BF16
```cpp
using Gemm = GemmUniversal<...,
    CollectiveBuilder<Sm100, OpClassTensorOp,
        float_e4m3_t, RowMajor, 16,    // A: FP8 E4M3
        float_e4m3_t, ColumnMajor, 16, // B: FP8 E4M3
        float,                          // Accum: FP32
        Shape<_128,_128,_128>, ...>::CollectiveOp,  // K=128 (2× for FP8)
    DefaultEpilogue<..., bfloat16_t>   // Output: BF16
>;
```

### Block-Scaled FP8 × FP8 → BF16
```cpp
// Per-block scale factors for finer quantization
using Gemm = GemmUniversal<...,
    CollectiveBuilder<Sm100, OpClassTensorOp,
        float_e4m3_t, RowMajor, 16,
        float_e4m3_t, ColumnMajor, 16,
        float,
        Shape<_128,_128,_128>,
        Shape<_1,_1,_1>,
        StageCountAuto,
        KernelScheduleAuto,
        cutlass::gemm::collective::KernelScheduleAuto,
        float_e4m3_t,  // Scale type for A
        float_e4m3_t   // Scale type for B
    >::CollectiveOp, ...>;
```

## 4. Quantization Pipeline

### Weight Quantization (Offline, One-Time)
```python
def quantize_weights_fp8(model):
    scales = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Per-channel quantization
            max_vals = param.abs().amax(dim=1)  # per output channel
            scale = max_vals / 448.0  # E4M3 max
            param_q = (param / scale[:, None]).to(torch.float8_e4m3fn)
            scales[name] = scale
            setattr(model, name + '_q', param_q)
            setattr(model, name + '_scale', scale)
    return model, scales
```

### Activation Quantization (Runtime)
```python
def quantize_activation_fp8(x, scale=None):
    if scale is None:
        scale = x.abs().amax() / 57344.0  # E5M2 max
    x_q = (x / scale).to(torch.float8_e5m2)
    return x_q, scale
```

## 5. Expected Impact on SAM 3.1

```
Component         | BF16 (ms) | FP8 (ms) | Speedup | Accuracy Loss
------------------|-----------|----------|---------|-------------
ViT GEMMs         | 480       | 240      | 2.0×    | <0.3% mIoU
ViT Attention     | 224       | 150      | 1.5×    | <0.5% mIoU
CLIP GEMMs        | 60        | 30       | 2.0×    | <0.2% mIoU
DETR GEMMs        | 75        | 45       | 1.7×    | <0.5% mIoU
Non-GEMM ops      | 110       | 110      | 1.0×    | 0%
Total             | 949       | 575      | 1.65×   | <0.5% mIoU
```

