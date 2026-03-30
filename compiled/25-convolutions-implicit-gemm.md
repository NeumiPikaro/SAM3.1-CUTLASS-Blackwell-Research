# CUTLASS Convolutions & Implicit GEMM for SAM 3.1

## 1. Convolution as GEMM

### Explicit GEMM (im2col)
Transform input patches into columns, then do GEMM:
```
Input: (N, C, H, W) → im2col → (N×H_out×W_out, C×K_h×K_w)
Filter: (K, C, K_h, K_w) → reshape → (K, C×K_h×K_w)
GEMM: (N×H_out×W_out, C×K_h×K_w) @ (C×K_h×K_w, K) = (N×H_out×W_out, K)
```
Problem: im2col creates large intermediate tensor (memory overhead).

### Implicit GEMM
CUTLASS computes GEMM without materializing im2col:
```
Threadblock (tb_m, tb_n) maps to:
  - tb_m → output spatial position (h_out, w_out)
  - tb_n → output channel k
  - Inner loop iterates over (c, kh, kw) — input channels and filter positions
  
Each thread computes: output[h,w,k] += input[h+kh, w+kw, c] × filter[k, c, kh, kw]
Address computation happens on-the-fly — no im2col buffer needed
```

## 2. SAM 3.1 FPN Convolutions

### FPN Architecture
```
ViT output: (H/14, W/14, 1024)
↓ Conv2d(1024→256, kernel=1×1, stride=1)
↓ Upsample 2×
Level 1: (H/7, W/7, 256)
↓ Conv2d(256→256, kernel=3×3, stride=1, pad=1) + Upsample 2×
Level 2: (H/3.5, W/3.5, 256)
↓ Conv2d(256→256, kernel=3×3, stride=1, pad=1)
Level 3: (H/3.5, W/3.5, 256)
```

### CUTLASS Implicit GEMM for 1×1 Conv
1×1 convolution is exactly a GEMM:
```
Input: (N×H×W, C) — flatten spatial
Filter: (C, K)
GEMM: (N×H×W, C) @ (C, K) = (N×H×W, K)
```
Use standard CUTLASS GemmUniversal directly.

### CUTLASS Implicit GEMM for 3×3 Conv
```cpp
// Using CUTLASS implicit GEMM convolution
using Conv2d = cutlass::conv::device::Conv2dUniversalAdapter<
    cutlass::conv::kernel::Conv2dUniversal<
        cutlass::conv::collective::CollectiveConv<
            cutlass::arch::Sm100,
            cutlass::arch::OpClassTensorOp,
            cutlass::conv::Operator::kFprop,
            cutlass::layout::TensorNHWC,
            cutlass::bfloat16_t,
            cutlass::layout::TensorNHWC,
            cutlass::bfloat16_t,
            cutlass::layout::TensorNHWC,
            cutlass::bfloat16_t,
            float,
            cutlass::layout::TensorNHWC,
            cutlass::bfloat16_t,
            Shape<_128, _128, _64>,   // Threadblock tile
            Shape<_1, _1, _1>,        // Cluster
            cutlass::gemm::collective::StageCountAuto,
            cutlass::gemm::collective::KernelScheduleAuto
        >,
        cutlass::epilogue::collective::DefaultEpilogue<...>
    >
>;

// Launch
Conv2d conv_op;
Conv2d::Arguments args{
    {N, H, W, C},           // Input shape
    {K, C, 3, 3},           // Filter shape  
    {0, 0, 0, 0},           // Padding
    {1, 1},                 // Stride
    {1, 1},                 // Dilation
    {N, H, W, K}            // Output shape
};
conv_op(args, workspace, stream);
```

## 3. Upsampling

Bilinear upsampling is not a GEMM — it's an element-wise interpolation:
```
Output[h, w] = bilinear_interp(Input, h/scale, w/scale)
```

### Options
1. **PyTorch F.interpolate** — uses cuDNN internally, optimized
2. **Custom CUDA kernel** — simple, low overhead
3. **Skip** — fold into next Conv2d's padding/stride if possible

**Recommendation:** Use PyTorch's F.interpolate — not worth customizing.

## 4. Depthwise Separable Convolution

If SAM 3.1 ever uses depthwise conv (e.g., MobileNet-style FPN):
```
Depthwise: each channel convolved independently
  Compute: C × K_h × K_w × H_out × W_out
  Not a standard GEMM — custom kernel or grouped conv

CUTLASS grouped GEMM can handle this:
  Group = C channels
  Each group: (H_out×W_out, 1) @ (1, K_h×K_w)
```

## 5. Conv2d Performance on Blackwell

For SAM 3.1 FPN (3 Conv2d layers):
```
1×1 Conv: (H×W/196, 1024) @ (1024, 256) = ~1.3 MFLOP per 1024×1024 image
3×3 Conv: (H×W/49, 256) @ (256×9, 256) = ~2.6 MFLOP per layer
Total FPN: ~6.5 MFLOP — negligible compared to ViT (614 GFLOP)
```
FPN is <1% of total compute — not worth heavy optimization.

