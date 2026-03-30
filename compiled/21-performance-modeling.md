# Performance Modeling & Roofline Analysis

## 1. RTX 5060 Theoretical Peak

### Compute
- SM count: 36 (estimated)
- Tensor Core ops per SM per cycle (BF16): 16384 (64x16x16)
- Clock: 2500 MHz (boost)
- Peak BF16 TFLOPS: 36 * 16384 * 2.5 = **1,474 TFLOPS**
- Peak FP8 TFLOPS: 36 * 32768 * 2.5 = **2,949 TFLOPS**

### Memory
- GDDR7 bandwidth: 448 GB/s
- L2 cache: 32 MB at ~5 TB/s
- SMEM per SM: 228 KB at ~100 TB/s

## 2. Roofline Model

### Compute-Bound Regime
AI > (Peak FLOPS / Peak BW) = 1474000 / 448 = **3290 FLOP/Byte**
- All large GEMMs (ViT QKV, MLP) are compute-bound ✓
- Tensor Core utilization is the bottleneck

### Memory-Bound Regime
AI < 3290 FLOP/Byte
- Attention (without Flash): AI=30 → memory-bound
- Small GEMMs (DETR decoder): may be memory-bound
- Solution: Flash Attention increases AI to 60-100

## 3. Per-Component Latency Model

### ViT Attention (per block)
```
QKV GEMM: 3 * (2 * 1024 * 1024 * 1024) / (1474e12 * 0.7) = 2.9 ms
Q@K^T:    16 * (2 * 1024 * 1024 * 64) / (1474e12 * 0.6) = 1.5 ms
P@V:      16 * (2 * 1024 * 64 * 1024) / (1474e12 * 0.6) = 1.5 ms
Output:   (2 * 1024 * 1024 * 1024) / (1474e12 * 0.7) = 1.0 ms
RoPE:     (1024 * 1024 * 2) ops / (448e9) = 0.005 ms (negligible)
Softmax:  (1024 * 1024 * 16) ops / (448e9) = 0.04 ms
Total per block: ~7 ms
32 blocks: ~224 ms
```

### ViT MLP (per block)
```
FC1 GEMM: (2 * 1024 * 4096 * 1024) / (1474e12 * 0.7) = 3.9 ms
GELU:     (1024 * 4096) ops / (448e9) = 0.01 ms (fused)
FC2 GEMM: (2 * 1024 * 1024 * 4096) / (1474e12 * 0.7) = 3.9 ms
Total per block: ~8 ms
32 blocks: ~256 ms
```

### Total ViT Estimate
- Attention: 224 ms
- MLP: 256 ms
- LayerNorms: ~32 ms
- **Total: ~512 ms**

### With Optimizations
- Epilogue fusion (no separate bias/GELU): -20% → 410 ms
- FP8 weights: -40% → 246 ms
- CUDA graphs + PDL: -5% → 234 ms
- **Optimized ViT: ~234 ms**

## 4. End-to-End Estimate
```
ViT Image Encoder: 234 ms (optimized)
CLIP Text Encoder: 60 ms (24 layers, seq=77, much faster)
DETR Encoder: 45 ms (6 layers, d=256)
DETR Decoder: 30 ms (6 layers, M=100)
FPN Neck: 15 ms
Seg Head: 6 ms
Total: ~390 ms
```

## 5. Comparison Table
```
Platform          | Model Version    | Precision | Resolution | Latency
------------------|------------------|-----------|------------|--------
Tesla T4 (ours)   | Meta native 3.1  | BF16      | 1024       | 2897 ms
RTX 5060 (proj.)  | Meta native 3.1  | BF16      | 1024       | ~650 ms
RTX 5060 (proj.)  | Meta native 3.1  | FP8       | 1024       | ~390 ms
H100 (SAM3-TRT)   | HF transformers  | FP16      | 4K         | 24.9 ms
RTX 3090 (SAM3-TRT)| HF transformers | FP16      | 4K         | 75 ms
```
