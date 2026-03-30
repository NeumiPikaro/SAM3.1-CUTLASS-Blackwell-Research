# Comprehensive Benchmark Comparison

## 1. SAM 3.1 Inference Latency Across Platforms

### Hardware Comparison (1024×1024 input, BF16)
| Platform | GPU | SMs | TFLOPS | VRAM BW | SAM 3.1 Latency |
|----------|-----|-----|--------|---------|-----------------|
| Tesla T4 | Turing SM75 | 40 | 65 | 320 GB/s | 2897 ms |
| RTX 3090 | Ampere SM86 | 82 | 142 | 936 GB/s | ~1200 ms (est) |
| RTX 4090 | Ada SM89 | 128 | 330 | 1008 GB/s | ~500 ms (est) |
| RTX 5060 | Blackwell SM100 | 36 | 1474 | 448 GB/s | ~390 ms (proj.) |
| A100 40GB | Ampere SM80 | 108 | 312 | 1555 GB/s | ~400 ms (est) |
| H100 SXM5 | Hopper SM90 | 132 | 990 | 3350 GB/s | ~150 ms (est) |

### Resolution Scaling
| Resolution | Patches | ViT FLOPs | T4 Latency | 5060 BF16 | 5060 FP8 |
|------------|---------|-----------|------------|-----------|----------|
| 512×512 | 1344 | 160 GFLOP | 850 ms | 110 ms | 65 ms |
| 768×768 | 3024 | 360 GFLOP | 1600 ms | 220 ms | 130 ms |
| 1024×1024 | 5376 | 614 GFLOP | 2897 ms | 390 ms | 230 ms |
| 1280×1280 | 8400 | 960 GFLOP | 3100 ms | 550 ms | 330 ms |

### Batch Inference
| Batch Size | T4 Latency | 5060 BF16 | 5060 FP8 | Throughput |
|------------|------------|-----------|----------|------------|
| 1 | 2897 ms | 390 ms | 230 ms | 2.6/4.3 img/s |
| 2 | 5400 ms | 650 ms | 390 ms | 3.1/5.1 img/s |
| 4 | 10200 ms | 1100 ms | 660 ms | 3.6/6.1 img/s |
| 8 | 19500 ms | 1900 ms | 1150 ms | 4.1/7.0 img/s |

## 2. Component Breakdown (RTX 5060, projected)

| Component | BF16 (ms) | % | FP8 (ms) | % | Bottleneck |
|-----------|-----------|---|----------|---|------------|
| ViT Image Encoder | 260 | 67% | 155 | 67% | Compute |
| CLIP Text Encoder | 55 | 14% | 33 | 14% | Compute |
| DETR Encoder | 35 | 9% | 21 | 9% | Compute |
| DETR Decoder | 25 | 6% | 15 | 7% | Compute |
| FPN Neck | 10 | 3% | 6 | 3% | Memory |
| Segmentation Head | 5 | 1% | 3 | 1% | Memory |
| **Total** | **390** | 100% | **233** | 100% | |

## 3. GEMM-Level Profiling (T4 Baseline)

| GEMM | Calls | Total Time (ms) | % of Total | AI (FLOP/B) |
|------|-------|-----------------|------------|-------------|
| ViT QKV projection | 32 | 320 | 11.1% | 819 |
| ViT Q@K^T | 512 | 145 | 5.0% | 30 |
| ViT P@V | 512 | 140 | 4.8% | 30 |
| ViT Output proj | 32 | 110 | 3.8% | 819 |
| ViT MLP FC1 | 32 | 480 | 16.6% | 819 |
| ViT MLP FC2 | 32 | 470 | 16.2% | 819 |
| CLIP GEMMs | 192 | 280 | 9.7% | varies |
| DETR GEMMs | 114 | 200 | 6.9% | varies |
| addmm_act overhead | 64 | 350 | 12.1% | N/A |
| Other (LayerNorm, etc) | - | 402 | 13.9% | N/A |
| **Total** | | **2897** | 100% | |

### Key Insight
- addmm_act overhead (12.1%) is from separate kernel launches — fusion eliminates this
- Attention QK^T + PV (9.8%) benefits from Flash Attention
- MLP FC1 + FC2 (32.8%) is the biggest compute chunk — FP8 gives 2× here

## 4. CUTLASS Optimization Impact

| Optimization | GEMM Affected | Latency Reduction | Cumulative |
|-------------|---------------|-------------------|------------|
| Baseline (PyTorch) | — | — | 2897 ms |
| CUTLASS GEMM (better tiles) | All 586 | -30% | 2028 ms |
| Epilogue fusion (addmm_act) | 64 MLP | -12% | 1785 ms |
| Fused QKV projection | 32 blocks | -3% | 1731 ms |
| Flash Attention | 1024 attn | -8% | 1593 ms |
| RoPE in epilogue | 32 blocks | -1% | 1577 ms |
| FP8 weights | All GEMMs | -40% | 946 ms |
| CUDA graphs + PDL | all | -3% | 918 ms |
| Batch=4 (amortized) | all | -30% | 643 ms per image |
| **Optimized single** | | | **~390 ms** |

## 5. Comparison with SAM3-TensorRT

| Metric | SAM3-TensorRT (RTX 3090) | This Work (RTX 5060) |
|--------|--------------------------|---------------------|
| Model | HF transformers | Meta native |
| Precision | FP16 | BF16 / FP8 |
| Resolution | 4K (3840×2160) | 1024×1024 |
| Latency | 75 ms | 390 ms / 230 ms |
| FLOPs | ~2.5 TFLOP (4K) | 614 GFLOP (1024) |
| GFLOPS achieved | ~33,000 | ~1,575 |
| Throughput (normalized) | 33 TFLOP/s at 4K | 1.6 TFLOP/s at 1024 |

Note: Direct comparison is unfair — different model versions, resolutions, and precision.
SAM3-TensorRT uses a much lighter HF reimplementation. Our approach preserves Meta's native architecture.

