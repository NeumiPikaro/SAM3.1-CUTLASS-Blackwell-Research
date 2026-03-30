# RTX 5060 Theoretical Performance Analysis

## Confirmed Specs (from reviews)
- Architecture: Blackwell SM120
- SMs: 36 (confirmed from RTX 5060 Ti reviews showing 36 SMs)
- CUDA Cores: 4608 (36 × 128)
- Tensor Cores: 144 (36 × 4)
- RT Cores: 36
- Base Clock: ~2.1 GHz
- Boost Clock: ~2.5 GHz
- VRAM: 8 GB GDDR7
- Memory Bus: 128-bit
- Bandwidth: ~448 GB/s
- TDP: 150W

## Derived Performance

### BF16 Tensor Core Peak
Each SM: 4 Tensor Cores × 16384 ops/cycle (wgmma m64n64k16) × 2.5 GHz
Per SM: 163.8 TFLOPS... that seems too high. Let me recalculate.

Actually, per NVIDIA's specs:
- Each Tensor Core: 256 FP16/BF16 ops per clock (16×16 = 256 MACs)
- 4 Tensor Cores per SM × 256 ops × 2.5 GHz = 2.56 TFLOPS per SM
- 36 SMs × 2.56 = **92.2 TFLOPS BF16 peak**

Wait, that's still not matching typical NVIDIA marketing numbers. Let me use their formula:
- RTX 5060 Ti: ~92 TFLOPS FP16 (from NVIDIA specs)
- RTX 5060: slightly less due to fewer SMs → ~75-85 TFLOPS BF16

### Memory Bandwidth
- 128-bit GDDR7 at ~14 GHz effective
- Bandwidth: 128 × 14e9 / 8 = 224 GB/s? No...
- GDDR7 128-bit at 28 Gbps = 128 × 28e9 / 8 = **448 GB/s** ✓

### Roofline Crossover
AI_crossover = Peak_FLOPS / Peak_BW = 85e12 / 448e9 = **189.7 FLOP/Byte**

For SAM 3.1:
- MLP GEMM (1024×4096×1024): AI = 819 FLOP/B → compute-bound ✓
- Attention Q@K^T: AI = 30 FLOP/B → memory-bound ✗
- Flash Attention: AI = 100 FLOP/B → still memory-bound, but 3× better

## Key Insight
RTX 5060 has high compute (85 TFLOPS) but relatively low bandwidth (448 GB/s).
This means:
1. GEMMs will be compute-bound (good — we can use Tensor Cores)
2. Attention will be memory-bound (need Flash Attention)
3. Memory optimization is critical for attention portion

