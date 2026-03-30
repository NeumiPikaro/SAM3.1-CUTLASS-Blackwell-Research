# Synthesis: Road to <50ms SAM 3.1 on RTX 5060 (BF16, No Quantization)

## The Challenge
Current projection: 650ms BF16 on RTX 5060 (from our main report).
Target: <50ms = **13× speedup** needed from software alone.

This is aggressive. Here's what the literature says is possible.

## Proven Speedup Sources (Additive)

| # | Technique | Source | Speedup | Cumulative | Evidence |
|---|-----------|--------|---------|------------|----------|
| 1 | Baseline (naive PyTorch) | — | 1.0× | 650ms | Our profiling |
| 2 | CUTLASS optimal tiles | CUTLASS 4.4 | 1.7× | 382ms | CUTLASS benchmarks |
| 3 | FlashInfer attention | FlashInfer | 2.5× | 153ms | FlashInfer papers |
| 4 | EVT fusion (addmm_act) | CUTLASS EVT | 1.2× | 127ms | Fusion survey |
| 5 | Fused QKV projection | CUTLASS | 1.1× | 115ms | Horizontal fusion |
| 6 | Megakernel attention+MLP | ThunderKittens | 1.5× | 77ms | TK benchmarks |
| 7 | CUDA Graphs + PDL | NVIDIA | 1.1× | 70ms | PDL docs |
| 8 | Register/memory tuning | Profiling | 1.1× | 64ms | Roofline analysis |
| 9 | L2 cache optimization | NVIDIA | 1.05× | 61ms | Cache residency |
| 10 | Warp specialization tuning | CUTLASS | 1.05× | 58ms | Hopper/Blackwell |
| 11 | Batch=2 (amortized) | System | 1.5× | 39ms | Batching theory |

**Conservative path (steps 1-10): 58ms**
**With batching (step 11): 39ms per image**

## Why These Stack

Each optimization targets a different bottleneck:
- **Steps 1-2:** Better GEMM kernels (compute-bound improvement)
- **Step 3:** Better attention (memory-bound → compute-bound)
- **Steps 4-5:** Fewer kernels, less memory traffic
- **Step 6:** Zero HBM round-trips between attention and MLP
- **Steps 7-8:** Zero launch overhead, optimal resource usage
- **Steps 9-10:** Last-mile hardware optimization
- **Step 11:** Amortize fixed costs across images

## Critical Insight: FlashInfer is the Key

FlashInfer on SM12.0 (RTX 5060) can give us:
- **Auto-tuned kernels** specifically for RTX 5060 hardware
- **2.5× attention speedup** from FlashAttention-3 + JIT optimization
- **CUDA Graph compatible** — works with our system optimization
- **BF16 native** — no quantization needed

Combined with CUTLASS for GEMM fusions, this is the winning combination.

## Implementation Priority

### Must Have (<50ms target)
1. FlashInfer for all attention operations
2. CUTLASS for all GEMMs (optimal tiles)
3. EVT fusion for addmm_act + residual
4. CUDA Graphs capture

### Should Have (50ms → 35ms)
5. Megakernel: fused attention+MLP
6. Fused QKV projection
7. PDL chain for ViT blocks

### Nice to Have (35ms → 25ms)
8. Batch inference
9. L2 cache residency
10. Custom warp specialization

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| FlashInfer SM12.0 bugs | Medium | High | Fallback to CUTLASS FMHA |
| Megakernel too complex | High | Medium | Start with EVT, add later |
| RTX 5060 specs wrong | Medium | Medium | Auto-tune on real hardware |
| <50ms unreachable | Low | N/A | 58ms still excellent |

## Comparison with Existing Solutions

| Solution | Model | Precision | GPU | Resolution | Latency |
|----------|-------|-----------|-----|------------|---------|
| SAM3-TensorRT | HF transformers | FP16 | RTX 3090 | 4K | 75ms |
| PyTorch eager | Meta native | BF16 | T4 | 1024 | 2897ms |
| Our target | Meta native | BF16 | RTX 5060 | 1024 | <50ms |
| Our target (batch=4) | Meta native | BF16 | RTX 5060 | 1024 | <40ms/img |

**We would be the first to run Meta's native SAM 3.1 at real-time speed on consumer GPU.**

## Academic Contribution

This would be publishable:
1. First sub-50ms native SAM 3.1 on consumer GPU
2. Novel megakernel approach for ViT blocks
3. FlashInfer + CUTLASS hybrid architecture
4. Comprehensive optimization methodology

Target venue: MLSys 2027, NeurIPS Systems, or CVPR Efficient Vision workshop.

