# MatMul-Free Transformers & Alternative Architectures

## MatMul-Free Transformer (Wang et al., 2024)
Replaces ALL matrix multiplications with element-wise ops and additions.
Uses ternary weights {-1, 0, +1}. 2-4x faster on some hardware.
**SAM 3.1:** Not applicable without retraining.

## Mamba / State Space Models
O(N) complexity replacing attention with selective SSMs.
No GEMM for sequence processing.
**SAM 3.1:** Not applicable — architecture is fixed.

## Linear Attention
Kernel trick for O(N) attention. Quality trade-off vs standard.
**SAM 3.1:** Not applicable without retraining.

## Bottom Line
None apply to native SAM 3.1. Only path to <50ms is kernel-level optimization:
1. CUTLASS GEMM with optimal Blackwell tiles
2. Flash Attention / FlashInfer
3. Kernel fusion (EVT, megakernel)
4. CUDA Graphs + PDL
5. Hardware-specific tuning
