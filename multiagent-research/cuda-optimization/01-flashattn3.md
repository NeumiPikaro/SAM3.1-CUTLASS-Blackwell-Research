# 01-flashattn3.md — Key Findings Summary

## Source
- **File**: `literature/attention/01-flashattn3.md`
- **Author**: Tri Dao et al. (Jul 2024)
- **Paper**: arXiv:2407.08608 "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"

## Relevance Score: ★★★★★ (5/5)

## Applicability to SAM 3.1

| Factor | Value |
|--------|-------|
| **Priority** | High — core optimization target |
| **Head configuration** | d=64 is FA-3's best case |
| **Precision** | FP8 provides 2× throughput with minimal accuracy loss |
| **Sequence length** | N=1024 is short but sufficient for good occupancy |
| **Attention type** | Non-causal is the faster path |
| **13× speedup achievable?** | **Yes** — combining FA tiling (5-10×), FP8 (2×), async overlap (1.5-2×) |

## Key Findings

1. FA-3 achieves 75% Tensor Core utilization on H100 (740 TFLOPS FP16) vs FA-2's 35%
2. Ping-pong warp scheduling overlaps softmax with GEMM → 8.8% improvement
3. WGMMA + TMA async overlap enables 3 operations in flight simultaneously
4. FP8 with incoherent processing gives 2.6× lower quantization error
5. d=64 non-causal outperforms cuDNN — exactly SAM 3.1's configuration
6. FA-4 (CuTeDSL) targets Blackwell directly — use for RTX 5060

## Recommended Implementation
- Use `flash-attn-4[cu13]` for Blackwell RTX 5060
- FP8 inference with per-block quantization
- Skip incoherent processing (vision tasks, fewer outliers)
- Fused QKV projection + attention + output projection kernel
