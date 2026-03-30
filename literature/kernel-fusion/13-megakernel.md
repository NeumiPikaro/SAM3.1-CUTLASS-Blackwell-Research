# Megakernel Approach for SAM 3.1

## The Concept
Entire ViT block (attention + MLP + residual + LayerNorm) in ONE CUDA kernel:
- Zero kernel launch overhead
- Zero HBM round-trips for intermediates
- Attention and MLP compute overlap via warp specialization

## ThunderKittens Implementation
HazyResearch achieves 86% Tensor Core utilization with fused attention kernels on H100.
Their FA-3 implementation in ThunderKittens is <100 lines and achieves 855 TFLOPS.

## SAM 3.1 Megakernel Design
```
14 warps per CTA:
  Warps 0-3 (DMA): Load X, W_qkv, W1, W2 from global
  Warps 4-9 (Attention): QKV GEMM -> RoPE -> Flash Attention -> Output GEMM
  Warps 10-13 (MLP): FC1 -> GELU -> FC2 (pipelined with attention)
  
Shared memory: Q, K, V, O tiles (attention side)
               FC1 intermediate (MLP side)
Register file: Accumulators for both GEMM streams

Flow:
1. DMA loads X, W_qkv -> SMEM
2. Attention warps: QKV GEMM -> RoPE in registers -> store Q,K,V to SMEM
3. Attention warps: Flash Attention Q@K^T -> softmax -> P@V
4. Meanwhile MLP warps: already computing FC1 with pre-loaded W1
5. Attention warps: Output GEMM -> store to SMEM
6. MLP warps: GELU in registers -> FC2 -> store to SMEM
7. All warps: residual add + LayerNorm -> write to HBM
```

## Challenges
1. Register pressure: attention + MLP in same kernel = 120+ regs per thread
2. Synchronization: attention and MLP warps must coordinate
3. SMEM partitioning: needs 80KB+ for both streams
4. Debugging: single kernel failure = entire block lost

## Feasibility on RTX 5060
- 228KB SMEM: enough for 80KB attention + 40KB MLP + overhead
- 65536 register file: enough for 14 warps with 80-100 regs each
- Blackwell TMA: enables efficient async loads for both streams
- **Verdict: Feasible. This is the path to <50ms.**

## Implementation Priority
Phase 1: CUTLASS EVT fusions (addmm_act, residual) — 2-3 weeks
Phase 2: Fused QKV + attention kernel — 2-3 weeks  
Phase 3: Full megakernel (attention + MLP) — 4-6 weeks

