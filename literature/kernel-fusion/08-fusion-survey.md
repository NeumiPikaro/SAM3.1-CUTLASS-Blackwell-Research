# Kernel Fusion Techniques — Comprehensive Survey

## 1. Taxonomy of Fusion

### Element-wise Fusion
Combine operations that touch the same elements:
```
x + bias → GELU → output
All three share same memory access pattern
Cost: 1 memory round-trip instead of 3
```

### Vertical Fusion (Producer-Consumer)
Output of one kernel feeds directly into next:
```
GEMM output → in registers → activation kernel input
No HBM round-trip for intermediate
```

### Horizontal Fusion (Siblings)
Independent operations in same kernel:
```
Q, K, V projections computed in single GEMM
Fused QKV: (seq, 3072) instead of 3 × (seq, 1024)
Better Tensor Core utilization (larger N)
```

### Loop Fusion
Merge loops that iterate over same data:
```
for i in range(seq): Q[i] = X[i] @ Wq
for i in range(seq): K[i] = X[i] @ Wk
→
for i in range(seq): Q[i], K[i], V[i] = X[i] @ W_qkv
```

## 2. Fusion Opportunities in SAM 3.1

### Opportunity 1: addmm_act (32 instances)
```
Current: GEMM → bias kernel → GELU kernel (3 launches, 2 HBM round-trips)
Fused:   GEMM + bias + GELU in epilogue (1 launch, 0 extra round-trips)
Savings: ~350ms on T4 (12% of total)
```

### Opportunity 2: Fused QKV (32 instances)
```
Current: 3 × GEMM(seq, 1024) @ (1024, 1024) = 3 launches
Fused:   1 × GEMM(seq, 1024) @ (1024, 3072) = 1 launch
Savings: 2 × kernel launch overhead per block = 128 launches saved
Better:  Larger N dimension → better Tensor Core utilization (~10% more)
```

### Opportunity 3: Attention + Output Projection
```
Current: Attention → HBM → Output GEMM
Fused:   Attention computes, then immediately @ Wo, result in SMEM/registers
Savings: Eliminates (seq, 1024) × 2B = 2KB per head × 16 = 32KB HBM traffic per block
```

### Opportunity 4: Residual + LayerNorm
```
Current: Add kernel → LayerNorm kernel (2 launches)
Fused:   Single kernel: norm(x + residual)
Savings: 1 launch, 1 HBM read of (seq, 1024) avoided
```

### Opportunity 5: MLP Block Fusion (THE BIG ONE)
```
Current: FC1 → bias → GELU → HBM → FC2 → bias → residual
Fused:   FC1 → GELU (register) → FC2 → +residual
Savings: Eliminates intermediate HBM round-trip (seq, 4096) × 2B = 8KB/seq
For seq=1024: 8MB saved per block × 32 = 256MB total
```

## 3. State-of-Art Fusion Frameworks

### CUTLASS EVT (Epilogue Visitor Tree)
Best for: GEMM post-processing fusion
```
Sm90EVT<Sm90Compute<GELU>,
  Sm90EVT<Sm90Compute<plus>,
    Sm90AccFetch,
    Sm90RowBroadcast<float>>>
```
Limitation: Can only fuse operations that fit in the epilogue (after GEMM).

### ThunderKittens
Best for: Attention + MLP megakernel fusion
- Fuses entire transformer block into single kernel
- Uses warp specialization for producer/consumer overlap
- Claims 86% Tensor Core utilization on H100

### Triton
Best for: Auto-fusion via compiler
- Automatically identifies fusion opportunities
- JIT compiles fused kernels
- Less control than CUTLASS but faster iteration

### XLA (TensorFlow/JAX)
Best for: Graph-level fusion
- Fuses at computation graph level
- Good for static shapes
- Limited CUDA-specific optimization

## 4. Performance Impact Analysis

### Memory Traffic Reduction
| Fusion | HBM Traffic Saved | % of Total |
|--------|-------------------|------------|
| addmm_act | 2 × seq × 4096 × 2B per block | 8% |
| Fused QKV | 2 × 2 × seq × 1024 × 2B per block | 4% |
| Attn + Output | seq × 1024 × 2B per block | 2% |
| Residual + LN | seq × 1024 × 2B per block | 2% |
| MLP fusion | seq × 4096 × 2B per block | 8% |
| **Total** | | **~24%** |

### Kernel Launch Reduction
```
Unfused: ~480 launches per ViT forward pass
Fused:   ~160 launches (67% reduction)
At 12μs per launch: 5.76ms → 1.92ms = 3.84ms saved
```

## 5. The Megakernel Dream

### What is a Megakernel?
Entire ViT block (attention + MLP + residuals + norms) in ONE CUDA kernel:
```
__global__ void vit_block_megakernel(...) {
    // All warps: load X from HBM → SMEM
    // Warps 0-7: QKV GEMM → RoPE → Attention → Output GEMM
    // Warps 8-15: MLP FC1 → GELU → FC2 (pipelined with attention)
    // All warps: residual add + LayerNorm → write to HBM
}
```

### Benefits
- Zero kernel launch overhead (1 launch for entire block)
- Zero HBM round-trips for intermediates
- Attention and MLP compute overlap (different warps)
- Single synchronization point

### Challenges
- Register pressure: attention + MLP in same kernel = many registers
- Complex warp coordination
- Hard to debug
- Blackwell TMEM could help (store attention scores)

### Feasibility for SAM 3.1
With RTX 5060's 228KB SMEM:
- Can hold Q, K, V, O tiles for attention
- Can hold FC1 intermediate for MLP
- Register file: 65536 regs — enough for 14 warps per role
- **Verdict: Feasible but high engineering cost. Start with CUTLASS EVT, evolve to megakernel.**

