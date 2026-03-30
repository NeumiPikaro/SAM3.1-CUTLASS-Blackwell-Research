# Collective MMA & Mainloop — The Compute Engine

## 1. Collective MMA Architecture

"Collective" means multiple warps cooperating on a single MMA operation. This is fundamentally different from the older per-warp MMA approach.

### Warp-Level MMA (Pre-Hopper)
Each warp (32 threads) independently computes a sub-tile:
```
Warp 0: computes C[0:16, 0:8] += A[0:16, 0:16] @ B[0:16, 0:8]
Warp 1: computes C[0:16, 8:16] += A[0:16, 0:16] @ B[0:16, 8:16]
...
```
Issue: each warp loads the same A row from shared memory → wasted bandwidth.

### Warp Group MMA (Hopper/Blackwell)
4 warps (128 threads) form a "warp group" and cooperate:
```
Warp Group 0 (4 warps):
  - Collectively loads A and B from SMEM using WGMMA instructions
  - Each warp holds partial results
  - Inter-warp reduction at epilogue
```
Benefit: single shared memory load serves all 4 warps → 4× SMEM bandwidth savings.

### sm90 Collective MMA (Hopper)
```cpp
using MmaOp = SM90_64x64x32_F16F16F16_SS;  // 64×64×32 tile, FP16
// SS = shared memory source operands
// Parameters: instruction tile (16×8×16), warp arrangement (4×1×1)
```

### sm100 Collective MMA (Blackwell)
Enhanced variants:
```cpp
SM100_64x64x32_F16F16F16_SS     // BF16/FP16 on Blackwell
SM100_128x128x64_F8F8F32_SS     // FP8 → FP32 accumulation
SM100_64x128x128_NVFP4_SS       // FP4 native (new on Blackwell)
```

## 2. Mainloop — Software Pipelining Engine

The mainloop orchestrates the inner K-dimension iteration with pipelining:

### Mainloop Structure (Hopper TMA Warp Specialized)
```
Roles:
  DMA WARPS (producer): Issue TMA_LOAD operations
  MMA WARPS (consumer): Execute WGMMA on loaded data
  
Pipeline:
  ┌──────────────────────────────────────────┐
  │ Stage 0: DMA loads A[0],B[0] → SMEM     │
  │ Stage 1: DMA loads A[1],B[1] → SMEM     │
  │          MMA computes A[0]@B[0]          │
  │ Stage 2: DMA loads A[2],B[2] → SMEM     │
  │          MMA computes A[1]@B[1]          │
  │ Stage 3: DMA loads A[3],B[3] → SMEM     │
  │          MMA computes A[2]@B[2]          │
  │ ...                                      │
  │ Final: MMA computes A[N-1]@B[N-1]       │
  └──────────────────────────────────────────┘
```

### Pipeline Synchronization
- **mbarrier:** hardware barrier for producer-consumer sync
  - Producer (DMA warp): `mbarrier.arrive` after TMA completes
  - Consumer (MMA warp): `mbarrier.wait` before reading SMEM
- **Commit group:** groups multiple TMA operations for single barrier signal

### Mainloop Variants in CUTLASS

| Variant | Description | Best For |
|---------|-------------|----------|
| `KernelTmaWarpSpecialized` | Basic warp specialization | General GEMM |
| `KernelTmaWarpSpecializedPingpong` | Alternates MMA between two warps | Latency hiding |
| `KernelTmaWarpSpecializedCooperative` | All MMA warps compute together | Large tiles |
| `KernelScheduleAuto` | CUTLASS picks best | Default choice |

### Stage Count Selection
Stages = how many tiles are "in flight" simultaneously.
- More stages = deeper pipeline = better latency hiding
- More stages = more SMEM usage = fewer threadblocks per SM
- Auto formula: `stages = SMEM_size / (A_tile_size + B_tile_size)`
- For (128,128,64) BF16: 228KB / 32KB = 7 stages max

## 3. Mainloop for Attention (Blackwell GQA Example)

CUTLASS Example 93 demonstrates a specialized mainloop for Grouped Query Attention:

**Architecture:**
- 7 warps per CTA: 1 DMA_Q, 1 DMA_KV, 1 MMA, 4 EPILOG (softmax + reduction)
- Flash Decoding: KV sequence split across CTAs in a cluster
- Cluster reduction: partial attention results combined across cluster SMs

**Data flow:**
```
DMA_Q warp:    Load Q tiles from global → SMEM
DMA_KV warp:   Load K, V tiles from global → SMEM
MMA warp:      WGMMA(Q @ K^T) → attention scores
EPILOG warps:  Online softmax → WGMMA(softmax @ V) → output
               Cluster reduction for Flash Decoding splits
```

**Supported configs:**
- dH = 64, 128 (head dimensions)
- BF16/FP8 KV cache
- Arbitrary sequence lengths
- Attention sink support
- Sliding window attention

## 4. SAM 3.1 Mainloop Mapping

### ViT Self-Attention (16 heads × 64-dim, seq=1024)
- Q shape: (1024, 1024) — 16 heads × 64-dim each
- Mainloop: standard TMA warp specialized
- Tile: (128, 128, 64) — one tile covers 2 heads
- K dimension iteration: 1024/64 = 16 iterations
- **Fusion opportunity:** QKV projection + RoPE can be fused into the mainloop

### DETR Cross-Attention (8 heads × 32-dim, Q=100, KV=HW)
- Short Q, long KV → asymmetric tile shapes
- Tile: (64, 128, 32) for Q dimension, (128, 128, 32) for KV
- Mainloop: needs variable-length KV handling
- **Fusion opportunity:** mask application in the mainloop

### DETR Self-Attention (8 heads × 32-dim, Q=KV=100)
- Tiny problem → StreamK mode or even SIMT (non-Tensor Core) may be better
- Tile: (64, 64, 32)
- Very few iterations → pipeline overhead dominates

