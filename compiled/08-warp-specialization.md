# Warp Specialization Patterns — Maximizing SM Utilization

## 1. The Utilization Problem

Traditional CUDA kernels have all warps doing everything:
```
Warp 0-31: Load data → Compute → Store data
```
Problem: when warps are loading, compute units are idle. When computing, memory units are idle.
Typical utilization: 40-60% of theoretical peak.

## 2. Warp Specialization Solution

Assign different roles to different warps:
```
DMA Warps (2-4):   Only issue TMA loads
MMA Warps (4-8):   Only execute WGMMA
EPILOGUE Warps(2-4): Only execute epilogue
```
All roles run concurrently → near 100% utilization of both compute and memory units.

## 3. Ping-Pong Schedule

Two groups of MMA warps alternate:
```
Group A: Compute tile 0, 2, 4, ...
Group B: Compute tile 1, 3, 5, ...

Time: ─────────────────────────────→
DMA:   [L0][L1][L2][L3][L4][L5]
Grp A:      [C0]    [C2]    [C4]
Grp B:         [C1]    [C3]    [C5]
```
Benefit: hides WGMMA pipeline fill latency.

## 4. Cooperative Schedule

All MMA warps cooperate on one tile:
```
DMA:   [L0][L1][L2][L3]
MMA:        [C0----][C1----][C2----]
```
Benefit: larger effective tile = better Tensor Core occupancy.
Best for: large problem sizes where each tile has enough work.

## 5. SAM 3.1 Warp Specialization Strategy

### ViT Attention (large tiles, compute-bound)
Use **cooperative** — tiles are (128, 128, 64), enough work for all warps.

### DETR Decoder (small tiles, latency-bound)
Use **ping-pong** — tiles are (64, 64, 32), alternating hides latency.

### MLP Blocks (wide matrices)
Use **cooperative** — N=4096 dimension gives large tiles.

### Configuration for RTX 5060
```
ViT Attention: 2 DMA + 10 MMA + 2 EPILOG = 14 warps (448 threads)
ViT MLP:       2 DMA + 10 MMA + 2 EPILOG = 14 warps
DETR:          4 DMA + 4 MMA + 4 EPILOG  = 12 warps (384 threads)
```

