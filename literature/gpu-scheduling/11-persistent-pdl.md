# Persistent Kernels & PDL for SAM 3.1

## Persistent Kernels
Launch one kernel with max SMs, process all 256 ViT GEMMs via device-side task queue.
- Saves: 256 x 6us launch overhead = 1.5ms
- L2 cache: weights stay hot between consecutive GEMMs
- CUTLASS: PersistentTileScheduler built-in

## Programmatic Dependent Launch (PDL)
Kernel A triggers Kernel B on same SMs, zero CPU overhead.
Block 0 Attn -> PDL -> Block 0 MLP -> PDL -> Block 1 Attn -> ...
- Saves: 64 transitions x 10us = 640us
- Blackwell native

## Combined Impact
~2ms saved. Essential alongside kernel optimization for <50ms target.
