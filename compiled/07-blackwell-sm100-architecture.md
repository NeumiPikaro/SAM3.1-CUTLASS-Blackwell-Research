# Blackwell SM100 Architecture — RTX 5060 Deep Dive

## 1. Blackwell Architecture Overview

NVIDIA Blackwell (2025-2026) is the successor to Hopper. Key GPUs:
- **GB100/GB200:** Data center (B100/B200)
- **GB300 (SM103):** Enhanced data center variant
- **RTX 5090 (SM100):** Consumer flagship
- **RTX 5080 (SM100):** Consumer high-end
- **RTX 5070 (SM100):** Consumer mid-high
- **RTX 5060 Ti (SM100):** Consumer mid-range
- **RTX 5060 (SM100):** Consumer mainstream — **our target**

## 2. RTX 5060 Specifications (Estimated/Confirmed)

| Parameter | RTX 5060 | RTX 5060 Ti | RTX 4060 (prev) |
|-----------|----------|-------------|----------------|
| Architecture | Blackwell | Blackwell | Ada Lovelace |
| SM Count | 30-36 | 36 | 24 |
| CUDA Cores | 3840-4608 | 4608 | 3072 |
| Tensor Cores | 120-144 | 144 | 96 |
| RT Cores | 30-36 | 36 | 24 |
| Base Clock | ~2.1 GHz | ~2.3 GHz | 1.83 GHz |
| Boost Clock | ~2.5 GHz | ~2.6 GHz | 2.46 GHz |
| VRAM | 8 GB GDDR7 | 16 GB GDDR7 | 8 GB GDDR6 |
| Memory Bus | 128-bit | 128-bit | 128-bit |
| Bandwidth | ~448 GB/s | ~448 GB/s | 272 GB/s |
| SMEM/SM | 228 KB | 228 KB | 100 KB |
| L2 Cache | 32 MB | 32 MB | 24 MB |
| TDP | 150W | 180W | 115W |
| Process | TSMC 4NP | TSMC 4NP | TSMC 4N |

### Key Specs for CUTLASS Kernel Design
- **SMEM per SM:** 228 KB — 2.28× more than Ada (100 KB)
- **Memory bandwidth:** 448 GB/s — 1.65× more than Ada (272 GB/s)
- **Tensor Core throughput:** ~2× Ada per SM (enhanced WGMMA)
- **L2 cache:** 32 MB — good for weight reuse across SMs

## 3. SM100 Streaming Multiprocessor

### SM100 Block Diagram
```
┌─────────────────────────────────────────────────┐
│                  SM100                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Sub-core │ │ Sub-core │ │ Sub-core │ ×4     │
│  │ 0        │ │ 1        │ │ 2/3      │       │
│  │ - 32 ALU │ │ - 32 ALU │ │ - 32 ALU │       │
│  │ - SFU    │ │ - SFU    │ │ - SFU    │       │
│  │ - LD/ST  │ │ - LD/ST  │ │ - LD/ST  │       │
│  └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────────────────────────────────┐      │
│  │     Tensor Core Array (4 units)      │      │
│  │     WGMMA capable, FP4-FP32          │      │
│  └──────────────────────────────────────┘      │
│  ┌──────────────────────────────────────┐      │
│  │     TMA Unit (async copy engine)     │      │
│  └──────────────────────────────────────┘      │
│  ┌──────────┐ 228 KB shared memory ┌────────┐  │
│  │ Register │ 65536 × 32-bit       │ L1/SMEM│  │
│  │ File     │                       │ Cache  │  │
│  └──────────┘                       └────────┘  │
└─────────────────────────────────────────────────┘
```

### Warp Schedulers
- 4 warp schedulers per SM
- Each can issue 1 instruction per cycle per sub-core
- WGMMA instructions issued by any scheduler, execute across all warps

## 4. Tensor Core 5th Generation

### Instruction Throughput (per SM, per cycle)
| Instruction | Data Type | Shape | Ops/Cycle |
|-------------|-----------|-------|-----------|
| wgmma.mma_async | FP16/BF16 | 64×8×16 | 16384 |
| wgmma.mma_async | FP8 E4M3 | 64×8×32 | 32768 |
| wgmma.mma_async | INT8 | 64×8×32 | 32768 |
| wgmma.mma_async | NVFP4 | 64×8×64 | 65536 |
| wgmma.mma_async | MXFP8 | 64×8×32 | 32768 |

### BF16 Peak Performance (RTX 5060, estimated)
- 36 SMs × 16384 ops/cycle × 2.5 GHz = **1,474 TFLOPS theoretical peak**
- Realistic utilization: 60-80% = **~900-1200 TFLOPS sustained**
- This is ~3× RTX 4060's ~480 TFLOPS BF16

### FP8 Peak Performance
- 36 SMs × 32768 ops/cycle × 2.5 GHz = **2,949 TFLOPS theoretical**
- **SAM 3.1 FP8 quantization could double throughput vs BF16**

## 5. TMEM (Tensor Memory) — New on Blackwell

TMEM is a new on-chip memory specifically for Tensor Core operands:
- **Size:** 256 KB per SM (separate from SMEM)
- **Purpose:** Hold intermediate results between GEMM stages
- **Benefit:** Avoids register file pressure for complex epilogues

### Application for SAM 3.1
In DETR decoder with multi-head attention:
```
GEMM(Q @ K^T) → TMEM (attention scores)
Softmax → TMEM → GEMM(softmax @ V) → output
```
TMEM avoids spilling attention scores to shared memory.

## 6. Thread Block Clusters on Blackwell

Blackwell supports larger clusters than Hopper:
- Hopper: max 8 SMs per cluster
- Blackwell: max 16 SMs per cluster

### Cluster Benefits for SAM 3.1
1. **Flash Decoding:** Split attention across cluster SMs, reduce in cluster
2. **Weight sharing:** TMA multicast loads weights to all cluster SMs
3. **Large tiles:** Cluster can compute tiles that don't fit single SM's SMEM

### Cluster Configuration for RTX 5060
With 36 SMs:
- Cluster size 4: 9 clusters available
- Cluster size 8: 4 clusters (some waste)
- **Optimal for SAM 3.1:** cluster size 2-4

## 7. Programmatic Dependent Launch (PDL)

PDL allows a kernel to trigger the next kernel without CPU involvement:
```cuda
// At end of kernel A:
cudaTriggerProgrammaticLaunchCompletion();
// Kernel B starts immediately on same SMs — no CPU overhead
```
### SAM 3.1 Application
Chain kernels without launch overhead:
```
ViT Block N:
  Attention GEMM → PDL → MLP GEMM → PDL → Layer Norm → PDL → Block N+1
```
Saves ~5-15μs per kernel launch × 32 blocks × 3-4 kernels = **500-2000μs total**.

