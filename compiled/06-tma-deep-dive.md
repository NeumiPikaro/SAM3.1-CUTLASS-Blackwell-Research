# TMA (Tensor Memory Accelerator) — Hardware Data Movement

## 1. What TMA Is

TMA is a **dedicated hardware unit** on Hopper (SM90) and Blackwell (SM100+) GPUs that performs asynchronous copies between global memory and shared memory. Before TMA, data movement used cp.async (Ampere) or synchronous loads — both requiring software-managed addressing.

### Key Advantages Over cp.async
| Feature | cp.async (Ampere) | TMA (Hopper/Blackwell) |
|---------|-------------------|----------------------|
| Address computation | Software (PTX) | Hardware (descriptor) |
| 2D/3D copies | No (must linearize) | Yes (native) |
| Multicast | No | Yes (1 load → N SMs) |
| Reduction | No | Yes (atomic across SMs) |
| Boundary handling | Software predicates | Hardware OOB handling |
| Swizzle | Software | Hardware (in descriptor) |
| Bandwidth utilization | ~80% | ~95%+ |

## 2. TMA Descriptor

A TMA descriptor encodes the full tensor geometry:
```
TMA Descriptor:
  - global_address: pointer to global memory
  - tensor_shape: [D0, D1, D2, ...] — full tensor dimensions
  - tensor_stride: [S0, S1, S2, ...] — stride per dimension
  - box_shape: [B0, B1, B2, ...] — size of transfer box
  - box_stride: stride within the box
  - element_type: FP16, BF16, FP8, INT8, etc.
  - swizzle_mode: None, 32B, 64B, 128B
```

### Construction in CUTLASS
```cpp
// Create TMA descriptor for A matrix
auto tma_A = make_tma_copy(
  SM90_TMA_LOAD{},           // TMA load operation
  gA_tensor,                 // Source tensor (global memory)
  sA_layout,                 // Destination layout (shared memory)
  TileShape<_128,_64>{},     // Box shape (transfer size)
  _1{}                       // Cluster multicast mask
);
```

## 3. TMA Copy Modes

### 1D TMA
Simple linear copy: one contiguous block.
```
Load [box_size] elements from global[offset] → shared memory
```
Used for: vectors, bias terms, small contiguous blocks.

### 2D TMA
Rectangle copy with stride:
```
For row in [0, box_rows):
  Load [box_cols] elements from global[base + row * stride] → shared[row, :]
```
Used for: matrix tiles, the primary GEMM data loading mode.

### 3D TMA
Volume copy:
```
For depth in [0, box_depth):
  For row in [0, box_rows):
    Load [box_cols] from global[...] → shared[depth, row, :]
```
Used for: 3D tensor slices, batched operations.

### TMA Im2Col
Special mode for convolutions:
```
Extract [filter_h × filter_w × channels] patches from input image
```
Used for: implicit GEMM convolution.

### TMA Reduction
Atomic reduction across SMs:
```
SM 0: partial_result → global[addr] (atomic add)
SM 1: partial_result → global[addr] (atomic add)
```
Used for: Flash Decoding final reduction, MoE expert output aggregation.

## 4. TMA Multicast

TMA can broadcast a single load to multiple SMs' shared memory:
```
Cluster of 4 SMs:
  TMA loads data from global → SMEM of all 4 SMs simultaneously
  Hardware multicast path: single memory transaction, 4 destinations
```
Benefit: 4× effective memory bandwidth for shared data (like K, V in attention).

### Multicast Mask
```cpp
// Load A to SMs in column 0 of a 2×2 cluster
constexpr uint16_t multicast_mask = 0b0101;  // SMs 0,2
auto tma_A = make_tma_copy(SM90_TMA_LOAD{}, gA, sA, tile, multicast_mask);
```

## 5. TMA in CUTLASS Pipeline

```
Producer Warps (DMA):
  1. Issue TMA_LOAD: tma_copy(gA, sA_stage_N, mbarrier)
  2. TMA hardware copies data asynchronously
  3. mbarrier.arrive when TMA completes

Consumer Warps (MMA):
  1. mbarrier.wait(stage_N) — wait for data
  2. WGMMA(sA_stage_N, sB_stage_N) — compute
  3. Signal pipeline: advance stage pointer
```

### Pipeline Stages with TMA
```
Time ──────────────────────────────────────→

DMA:    [Load S0][Load S1][Load S2][Load S3][Load S0]
MMA:             [Comp S0][Comp S1][Comp S2][Comp S3]
EPIO:                    [Stor S0][Stor S1][Stor S2]
```
4 stages = 4 tiles in flight. TMA ensures zero bubbles.

## 6. Blackwell TMA Enhancements

### SM100 TMA
- Larger box shapes supported
- Enhanced multicast for 8+ SM clusters
- TMA + PDL integration (trigger next kernel from TMA completion)

### SM103 TMA (GB300)
- Additional 3D copy optimizations
- TMA for TMEM (tensor memory) loads
- Inter-SM TMA for distributed shared memory

## 7. SAM 3.1 TMA Application

### ViT Attention — Loading Q, K, V
Each attention layer loads:
- Q: (seq, head_dim) = (1024, 64) per head
- K: same
- V: same
- With TMA multicast: load Q once → broadcast to cluster SMs

### DETR — Loading Encoder Output
Cross-attention loads encoder features:
- Encoder output: (H×W, 256) for DETR (e.g., 1024 features from 32×32 feature map)
- TMA 2D: one box per threadblock covers a tile of features
- Boundary handling automatic for non-tile-aligned H×W

### Memory Bandwidth Impact
Without TMA: cp.async loads 128B per instruction, needs 4 instructions for 512B
With TMA: single descriptor covers 512B, hardware optimizes DRAM burst patterns
**Estimated bandwidth improvement: 15-25% for SAM 3.1's tile loads.**

