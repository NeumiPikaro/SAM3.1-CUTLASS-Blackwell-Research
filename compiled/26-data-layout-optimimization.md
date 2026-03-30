# Data Layout & Memory Access Optimization

## 1. Tensor Layout Strategies

### Row-Major vs Column-Major
```
Row-Major (C/CUDA default):
  address = row * stride + col
  Sequential access along columns — good for A in A@B

Column-Major (Fortran/BLAS default):
  address = col * stride + row
  Sequential access along rows — good for B in A@B
```

### CUTLASS Layout Recommendations
```
For GEMM: A (M,K) = Row-Major, B (K,N) = Column-Major
  → Both A and B reads are coalesced
  → Standard CUTLASS configuration

For attention: Q (seq,d) = Row-Major, K (seq,d) = Row-Major
  → Q@K^T requires K transposed
  → CUTLASS handles transpose in shared memory (free with swizzle)
```

### SAM 3.1 Weight Layout
Meta stores SAM 3.1 weights in PyTorch default (row-major):
```
W_qkv: (3072, 1024) row-major — needs transpose for CUTLASS
W_mlp1: (4096, 1024) row-major — same
W_mlp2: (1024, 4096) row-major — same
```

**Strategy:** Transpose weights once at load time, store in optimal layout.
Cost: one-time ~100ms. Benefit: 10-20% faster GEMM from coalesced access.

## 2. Weight Packing

CUTLASS can pack weights for optimal Tensor Core access:
```cpp
// Pack B matrix for Tensor Core layout
auto packed_B = cutlass::reference::device::TensorRef<Element, Layout>(
    B_data, cutlass::layout::ColumnMajor(N, K, packed_stride));
```

### For FP8 Weights
FP8 weights need special packing for WGMMA:
```
WGMMA expects: 16×32 tiles for FP8
Natural layout: may not align to 16×32
Packing: reorganize into WGMMA-friendly tiles
```

## 3. Activation Memory Layout

### Interleaved Layout for Multi-Head Attention
Standard: all heads concatenated — (seq, 16×64)
Interleaved: heads interleaved — (seq, 64) with stride 16×64

```
Standard: Q = [h0_0, h0_1, ..., h0_63, h1_0, h1_1, ..., h15_63]
Interleaved: Q = [h0_0, h1_0, ..., h15_0, h0_1, h1_1, ..., h15_1, ...]
```

Interleaved benefits:
- Each head's data is contiguous in memory — better for per-head GEMM
- Less bank conflict in shared memory

**Recommendation:** Keep PyTorch default (standard) for compatibility. Optimize later if needed.

## 4. KV Cache Layout (Future)

For autoregressive generation (not SAM 3.1 currently):
```
K_cache: (max_seq, n_heads, head_dim) — preallocated
V_cache: (max_seq, n_heads, head_dim) — preallocated

Append new K, V at each step
Attention reads entire cache
```

### Layout Optimization
```
Page-based layout: divide into pages of P tokens
Each page: (P, n_heads, head_dim) — contiguous
Page table: maps logical position → physical page
```
Benefit: avoids memory fragmentation, enables variable-length batches.

## 5. Shared Memory Bank Conflict Analysis

### 32 Banks, 4-Byte Stride
```
Bank = (address / 4) % 32
Access conflict: two threads access same bank in same cycle
```

### BF16 Tile (128×64)
```
Each element: 2 bytes
Row access: threads 0-31 read elements 0-31 (addresses 0, 2, 4, ..., 62)
Banks: 0, 1, 2, ..., 31 → all different → no conflict ✓

Column access: threads 0-31 read elements 0-31 in column
Addresses: 0, 128, 256, ..., 3968
Banks: 0, 0, 0, ..., 0 → ALL conflict ✗
```

### Solution: Swizzle<3, 3, 3>
```
Swizzle XOR bits [8:6] of address
Maps column access to different banks
Cost: XOR operation in address computation (free on modern GPUs)
```

## 6. TMA Alignment Requirements

TMA requires 16-byte alignment:
```
BF16: 16 bytes = 8 elements → alignment = 8
FP8: 16 bytes = 16 elements → alignment = 16
FP4: 16 bytes = 32 elements → alignment = 32
```

### Impact on Tile Sizes
```
K dimension must be divisible by alignment:
BF16: K % 8 == 0 (64, 128, 256 all OK)
FP8: K % 16 == 0 (64, 128, 256 all OK)
FP4: K % 32 == 0 (64, 128, 256 all OK)
```

SAM 3.1 K=1024 is divisible by all alignments ✓

