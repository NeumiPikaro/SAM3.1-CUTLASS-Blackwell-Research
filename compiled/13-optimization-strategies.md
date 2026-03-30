# SAM 3.1 Optimization Strategies for Blackwell RTX 5060

## 1. Performance Budget

Current profiling (Tesla T4, BF16):
- End-to-end: 2897 ms
- ViT Image Encoder: ~2200 ms (76%) — **primary target**
- CLIP Text Encoder: ~296 ms (10%)
- DETR Encoder/Decoder: ~300 ms (10%)
- FPN + Seg Head: ~100 ms (4%)

**RTX 5060 vs T4 comparison:**
- SM count: 36 vs 40 (5060 fewer)
- Tensor Core throughput/SM: ~3× (WGMMA vs older mma.sync)
- Memory bandwidth: 448 GB/s vs 320 GB/s (1.4×)
- Clock: 2.5 GHz vs 1.59 GHz (1.57×)
- **Net speedup estimate: 4-5× vs T4**

**Target end-to-end on RTX 5060: ~600-700ms (BF16), ~300-400ms (FP8)**

## 2. Optimization Tiers

### Tier 1: Drop-In CUTLASS GEMM Replacement (1 week)
Replace Meta's custom GEMMs with CUTLASS GemmUniversal:

**What to do:**
1. Build CUTLASS 4.4 with `-DCUTLASS_NVCC_ARCHS=100a` for SM100
2. For each GEMM in SAM 3.1, instantiate appropriate CUTLASS kernel:
```cpp
// ViT QKV projection (seq, 1024) @ (1024, 3072)
using QKV_Gemm = GemmUniversal<
  Shape<int,int,int>,
  CollectiveBuilder<Sm100, OpClassTensorOp,
    half_t, RowMajor, 8,
    half_t, ColumnMajor, 8,
    float,
    Shape<_128,_256,_64>,
    Shape<_1,_1,_1>,
    StageCountAuto,
    KernelScheduleAuto
  >::CollectiveOp,
  DefaultEpilogue<...>
>;
```
3. Replace PyTorch's matmul calls with CUTLASS kernels

**Expected speedup:** 1.5-2× from better Tensor Core utilization.

### Tier 2: Epilogue Fusion (2 weeks)
Fuse operations into GEMM epilogues using EVT:

**addmm_act fusion:**
```cpp
// Replace: GEMM + separate bias_gelu kernel with:
using MLP_Epilogue = Sm90EVT<
  Sm90Compute<cutlass::epilogue::thread::GELU, float>,
  Sm90EVT<
    Sm90Compute<cutlass::plus, float>,
    Sm90AccFetch,
    Sm90RowBroadcast<float>  // bias vector
  >
>;
```

**QKV + RoPE fusion:**
```cpp
// Custom EVT node for rotary embedding
struct RotaryEmbedOp {
  __device__ float operator()(float val, int row, int col) {
    // Apply rotary: val *= cos(theta * pos) for even indices
    //               val = -original * sin(theta * pos) for odd indices
  }
};
```

**Residual connection fusion:**
```cpp
// GEMM + residual in epilogue
using Residual_Epilogue = Sm90EVT<
  Sm90Compute<cutlass::plus, float>,
  Sm90AccFetch,
  Sm90AuxLoad<float, ...>  // load residual tensor
>;
```

**Expected speedup:** 1.3-1.5× from eliminated kernel launches and memory round-trips.

### Tier 3: Flash Attention (1 week)
Replace standard attention with CUTLASS FMHA:

**Implementation:**
- Use CUTLASS `fmha.py` (CuTe DSL) as template
- Adapt for SAM 3.1's 16-head × 64-dim configuration
- Handle RoPE integration (apply before flash attention)

**Expected speedup:** 2-3× for attention portion (20% of total).

### Tier 4: Custom Fused Kernels (4 weeks)
Write custom CUTLASS/CuTe DSL kernels for SAM 3.1-specific patterns:

**4a. Fused MLP Kernel:**
```
Input: x (seq, 1024)
→ W1 GEMM → bias+GELU (in register) → W2 GEMM → +residual
Single kernel, intermediate stays in register file
```
Challenge: Need custom mainloop that chains two GEMMs.
Solution: CuTe DSL custom kernel with explicit register management.

**4b. Fused QKV+RoPE+Attention Kernel:**
```
Input: x (seq, 1024)
→ W_qkv GEMM → split Q,K,V → RoPE on Q,K → Flash Attention
Single kernel: GEMM → attention pipeline
```
Challenge: Very complex — GEMM output feeds directly into attention.
Solution: Custom CuTe DSL kernel with TMEM staging.

**4c. DETR Decoder Optimization:**
```
Handle M=100 tiny GEMMs efficiently
Batch multiple layers' QKV into single GEMM
Use StreamK for small-M GEMMs
```

**Expected speedup:** 1.5-2× additional.

### Tier 5: Quantization (2 weeks)

**FP8 Weights:**
```
Pre-quantize all weights to FP8 E4M3
Store in VRAM as FP8 (half the memory)
GEMM: FP8 @ FP8 → FP32 accumulator → BF16 output
```
Expected: 1.5-2× GEMM speedup, 50% memory reduction.

**FP8 Activations (more aggressive):**
```
Quantize activations to FP8 at layer boundaries
Requires per-token scale factors
CUTLASS block-scaled GEMM handles this natively
```
Expected: 1.8-2× GEMM speedup, some accuracy loss.

### Tier 6: System Optimizations (1 week)

**CUDA Graphs:**
```
Capture entire ViT forward pass as CUDA graph
Replay with zero kernel launch overhead
Saves: ~5-15μs per kernel × 256 GEMMs = 1-4ms
```

**Programmatic Dependent Launch (PDL):**
```
Chain kernels without CPU round-trip
Attention → MLP → Attention → MLP (all GPU-side)
Saves: ~2-5ms from eliminated launch latency
```

**Weight Preloading:**
```
Load all ViT weights into L2 cache at inference start
307M params × 2B = 614 MB — fits in 32MB L2? No.
Strategy: Preload current layer's weights (~20MB per layer)
```

**Batch Inference:**
```
Process multiple images simultaneously
Batch GEMM: (batch × seq, ...) @ (...)
Better Tensor Core utilization
```

## 3. Estimated Performance (RTX 5060)

| Tier | Speedup | End-to-End | Cumulative |
|------|---------|-----------|------------|
| Baseline (PyTorch) | 1.0× | ~2800ms | — |
| T1: CUTLASS GEMMs | 1.7× | ~1650ms | 1.7× |
| T2: Epilogue fusion | 1.4× | ~1180ms | 2.4× |
| T3: Flash Attention | 1.2× | ~980ms | 2.9× |
| T4: Custom kernels | 1.5× | ~650ms | 4.3× |
| T5: FP8 quantization | 1.5× | ~430ms | 6.5× |
| T6: System opts | 1.1× | ~390ms | 7.2× |

**Target: ~390ms end-to-end on RTX 5060 (7.2× faster than T4 baseline)**

This would be competitive with the SAM3-TensorRT repo's numbers:
- Their RTX 3090: 75ms at 4K (different model version, FP16 TensorRT)
- Our RTX 5060 estimate: ~390ms at 1024×1024 (SAM 3.1, CUTLASS)

Note: They use transformers-based SAM3 (much lighter), we use Meta's native (heavier but more capable).

