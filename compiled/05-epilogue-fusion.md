# Epilogue Design & Fusion — The Post-Processing Power

## 1. What the Epilogue Does

After the GEMM accumulation loop, the epilogue:
1. Converts accumulator (FP32) to output type (BF16/FP16/FP8)
2. Applies α × result + β × C (linear combination)
3. Optionally fuses: bias, activation, scaling, per-channel quantization

### Why Epilogue Fusion Matters
Without fusion, each operation is a separate kernel launch:
```
GEMM kernel → launch overhead (5-15μs)
Bias kernel → launch overhead + memory round-trip  
Activation kernel → launch overhead + memory round-trip
```
With fusion:
```
GEMM + bias + activation kernel → single launch, zero extra memory traffic
```

**For SAM 3.1's ViT with 32 MLP blocks**, fusing addmm_act (fc1 → activation) saves 32 kernel launches and 32 memory round-trips.

## 2. Thread-Level Epilogue (Legacy)

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  ElementOutput,    // BF16 output
  Alignment,        // 8 elements
  ElementAccum,     // FP32 accumulator
  ElementCompute    // FP32 compute
>;
```
Each thread applies `output = α * accumulator + β * C` independently.

### Activation Fusions
```cpp
// LinearCombination + ReLU
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
  ElementOutput, Alignment, ElementAccum, ElementCompute
>;

// LinearCombination + GELU
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
  ElementOutput, Alignment, ElementAccum, ElementCompute
>;

// LinearCombination + SiLU/Swish
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
  ElementOutput, Alignment, ElementAccum, ElementCompute
>;
```

## 3. Collective Epilogue (Hopper/Blackwell)

The collective epilogue uses TMA for storing results, avoiding SMEM bottleneck:

```cpp
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
  StrideC, StrideD,    // Memory strides
  EpilogueOp           // Thread-level operation
>;
```

### TMA-Based Store
On Hopper/Blackwell, the epilogue writes results directly via TMA:
- Register file → SMEM (one warp)
- SMEM → global memory (TMA store, async)

This overlaps the last tile's epilogue with the next tile's compute.

## 4. Epilogue Visitor Tree (EVT) — Custom Fusions

EVT allows arbitrary computation trees in the epilogue:

```cpp
// Example: GEMM → bias → GELU → scale → output
using EVT = Sm90EVT<
  Sm90Compute<cutlass::multiplies, float>,  // scale
  Sm90EVT<
    Sm90Compute<cutlass::gelu, float>,       // GELU
    Sm90EVT<
      Sm90Compute<cutlass::plus, float>,    // bias add
      Sm90AccFetch,                          // accumulator from GEMM
      Sm90ScalarBroadcast<float>             // bias vector
    >
  >,
  Sm90ScalarBroadcast<float>                // scale factor
>;
```

### EVT Operations Available
- **Sm90AccFetch:** grab the GEMM accumulator
- **Sm90ScalarBroadcast:** broadcast a scalar (alpha, beta)
- **Sm90RowBroadcast:** broadcast a row vector (per-column bias)
- **Sm90ColBroadcast:** broadcast a column vector (per-row bias)
- **Sm90Compute:** apply any binary/unary function (add, mul, relu, gelu, silu)
- **Sm90AuxLoad:** load auxiliary data (scale factors, masks)

## 5. SAM 3.1 Epilogue Opportunities

### ViT MLP Blocks — addmm_act Fusion
Meta's native `addmm_act` does: `fc1(x) → gelu → fc2(activated)` in one fused CUDA kernel.

With CUTLASS EVT, we can fuse:
```
GEMM(fc1) → EVT: bias → GELU → store as intermediate
Then:
GEMM(fc2) → EVT: scale → BF16 output
```
Or better, fuse the entire MLP if fc1 output is kept in registers:
```
GEMM(fc1) → GELU in registers → GEMM(fc2) register input
```
This requires a **custom mainloop** that feeds fc1's output directly into fc2. CUTLASS doesn't natively support chained GEMMs, but we can:
1. Write a custom collective that runs both GEMMs
2. Use the output of one WGMMA as input to the next in register file
3. Only one SMEM round-trip instead of two

### ViT Attention — QKV Fusion
```
Fused QKV projection: single GEMM with 3× output
  Input: X (seq, 1024), Weight: W_qkv (1024, 3072)
  Output: Q || K || V (seq, 3072)
  Split in epilogue: no extra kernel needed
```

### Attention Output Projection + Residual
```
GEMM(O @ proj) → EVT: + residual connection → layer_norm
```
Layer norm in epilogue is possible but complex — needs row-wise reduction.

### Quantization in Epilogue
```
GEMM(BF16 @ BF16) → EVT: per-channel scale → INT8/FP8 output
```
CUTLASS supports this directly with `Sm90Compute<quantize>`.

