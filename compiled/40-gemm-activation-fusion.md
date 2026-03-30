# GEMM + Activation Fusion — CUTLASS 3.x Variants & Pip Package Analysis

## 1. SAM 3.1's Critical Bottleneck: addmm_act

SAM 3.1's `addmm_act` operation (GEMM + per-row bias + activation) appears in every ViT MLP block and DETR feedforward layer. In PyTorch this is `F.silu(X @ W1 + b1)` or `F.gelu(X @ W1 + b1)`. Fusing this into a single GEMM kernel eliminates one full memory round-trip (write bias+act result, read back for next op).

---

## 2. CUTLASS 3.x GEMM + Activation Variant Table

All variants use `ElementD=f32` epilogue compute. The naming pattern:
```
gemm_sm90_{TypeA}_{TypeB}_{TypeC}_{TypeD}_{TypeAccum}_tensorop_f32_epilogue[_bias_{activation}]
```

### Basic Variants (No Activation)
| Variant | Input | Output | Epilogue | Description |
|---------|-------|--------|----------|-------------|
| `gemm` (default) | e4m3/bf16/f16 | matches input | f32 | Basic GEMM |
| `gemm_with_f32_epilogue` | e4m3/bf16/f16 | f32 | f32 | Cast output to f32 |

### Per-Column Bias + Activation (CUTLASS 3.x Native)
| Variant | Input | Bias | Activation | Test Count |
|---------|-------|------|------------|------------|
| `_bias_relu` | f16/bf16/e4m3 | per-col | ReLU | 12 tests |
| `_bias_gelu` | f16/bf16/e4m3 | per-col | GELU | 12 tests |
| `_bias_silu` | f16/bf16/e4m3 | per-col | SiLU | 12 tests |
| `_bias_hardswish` | f16/bf16/e4m3 | per-col | HardSwish | 12 tests |

### Per-Row Bias + Activation (EVT System)
| Variant | Input | Bias | Activation | Test Count |
|---------|-------|------|------------|------------|
| `_per_row_bias_relu` | f16/bf16 | per-row | ReLU | 2 tests |
| `_per_row_bias_gelu` | f16/bf16 | per-row | GELU | 2 tests |
| `_per_row_bias_silu` | f16/bf16 | per-row | SiLU | 2 tests |
| `_silu` (no bias) | f16/bf16 | none | SiLU | 1 test |

**Critical limitation:** The per-column bias variants are CUTLASS 3.x native (pre-built). Per-row bias requires the EVT (Epilogue Visitor Tree) system — SAM 3.1 needs per-row bias, so EVT is mandatory.

---

## 3. Epilogue Visitor Tree (EVT) for Custom Fusion

CUTLASS 3.x uses a tree-based composable epilogue system. Each node either fetches data, performs compute, or stores results.

### EVT for D = GELU(α·acc + β·C + per_row_bias)
```cpp
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<     // Root: GELU
        cutlass::epilogue::thread::GELU, float, float>,
    cutlass::epilogue::fusion::Sm90EVT<         // Left: add bias
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus, float, float>,
        cutlass::epilogue::fusion::Sm90AccFetch,          // Fetch accumulator
        cutlass::epilogue::fusion::Sm90RowBroadcast<      // Per-row bias
            cutlass::layout::RowMajor, float>
    >
>;
```

### EVT Node Types
| Node | Purpose | Example |
|------|---------|---------|
| `Sm90AccFetch` | Fetch accumulator from register file | Leaf node |
| `Sm90SrcFetch` | Fetch source C from global memory | Leaf node |
| `Sm90ColBroadcast<>` | Broadcast per-column bias | Leaf node |
| `Sm90RowBroadcast<>` | Broadcast per-row bias | Leaf node |
| `Sm90Compute<Op>` | Apply compute op (add, GELU, etc.) | Unary or binary |

---

## 4. Activation Function Implementations

### GELU (kIsHeavy = true, ~8 ops)
```cpp
// Fast tanh-based approximation
T inner = kSqrtTwoOverPi * (x + 0.044715 * x³);
return 0.5 * x * (1 + fast_tanh(inner));
```

### SiLU / Swish (kIsHeavy = true, ~7 ops)
```cpp
return x * sigmoid(x);  // Uses Sigmoid<T> internally
```

### ReLU (kIsHeavy = false, 1 op)
```cpp
return maximum<T, PropagateNaN=true>(x, T(0));  // NaN-propagating
```

### HardSwish (kIsHeavy = false, ~5 ops)
```cpp
return x * ReLU6(x + 3) / 6;  // Approximate version
```

The `kIsHeavy` flag guides the scheduler: heavy activations (GELU, SiLU) may be overlapped with memory operations; light ones (ReLU) are inlined.

---

## 5. 168 Generated Tests: What They Cover

The CUTLASS build system generates 168 GEMM test files from the Python generator (`python/cutlass_library/generator.py`). These aren't hand-written — they're generated from the operation table.

### Test Coverage Matrix (SM90)
| Type Combo | No Act | ReLU | GELU | SiLU | HardSwish |
|-----------|--------|------|------|------|-----------|
| f16→f16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| f16→f32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| bf16→bf16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| bf16→f32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| e4m3→e4m3 | ✅ | — | — | — | — |
| e4m3→f32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| f32→f32 | ✅ | — | — | — | — |

---

## 6. Pip Package vs Source: What's Where

### In pip `nvidia-cutlass==3.5.1`
- `cutlass_library.generator` — can generate CUTLASS 3.x operations
- `cutlass.emit()` for JIT compilation of GEMMs
- Operation table with all GEMM + activation variants
- Python bindings for kernel generation

### NOT in pip (source-only)
- Pre-compiled CUDA test binaries
- `test/unit/gemm/device/sm90_evt/` test suite
- `test/unit/gemm/device/sm90_gemm_with_f32_epilogue/` tests
- `examples/49_hopper_gemm_with_collective_builder/` (3.x API example)

---

## 7. SAM 3.1 Recommendations

### For ViT MLP blocks (addmm_act: GEMM + per-row bias + GELU/SiLU)
**Best:** CUTLASS 3.x EVT with `Sm90RowBroadcast` + `Sm90Compute<GELU>`. This is the CUTLASS-native approach for per-row bias + activation.

### For BF16 inference
Use `bf16→f32` variants with EVT per-row bias + activation. The f32 epilogue preserves accuracy during activation computation.

### For FP8 quantized models
Use `ScaledLinCombPerRowBiasEltAct` from CUTLASS 3.x fusion operations — supports per-row scaling + bias + activation + block scaling in a single pass.

### For ViT attention (QKV projection + activation is rare)
Most attention uses linear projections without activation. If SiLU/GELU is needed, use the same EVT pattern with appropriate activation template parameter.

---

## References

- `include/cutlass/epilogue/thread/activation.h` — All activation implementations
- `include/cutlass/epilogue/fusion/sm90_visitor_load_tma.hpp` — EVT node definitions
- `test/unit/gemm/device/sm90_evt/` — EVT test suite
- `test/unit/gemm/device/sm90_gemm_with_f32_epilogue/` — F32 epilogue tests
- `python/cutlass_library/generator.py` — Test generation logic
- CUTLASS examples: 49 (Hopper collective builder), 57 (Hopper grouped GEMM)
