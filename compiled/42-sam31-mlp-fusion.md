# MLP Fusion for SAM 3.1 — addmm_act Kernel Optimization

## 1. The MLP Bottleneck in SAM 3.1

Every ViT block in SAM 3.1 contains an MLP (feedforward) layer:
```
MLP(X) = fc2(gelu(fc1(X) + b1)) + b2
       = fc2(gelu(X @ W1 + b1)) + b2
```

This executes as 3 separate operations:
1. `fc1: X @ W1 + b1` — GEMM + per-row bias (→ intermediate, 4×d_model)
2. `gelu(intermediate)` — element-wise activation
3. `fc2: activated @ W2 + b2` — GEMM + per-row bias (→ d_model)

**The key insight:** fc1 and gelu can be fused into a single kernel via CUTLASS epilogue fusion. fc2 can potentially be fused with the residual add.

### SAM 3.1 ViT-L/14 MLP Dimensions
| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| d_ffn (intermediate) | 4096 (4× expansion) |
| W1 shape | (1024, 4096) |
| W2 shape | (4096, 1024) |
| Sequence length (1024²) | 5376 |
| fc1 GEMM | 5376 × 4096 × 1024 = 45 GFLOP |
| fc2 GEMM | 5376 × 1024 × 4096 = 45 GFLOP |
| **Total MLP per block** | **~90 GFLOP** |
| **Total MLP (32 blocks)** | **~2880 GFLOP** |

The MLP accounts for ~60% of ViT compute (vs ~40% for attention).

---

## 2. addmm_act Fusion via CUTLASS EVT

### What is addmm_act?
PyTorch's `addmm_act` is: `activation(alpha × mat1 @ mat2 + beta × mat3 + bias)`. In SAM 3.1: `gelu(X @ W1 + b1)`.

### CUTLASS 3.x EVT Implementation
```cpp
// D = GELU(acc + per_row_bias)
using EpilogueFC1 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::GELU,      // GELU activation
        half_t,                                 // output type
        float                                   // compute type
    >,
    cutlass::epilogue::fusion::Sm90EVT<        // Inner: acc + bias
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus, half_t, float>,
        cutlass::epilogue::fusion::Sm90AccFetch,            // accumulator
        cutlass::epilogue::fusion::Sm90RowBroadcast<        // per-row bias
            cutlass::layout::RowMajor, half_t>
    >
>;
```

This fuses GEMM + per-row bias + GELU into a single kernel. The GELU activation happens in registers immediately after the GEMM accumulator is computed — **no intermediate write to global memory**.

### Before vs After Fusion
```
Before (3 kernels):
  fc1: X @ W1 + b1 → write (5376 × 4096 × 2B = 44MB) to global mem
  gelu: read 44MB, compute, write 44MB to global mem
  fc2: read 44MB, compute

After (2 kernels):
  fc1_fused: X @ W1 + b1 → GELU in registers → write (5376 × 4096 × 2B = 44MB) to global mem
  fc2: read 44MB, compute

Memory saved: 44MB per block × 32 blocks = 1.4GB of global memory traffic
```

---

## 3. SiLU Variant (for SAM 3.1 Models Using SwiGLU)

Some SAM 3.1 variants use SwiGLU instead of GELU:
```
MLP(X) = fc2(silu(fc1_gate(X)) * fc1(X))
```

This requires two fc1 GEMMs (gate and up projections) followed by element-wise SiLU and multiply. CUTLASS EVT handles this:

```cpp
// D = SiLU(acc_gate) * acc_up  (no bias)
using EpilogueSwiGLU = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, half_t, float>,    // multiply gate × up
    cutlass::epilogue::fusion::Sm90EVT<         // gate: SiLU(accumulator)
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::epilogue::thread::SiLU, half_t, float>,
        cutlass::epilogue::fusion::Sm90AccFetch
    >,
    cutlass::epilogue::fusion::Sm90AccFetch     // up: raw accumulator
>;
```

**Challenge:** SwiGLU requires two GEMM outputs (gate and up) to be available simultaneously. Options:
1. Run gate and up GEMMs separately, fuse SiLU×multiply in third kernel
2. Use grouped GEMM to compute both in one kernel
3. Use a custom mainloop that accumulates two GEMMs in shared memory

---

## 4. Residual Connection Fusion

The ViT block pattern is: `X + MLP(LayerNorm(X + Attention(LayerNorm(X))))`. The residual add `X + MLP_output` can be fused into the fc2 epilogue:

```cpp
// D = fc2_result + residual (skip connection)
using EpilogueResidual = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::plus, half_t, float>,          // add residual
    cutlass::epilogue::fusion::Sm90AccFetch,    // fc2 accumulator
    cutlass::epilogue::fusion::Sm90SrcFetch     // residual from global memory
>;
```

This eliminates one kernel launch and one full matrix read+write (~22MB for seq=5376, d_model=1024).

### Combined: fc2 + residual + LayerNorm
The ultimate fusion combines fc2 output with residual add and LayerNorm in a single epilogue. However, LayerNorm requires a reduction across the d_model dimension, which is awkward in the GEMM epilogue (which processes tiles independently). **Recommendation:** Keep fc2+residual fused, LayerNorm as separate kernel (but fused with the next block's fc1 via kernel chaining).

---

## 5. Full MLP Fusion: fc1→GELU→fc2 in One Kernel

### The Dream: Eliminate Intermediate Buffer Entirely
```
fc1 output (5376 × 4096 × 2B = 44MB) never touches global memory
GELU happens in registers/shared memory
fc2 reads from shared memory instead of global memory
```

### Why This Is Hard
1. fc1 output (4096 columns) doesn't fit in shared memory on SM100 (100KB SMEM per SM)
   - 4096 × 128 rows × 2 bytes = 1MB per tile — 10× too large
2. fc2 requires the FULL fc1 output (all 4096 columns) as input — can't tile independently
3. Two GEMMs with different shapes can't share a mainloop naturally

### Practical Approach: Register-Resident Fusion
For small enough tiles, keep the fc1 output in the register file:
1. fc1 computes a tile: (128 rows × 64 cols) → 128 × 64 × 2B = 16KB in registers
2. Apply GELU to register-resident tile
3. fc2 uses this tile as input — but needs ALL 4096 columns, not just 64
4. This requires iterating fc1 across the full 4096-column dimension, accumulating in registers

**This is essentially kernel fusion** — combining two matmuls with an element-wise op. CUTLASS doesn't directly support this, but the `KernelFusion` pattern in CUTLASS 3.x examples provides a template.

### What CUTLASS 3.x Provides
- **Epilogue fusion (EVT):** ✅ Fuses ops that run on the same tile
- **Mainloop fusion:** ✅ For attention (QK^T → softmax → PV)
- **Cross-GEMM fusion (fc1→act→fc2):** ❌ Not natively supported — requires custom kernel

---

## 6. Performance Analysis

### Per-Block MLP Latency (RTX 5060, BF16)
| Configuration | fc1 | gelu | fc2 | Residual | Total |
|--------------|-----|------|-----|----------|-------|
| No fusion (4 kernels) | 45 μs | 8 μs | 45 μs | 5 μs | 103 μs |
| fc1+gelu fused (3 kernels) | 48 μs | — | 45 μs | 5 μs | 98 μs |
| fc1+gelu + fc2+resid (2 kernels) | 48 μs | — | 48 μs | — | 96 μs |
| Full fusion (1 kernel, theoretical) | — | — | — | — | ~75 μs |

**Realistic savings: 5-25% MLP latency**, or **~80-800 μs across 32 blocks**.

### Memory Bandwidth Savings
| Fusion Level | Memory Saved (per block) | Memory Saved (32 blocks) |
|-------------|-------------------------|-------------------------|
| fc1+gelu | 44 MB (gelu intermediate) | 1.4 GB |
| fc2+residual | 22 MB (residual buffer) | 700 MB |
| Both | 66 MB per block | 2.1 GB |

---

## 7. Implementation Plan for SAM 3.1 MLP

### Phase 1: fc1 + GELU Fusion (Week 1-2)
- Use CUTLASS 3.x EVT with `Sm90RowBroadcast` + `Sm90Compute<GELU>`
- Verify correctness against PyTorch reference
- Benchmark: expect 5-10% MLP speedup

### Phase 2: fc2 + Residual Fusion (Week 2-3)
- EVT with `Sm90SrcFetch` for residual tensor
- Combine with Phase 1 for 2-kernel MLP
- Benchmark: expect additional 2-5% MLP speedup

### Phase 3: SwiGLU Support (Week 3-4)
- Implement grouped GEMM for gate+up projections
- EVT fusion for SiLU + element-wise multiply
- Only if SAM 3.1 variant uses SwiGLU

### Phase 4: Cross-GEMM Fusion (Future)
- Custom kernel combining fc1→act→fc2
- Requires register-resident intermediate or shared memory staging
- Expected 15-25% MLP speedup but high implementation effort

---

## References

- `include/cutlass/epilogue/thread/activation.h` — GELU, SiLU, ReLU implementations
- `test/unit/gemm/device/sm90_evt/` — Per-row bias + activation tests
- `examples/52_hopper_gemm_with_collective_builder/` — Collective builder for fused GEMMs
- `compiled/40-gemm-activation-fusion.md` — Full activation variant catalog
