# FlashInfer — Deep Technical Analysis

## Overview
FlashInfer is the state-of-the-art GPU attention library as of 2026, used by SGLang, vLLM, TensorRT-LLM, and TGI. It provides unified APIs for attention, GEMM, and MoE with automatic backend selection (FlashAttention-2/3, cuDNN, CUTLASS, TensorRT-LLM).

## Key Features for SAM 3.1

### 1. Multi-Backend Architecture
FlashInfer doesn't implement one attention kernel — it wraps multiple and selects the best:
- FlashAttention-2/3 for standard attention
- cuDNN for certain shapes
- CUTLASS-based for custom patterns
- JIT-compiled kernels for specific shapes

**SAM 3.1 implication:** FlashInfer can auto-select the best attention kernel for each of SAM 3.1's attention shapes (ViT 16×1024×64, CLIP 16×77×64, DETR 8×100×32).

### 2. Blackwell Support (SM10.0, 10.3, 12.0, 12.1)
FlashInfer explicitly supports:
- SM10.0 (B200) and SM10.3 (B300) — data center Blackwell
- SM12.0/SM12.1 (RTX 50 series, DGX Spark, Jetson Thor) — **consumer Blackwell including RTX 5060**

This is critical — FlashInfer is one of the few libraries with RTX 5060 (SM12.0) support.

### 3. BF16 GEMM for SM10.0+
Native BF16 matrix multiplication — exactly what SAM 3.1 needs.

### 4. JIT Compilation
Kernels are compiled on first use for the specific GPU. This means:
- Optimal code for RTX 5060 specifically (not generic SM100)
- Auto-tuned tile sizes for SAM 3.1's exact matrix dimensions
- No binary compatibility overhead

### 5. CUDA Graph Compatible
FlashInfer is explicitly CUDA Graph compatible — essential for our <50ms target.

## Performance Comparison

FlashInfer claims state-of-the-art performance across prefill, decode, and mixed batching. For SAM 3.1's use case (prefill-only, non-autoregressive):
- Prefill kernels: optimized for large Q, K, V
- Block-sparse attention: skip computation for masked/zero regions
- POD-Attention: fused prefill+decode (not relevant for SAM 3.1)

## SAM 3.1 Application Strategy

### Option A: Use FlashInfer directly
```python
import flashinfer
# For ViT attention (16 heads × 64-dim, seq=1024)
output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)
```
Pro: Simple, auto-optimized, Blackwell support
Con: Adds dependency, may not fuse with our CUTLASS GEMMs

### Option B: Borrow FlashInfer's kernel techniques
Study FlashInfer's source for:
- Optimal tile sizes on SM12.0 (RTX 5060)
- Warp specialization patterns
- Online softmax implementation
Then implement in CUTLASS ourselves.

### Option C: FlashInfer for attention + CUTLASS for GEMM
- Use FlashInfer for all attention operations (Q@K^T, softmax, P@V)
- Use CUTLASS for QKV projection, MLP, output projection
- Risk: interface overhead between the two

**Recommendation:** Option A — FlashInfer is production-ready and handles Blackwell optimization. Focus our CUTLASS effort on the GEMM fusions (addmm_act) that FlashInfer doesn't cover.

## Expected Impact
FlashInfer's optimized attention + our CUTLASS GEMM fusions:
- Attention portion: 2-3× faster than naive implementation
- Combined with GEMM fusion: could reach <50ms target

## Installation
```bash
pip install flashinfer-python flashinfer-cubin
```

