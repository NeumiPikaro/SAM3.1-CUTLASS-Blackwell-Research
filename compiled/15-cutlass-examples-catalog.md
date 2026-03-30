# CUTLASS Examples Catalog — Relevant Examples for SAM 3.1

## Blackwell (SM100) Examples

### C++ Examples
| # | Name | Description | SAM 3.1 Relevance |
|---|------|-------------|-------------------|
| 93 | Blackwell Low-Latency GQA | Grouped Query Attention with Flash Decoding, cluster reduction | **Critical** — attention optimization |
| 94 | Ada FP8 Blockwise | FP8×FP8→BF16 with blockwise dequantization | FP8 quantization |
| 111 | Hopper SSD | State Space Decomposition | Algorithm reference |
| 112 | Blackwell SSD | SSD on SM100 | Algorithm reference |
| 55 | Hopper Mixed-Dtype | FP16×INT8, FP8×INT4 GEMM | Mixed precision |

### CuTe DSL Examples (Python)
| Example | Description | SAM 3.1 Relevance |
|---------|-------------|-------------------|
| `dense_gemm.py` | Basic Blackwell GEMM | Foundation for all GEMMs |
| `dense_gemm_persistent.py` | Persistent kernel GEMM | Production GEMM pattern |
| `dense_gemm_persistent_prefetch.py` | Prefetch-optimized GEMM | Latency hiding |
| `dense_gemm_software_pipeline.py` | Manual pipeline control | Fine-tuned pipelining |
| `dense_blockscaled_gemm_persistent.py` | FP8 block-scaled GEMM | **Critical** — FP8 quantization |
| `dense_blockscaled_gemm_persistent_amax.py` | Block-scaled + AMAX | Quantization stats |
| `fmha.py` | Flash Multi-Head Attention | **Critical** — attention kernel |
| `fmha_bwd.py` | FMHA backward pass | Training support |
| `mixed_input_fmha/` | FMHA with INT4/INT8 KV | KV cache quantization |
| `mla/` | Multi-Latent Attention | Advanced attention variant |
| `grouped_gemm.py` | Grouped GEMM | Multi-head attention |
| `grouped_blockscaled_gemm.py` | Grouped block-scaled | Quantized multi-head |
| `blockwise_gemm/` | Blockwise quantized GEMM | Weight quantization |
| `mamba2_ssd/` | SSD kernel | Alternative architecture |
| `rmsnorm.py` | RMSNorm | Layer norm variant |
| `reduce.py` | Reduction kernel | Softmax, statistics |
| `programmatic_dependent_launch.py` | PDL example | Kernel chaining |
| `epilogue/` | Epilogue examples | Fusion patterns |
| `tutorial_gemm/` | GEMM tutorial | Learning resource |

## Hopper (SM90) Examples

| Example | Description | SAM 3.1 Relevance |
|---------|-------------|-------------------|
| 55 | Mixed-Dtype GEMM | FP8, INT4, INT8 support |
| Hopper FMHA | Flash Attention | Attention baseline |

## Key Examples for SAM 3.1 Implementation

### Priority 1: Start Here
1. **`dense_gemm_persistent.py`** — Template for all ViT GEMMs
2. **`fmha.py`** — Template for attention kernel
3. **`dense_blockscaled_gemm_persistent.py`** — Template for FP8 GEMMs
4. **`93_blackwell_low_latency_gqa`** — Template for optimized attention

### Priority 2: Optimization
5. **`dense_gemm_persistent_prefetch.py`** — Latency hiding
6. **`grouped_gemm.py`** — Multi-head attention batching
7. **`mixed_input_fmha/`** — INT8/INT4 KV cache
8. **`programmatic_dependent_launch.py`** — Kernel chaining

### Priority 3: Advanced
9. **`blockwise_gemm/`** — Fine-grained quantization
10. **`mla/`** — If exploring attention alternatives
11. **`dense_blockscaled_gemm_persistent_amax.py`** — Quantization calibration

