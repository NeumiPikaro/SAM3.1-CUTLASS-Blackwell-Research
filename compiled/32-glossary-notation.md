# Glossary & Notation Reference

## CUTLASS Terms

| Term | Definition |
|------|-----------|
| **GEMM** | General Matrix Multiply: D = α×A×B + β×C |
| **CuTe** | CUDA Tensor Templates — layout abstraction library |
| **TMA** | Tensor Memory Accelerator — hardware async copy engine |
| **WGMMA** | Warp Group Matrix Multiply-Accumulate — Hopper/Blackwell Tensor Core instruction |
| **EVT** | Epilogue Visitor Tree — composable epilogue fusion system |
| **Collective** | Multiple warps cooperating on a single operation |
| **Mainloop** | The K-dimension iteration loop with pipelining |
| **Epilogue** | Post-GEMM operations (type conversion, bias, activation) |
| **Tile** | Sub-matrix processed by one threadblock |
| **Stage** | One pipeline buffer — data for one tile loaded ahead |
| **Cluster** | Group of threadblocks on multiple SMs sharing SMEM |
| **Swizzle** | XOR-based address remapping to avoid SMEM bank conflicts |
| **Persistent** | Kernel launched once, distributes work device-side |
| **StreamK** | GEMM mode that divides K across threadblocks for load balance |
| **PDL** | Programmatic Dependent Launch — GPU-triggered kernel launch |

## SAM 3.1 Terms

| Term | Definition |
|------|-----------|
| **ViT** | Vision Transformer — SAM 3.1's image encoder backbone |
| **ViT-L/14** | Large ViT with 14×14 pixel patches, 1024-dim, 32 layers |
| **CLIP** | Contrastive Language-Image Pre-training — text encoder |
| **DETR** | Detection Transformer — prompt-to-mask decoder |
| **FPN** | Feature Pyramid Network — multi-scale feature extractor |
| **RoPE** | Rotary Position Embeddings — position encoding via rotation |
| **addmm_act** | Meta's fused kernel: matmul + bias + GELU activation |
| **SDPA** | Scaled Dot-Product Attention |
| **GQA** | Grouped Query Attention — multiple Q heads share KV heads |
| **MHA** | Multi-Head Attention — standard (SAM 3.1 uses this) |
| **MLA** | Multi-Latent Attention — compressed KV (DeepSeek) |

## Notation

| Symbol | Meaning |
|--------|---------|
| A ∈ ℝ^(M×K) | Matrix A with M rows and K columns |
| op(A) | Transposed or identity operation on A |
| @ | Matrix multiplication |
| ⊙ | Element-wise multiplication |
| σ(x) | Sigmoid activation |
| GELU(x) | Gaussian Error Linear Unit activation |
| softmax(x) | exp(x) / sum(exp(x)) |
| √d | Square root of head dimension |
| BF16 | Brain Float 16 (1-8-7) |
| FP8 E4M3 | 8-bit float (1-4-3) |
| FP4 NVFP4 | 4-bit float (1-2-1) |
| SM | Streaming Multiprocessor |
| SMEM | Shared Memory |
| HBM | High Bandwidth Memory (global memory) |
| DRAM | Dynamic Random Access Memory |

## Architecture Shorthand

| Code | Architecture | GPUs |
|------|-------------|------|
| SM70 | Volta | V100 |
| SM75 | Turing | RTX 2080 |
| SM80 | Ampere | A100 |
| SM86 | Ampere | RTX 3090 |
| SM89 | Ada Lovelace | RTX 4090 |
| SM90 | Hopper | H100, H200 |
| SM100 | Blackwell | RTX 5060-5090, B100/B200 |
| SM103 | Blackwell (GB300) | B300 |

## CUTLASS Version History

| Version | Date | Key Features |
|---------|------|-------------|
| 1.0 | 2017 | Basic GEMM templates |
| 2.0 | 2019 | Ampere support, cp.async |
| 3.0 | 2022 | CuTe, Hopper support, TMA |
| 3.5 | 2024 | Performance improvements |
| 4.0 | 2025 | CuTe DSL (Python), Blackwell SM100 |
| 4.4 | 2026-02 | SM103, AoT compilation, JAX support |

