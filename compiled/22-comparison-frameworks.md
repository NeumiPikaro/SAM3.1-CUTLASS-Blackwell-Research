# Comparison with Alternative Frameworks

## 1. CUTLASS vs TensorRT

| Aspect | CUTLASS | TensorRT |
|--------|---------|----------|
| Input | Raw CUDA C++/Python | ONNX/TorchScript export |
| Flexibility | Full (any custom op) | Limited to supported ops |
| Meta's native SAM 3.1 | Direct support | Cannot export (blockers) |
| HF transformers SAM 3 | Works | Works (SAM3-TensorRT repo) |
| Build time | Seconds (headers) | Minutes (engine build) |
| Custom fusion | EVT, custom kernels | Plugin system (limited) |
| Optimization level | Near-optimal | Optimal for supported ops |
| Blackwell support | Day 1 (SM100) | Delayed (TRT 10.x) |
| FP4/FP8 | Native CUTLASS | TRT quantization toolkit |

**Verdict:** CUTLASS wins for Meta's native SAM 3.1 (TensorRT can't export it). TensorRT wins for HF transformers version.

## 2. CUTLASS vs Triton

| Aspect | CUTLASS | Triton |
|--------|---------|--------|
| Language | C++ templates | Python DSL |
| Compilation | CMake + nvcc | JIT (seconds) |
| Performance ceiling | Highest (hand-tuned) | High (auto-tuned) |
| Blackwell support | Full (SM100) | Partial (catching up) |
| Epilogue fusion | EVT system | Custom code |
| Learning curve | Steep | Moderate |
| Production readiness | Battle-tested | Maturing |

**Verdict:** CUTLASS for production, Triton for rapid prototyping.

## 3. CUTLASS vs cuBLAS

| Aspect | CUTLASS | cuBLAS |
|--------|---------|--------|
| API | Templates (compile-time) | Functions (runtime) |
| Custom fusion | Yes | No |
| Blackwell GEMM | Optimized | Optimized |
| Epilogue operations | EVT (any fusion) | Limited (alpha/beta) |
| Custom ops (RoPE etc) | Yes | No |
| Maintenance | NVIDIA + community | NVIDIA |

**Verdict:** Use cuBLAS for simple GEMMs, CUTLASS for fused/custom ops.

## 4. CUTLASS vs Flash Attention 2/3 (Dao-AILab)

| Aspect | CUTLASS FMHA | FA2/FA3 |
|--------|-------------|---------|
| Integration | Part of CUTLASS | Standalone library |
| Blackwell support | Native | FA3 has experimental |
| Customization | Full (CuTe DSL) | Limited API |
| Backward pass | Example included | Optimized |
| Variable length | Flash Decoding | FlashDecoding |
| MLA/MQA/GQA | Example 93 | Supported |

**Verdict:** For SAM 3.1, CUTLASS FMHA is better (same ecosystem, easier customization).

## 5. Framework Selection Matrix for SAM 3.1

| Component | Best Framework | Why |
|-----------|---------------|-----|
| ViT GEMMs | CUTLASS | Fused addmm_act, custom RoPE |
| ViT Attention | CUTLASS FMHA | Customizable, Blackwell native |
| CLIP Text | CUTLASS | Same as ViT |
| DETR Encoder | CUTLASS | Cross-attention support |
| DETR Decoder | cuBLAS or CUTLASS | Tiny GEMMs — test both |
| FPN Conv2d | CUTLASS implicit GEMM | Or cuDNN |
| Seg Head | cuBLAS | Simple, not a bottleneck |
| LayerNorm | Custom CUDA | Not a GEMM |
| Softmax | Fused in FMHA | No separate kernel |
