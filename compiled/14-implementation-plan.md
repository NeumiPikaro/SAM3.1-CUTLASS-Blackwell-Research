# Implementation Plan — From Research to Production

## Phase 1: Foundation (Week 1-2)

### 1.1 Environment Setup
```bash
# Clone CUTLASS 4.4
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout v4.4.2

# Build for Blackwell
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=100a \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_LIBRARY=OFF  # We only need headers
make -j16

# Install CuTe DSL (optional, for Python prototyping)
pip install nvidia-cutlass-dsl
```

### 1.2 GEMM Benchmarking
Create a benchmark suite for all SAM 3.1 GEMM shapes:
```cpp
// benchmark_sam31_gemms.cu
// Test each GEMM with different tile shapes, pipeline stages, cluster configs

struct GemmConfig {
  int M, N, K;
  const char* name;
};

GemmConfig sam31_gemms[] = {
  {1024, 1024, 1024, "ViT QKV projection"},
  {1024, 1024, 1024, "ViT output projection"},
  {1024, 4096, 1024, "ViT MLP FC1"},
  {1024, 1024, 4096, "ViT MLP FC2"},
  {1024, 1024, 64,   "ViT attention Q@K^T (per head)"},
  {1024, 64,   1024, "ViT attention P@V (per head)"},
  {77,   1024, 1024, "CLIP QKV"},
  {77,   4096, 1024, "CLIP MLP FC1"},
  {1024, 256,  256,  "DETR self-attn QKV"},
  {100,  256,  256,  "DETR decoder QKV"},
  {100,  2048, 256,  "DETR decoder MLP FC1"},
};

// For each config, sweep:
// - Tile shapes: (64,64,32), (128,64,64), (128,128,64), (256,128,64)
// - Pipeline stages: 2, 3, 4, 5, 6
// - Cluster shapes: (1,1,1), (1,2,1), (2,1,1), (2,2,1)
// - Schedule: Auto, PingPong, Cooperative
```

### 1.3 Baseline Measurements
Run each GEMM config with:
- cuBLAS (reference)
- CUTLASS default
- CUTLASS with best tile from sweep

Record: GFLOPS, latency, memory bandwidth, SM occupancy.

## Phase 2: GEMM Replacement (Week 3-4)

### 2.1 Custom CUDA Kernels for CUTLASS GEMMs
Write wrapper functions that replace PyTorch's matmul:
```cpp
// sam31_cutlass_gemms.h
template<typename T>
void vit_qkv_gemm(const T* X, const T* W_qkv, T* QKV, 
                  int seq_len, cudaStream_t stream);

template<typename T>
void vit_mlp_fc1_gemm(const T* X, const T* W1, const T* bias1, T* out,
                      int seq_len, cudaStream_t stream);
```

### 2.2 PyTorch Integration
```cpp
// Register as PyTorch custom ops
TORCH_LIBRARY(sam31_ops, m) {
  m.def("vit_qkv_gemm(Tensor X, Tensor W_qkv) -> Tensor");
  m.def("vit_mlp_fc1_with_bias_gelu(Tensor X, Tensor W1, Tensor bias1) -> Tensor");
}

// In Python:
import torch.ops.sam31_ops
QKV = torch.ops.sam31_ops.vit_qkv_gemm(X, W_qkv)
```

### 2.3 Integration Testing
- Run full SAM 3.1 forward pass with CUTLASS GEMMs
- Compare output with original (cosine similarity > 0.999)
- Benchmark end-to-end latency

## Phase 3: Fusion (Week 5-6)

### 3.1 Epilogue Fusions
Implement EVT-based fusions:
1. `fc1 + bias + GELU` — replaces addmm_act
2. `QKV + RoPE` — custom EVT for rotary embedding
3. `output_proj + residual` — skip connection in epilogue
4. `attention + softmax` — in mainloop

### 3.2 Flash Attention Integration
Adapt CUTLASS FMHA for SAM 3.1:
1. Port `fmha.py` (CuTe DSL) to C++ for production
2. Add RoPE pre-processing before attention
3. Test with SAM 3.1's 16-head × 64-dim configuration
4. Benchmark vs standard attention

### 3.3 Chain Fusion (Advanced)
Implement fused MLP (fc1 → GELU → fc2):
1. Custom CuTe DSL kernel
2. Intermediate in register file (4096 × 4 bytes = 16KB per row)
3. Challenge: register file pressure — need careful register allocation

## Phase 4: Quantization (Week 7-8)

### 4.1 Weight Quantization
```python
# Quantize all SAM 3.1 weights to FP8
import torch
from torch.quantization import quantize_dynamic

for name, param in model.named_parameters():
    if 'weight' in name:
        # Per-channel quantization to FP8 E4M3
        scale = param.abs().max() / 448.0  # E4M3 max value
        param_q = (param / scale).to(torch.float8_e4m3fn)
        # Store scale alongside quantized weight
```

### 4.2 CUTLASS FP8 GEMM
Instantiate FP8 variants of all GEMM kernels:
```cpp
using FP8_QKV_Gemm = GemmUniversal<
  Shape<int,int,int>,
  CollectiveBuilder<Sm100, OpClassTensorOp,
    float_e4m3_t, RowMajor, 16,
    float_e4m3_t, ColumnMajor, 16,
    float,
    Shape<_128,_128,_128>,  // K dimension 128 for FP8 (2× BF16)
    Shape<_1,_1,_1>,
    StageCountAuto,
    KernelScheduleAuto
  >::CollectiveOp,
  DefaultEpilogue<...>
>;
```

### 4.3 Accuracy Validation
- Run SAM 3.1 on COCO validation set
- Compare FP8 vs BF16 mIoU
- Target: <0.5% mIoU degradation

## Phase 5: System Optimization (Week 9-10)

### 5.1 CUDA Graphs
```python
# Capture entire forward pass
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = sam31_model(input_image, text_prompt)

# Replay
g.replay()
```

### 5.2 Programmatic Dependent Launch
Chain ViT blocks with PDL:
```cuda
// At end of each ViT block kernel:
cudaTriggerProgrammaticLaunchCompletion();
// Next block starts immediately — no CPU overhead
```

### 5.3 Memory Optimization
- Weight sharing across decoder layers
- Activation checkpointing for training
- KV cache for autoregressive generation (future)

## Phase 6: Testing & Benchmarking (Week 11-12)

### 6.1 Correctness Testing
- Output comparison: cosine similarity, max absolute error
- Visual quality: segmentation mask comparison
- Edge cases: different resolutions, different text prompts

### 6.2 Performance Benchmarking
- End-to-end latency at various resolutions
- Per-component breakdown
- Comparison with:
  - Original PyTorch SAM 3.1
  - SAM3-TensorRT (transformers-based)
  - ONNX Runtime
  - TensorRT (via transformers export)

### 6.3 Memory Profiling
- Peak VRAM usage
- Weight memory vs activation memory
- SMEM utilization per SM

## Deliverables

1. **CUTLASS SAM 3.1 Library:** Reusable C++ library with all optimized kernels
2. **PyTorch Extension:** Drop-in replacement for SAM 3.1 inference
3. **Benchmark Suite:** Automated performance testing
4. **Documentation:** Kernel selection guide, tuning guide
5. **Paper Draft:** "Optimizing SAM 3.1 for Blackwell: A CUTLASS Approach" — if results are publishable

