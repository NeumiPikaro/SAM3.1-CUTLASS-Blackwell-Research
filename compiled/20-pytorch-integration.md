# PyTorch Integration Guide

## 1. Custom CUDA Op Registration

```cpp
// sam31_ops.cu
#include <torch/extension.h>
#include "sam31_cutlass_gemms.h"

torch::Tensor vit_qkv_forward(torch::Tensor X, torch::Tensor W_qkv) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kBFloat16);
    TORCH_CHECK(W_qkv.is_cuda() && W_qkv.dtype() == torch::kBFloat16);
    
    int seq_len = X.size(0);
    auto QKV = torch::empty({seq_len, 3072}, X.options());
    
    vit_qkv_gemm(
        X.data_ptr<cutlass::bfloat16_t>(),
        W_qkv.data_ptr<cutlass::bfloat16_t>(),
        QKV.data_ptr<cutlass::bfloat16_t>(),
        seq_len);
    
    return QKV;
}

torch::Tensor vit_mlp_fc1_fused_forward(
    torch::Tensor X, torch::Tensor W1, torch::Tensor bias1) {
    int seq_len = X.size(0);
    auto out = torch::empty({seq_len, 4096}, X.options());
    vit_mlp_fc1_fused(
        X.data_ptr<cutlass::bfloat16_t>(),
        W1.data_ptr<cutlass::bfloat16_t>(),
        bias1.data_ptr<float>(),
        out.data_ptr<cutlass::bfloat16_t>(),
        seq_len);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vit_qkv", &vit_qkv_forward, "ViT QKV projection (CUTLASS)");
    m.def("vit_mlp_fc1_fused", &vit_mlp_fc1_fused_forward, "ViT MLP FC1 + bias + GELU");
}
```

## 2. Python Module Replacement

```python
# sam31_cutlass.py
import torch
import sam31_ops_cuda  # compiled extension

class CutlassViTBlock(torch.nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.W_qkv = original_block.attn.qkv.weight
        self.W_o = original_block.attn.proj.weight
        self.W1 = original_block.mlp.fc1.weight
        self.b1 = original_block.mlp.fc1.bias
        self.W2 = original_block.mlp.fc2.weight
        self.b2 = original_block.mlp.fc2.bias
        # Copy all layer norms etc.
    
    def forward(self, x):
        # Attention path — CUTLASS fused
        residual = x
        x_norm = self.norm1(x)
        qkv = sam31_ops_cuda.vit_qkv(x_norm, self.W_qkv)
        Q, K, V = qkv.chunk(3, dim=-1)
        attn = flash_attention(Q, K, V)  # CUTLASS FMHA
        x = residual + sam31_ops_cuda.vit_output_proj(attn, self.W_o)
        
        # MLP path — CUTLASS fused addmm_act
        residual = x
        x_norm = self.norm2(x)
        h = sam31_ops_cuda.vit_mlp_fc1_fused(x_norm, self.W1, self.b1)
        x = residual + sam31_ops_cuda.vit_mlp_fc2(h, self.W2, self.b2)
        return x

def replace_with_cutlass(model):
    """Replace all ViT blocks with CUTLASS-optimized versions."""
    for i, block in enumerate(model.image_encoder.trunk.blocks):
        model.image_encoder.trunk.blocks[i] = CutlassViTBlock(block)
    return model
```

## 3. CUDA Graph Integration

```python
def inference_with_cuda_graph(model, image, prompt):
    # Warmup
    for _ in range(3):
        _ = model(image, prompt)
    torch.cuda.synchronize()
    
    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(image, prompt)
    
    # Replay (zero launch overhead)
    g.replay()
    return static_output
```

## 4. Build System

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sam31_cutlass',
    ext_modules=[
        CUDAExtension(
            name='sam31_ops_cuda',
            sources=['sam31_ops.cu'],
            include_dirs=[
                'cutlass/include',
                'cutlass/tools/util/include',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '--use_fast_math',
                    '-gencode=arch=compute_100a,code=sm_100a',
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```
