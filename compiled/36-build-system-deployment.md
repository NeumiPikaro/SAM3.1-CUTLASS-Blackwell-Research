# Build System & Deployment Guide

## 1. Project Structure

```
sam31-cutlass/
├── CMakeLists.txt                 # Root build config
├── cutlass/                       # CUTLASS 4.4 (git submodule)
├── include/
│   └── sam31/
│       ├── sam31_gemms.h          # CUTLASS GEMM wrappers
│       ├── sam31_attention.h      # Flash Attention wrapper
│       ├── sam31_fused_ops.h      # Epilogue fusions
│       └── sam31_config.h         # Kernel configurations
├── src/
│   ├── sam31_ops.cu               # CUDA kernel implementations
│   ├── sam31_attention.cu         # Attention kernel
│   └── sam31_layernorm.cu         # LayerNorm kernel
├── python/
│   ├── sam31_cutlass.cpp          # PyTorch extension binding
│   ├── sam31_model.py             # Modified SAM 3.1 model
│   └── setup.py                   # Build script
├── tests/
│   ├── test_gemm_correctness.py
│   ├── test_attention_correctness.py
│   └── test_end_to_end.py
├── benchmarks/
│   ├── bench_gemm.py
│   ├── bench_attention.py
│   └── bench_end_to_end.py
└── docker/
    └── Dockerfile                 # Reproducible build environment
```

## 2. CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(sam31_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUTLASS
set(CUTLASS_DIR ${CMAKE_SOURCE_DIR}/cutlass)
add_subdirectory(${CUTLASS_DIR} ${CMAKE_BINARY_DIR}/cutlass)

# Target architecture
set(CMAKE_CUDA_ARCHITECTURES 100a)

# PyTorch
find_package(Torch REQUIRED)

# Our library
add_library(sam31_ops SHARED
    src/sam31_ops.cu
    src/sam31_attention.cu
    src/sam31_layernorm.cu
)

target_include_directories(sam31_ops PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sam31_ops ${TORCH_LIBRARIES})

# PyTorch extension
torch_python_library(sam31_ops_cuda
    python/sam31_cutlass.cpp
)
target_link_libraries(sam31_ops_cuda sam31_ops)
```

## 3. Docker Build

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Python
RUN apt-get update && apt-get install -y python3 python3-pip cmake
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu126

# CUTLASS
RUN git clone --branch v4.4.2 https://github.com/NVIDIA/cutlass.git /opt/cutlass

# Build
WORKDIR /workspace
COPY . .
RUN mkdir build && cd build && \
    cmake .. -DCUTLASS_DIR=/opt/cutlass -DCMAKE_CUDA_ARCHITECTURES=100a && \
    make -j$(nproc)

# Install
RUN cd python && pip3 install -e .
```

## 4. Installation

```bash
# Clone
git clone https://github.com/your-repo/sam31-cutlass.git
cd sam31-cutlass

# Build
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=100a
make -j16

# Install Python package
pip install -e python/

# Verify
python -c "import sam31_ops_cuda; print('OK')"
```

## 5. Usage

```python
import torch
from sam31_model import Sam31CutlassModel

# Load model with CUTLASS kernels
model = Sam31CutlassModel.from_pretrained("facebook/sam3.1")
model = model.cuda().bfloat16()

# Optional: quantize to FP8
model.quantize_fp8()

# Optional: capture CUDA graph
model.capture_graph(resolution=1024)

# Inference
image = torch.randn(1, 3, 1024, 1024, device='cuda', dtype=torch.bfloat16)
masks = model(image, text_prompt="segment all objects")

# Benchmark
import time
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    masks = model(image, text_prompt="segment all objects")
torch.cuda.synchronize()
print(f"Latency: {(time.perf_counter() - start) / 100 * 1000:.1f} ms")
```

## 6. Benchmarking Script

```python
#!/usr/bin/env python3
"""SAM 3.1 CUTLASS Benchmark Suite"""

import torch
import time
import json
from sam31_model import Sam31CutlassModel

def benchmark(model, resolution, batch_size, precision, num_warmup=10, num_iter=100):
    H = W = resolution
    image = torch.randn(batch_size, 3, H, W, device='cuda')
    if precision == 'bf16':
        image = image.bfloat16()
    
    prompt = "segment all objects"
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(image, prompt)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = model(image, prompt)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iter * 1000
    
    return {
        'resolution': resolution,
        'batch_size': batch_size,
        'precision': precision,
        'latency_ms': round(elapsed, 1),
        'throughput_fps': round(batch_size * 1000 / elapsed, 2),
        'vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }

# Run benchmarks
results = []
model = Sam31CutlassModel.from_pretrained("facebook/sam3.1").cuda()

for prec in ['bf16', 'fp8']:
    if prec == 'fp8':
        model.quantize_fp8()
    for res in [512, 768, 1024, 1280]:
        for bs in [1, 2, 4]:
            r = benchmark(model, res, bs, prec)
            results.append(r)
            print(json.dumps(r))

with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

