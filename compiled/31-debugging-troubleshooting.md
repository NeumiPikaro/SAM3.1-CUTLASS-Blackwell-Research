# Debugging & Troubleshooting Guide

## 1. Common CUTLASS Build Errors

### Error: "No kernel found for SM100"
**Cause:** Building with wrong architecture flag
**Fix:** Use `-DCUTLASS_NVCC_ARCHS=100a` (not `100`)
```bash
cmake .. -DCUTLASS_NVCC_ARCHS=100a
```

### Error: "Template instantiation exceeded maximum depth"
**Cause:** Complex nested templates in CUTLASS
**Fix:** Simplify tile shape, reduce cluster size, or use `KernelScheduleAuto`

### Error: "Shared memory limit exceeded"
**Cause:** Too many pipeline stages for tile size
**Fix:** Reduce stages: `StageCount<3>` instead of `StageCountAuto`

### Error: "FP8 type not supported"
**Cause:** Using old CUDA toolkit (<12.0)
**Fix:** Upgrade to CUDA 12.0+ for FP8 support

## 2. Runtime Errors

### CUDA Error: "invalid configuration argument"
**Cause:** Grid/block dimensions exceed GPU limits
**Fix:** Check `cudaGetDeviceProperties` for max grid/block sizes

### CUDA Error: "out of memory"
**Cause:** Model too large for VRAM
**Fixes:**
1. Use FP8 weights (half memory)
2. Reduce batch size
3. Reduce resolution
4. Use gradient checkpointing (training)

### NaN in Output
**Cause:** Numerical instability in attention or GELU
**Debug:** Add NaN checks after each kernel:
```python
assert not torch.isnan(x).any(), f"NaN detected after {layer_name}"
```

## 3. Performance Debugging

### Slow GEMM Despite CUTLASS
**Check 1:** Are you compiling for the right architecture?
```bash
nvcc --gpu-architecture=sm_100a  # Must match your GPU
```

**Check 2:** Is FP32 accumulation enabled?
```cpp
using ElementAccumulator = float;  // Not bfloat16_t!
```

**Check 3:** Is TMA being used?
```cpp
// Check Nsight for SM90/SM100_TMA_LOAD instructions
// If missing, you're on Ampere schedule
```

### Attention Slower Than Expected
**Check:** Are you using Flash Attention or standard?
```python
# Standard (slow):
attn = torch.matmul(Q, K.T) / scale
attn = torch.softmax(attn, dim=-1)
out = torch.matmul(attn, V)

# Flash (fast):
out = flash_attention(Q, K, V)  # CUTLASS FMHA
```

## 4. Correctness Debugging

### Output Differs from Reference
```python
def debug_gemm(M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    
    # Reference
    C_ref = torch.matmul(A, B)
    
    # CUTLASS
    C_cutlass = cutlass_gemm(A, B)
    
    # Compare
    diff = (C_ref - C_cutlass).abs()
    print(f"Max error: {diff.max()}")
    print(f"Mean error: {diff.mean()}")
    print(f"Cosine sim: {F.cosine_similarity(C_ref.flatten(), C_cutlass.flatten(), dim=0)}")
    
    # Find worst elements
    worst_idx = diff.argmax()
    row, col = worst_idx // N, worst_idx % N
    print(f"Worst: [{row},{col}] ref={C_ref[row,col]:.6f} cutlass={C_cutlass[row,col]:.6f}")
```

## 5. Memory Debugging

### Check VRAM Usage
```python
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

### Memory Leak Detection
```python
torch.cuda.reset_peak_memory_stats()
# Run inference
peak = torch.cuda.max_memory_allocated()/1e9
print(f"Peak VRAM: {peak:.2f} GB")
# Expected: ~5-6 GB for SAM 3.1 BF16, ~3-4 GB for FP8
```

## 6. Build System Issues

### CUTLASS Include Path
```cmake
target_include_directories(your_target PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)
```

### CUDA + C++17
```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# Ensure nvcc supports C++17
```

