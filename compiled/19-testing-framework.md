# Testing & Validation Framework

## 1. Unit Testing Strategy

### GEMM Correctness
```cpp
// Test each CUTLASS GEMM against cuBLAS reference
void test_gemm_correctness(int M, int N, int K, const char* name) {
    // Allocate
    cutlass::HostTensor<bfloat16_t, RowMajor> A({M, K});
    cutlass::HostTensor<bfloat16_t, ColumnMajor> B({K, N});
    cutlass::HostTensor<bfloat16_t, RowMajor> C_cutlass({M, N});
    cutlass::HostTensor<bfloat16_t, RowMajor> C_cublas({M, N});
    
    // Fill with random data
    A.fill_random(-1, 1);
    B.fill_random(-1, 1);
    
    // Run CUTLASS
    run_cutlass_gemm(A, B, C_cutlass);
    
    // Run cuBLAS reference
    run_cublas_gemm(A, B, C_cublas);
    
    // Compare
    double cos_sim = cosine_similarity(C_cutlass, C_cublas);
    double max_err = max_absolute_error(C_cutlass, C_cublas);
    
    printf("[%s] cos_sim=%.6f max_err=%.6f %s\n",
        name, cos_sim, max_err,
        (cos_sim > 0.9999 && max_err < 0.01) ? "PASS" : "FAIL");
}
```

### Fusion Correctness
```cpp
// Test fused addmm_act vs unfused reference
void test_fused_addmm_act(int M, int N, int K) {
    // Reference: GEMM then separate bias+GELU
    run_gemm(X, W, temp);
    add_bias(temp, bias, temp);
    gelu_activation(temp, ref_output);
    
    // CUTLASS: Fused GEMM with EVT bias+GELU
    run_fused_gemm_evt(X, W, bias, fused_output);
    
    // Compare
    assert(cosine_similarity(ref_output, fused_output) > 0.9999);
}
```

### Flash Attention Correctness
```python
def test_flash_attention(Q, K, V):
    # Reference: standard attention
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(64)
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V)
    
    # Flash Attention
    O_flash = flash_attention_kernel(Q, K, V)
    
    # Compare
    cos_sim = F.cosine_similarity(O_ref.flatten(), O_flash.flatten(), dim=0)
    assert cos_sim > 0.999
```

## 2. End-to-End Validation

### Pipeline
1. Load SAM 3.1 model weights (BF16)
2. Load test image + text prompt
3. Run original PyTorch forward pass → save all intermediate outputs
4. Run CUTLASS-optimized forward pass → compare intermediates
5. Compare final segmentation masks

### Metrics
- Per-layer cosine similarity (target > 0.999)
- Max absolute error per layer (target < 0.01)
- Final mIoU on COCO val set (target < 0.5% loss)
- Visual quality: side-by-side mask comparison

## 3. Performance Testing

### Micro-benchmarks
- Individual GEMM latency (all 586 GEMMs)
- Memory bandwidth (stream benchmark)
- Tensor Core utilization (Nsight Compute)
- SM occupancy (Nsight Compute)

### Macro-benchmarks
- End-to-end inference latency
- Per-component breakdown (ViT, CLIP, DETR, FPN)
- Latency vs resolution scaling
- Batch inference throughput

### Roofline Validation
```python
# For each GEMM:
theoretical_peak_tflops = sm_count * ops_per_sm_per_cycle * clock_ghz
actual_tflops = 2 * M * N * K / latency_ns * 1e-6
efficiency = actual_tflops / theoretical_peak_tflops

# Target: >60% Tensor Core utilization for large GEMMs
# Target: >80% memory bandwidth for attention
```

## 4. Regression Testing

### CI Pipeline
```yaml
test_sam31_cutlass:
  script:
    - cmake .. -DCUTLASS_NVCC_ARCHS=100a
    - make sam31_tests -j16
    - ./sam31_tests --all-configs
    - python validate_outputs.py --reference outputs_ref/ --test outputs/
  artifacts:
    - benchmark_results.json
    - correctness_report.txt
```

## 5. Nsight Compute Profiling Guide

### Key Metrics to Monitor
- **sm__throughput.avg.pct_of_peak_sustained_elapsed** — overall SM utilization
- **sm__pipe_tensor_op_active.avg.pct_of_peak_sustained_elapsed** — Tensor Core utilization
- **l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum** — SMEM read throughput
- **gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed** — memory bandwidth utilization
- **sm__warps_active.avg.per_cycle_active** — warp occupancy
- **launch__occupancy** — theoretical occupancy

### Profile Command
```bash
ncu --set full --target-processes all \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_active.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
    ./sam31_inference --image test.jpg --prompt "object"
```
