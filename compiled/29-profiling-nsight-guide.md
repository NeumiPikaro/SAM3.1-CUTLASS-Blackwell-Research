# Profiling & Nsight Compute Deep Guide

## 1. Why Profile?

CUTLASS provides near-optimal kernels, but optimal *configuration* requires profiling:
- Tile size selection (128×128 vs 256×128 vs 128×256)
- Pipeline stage count (3 vs 4 vs 5)
- Cluster shape ((1,1,1) vs (1,2,1) vs (2,2,1))
- Warp specialization (pingpong vs cooperative)

## 2. Nsight Compute Metrics That Matter

### Compute Efficiency
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
  → Overall SM utilization. Target: >70%
  
sm__pipe_tensor_op_active.avg.pct_of_peak_sustained_elapsed
  → Tensor Core utilization. Target: >60% for compute-bound GEMMs
  
sm__inst_executed_pipe_tensor_op.avg.per_cycle_active
  → Tensor Core instructions per cycle. Target: >0.5
```

### Memory Efficiency
```
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
  → DRAM bandwidth utilization. Target: >80% for memory-bound ops

l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum
  → Shared memory read sectors. Compare against ideal.

l1tex__t_sector_hit_rate.pct
  → L1/SMEM cache hit rate. Target: >95%
```

### Occupancy & Scheduling
```
launch__occupancy
  → Theoretical occupancy. Sweet spot: 50-75%

sm__warps_active.avg.per_cycle_active
  → Active warps per cycle. Higher = better latency hiding

smsp__thread_inst_executed_per_inst_executed.ratio
  → Warp divergence indicator. Target: >30 (close to 32)
```

## 3. SAM 3.1 Profiling Script

```bash
#!/bin/bash
# profile_sam31_cutlass.sh

KERNELS="vit_qkv vit_mlp_fc1 vit_mlp_fc2 flash_attention detr_cross_attn"

for KERNEL in $KERNELS; do
    echo "=== Profiling: $KERNEL ==="
    
    ncu --set full \
        --target-processes all \
        --kernel-name "$KERNEL" \
        --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_active.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy,\
sm__warps_active.avg.per_cycle_active,\
l1tex__t_sector_hit_rate.pct \
        --output "profile_${KERNEL}" \
        ./sam31_inference --profile-kernel $KERNEL
        
    ncu --import "profile_${KERNEL}.ncu-rep" --csv > "profile_${KERNEL}.csv"
done
```

## 4. Expected Nsight Output (ViT QKV GEMM)

```
Metric                                          | Value  | Target | Status
------------------------------------------------|--------|--------|-------
SM throughput                                    | 82%    | >70%   | ✓
Tensor Core utilization                          | 71%    | >60%   | ✓
DRAM bandwidth utilization                       | 45%    | varies | ✓ (compute-bound)
Theoretical occupancy                            | 62%    | 50-75% | ✓
Active warps per cycle                           | 42     | >30    | ✓
L1/SMEM hit rate                                 | 97%    | >95%   | ✓
Shared memory bank conflicts                     | 2%     | <5%    | ✓
```

## 5. Common Issues & Fixes

### Low Tensor Core Utilization (<50%)
**Cause:** Tile size too small, not enough work per warp
**Fix:** Increase tile M or N dimension. Try (256,128,64) instead of (128,128,64)

### Low SM Throughput with High Tensor Core Utilization
**Cause:** Non-matmul operations (epilogue, softmax) dominating
**Fix:** Fuse more operations into epilogue/EVT

### High DRAM Utilization with Low Compute
**Cause:** Memory-bound (small GEMMs, attention without Flash)
**Fix:** Use Flash Attention, batch small GEMMs, use FP8

### Low Occupancy (<30%)
**Cause:** Too much SMEM per threadblock
**Fix:** Reduce pipeline stages, reduce tile size, or reduce cluster size

### High Bank Conflicts (>10%)
**Cause:** Wrong swizzle pattern
**Fix:** Use Swizzle<3,3,3> for BF16, Swizzle<4,3,3> for FP8

## 6. Roofline Plot Generation

```python
import matplotlib.pyplot as plt
import numpy as np

# SAM 3.1 GEMMs: (AI in FLOP/Byte, achieved TFLOPS)
gemms = {
    'ViT QKV':     (819, 980),
    'ViT MLP FC1': (819, 1050),
    'ViT MLP FC2': (819, 1020),
    'Attention QK': (30,  450),
    'Attention PV': (30,  440),
    'DETR FC1':    (85,  380),
    'Flash Attn':  (100, 680),
}

# Roofline
bw = 448  # GB/s
peak = 1474  # TFLOPS
ai = np.logspace(0, 4, 100)
roofline = np.minimum(ai * bw, peak)

plt.figure(figsize=(10, 6))
plt.loglog(ai, roofline, 'k-', linewidth=2, label='Roofline')
for name, (ai_val, tflops) in gemms.items():
    plt.loglog(ai_val, tflops, 'o', markersize=10, label=name)
plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
plt.ylabel('Performance (TFLOPS)')
plt.title('SAM 3.1 Roofline - RTX 5060')
plt.legend()
plt.grid(True)
plt.savefig('roofline_sam31.png', dpi=150)
```

