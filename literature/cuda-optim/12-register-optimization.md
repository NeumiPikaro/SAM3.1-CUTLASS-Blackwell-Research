# CUDA Register Optimization for SAM 3.1

## Register File on RTX 5060
- 65536 registers per SM (32-bit each) = 256 KB
- 2048 max threads per SM = 32 registers per thread at 100% occupancy
- 512 threads = 128 registers per thread

## GEMM Register Usage
For tile (128, 128, 64) BF16:
- A operand: 128x64x2B / 256 threads = 64 bytes = 16 regs
- B operand: 64x128x2B / 256 threads = 64 bytes = 16 regs
- Accumulator: 128x128x4B / 256 threads = 256 bytes = 64 regs
- Total: ~96 registers per thread

With 512 threads: 96 x 512 = 49152 regs = 75% of register file. Leaves room for other warps.

## CUTLASS Register Strategy
CUTLASS automatically manages register allocation via template parameters. Key controls:
- maxrregcount: compiler flag to limit registers per thread
- Warp specialization: producer warps use fewer registers (40), consumer warps use more (232)
- ThunderKittens approach: explicitly set register counts per warp group

## Optimization for SAM 3.1
- Attention warps: need registers for Q, K, V, S, O tiles
- MLP warps: need registers for X, W, intermediate
- Separate warp groups with different register budgets
- Target: 75-85% register utilization without spilling

## Profiling
```bash
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./sam31_inference
```
Target: ratio > 30 (minimal warp divergence, efficient register use).

