# GPU Memory Management for SAM 3.1 Inference

## 1. The Allocation Problem

cudaMalloc takes ~1ms per call. SAM 3.1 needs ~500MB of activation memory per inference.
If we allocate/free per inference: 50 allocations × 1ms = 50ms overhead — unacceptable for <50ms target.

## 2. Solution: Pre-allocate Everything

### Static Memory Pool
```python
# Pre-allocate all memory at model load time
workspace = torch.empty(500 * 1024 * 1024, dtype=torch.uint8, device='cuda')  # 500MB
# Use views into workspace for each tensor
Q = workspace[0:seq*1024*2].view(torch.bfloat16).reshape(seq, 1024)
K = workspace[seq*2048:seq*4096].view(torch.bfloat16).reshape(seq, 1024)
# ... etc
```

### CUDA Memory Pools
```python
pool = torch.cuda.MemPool()
with torch.cuda.mem_pool(pool):
    # All allocations come from pool — no system calls
    Q = torch.empty(seq, 1024, dtype=torch.bfloat16, device='cuda')
```

### PyTorch Caching Allocator
PyTorch already caches allocations. But:
- First inference is slow (cold start)
- Fragmentation can cause OOM
- Solution: warmup with dummy inference before benchmarking

## 3. Alignment for TMA

TMA requires 16-byte (128-bit) alignment:
```python
# Ensure all tensors are 16-byte aligned
Q = torch.empty(seq, 1024, dtype=torch.bfloat16, device='cuda')
assert Q.data_ptr() % 16 == 0  # PyTorch guarantees this
```

## 4. SAM 3.1 Memory Budget (RTX 5060, 8GB VRAM)

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (BF16) | 3.4 GB | Load once |
| Activations (per inference) | 500 MB | Pre-allocate |
| Scratch workspace | 100 MB | Pre-allocate |
| CUDA overhead | 200 MB | Context, streams |
| FlashInfer workspace | 50 MB | JIT kernels |
| **Total** | **4.25 GB** | **Fits in 8 GB** |

With FP8 weights: 1.7 GB + 500 MB + overhead = ~2.5 GB — very comfortable.

## 5. Best Practices
1. Pre-allocate all tensors at model load
2. Use torch.cuda.memory_pool for systematic allocation
3. Run warmup inference before benchmarking
4. Monitor fragmentation with torch.cuda.memory_stats()
5. Use 256-byte alignment for optimal TMA performance

