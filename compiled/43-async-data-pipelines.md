# Async Data Movement Pipelines for SAM 3.1

## 1. The Memory Wall Problem

SAM 3.1's ViT-L/14 processes a 1024×1024 image through 32 transformer blocks. Each block reads its input from global memory, computes attention + MLP, and writes output back. The RTX 5060's 448 GB/s memory bandwidth is the bottleneck:

| Operation | Data Movement | Bandwidth Needed | Compute |
|-----------|--------------|-----------------|---------|
| fc1 (per block) | Read X: 11MB, Read W1: 8MB, Write out: 44MB | 63 MB | 45 GFLOP |
| Attention (per block) | Read X: 11MB, Read QKV weights: 24MB, Read/Write attention | ~100 MB | 52 GFLOP |
| fc2 (per block) | Read act: 44MB, Read W2: 8MB, Write out: 22MB | 74 MB | 45 GFLOP |
| **Total per block** | **~237 MB** | | **~142 GFLOP** |

**Arithmetic intensity:** 142 GFLOP / 237 MB = 0.6 FLOP/byte. The RTX 5060 needs ~33 FLOP/byte to saturate compute. This means the GPU is **heavily bandwidth-bound**.

Async pipelines hide memory latency by overlapping transfers with computation.

---

## 2. CUDA Async Copy Architecture

### Hopper/Blackwell Async Copy Hierarchy
```
Global Memory (HBM)  →  Shared Memory (SMEM)  →  Register File
         ↑                        ↑                      ↑
    TMA (hardware)          cp.async / TMA           LDGMMA
   (zero SM overhead)     (warp-level async)      (Tensor Core load)
```

### TMA (Tensor Memory Accelerator)
TMA is a hardware unit that handles global→SMEM copies asynchronously:
- **Zero SM overhead:** TMA engine copies data while warps compute
- **Multidimensional:** Supports 2D/3D tiled copies with swizzle
- **Multicast:** One TMA copy can deliver to multiple SMs in a cluster

```cpp
// TMA async copy: initiate transfer, continue computing
tma::cp_async_bulk<gmem_to_smem>(&smem_buf, gmem_ptr, tma_desc);
// ^ Returns immediately. Data arrives in SMEM later.
// Meanwhile, warps continue computing with previous tile's data.
```

---

## 3. Software Pipelining in CUTLASS Mainloop

### The Core Pattern: Load Stage N+1 While Computing Stage N
CUTLASS mainloops maintain multiple "stages" in shared memory — circular buffers that allow overlapping load and compute:

```
Time →
Stage 0: [LOAD K=0] [COMPUTE K=0] [LOAD K=4] [COMPUTE K=4] ...
Stage 1:           [LOAD K=1] [COMPUTE K=1] [LOAD K=5] ...
Stage 2:                     [LOAD K=2] [COMPUTE K=2] ...
Stage 3:                               [LOAD K=3] [COMPUTE K=3] ...
```

With 4 stages, the GPU is simultaneously:
- Loading K-tile 4 into Stage 0 (TMA, hardware)
- Computing K-tile 3 using Stage 3 (Tensor Cores)
- Previous stages idle (awaiting reuse)

### Pipeline Depth Selection
| Stages | SMEM Usage | Latency Hiding | Best For |
|--------|-----------|----------------|----------|
| 2 | Minimal | 1 transfer ahead | Short K, SMEM-constrained |
| 3 | Moderate | 2 transfers ahead | Balanced (default) |
| 4 | High | 3 transfers ahead | Long K, hiding TMA latency |

For SAM 3.1's fc1 (M=5376, N=4096, K=1024), 3 stages is optimal:
- K=1024 ÷ 64 (tile_K) = 16 iterations
- 3 stages = pipeline fills in 2 iterations, steady state for 14 iterations
- Pipeline fill overhead: 2/16 = 12.5%

---

## 4. TMA Pipeline Barriers

### Producer-Consumer Synchronization
TMA uses hardware barriers (not software fences) for synchronization between producer (TMA engine) and consumer (MMA warps):

```cpp
// Producer (TMA) side:
pipeline.producer_acquire(barrier);     // Wait for consumer to release buffer
tma::cp_async_bulk(smem_buf, gmem, desc);  // Initiate async copy
pipeline.producer_commit(barrier);      // Signal copy initiated

// Consumer (MMA) side:
pipeline.consumer_wait(barrier);        // Wait for copy to complete
wgmma.execute(smem_buf_a, smem_buf_b);  // Compute with loaded data
pipeline.consumer_release(barrier);     // Release buffer for reuse
```

### Phase Bit for Circular Buffer Wrap
```cpp
struct PipelineState {
    int index_;       // Current slot [0, Stages)
    uint32_t phase_;  // Alternates 0/1 on wrap-around
    uint32_t count_;  // Total advances
};
```

The phase bit solves the ABA problem: when the circular buffer wraps, a barrier can distinguish "buffer full from previous cycle" vs "buffer empty in current cycle."

---

## 5. Overlapping Compute and Memory Transfers in SAM 3.1

### Attention Block Pipeline
```
Timeline for one ViT attention block:
                                    
TMA:     [Load QKV tiles] [Load K/V tiles for attn] [Load output proj weights]
Tensor:                  [QKV GEMM]                [SDPA compute]   [Output GEMM]
Epilogue:                                         [Write attn out]

Ideal overlap: TMA always 1-2 tiles ahead of compute
```

### MLP Block Pipeline
```
TMA:     [Load X tile + W1]  [Load W2 + residual]  [Load next block's X]
Tensor:  [fc1 GEMM]          [GELU + fc2 GEMM]    
Epilogue:                     [Write fc2 + residual]
```

### End-to-End ViT Pipeline (Across Blocks)
The biggest opportunity is overlapping **block N+1's weight loads with block N's computation:**

```
Block N compute:    [Attn GEMMs → SDPA → MLP GEMMs]
Block N+1 loads:              [Prefetch Wqkv, W1, W2 for block N+1]
```

This requires:
1. **Double-buffered weight staging:** Weights for block N+1 loaded into a separate SMEM region while block N computes
2. **TMA multicast (cluster mode):** One TMA copy delivers weights to all SMs processing block N+1
3. **Persistent kernel:** Same kernel processes all 32 blocks, swapping weight pointers

---

## 6. Cluster-Level Async Pipelines (Blackwell SM100)

### Thread Block Clusters
Blackwell supports clusters of up to 16 thread blocks that share SMEM via distributed shared memory (DSM):

```
Cluster (2×2 blocks):
  [Block 0] ←SMEM→ [Block 1]
     ↕ DSM            ↕ DSM
  [Block 2] ←SMEM→ [Block 3]
```

### Multicast TMA
A single TMA copy can deliver data to multiple blocks in a cluster:
```cpp
// Copy from global memory to SMEM of all blocks in cluster simultaneously
tma::cp_async_bulk_multicast<gmem_to_smem>(
    smem_buf, gmem_ptr, tma_desc, cluster_mask);
```

**SAM 3.1 benefit:** Weight matrices (Wqkv: 3×1024×64×2B = 384KB per head) can be multicast to all blocks processing different heads — 4× reduction in HBM reads for shared weights.

---

## 7. CUDA Streams for Pipeline Parallelism

### Stream-Based Overlapping
Different pipeline stages can run on different CUDA streams:

```cpp
cudaStream_t compute_stream, transfer_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&transfer_stream);

// Stream 1: H2D weight transfer for next batch
cudaMemcpyAsync(d_weights_next, h_weights, size, 
                cudaMemcpyHostToDevice, transfer_stream);

// Stream 2: Compute current batch (independent of transfer)
run_vit_forward(d_input, d_weights_current, compute_stream);

// Both run concurrently — transfer hidden behind compute
```

### Multi-Stream for SAM 3.1
```
Stream 0: ViT forward pass (compute)
Stream 1: H2D weight prefetch (next block/layer)
Stream 2: D2H result copy (previous block output if needed)
Stream 3: CLIP text encoder (independent of ViT)
```

---

## 8. Practical Implementation: CUTLASS Pipeline Setup

### Mainloop Configuration (Hopper/Blackwell)
```cpp
using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmma<
    3,           // Stages (K_PIPE_MAX)
    Shape<_128, _128, _64>,   // Cluster shape
    2            // PipelineAsyncMmaStages (K_PIPE_MMAS)
>;

// K_PIPE_TMAS = K_PIPE_MAX - K_PIPE_MMAS = 3 - 2 = 1
// Meaning: 1 TMA in-flight, 2 MMAs in-flight
```

### TMA Descriptor Setup
```cpp
// Create TMA descriptor for A matrix tile
auto tma_desc_A = make_tma_copy(
    SM90_TMA_LOAD{},                    // TMA load operation
    make_tensor(gmem_A, layout_A),      // Global memory tensor
    SmemLayoutA{},                      // Shared memory layout
    make_shape(128, 64),                // Tile shape
    _1{}                                // Pipeline stages (1 for async)
);
```

---

## 9. Expected Performance Impact

| Optimization | Technique | Latency Saved | Complexity |
|-------------|-----------|--------------|------------|
| Software pipelining (3-stage) | CUTLASS mainloop | 30-40% per GEMM | Automatic (template param) |
| TMA async copies | Hardware unit | Zero SM overhead | Requires SM90+ |
| Block-to-block prefetch | CUDA graphs + persistent kernel | 100-200 μs total | Medium |
| Multicast TMA | Cluster mode (SM100) | 25% weight reads | Requires SM100 |
| Multi-stream overlap | CUDA streams | 50-100 μs (H2D overlap) | Low |

**Combined pipeline optimization: ~15-25% end-to-end latency reduction** on SAM 3.1 inference.

---

## References

- `include/cutlass/pipeline/` — CUTLASS pipeline primitives (PipelineAsync, PipelineTma)
- `include/cutlass/arch/mma_sm90.h` — TMA instruction definitions
- `examples/52_hopper_gemm_with_collective_builder/` — TMA pipeline example
- `compiled/06-tma-deep-dive.md` — Full TMA reference
- `compiled/04-collective-mma-mainloop.md` — Mainloop software pipelining
