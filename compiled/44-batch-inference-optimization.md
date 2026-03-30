# Batch Inference Optimization for SAM 3.1 on Blackwell

## 1. Why Batch Inference Matters

SAM 3.1 is compute-bound at batch=1 but bandwidth-bound at larger batches. The RTX 5060's 1474 TFLOPS (BF16) can process multiple images nearly in parallel because:
- Weight reads are shared across the batch (read once, use for all images)
- Larger GEMMs have higher arithmetic intensity
- GPU occupancy improves with more work

### SAM 3.1 Batch Scaling (Projected RTX 5060 BF16)
| Batch Size | Latency | Throughput | Efficiency |
|-----------|---------|------------|------------|
| 1 | 390 ms | 2.6 img/s | 100% (baseline) |
| 2 | 650 ms | 3.1 img/s | 119% |
| 4 | 1100 ms | 3.6 img/s | 138% |
| 8 | 1900 ms | 4.2 img/s | 162% |

Batch=8 gives 1.62× the throughput of batch=1 while using only 4.9× the time — significant efficiency gain.

---

## 2. Batched GEMM: The Core Technique

### What Changes with Batch Size
At batch=1, each GEMM operates on (seq, d_model). With batch B, the GEMM becomes (B×seq, d_model) — a single larger GEMM, or a batched GEMM with B independent problems.

### CUTLASS Batched GEMM Options

**Option A: Single large GEMM (merge batch into M dimension)**
```
Batch=1:  GEMM(M=5376, N=3072, K=1024)
Batch=4:  GEMM(M=21504, N=3072, K=1024)   // 4× M dimension
```
- Pro: Simplest, highest arithmetic intensity
- Con: Requires contiguous input layout (B×seq, d_model)
- Con: Can't handle variable image sizes in batch

**Option B: Batched GEMM (strided)**
```
Batch=4:  BatchedGEMM(M=5376, N=3072, K=1024, batch=4)
```
- Pro: Independent problems, natural layout
- Pro: CUTLASS `GemmBatched` or `GemmUniversal` with batch stride
- Con: Slightly lower occupancy than merged GEMM

**Option C: Grouped GEMM (variable sizes)**
```
Batch=4:  GroupedGEMM with 4 groups, each with different M
```
- Pro: Handles variable image resolutions per batch element
- Pro: CUTLASS 3.x `GroupProblemShape`
- Con: Most complex API

### For SAM 3.1: Option A (merged) for fixed resolution, Option C for variable resolution

---

## 3. Weight Sharing Across Batch

### The Key Insight
Weights (Wqkv, W1, W2, Wo, etc.) are identical for all batch elements. Only the input activations differ. This means:

1. **Weights load once per kernel** — same SMEM buffers reused for all batch elements
2. **L2 cache benefit** — second and subsequent batch elements hit cached weights
3. **Register reuse** — weight tiles in Tensor Core registers stay valid across batch iterations

### Quantifying Weight Read Savings
| Weights | Size | Per-Block Reads (B=1) | Per-Block Reads (B=4) | Savings |
|---------|------|-----------------------|-----------------------|---------|
| Wqkv | 6 MB | 6 MB | 6 MB (shared) | 3× |
| W1 | 8 MB | 8 MB | 8 MB (shared) | 3× |
| W2 | 8 MB | 8 MB | 8 MB (shared) | 3× |
| **Total weights** | **~22 MB** | **22 MB** | **22 MB** | **66 MB saved** |

---

## 4. Batch-Aware Tiling Strategy

### Tile Shape Selection for Batch
Larger effective M (B×seq) enables more efficient tile usage:

| Batch | Effective M | Tiles (128-wide) | Tile Utilization |
|-------|------------|-------------------|-----------------|
| 1 | 5376 | 42 tiles | 5376/5376 = 100% |
| 2 | 10752 | 84 tiles | 100% |
| 4 | 21504 | 168 tiles | 100% |
| 8 | 43008 | 336 tiles | 100% |

Larger batches → more tiles → better GPU occupancy (more independent work units for the scheduler).

### Cluster Shape for Batch
With batch=4, use larger clusters to improve data reuse:
```
Batch=1:  ClusterShape<1,1,1> (no cluster, each block independent)
Batch=4:  ClusterShape<2,2,1> (4 blocks share weight loads via multicast TMA)
Batch=8:  ClusterShape<2,4,1> or <4,2,1>
```

---

## 5. Memory Allocation Strategy

### Per-Batch Memory Budget
For SAM 3.1 at 1024×1024 resolution, BF16:

| Component | Per-Image | B=1 | B=4 | B=8 |
|-----------|----------|-----|-----|-----|
| Input image | 6 MB | 6 MB | 24 MB | 48 MB |
| ViT activations | 44 MB | 44 MB | 176 MB | 352 MB |
| Attention scores | 115 MB | 115 MB | 460 MB | 920 MB |
| DETR activations | 22 MB | 22 MB | 88 MB | 176 MB |
| Masks output | 16 MB | 16 MB | 64 MB | 128 MB |
| **Total** | **~203 MB** | **203 MB** | **812 MB** | **1624 MB** |
| **Weights (shared)** | **~150 MB** | **150 MB** | **150 MB** | **150 MB** |

RTX 5060 with 8GB VRAM can handle B=8 comfortably. With 4GB: B=4 max.

### Activation Memory Optimization
Use **activation checkpointing** — don't store all intermediate activations for backprop (inference-only):
- Re-run LayerNorm during backward (not needed for inference)
- Free attention score matrices immediately after SDPA
- Re-use activation buffers between blocks via persistent allocation

---

## 6. CUDA Graph Capture for Batch

### Why Graphs Matter for Batch
At B=8, the ViT executes ~500+ kernel launches (32 blocks × ~16 kernels per block). Each launch has ~5-10 μs overhead → 2.5-5 ms total overhead. CUDA graphs eliminate this:

```cpp
// Capture one ViT block
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
attention_block(input, weights, stream);
mlp_block(input, weights, stream);
cudaStreamEndCapture(stream, &block_graph);

// Replay 32 times with different weight pointers
for (int i = 0; i < 32; i++) {
    update_graph_weights(block_graph, layer_weights[i]);
    cudaGraphLaunch(block_graph, stream);
}
```

### Batch-Specific Graph Optimization
- Capture separate graphs for each batch size (B=1,2,4,8)
- Each graph has optimized kernel parameters for that batch size
- Graph instantiation overhead: ~100-500 μs (amortized over many inferences)

---

## 7. Dynamic Batch Sizing

### Mixed Batch Dispatch
Real-world inference may have varying numbers of images. Strategy:
1. Accumulate images until batch target reached or timeout
2. Dispatch to pre-captured CUDA graph for that batch size
3. Pad incomplete batches to nearest power of 2 (B=1,2,4,8)

### Padding Waste Analysis
| Actual Batch | Padded To | Compute Waste |
|-------------|-----------|---------------|
| 3 | 4 | 33% |
| 5 | 8 | 60% |
| 7 | 8 | 14% |
| 6 | 8 | 33% |

**Better approach:** Use grouped GEMM with per-image M sizes — no padding needed. Each image gets its own M dimension (may differ due to different resolutions or prompt counts).

---

## 8. Multi-Image Parallel Strategies

### Strategy A: Data Parallel (Same Model, Different Images)
All images processed through the same model with batched GEMMs. Simplest approach.

### Strategy B: Pipeline Parallel (Different Model Stages)
Image 1 in ViT while Image 2 in DETR while Image 3 in CLIP:
```
Time:   [ViT(img1)] [DETR(img1)] [SegHead(img1)]
              [ViT(img2)] [DETR(img2)] [SegHead(img2)]
```
- Pro: Better hardware utilization across different pipeline stages
- Con: Requires careful memory management, higher latency per image

### Strategy C: Hybrid
Batch within each pipeline stage, pipeline across stages:
```
ViT: process [img1, img2, img3, img4] as batch
  → output feeds into
DETR: process [img1, img2, img3, img4] as batch
  → output feeds into
SegHead: process [img1, img2, img3, img4] as batch
```
Best throughput for SAM 3.1 — combines batched GEMM efficiency with pipeline structure.

---

## 9. Expected Performance Summary

| Configuration | Latency | Throughput | VRAM Usage |
|--------------|---------|------------|------------|
| B=1, no optimization | 390 ms | 2.6 img/s | 350 MB |
| B=1, with CUDA graphs | 350 ms | 2.9 img/s | 350 MB |
| B=4, batched GEMM | 1100 ms | 3.6 img/s | 960 MB |
| B=4, batched + graphs | 1000 ms | 4.0 img/s | 960 MB |
| B=8, batched + graphs | 1800 ms | 4.4 img/s | 1770 MB |

---

## References

- `compiled/27-cuda-graphs-deep-dive.md` — CUDA Graphs full reference
- `compiled/39-grouped-gemm.md` — Grouped GEMM for variable batch sizes
- `compiled/21-performance-modeling.md` — Roofline model and projections
- CUTLASS `examples/32_batched_gemm/` — Batched GEMM examples
