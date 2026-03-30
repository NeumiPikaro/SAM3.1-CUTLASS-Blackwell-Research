# Advanced Optimization Techniques

## 1. Programmatic Dependent Launch (PDL)

PDL allows kernel B to start on the same SMs as kernel A without CPU intervention.

### Implementation
```cpp
// Kernel A: ViT Attention
__global__ void vit_attention_kernel(...) {
    // ... compute attention ...
    
    // Signal next kernel to start
    cudaTriggerProgrammaticLaunchCompletion();
}

// Kernel B: ViT MLP — starts immediately on same SMs
__global__ void vit_mlp_kernel(...) {
    // ... compute MLP ...
    cudaTriggerProgrammaticLaunchCompletion();
}
```

### SAM 3.1 Chain
```
Block 0 Attn → PDL → Block 0 MLP → PDL → Block 1 Attn → PDL → ...
```
Saves ~5-15μs per transition × 64 transitions (32 blocks × 2) = **320-960μs**

## 2. Persistent Kernels

Instead of launching 256 separate GEMMs, launch one persistent kernel:
```cpp
__global__ void persistent_vit_kernel(GemmTask* tasks, int num_tasks) {
    int task_id = blockIdx.x;
    while (task_id < num_tasks) {
        // Execute GEMM task[task_id]
        execute_gemm(tasks[task_id]);
        task_id += gridDim.x;  // Next task for this block
    }
}
```
Benefit: eliminated kernel launch overhead, better L2 cache reuse.

## 3. Weight Prefetching

Preload next layer's weights into L2 cache while current layer computes:
```cpp
// During ViT block N computation:
cudaMemPrefetchAsync(W_qkv_next, W_size, device, stream);
// Next block's weights are in L2 when needed
```
L2 cache (32 MB) can hold ~1.5 layers' weights (~20 MB each).

## 4. StreamK for Variable-Length DETR

StreamK divides K-dimension across SMs for better load balancing:
```cpp
// Standard: 1 block per (M_tile, N_tile) — some SMs idle if M small
// StreamK: split K across blocks, partial results need reduction
using DETR_Gemm = GemmUniversal<
    Shape<int,int,int>,
    CollectiveBuilder<...,
        cutlass::gemm::StreamKScheduler  // StreamK mode
    >::CollectiveOp, ...>;
```
For DETR decoder (M=100): StreamK keeps all 36 SMs busy.

## 5. Asynchronous Pipeline Depth Tuning

For RTX 5060 with 228 KB SMEM:
```
BF16 tile (128, 128, 64): 16 KB + 16 KB = 32 KB per stage
Max stages: 228 / 32 = 7
Sweet spot: 4-5 stages (balance pipeline depth vs occupancy)
```

For attention (needs Q, K, V, O in SMEM):
```
Q(64x64x2B) + K(64x64x2B) + V(64x64x2B) + S(64x64x4B) = 40 KB per stage
Max stages: 228 / 40 = 5
Use 3 stages (leave room for other SMEM needs)
```

## 6. Tensor Memory (TMEM) for DETR

Blackwell's TMEM can hold intermediate attention scores:
```
Q@K^T → TMEM (attention scores, 100x1024 x FP32 = 400 KB)
But TMEM is 256 KB per SM — need to tile
Solution: process attention in chunks that fit TMEM
```

## 7. Mixed Precision Scheduling

```cpp
// Different precision for different components
struct PrecisionPolicy {
    // ViT attention: BF16 (accuracy sensitive)
    using AttnQKV = bfloat16_t;
    // ViT MLP: FP8 (compute heavy, less accuracy sensitive)  
    using MLP = float_e4m3_t;
    // DETR: BF16 (small, accuracy critical)
    using DETR = bfloat16_t;
    // FPN: BF16
    using FPN = bfloat16_t;
};
```

## 8. L2 Cache Residence Hints

```cpp
// Keep frequently-used weights in L2
cudaMemAdvise(W_qkv, size, cudaMemAdviseSetAccessedBy, device);
cudaMemAdvise(W_qkv, size, cudaMemAdviseSetPreferredLocation, device);

// For ViT: process 2 blocks at a time (~40 MB fits in 32 MB L2 with eviction)
```
