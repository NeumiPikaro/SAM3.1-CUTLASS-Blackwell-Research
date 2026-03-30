# SAM 3.1 × CUTLASS × Blackwell (5060) — Master Research Plan

## Goal
100+ page deep technical report on using NVIDIA CUTLASS to optimize Meta's native SAM 3.1 for Blackwell architecture (RTX 5060 / SM100).

## Architecture
- 50+ sub-agents, each studying a specific slice
- Each agent writes findings to `research/<agent-id>.md`
- Final compilation → `compiled/FINAL_REPORT.md`

## Research Areas

### A. CUTLASS Core (1-12)
1. Library overview + architecture hierarchy
2. CuTe DSL fundamentals
3. GEMM implementations (basic → high-performance)
4. Collective MMA operations
5. Collective mainloop (software pipelining)
6. Epilogue design + fusion
7. Tiled MMA and warp-level
8. TMA (Tensor Memory Accelerator) deep dive
9. Swizzle patterns and shared memory layout
10. Problem shapes and data types
11. Grouped GEMM
12. CUTLASS examples analysis

### B. Blackwell / SM100 (13-18)
13. SM100 architecture + WGMMA
14. TMA on Blackwell (2D/3D copies, multicast)
15. Warp specialization patterns
16. Cluster-level shared memory
17. FP8/INT8/FP4 on Blackwell
18. CUTLASS 4.x new features for Blackwell

### C. Attention Mechanisms (19-24)
19. Flash Attention in CUTLASS (FlashDecoding)
20. Multi-head attention patterns
21. Cross-attention optimization
22. Causal attention + mask handling
23. Variable-length sequence attention
24. Attention + RoPE fusion

### D. SAM 3.1 Architecture Analysis (25-36)
25. ViT-L/14 forward pass — full operator breakdown
26. ViT attention blocks — QKV projection + SDPA
27. ViT MLP blocks — addmm_act fused kernel analysis
28. ViT RoPE implementation — complex tensor ops
29. FPN neck — multi-scale feature extraction
30. CLIP text encoder — operator mapping
31. DETR encoder — cross-attention with mask priors
32. DETR decoder — masked multi-head attention
33. Geometry encoder — 10 sub-module analysis
34. Pixel decoder — conv + upsample
35. Segmentation head — final layers
36. Data flow and memory bandwidth analysis

### E. Optimization Strategies (37-48)
37. GEMM fusion opportunities in SAM 3.1
38. Attention fusion (QKV→SDPA→proj)
39. MLP fusion (fc1→act→fc2 with addmm_act)
40. Custom CUDA kernels for RoPE
41. Memory hierarchy optimization
42. Mixed precision strategy (BF16/FP16/FP8)
43. Kernel launch overhead reduction
44. Async data movement pipelines
45. Graph-level optimization (CUDA graphs)
46. Batch inference optimization
47. Dynamic shape handling
48. End-to-end latency budget decomposition

### F. Implementation & Benchmarking (49-52)
49. CUTLASS kernel implementation patterns
50. Testing and validation framework
51. Benchmark methodology
52. Expected performance analysis vs current

## Status
- Created: 2026-03-30
- Agents spawned: 0/52
