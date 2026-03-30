# Efficient Vision Transformers Survey

## EfficientViT (MIT)
Linear attention replaces O(N²) self-attention. 10× faster than ViT-L. Hardware-aware architecture design.
**SAM 3.1:** Can't change architecture (must use Meta weights). But hardware-aware kernel design principles apply.

## Token Merging (ToMe)
Merge similar tokens mid-forward. 2-3× speedup, <1% accuracy loss.
**SAM 3.1:** Risky for segmentation — loses spatial detail. Not recommended for image encoder.

## Early Exit
Stop at layer N if confident. 30-50% compute savings.
**SAM 3.1:** Segmentation needs all features. Not safe for production.

## DynamicViT
Learned token pruning policy.
**SAM 3.1:** Can't predict which patches are important a priori. Not applicable.

## Bottom Line
None of these work without retraining. For <50ms, focus on kernel-level optimization (CUTLASS, FlashInfer, fusion).

