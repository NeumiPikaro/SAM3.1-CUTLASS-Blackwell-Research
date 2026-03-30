# Attention Sink Optimization for SAM 3.1

## What Are Attention Sinks?
In softmax attention, some tokens receive disproportionate attention weight (often 30-50% of total). These "sink" tokens absorb attention that can't be assigned elsewhere due to softmax normalization.

## In Vision Transformers
ViTs with CLS tokens often have the CLS token as a sink — it attends heavily to a few patches.
Without CLS (SAM 3.1 ViT has no CLS token): sinks may appear at first/last patches or at patches with high visual salience.

## Can We Exploit This?
If sink patterns are predictable:
1. Precompute sink attention weight → skip softmax for sink positions
2. Cache sink contribution to output → reuse across inferences
3. Reduce computation for sink tokens (they don't need fine-grained attention)

## SAM 3.1 Investigation Needed
```python
# Analyze attention patterns in SAM 3.1
for layer in model.image_encoder.trunk.blocks:
    attn_weights = layer.attn.attention_weights  # (heads, seq, seq)
    max_per_head = attn_weights.max(dim=-1)  # which tokens get most attention?
    # If max attention > 0.5 consistently → sink pattern exists
```

## CUTLASS Support
Example 93 GQA kernel supports attention sinks natively:
```
--attention_sink  # flag to handle sink tokens specially
```

## Expected Impact
If sink optimization saves 10-20% of attention compute:
- Attention portion: 153ms → 122-138ms
- Total: 58ms → 49-52ms

## Verdict
Research opportunity — need to measure sink patterns in SAM 3.1 before optimizing.

