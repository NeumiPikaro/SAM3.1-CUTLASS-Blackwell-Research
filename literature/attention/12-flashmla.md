# FlashMLA — DeepSeek's Multi-Latent Attention

## What is MLA
Multi-Latent Attention (DeepSeek-V2/V3) compresses KV into latent vectors:
```
Standard: K (seq, d), V (seq, d) → 2 × seq × d memory
MLA: c (seq, d_c), K = W_k @ c, V = W_v @ c → seq × d_c memory (d_c << d)
```
Memory reduction: d/d_c (e.g., 1024/128 = 8×)

## FlashInfer MLA Support
FlashInfer has native MLA attention support. Key features:
- Compressed KV cache
- On-the-fly K, V decompression
- Optimized for decode phase

## SAM 3.1 Applicability
SAM 3.1 does NOT use MLA — it uses standard MHA.
MLA would require:
- Retraining the model (not possible)
- Different weight format (compressed vs full)

**Not applicable without retraining.**

## Why Include Here
Understanding MLA's memory savings inspires similar approaches:
- Could we compress KV cache for SAM 3.1 DETR? (decoder attends to encoder output repeatedly)
- SVD-based weight compression? (not quantization, but structural)
- These are research ideas, not immediate optimizations

