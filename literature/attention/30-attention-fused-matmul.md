# Fused Attention MatMul — Eliminating Materialization

## The Problem
Standard attention materializes the S = Q@K^T matrix in HBM:
```
Q: (seq, d) × K: (seq, d)^T → S: (seq, seq) in HBM
S is 1024×1024 × 4B = 4MB per head × 16 heads = 64MB
```
This is pure memory traffic that Flash Attention eliminates.

## But Can We Go Further?

### Fused QKV + Attention
Instead of:
```
Q = X @ Wq  → HBM
K = X @ Wk  → HBM
V = X @ Wv  → HBM
Attn(Q, K, V) → HBM
```
Do:
```
Q,K,V = X @ W_qkv (fused) → keep Q,K,V in registers/SMEM
Attn(Q, K, V) → directly in same kernel
```
Saves: 3 HBM round-trips for Q, K, V intermediate

### Fused Attention + Output Projection
```
Standard: Attn → HBM → O = Attn @ Wo → HBM
Fused: O = Attn @ Wo (Wo loaded into attention kernel's SMEM)
Saves: 1 HBM round-trip of (seq, d) per block
```

### Ultimate: QKV + Attention + Output in One Kernel
```
Q,K,V = X @ W_qkv (GEMM in registers)
O = Attention(Q,K,V) (in SMEM)
O = O @ Wo (GEMM with Wo in SMEM)
Write O to HBM

Total HBM traffic: read X + read W_qkv + read Wo + write O
No intermediate writes!
```

## CUTLASS Implementation Strategy

### Step 1: GEMM → SMEM staging
Use CUTLASS epilogue to write Q,K,V to SMEM instead of HBM.

### Step 2: Attention kernel reads from SMEM
Flash Attention reads Q,K,V from previous kernel's SMEM (via distributed SMEM or PDL).

### Step 3: Output projection also from SMEM
Attention result stays in SMEM, output GEMM reads from SMEM.

## Challenge
Distributed shared memory between kernels requires:
- PDL (kernels on same SM can share SMEM)
- Or cluster-level SMEM sharing
- CUTLASS 4.4 supports both

## Expected Impact
- Eliminates 3 × (seq, d) × 2B = 6KB per head × 16 = 96KB HBM traffic per block
- Plus output projection: +64KB saved
- Total: ~160KB saved per block × 32 blocks = 5MB HBM traffic saved
- At 448 GB/s: 5MB / 448e9 = 11μs saved
- **Not huge, but contributes to <50ms target**

