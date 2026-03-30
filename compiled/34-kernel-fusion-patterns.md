# Kernel Fusion Patterns for SAM 3.1

## 1. Fusion Taxonomy

### Type 1: Epilogue Fusion (Zero-Cost)
Operations fused into GEMM epilogue — no extra memory traffic:
```
GEMM + bias → EVT: Sm90RowBroadcast + Sm90Compute<plus>
GEMM + GELU → EVT: Sm90Compute<gelu>
GEMM + bias + GELU → EVT: chain of Sm90EVT nodes
```

### Type 2: Mainloop Fusion
Operations fused into GEMM computation loop:
```
Attention: Q@K^T → softmax → P@V in single mainloop
Flash Attention: online softmax fused with GEMM
```

### Type 3: Kernel Chain Fusion
Multiple kernels merged by sharing data through registers/TMEM:
```
FC1 output → GELU → FC2 input (all in register file)
No SMEM round-trip for intermediate
```

### Type 4: Graph-Level Fusion
Multiple operations captured in CUDA graph:
```
Block N: Attn → MLP → Norm → Block N+1 Attn (single graph)
```

## 2. ViT Block Fusion Analysis

### Unfused (Baseline)
```
Op 1: LayerNorm(x)                    — 1 kernel
Op 2: Q = X @ Wq                      — 1 GEMM kernel
Op 3: K = X @ Wk                      — 1 GEMM kernel
Op 4: V = X @ Wv                      — 1 GEMM kernel
Op 5: RoPE(Q, K)                      — 1 kernel
Op 6: S = Q @ K^T / sqrt(d)          — 1 GEMM kernel (×16 heads)
Op 7: P = softmax(S)                  — 1 kernel (×16 heads)
Op 8: C = P @ V                       — 1 GEMM kernel (×16 heads)
Op 9: O = C @ Wo                      — 1 GEMM kernel
Op 10: x = x + O                      — 1 kernel (residual)
Op 11: LayerNorm(x)                   — 1 kernel
Op 12: h = X @ W1 + bias              — 1 GEMM kernel
Op 13: h = GELU(h)                    — 1 kernel
Op 14: o = h @ W2                     — 1 GEMM kernel
Op 15: x = x + o                      — 1 kernel (residual)

Total: 15 kernel launches per block × 32 blocks = 480 launches
```

### Partially Fused (CUTLASS Epilogue)
```
Op 1: LayerNorm(x)                    — 1 kernel
Op 2: Q||K||V = X @ W_qkv            — 1 GEMM kernel (fused QKV)
Op 3: RoPE(Q, K)                      — 1 kernel (or fused into Op 2)
Op 4: Flash Attention(Q, K, V)        — 1 kernel (fused QK^T + softmax + PV)
Op 5: O = C @ Wo + residual           — 1 GEMM kernel (fused residual)
Op 6: LayerNorm(x)                    — 1 kernel
Op 7: h = X @ W1 + bias + GELU       — 1 GEMM kernel (fused addmm_act)
Op 8: o = h @ W2 + residual           — 1 GEMM kernel (fused residual)

Total: 8 kernel launches per block × 32 blocks = 256 launches
Reduction: 46% fewer launches
```

### Fully Fused (Custom Kernels)
```
Op 1: LayerNorm(x)                    — 1 kernel
Op 2: Q||K||V = X @ W_qkv + RoPE     — 1 GEMM kernel (fused all)
Op 3: Flash Attention(Q, K, V) + proj — 1 kernel (fused output proj)
Op 4: LayerNorm(x)                    — 1 kernel
Op 5: MLP(x) = W2(GELU(X@W1+b1))+b2 — 1 kernel (fully fused MLP)

Total: 5 kernel launches per block × 32 blocks = 160 launches
Reduction: 67% fewer launches
```

## 3. DETR Block Fusion

### Cross-Attention Fusion
```
Standard:
  Q = dec_out @ Wq     — GEMM
  K = enc_out @ Wk     — GEMM  
  V = enc_out @ Wv     — GEMM
  S = Q @ K^T          — GEMM
  P = softmax(S)       — kernel
  C = P @ V            — GEMM
  O = C @ Wo           — GEMM

Fused:
  Q = dec_out @ Wq     — GEMM
  KV = enc_out @ Wkv   — 1 GEMM (fused K and V)
  Flash Attn(Q, K, V) + proj — 1 kernel
```

### MLP Fusion
Same as ViT: FC1 + bias + GELU + FC2 + residual in one or two kernels.

## 4. Attention + Output Projection Fusion

Standard attention writes intermediate to HBM, then output proj reads it:
```
Attn output (seq, d) → HBM → Output GEMM reads from HBM
Memory traffic: 2 × seq × d × 2 bytes
```

Fused attention + output projection:
```
Attn computes O = P@V, then immediately does O @ Wo
Intermediate stays in SMEM or registers
Memory traffic: 0 extra bytes
```

CUTLASS approach: custom mainloop that includes output projection as final step.

## 5. Residual Connection Fusion

Standard:
```
GEMM writes to HBM
Add kernel reads GEMM output + residual from HBM, writes to HBM
Total: 3 × M × N × sizeof(element) memory traffic
```

Fused:
```
GEMM epilogue loads residual tile from HBM (or SMEM if prefetched)
Adds to accumulator in register file
Writes result to HBM
Total: 2 × M × N × sizeof(element) memory traffic (33% savings)
```

CUTLASS EVT:
```cpp
Sm90EVT<Sm90Compute<plus>, Sm90AccFetch, Sm90AuxLoad<float>>
```

## 6. LayerNorm Fusion with Attention

LayerNorm can be fused with the preceding residual add:
```
Standard:
  x = x + attn_out  — add kernel
  x = layernorm(x)  — norm kernel
  
Fused:
  x = layernorm(x + attn_out)  — single kernel
  
Savings: 1 kernel launch, 1 HBM round-trip
```

Custom CUDA kernel:
```cuda
__global__ void fused_residual_layernorm(
    float* out, const float* x, const float* residual,
    const float* gamma, const float* beta, int N, int D) {
    
    int row = blockIdx.x;
    // Compute mean and variance of (x + residual)
    float mean = 0, var = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[row*D + i] + residual[row*D + i];
        mean += val;
    }
    // ... block reduction for mean, var
    // Normalize and apply gamma, beta
}
```

