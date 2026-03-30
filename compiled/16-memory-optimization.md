# Memory Hierarchy & Bandwidth Optimization

## 1. Memory Hierarchy on RTX 5060

```
┌─────────────────────────────────┐
│  Global Memory (GDDR7)          │  8 GB, 448 GB/s
│  ┌───────────────────────────┐  │
│  │ L2 Cache                  │  │  32 MB, ~5000 GB/s
│  │ ┌─────────────────────┐   │  │
│  │ │ SM Shared Memory     │   │  │  228 KB per SM, ~100 TB/s
│  │ │ ┌─────────────────┐  │   │  │
│  │ │ │ Register File    │  │   │  │  64K × 32-bit per SM
│  │ │ │ ┌─────────────┐  │  │   │  │
│  │ │ │ │ Tensor Core  │  │  │   │  │  Direct operand feed
│  │ │ │ └─────────────┘  │  │   │  │
│  │ │ └─────────────────┘  │   │  │
│  │ └─────────────────────┘   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

## 2. Bandwidth at Each Level

| Level | Size | Bandway | Latency |
|-------|------|---------|---------|
| Register File | 256 KB/SM | ~500 TB/s | 0 cycles |
| Shared Memory | 228 KB/SM | ~100 TB/s | ~20 cycles |
| L2 Cache | 32 MB | ~5 TB/s | ~100 cycles |
| Global (GDDR7) | 8 GB | 448 GB/s | ~300 cycles |

## 3. Arithmetic Intensity Analysis

Arithmetic Intensity (AI) = FLOPs / Bytes moved

### ViT Attention Q@K^T
```
FLOPs: 2 × 1024 × 1024 × 64 = 134 MFLOP (per head)
Bytes: Q(1024×64×2B) + K(64×1024×2B) + S(1024×1024×4B) = 4.5 MB
AI: 134M / 4.5M = 29.8 FLOP/Byte
Roofline: Well above bandwidth knee → compute-bound ✓
```

### ViT MLP FC1
```
FLOPs: 2 × 1024 × 4096 × 1024 = 8.6 GFLOP
Bytes: X(1024×1024×2B) + W(1024×4096×2B) + Out(1024×4096×2B) = 10.5 MB
AI: 8600M / 10.5M = 819 FLOP/Byte
Roofline: Deep compute-bound — Tensor Core limited ✓
```

### DETR Decoder FC1
```
FLOPs: 2 × 100 × 2048 × 256 = 102 MFLOP
Bytes: X(100×256×2B) + W(256×2048×2B) + Out(100×2048×2B) = 1.2 MB
AI: 102M / 1.2M = 85 FLOP/Byte
Roofline: Compute-bound but small — latency matters more
```

## 4. Optimization Strategies

### 4.1 Reduce Global Memory Traffic
- **Weight sharing:** Load weights once, reuse across batch
- **Activation checkpointing:** Recompute instead of store (training only)
- **KV cache:** For decoder, store K,V from previous steps
- **FP8 weights:** Half the memory traffic for weight loads

### 4.2 Maximize L2 Cache Hit Rate
```
ViT layer weights per block: ~20 MB (Wq,Wk,Wv,Wo,W1,W2)
L2 cache: 32 MB — can hold 1.5 layers' weights
Strategy: Process 1-2 layers at a time, let weights stay hot in L2
```

### 4.3 Shared Memory Efficiency
- **Swizzle:** Use 128B swizzle for BF16 tiles (avoids bank conflicts)
- **Double-buffering:** Overlap load and compute via 2+ stages
- **Avoid SMEM spills:** Keep intermediate results in registers

### 4.4 Register File Management
```
Per SM: 64K registers × 4 bytes = 256 KB
Per thread (2048 threads): 32 registers
Per thread (512 threads): 128 registers

For GEMM with 512 threads:
- A operand tile: 128×64 × 2B / 512 = 32 bytes = 8 registers
- B operand tile: 64×128 × 2B / 512 = 32 bytes = 8 registers
- Accumulator: 128×128 × 4B / 512 = 128 bytes = 32 registers
- Total: ~48 registers → plenty of headroom
```

## 5. Roofline Analysis for RTX 5060

```
Peak Compute: 1474 TFLOPS (BF16)
Peak Bandwidth: 448 GB/s
Compute knee: 1474000/448 = 3290 FLOP/Byte

GEMM AI > 3290 → compute-bound
Attention AI = 30 → bandwidth-bound (without Flash Attention)
MLP AI = 819 → compute-bound

With Flash Attention: AI improves to 60-100 → approaching compute-bound
```

## 6. Swizzle Configuration for SAM 3.1

### BF16 Tiles (2 bytes/element)
```
Tile (128, 64): 128×64×2B = 16 KB in SMEM
32 banks, 4-byte stride
Bank = (address / 4) % 32
For row-major: consecutive elements hit different banks ✓
For column-major: every 2nd element same bank ✗

Solution: 64B swizzle (Swizzle<3,3,3>)
Remaps XOR bits [2:0] of address → eliminates column conflicts
```

### FP8 Tiles (1 byte/element)
```
Tile (128, 128): 128×128×1B = 16 KB
Bank conflicts worse (denser packing)
Solution: 128B swizzle (Swizzle<4,3,3>)
```

