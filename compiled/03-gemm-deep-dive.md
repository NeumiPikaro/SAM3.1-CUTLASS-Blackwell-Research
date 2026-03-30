# CUTLASS GEMM Implementation — Deep Technical Analysis

## 1. GEMM Mathematical Foundation

GEMM: D = α × op(A) × op(B) + β × C

Where:
- A ∈ ℝ^(M×K), B ∈ ℝ^(K×N), C,D ∈ ℝ^(M×N)
- α, β are scalars
- op() is identity or transpose

SAM 3.1's GEMM operations (per ViT-L/14 block):
- Q projection: X @ Wq — (num_patches × 1024) @ (1024 × 1024) = (1024 × 1024)
- K projection: X @ Wk — same shape
- V projection: X @ Wv — same shape
- Attention: Q @ K^T — (heads × seq × 64) @ (heads × 64 × seq)
- MLP fc1: X @ W1 — (seq × 1024) @ (1024 × 4096) = (seq × 4096)
- MLP fc2: X @ W2 — (seq × 4096) @ (4096 × 1024) = (seq × 1024)

## 2. Threadblock Tiling Strategy

### The Tiling Loop
```
For each threadblock (bid_m, bid_n):
  Load A_tile[Mb, K] from global → shared memory
  Load B_tile[K, Nb] from global → shared memory
  For k = 0 to K step Kb:  // K-dimension iteration
    Compute += A_tile[:, k:k+Kb] @ B_tile[k:k+Kb, :]  // warp-level MMA
  Store C_tile[Mb, Nb] = α * Compute + β * C_tile  // epilogue
```

### Tile Size Selection for SAM 3.1

**ViT Attention QKV projection (M=1024, N=1024, K=1024, BF16):**
- Tile (128, 128, 64): 8×8×16 = 1024 threadblocks
- Tile (256, 128, 64): 4×8×16 = 512 threadblocks
- Tile (128, 256, 64): 8×4×16 = 512 threadblocks
- Best for 5060: (128, 128, 64) — balances occupancy and SMEM usage

**MLP fc1 (M=seq, N=4096, K=1024, BF16):**
- Wider N dimension → favor (128, 256, 64) tiles
- K=1024 → 16 stages of Kb=64
- High arithmetic intensity → compute-bound, Tensor Core limited

**MLP fc2 (M=seq, N=1024, K=4096, BF16):**
- Large K → more stages, deeper pipeline
- (128, 128, 64) with 4-8 stages optimal

## 3. Software Pipelining

### Ampere (SM80): cp.async Pipeline
```
Stage 0: Load A[0], B[0] → SMEM (cp.async)
Stage 1: Load A[1], B[1] → SMEM (cp.async)
         Compute += SMEM_A[0] @ SMEM_B[0]  // mma.sync
Stage 2: Load A[2], B[2] → SMEM (cp.async)
         Compute += SMEM_A[1] @ SMEM_B[1]  // mma.sync
...
```
2-3 stages typical. Each stage = one tile load from global memory.

### Hopper (SM90): TMA Pipeline
```
DMA Warp:     Issue TMA_LOAD for stage N+1 → SMEM
MMA Warp:     WGMMA on SMEM_A[stage_N] @ SMEM_B[stage_N]
Epilogue Warp: Write results from previous tile
```
3-4 stages typical. TMA runs asynchronously — DMA warp just issues descriptors, hardware does the copy.

### Blackwell (SM100): Enhanced TMA + TMEM
```
Same pattern as Hopper but:
- TMA supports 3D descriptors natively
- TMEM (Tensor Memory) can hold intermediate results
- WGMMA throughput is higher per SM
- PDL: kernel can trigger next kernel without CPU round-trip
```
4-8 stages possible with larger SMEM on consumer Blackwell.

## 4. Occupancy & Resource Constraints

### RTX 5060 (SM100, estimated specs)
- SM count: ~36-48 (based on 5060 Ti being 36, 5060 being cut-down)
- SMEM per SM: 228 KB (shared between threadblocks in cluster)
- Register file: 65536 registers per SM (32-bit each)
- Max threads per SM: 2048 (64 warps)
- Max warps per threadblock: 32 (1024 threads)
- Clock: ~2.5 GHz (estimated)

### Occupancy Calculation for BF16 GEMM
With tile (128, 128, 64):
- SMEM per threadblock: A_tile(128×64×2B) + B_tile(64×128×2B) = 16KB + 16KB = 32KB per stage
- 4 stages: 128KB → fits in 228KB with room for epilogue buffers
- Registers: ~64 regs/thread × 256 threads = 16384 regs → 25% of register file
- Blocks per SM: floor(228KB / 128KB) = 1 → 1 block per SM, 256 threads
- Occupancy: 256/2048 = 12.5% → low! 
- Better: 3 stages (96KB SMEM) → 2 blocks per SM → 25% occupancy

**Key insight:** RTX 5060's 228KB SMEM is generous for consumer, but attention kernels (which need Q, K, V, O in SMEM) are SMEM-hungry. Tile selection must balance compute utilization vs occupancy.

## 5. StreamK Load Balancing

Traditional GEMM assigns one tile per threadblock. Problem: if M/N isn't divisible by tile size, some SMs are idle.

StreamK solution: divide K-dimension into chunks and distribute:
```
Standard:  Block0 gets (M0, N0, ALL_K)
StreamK:   Block0 gets (M0, N0, K0:K31), Block1 gets (M0, N0, K31:K63), ...
           Then reduce partial results
```
Benefit: better utilization when M or N is small relative to tile size.

SAM 3.1 DETR has small batch sizes (1-16 images) → StreamK helps when seq_len < tile_size.

