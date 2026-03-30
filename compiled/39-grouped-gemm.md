# Grouped GEMM in CUTLASS — Deep Technical Analysis

## 1. What Is Grouped GEMM

Grouped GEMM executes **multiple GEMM operations with potentially different (Mᵢ, Nᵢ, Kᵢ) sizes in a single kernel launch.** Unlike batched GEMM (identical sizes, stride-separated) or batched array GEMM (identical sizes, pointer arrays), each group in a grouped GEMM has its own arbitrary dimensions.

| Pattern | Problem Sizes | Launch Model |
|---------|--------------|--------------|
| Single GEMM | One (M, N, K) | One kernel |
| Batched GEMM | All identical (M, N, K), stride-separated | One kernel |
| Batched Array GEMM | All identical (M, N, K), pointer arrays | One kernel |
| **Grouped GEMM** | **Each group has its own (Mᵢ, Nᵢ, Kᵢ)** | **One kernel** |

**SAM 3.1 relevance:** The model runs 16 attention heads that may process different sequence lengths. A grouped GEMM processes all 16 heads in one kernel, avoiding per-head launch overhead while handling heterogeneous sizes.

---

## 2. CUTLASS Grouped GEMM API (2.x)

### Class Hierarchy
```
cutlass::gemm::device::GemmGrouped<GemmKernel>      (device API)
  └── cutlass::gemm::device::BaseGrouped<GemmKernel>  (implementation)
        └── cutlass::gemm::kernel::GemmGrouped<...>    (kernel)
              └── GemmGroupedProblemVisitor<>           (scheduling)
```

### Arguments Structure
```cpp
struct Arguments {
    GemmCoord *problem_sizes;      // Array of (M, N, K) per group
    int problem_count;             // Number of groups
    int threadblock_count;         // Total threadblocks to launch
    ElementA **ptr_A;              // Array of A matrix pointers
    ElementB **ptr_B;              // Array of B matrix pointers
    ElementC **ptr_C;              // Array of C matrix pointers
    ElementC **ptr_D;              // Array of D (output) pointers
    int64_t *lda, *ldb, *ldc, *ldd;  // Leading dimensions per group
};
```

Each group computes: **Dᵢ = α · Aᵢ × Bᵢ + β · Cᵢ**

---

## 3. Problem Sorting and Threadblock Assignment

### Sort-by-K Heuristic
`BaseGrouped::sort_problems()` sorts all problems **in descending order by K dimension.** Rationale:
- Groups compute-heavy problems together → better load balancing
- Similar K → similar loop iteration counts → improved I-cache locality
- Avoids one threadblock finishing tiny K while neighbor runs massive K

### Tile Count and Grid Scheduling
Each problem decomposes into tiles: `grid_m = ceil(Mᵢ / TileM)`, `grid_n = ceil(Nᵢ / TileN)`. Total tiles = Σ grid_m × grid_n across all groups.

Grid size = `min(total_tiles, SMs × max_blocks_per_SM)`. Launching more threadblocks than tiles wastes cycles — idle blocks iterate through problems discovering nothing to do.

---

## 4. Scheduling Modes

### kDeviceOnly (Device-Side)
Each threadblock independently determines which problem/tile to process using warp-level prefix sums:
1. Threadblocks launched with 1D grid, each starts at `tile_idx = blockIdx.x`
2. Warp processes 32 problems at a time — each thread loads one problem's tile count
3. Inclusive prefix sum computes cumulative offsets; `__ballot_sync` finds the containing problem
4. Persistent loop: `advance(gridDim.x)` jumps to next unassigned tile

**Pros:** No host-device sync, no precomputation overhead.
**Cons:** O(problems/32) scheduling per threadblock. For many small problems, overhead grows.

### kHostPrecompute (Host-Computed Schedule)
Host builds exact tile→threadblock assignment array, copies to device workspace:
1. `ProblemVisitor::host_precompute()` builds flat `(problem_idx, tile_start)` array
2. Copied to device memory via `cudaMemcpyAsync`
3. Threadblocks index directly into precomputed table → O(1) per tile
4. Schedule prefetched into shared memory in chunks

**Pros:** Zero device-side scheduling overhead. Better for many small problems.
**Cons:** Requires host-device sync + workspace allocation.

### For SAM 3.1: kDeviceOnly is sufficient
16 heads → warp processes all 16 in a single shuffle iteration. Scheduling overhead is negligible.

---

## 5. CUTLASS 3.x: Hopper/Blackwell Grouped GEMM

The CUTLASS 3.x API uses `GroupProblemShape` abstraction with TMA-based descriptor modification:

```cpp
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
```

| Feature | CUTLASS 2.x (`GemmGrouped`) | CUTLASS 3.x (`GroupProblemShape`) |
|---------|---------------------------|----------------------------------|
| Kernel | `GemmGrouped` kernel struct | `GemmUniversal` with `GroupProblemShape` |
| Scheduling | `ProblemVisitor` warp shuffles | TMA-based, device-modified descriptors |
| TMA support | N/A | On-the-fly TMA descriptor modification |
| Data types | Standard | FP8 with fast accumulation |
| Warp specialization | No | Yes (cooperative or pingpong) |

**Key advantage:** TMA descriptors are modified on-the-fly to switch between groups — no separate pointer-load instructions needed. The TMA descriptor encodes address + stride for the current group.

Hopper config options:
- **Cooperative:** `TileShape<256,128,128>`, `ClusterShape<1,2,1>` — larger tiles for big problems
- **Pingpong:** `TileShape<128,128,128>`, `ClusterShape<2,1,1>` — smaller tiles for many small problems

Blackwell (SM100) extends with example 75 (`75_blackwell_grouped_gemm`).

---

## 6. Multi-Head Attention with Grouped GEMM

### Computation Pattern
```
For head h in [0, H):
    Q_h = X · W_Q[h]     // (seq_len × head_dim)
    K_h = X · W_K[h]
    V_h = X · W_V[h]
    Attention_h = softmax(Q_h · K_h^T / √d) · V_h
```

### SAM 3.1 Application
With 16 heads and variable sequence lengths:
1. **Q·Kᵀ:** 16 groups, each (Mᵢ × d_k) — Mᵢ varies per head
2. **Attn·V:** 16 groups, each (Mᵢ × d_v)
3. **Single kernel launch** — no per-head loop overhead

Alternative: pad all heads to max sequence length → batched GEMM wastes compute on padding tokens. Grouped GEMM avoids this entirely.

CUTLASS example 41 (`41_multi_head_attention`) directly demonstrates non-fixed sequence length attention.

---

## 7. StreamK + Grouped GEMM

StreamK addresses **load imbalance from heterogeneous K dimensions** in grouped GEMM:
- Without StreamK: threadblock assigned to K=128 finishes quickly while K=8192 neighbor runs 64× longer → SMs idle
- With StreamK: total K-work across all problems divided into chunks; threadblocks steal K-dimension work from other problems

For SAM 3.1 with 16 heads: benefit is modest (few groups), but helps if head sizes vary dramatically.

---

## 8. Performance: Grouped vs Individual GEMMs

| Metric | Individual GEMMs (16) | Grouped GEMM |
|--------|----------------------|--------------|
| Kernel launches | 16 | 1 |
| Launch overhead | 80-160 μs (5-10 μs each) | ~5-10 μs |
| CPU scheduling | Per-launch setup | Single setup |
| GPU idle gaps | Pipeline bubbles between launches | None |
| Memory access | Contiguous per problem | Non-contiguous (scattered pointers) |

CUTLASS benchmarks show **2-5× speedup** over individual launches for moderate group counts (10-100). Near-parity with batched GEMM when all problems have same size.

---

## 9. Limitations

1. **Alignment:** All matrices must satisfy tile-size alignment constraints. Misaligned problems waste memory/compute via padding.
2. **Homogeneous data types:** All groups must use same types, layouts, tile configs. No mixing FP16/BF16 in one call.
3. **Shared tile shape:** Small problems waste most of a large tile. Tile must accommodate smallest expected problem.
4. **No split-K across groups:** Each problem's K handled by single threadblock. StreamK partially addresses this.
5. **Hopper+ only for TMA-based:** CUTLASS 3.x grouped GEMM requires SM90+. Ampere uses CUTLASS 2.x API.
6. **No inter-problem fusion:** Each group's epilogue is independent.

---

## 10. Recommendations for SAM 3.1

```
Target:     Hopper (SM90) or Blackwell (SM100)
API:        CUTLASS 3.x with GroupProblemShape
Schedule:   kDeviceOnly (16 heads → trivial scheduling)
Tile:       128×128×128 (pingpong) for moderate per-head sizes
Data types: FP16/BF16 accumulators, FP8 inputs if available
Sort:       Descending by sequence length (proxy for K dimension)
```

Key CUTLASS examples: `24_gemm_grouped` (2.x), `41_multi_head_attention`, `57_hopper_grouped_gemm` (3.x), `75_blackwell_grouped_gemm` (SM100).

---

## References

- `include/cutlass/gemm/device/base_grouped.h` — BaseGrouped implementation
- `include/cutlass/gemm/device/gemm_grouped.h` — GemmGrouped device API
- `include/cutlass/gemm/kernel/grouped_problem_visitor.h` — Scheduling algorithms
- `examples/24_gemm_grouped/` — CUTLASS 2.x grouped GEMM
- `examples/41_multi_head_attention/` — Variable sequence length attention
- `examples/57_hopper_grouped_gemm/` — Hopper grouped GEMM with FP8
- `examples/75_blackwell_grouped_gemm/` — Blackwell SM100 grouped GEMM
