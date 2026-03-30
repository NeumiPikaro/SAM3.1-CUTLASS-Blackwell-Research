# Grouped GEMM in CUTLASS: Deep Technical Analysis

## 1. Introduction: What Is Grouped GEMM?

Grouped GEMM is a computation pattern where **multiple GEMM operations with potentially different problem sizes (M, N, K)** are launched in a single kernel invocation. This stands in contrast to three related patterns:

| Pattern | Problem Sizes | Launch Model |
|---------|--------------|--------------|
| **Single GEMM** | One (M, N, K) | One kernel |
| **Batched GEMM** | All identical (M, N, K), stride-separated | One kernel |
| **Batched Array GEMM** | All identical (M, N, K), pointer arrays | One kernel |
| **Grouped GEMM** | **Each group has its own (Mᵢ, Nᵢ, Kᵢ)** | One kernel |

The critical distinction: in grouped GEMM, each "group" (problem) in the batch can have **arbitrarily different dimensions**. Problem metadata (sizes, pointers, leading dimensions) is stored in device-memory arrays and loaded by the kernel at runtime.

**Why this matters for SAM 3.1:** The Segment Anything Model 3.1 runs 16 attention heads, each potentially processing different sequence lengths or operating on different-sized slices. A grouped GEMM can process all 16 heads in a single kernel launch, avoiding the overhead of 16 separate launches while handling heterogeneous problem sizes naturally.

---

## 2. CUTLASS Grouped GEMM API Architecture

### 2.1 Class Hierarchy

CUTLASS's grouped GEMM is built on a clean layered architecture:

```
cutlass::gemm::device::GemmGrouped<GemmKernel>     (device-level API)
  └── cutlass::gemm::device::BaseGrouped<GemmKernel> (base implementation)
        └── cutlass::gemm::kernel::GemmGrouped<...>   (kernel-level)
              └── GemmGroupedProblemVisitor<>          (scheduling logic)
                    └── GroupedProblemVisitor<>         (device/host scheduler)
```

**`GemmGrouped`** (in `gemm_grouped.h`) is a thin wrapper — just a type alias inheriting from `BaseGrouped`:

```cpp
template <typename GemmKernel_>
class GemmGrouped : public BaseGrouped<GemmKernel_> {
public:
  using GemmKernel = GemmKernel_;
};
```

All real logic lives in **`BaseGrouped`** (`base_grouped.h`).

### 2.2 The Arguments Structure

The kernel's `Arguments` struct holds arrays of per-problem metadata in device memory:

```cpp
struct Arguments {
    GemmCoord *problem_sizes;      // Array of (M, N, K) per group
    int problem_count;             // Number of groups
    int threadblock_count;         // Total threadblocks to launch

    ElementA **ptr_A;              // Array of A matrix pointers
    ElementB **ptr_B;              // Array of B matrix pointers
    ElementC **ptr_C;              // Array of C matrix pointers
    ElementC **ptr_D;              // Array of D (output) matrix pointers

    int64_t *lda, *ldb, *ldc, *ldd;  // Leading dimensions per group

    GemmCoord *host_problem_sizes; // Host-side copy for precomputation
};
```

Each group `i` computes: **Dᵢ = α · Aᵢ × Bᵢ + β · Cᵢ**, where Aᵢ is Mᵢ×Kᵢ, Bᵢ is Kᵢ×Nᵢ, and Dᵢ is Mᵢ×Nᵢ.

### 2.3 Workspace and Initialization

`BaseGrouped` manages a workspace for precomputed scheduling data:

- **`get_workspace_size()`** — queries the `ProblemVisitor` for required workspace bytes
- **`precompute()`** — calls `ProblemVisitor::host_precompute()` to build the schedule on the host, then copies it to device memory via `cudaMemcpyAsync`
- **`initialize()`** — builds params from arguments + workspace
- **`update()`** — lightweight re-initialization when only problem sizes change (reuse workspace)

---

## 3. Problem Sorting and Threadblock Assignment

### 3.1 Sort-by-K Heuristic

`BaseGrouped` provides a static `sort_problems()` method that sorts all problems **in descending order by K dimension**:

```cpp
static void sort_problems(int problem_count,
                          GemmCoord* problem_sizes_ptr,
                          int64_t* lda_host_ptr, int64_t* ldb_host_ptr,
                          int64_t* ldc_host_ptr, int64_t* ldd_host_ptr,
                          int64_t* offset_A_ptr, int64_t* offset_B_ptr,
                          int64_t* offset_C_ptr, int64_t* offset_D_ptr)
{
    std::vector<size_t> indices(problem_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
      [&problem_sizes_ptr](size_t i, size_t j) {
        return problem_sizes_ptr[i].k() > problem_sizes_ptr[j].k();
      });
    // Reorder all arrays according to sorted indices...
}
```

**Rationale:** Sorting by K (the reduction dimension) groups compute-heavy problems together. This helps with:
- **Warp-level scheduling:** Warps that are co-located in the same scheduling wave process problems of similar computational weight
- **Load balancing:** Avoids a scenario where one threadblock gets a tiny problem while its neighbor gets a massive one, causing serialization
- **Cache behavior:** Similar K dimensions mean similar loop iteration counts, improving instruction cache locality

### 3.2 Tile Count Computation

Each problem is decomposed into tiles. The grid shape for a single problem is:

```cpp
static GemmCoord grid_shape(const GemmCoord& problem) {
    return GemmCoord(
        ceil_div(problem.m(), ThreadblockShape::kM),
        ceil_div(problem.n(), ThreadblockShape::kN),
        1);
}
```

The total tile count across all groups is the sum of `grid_m × grid_n` for each problem.

### 3.3 Threadblock Count Selection

`sufficient()` computes the optimal number of threadblocks:

```cpp
static int sufficient(const GemmCoord* problem_sizes_ptr,
                      int problem_count, int available_sm_count) {
    // 1. Compute occupancy-based block count: SMs × max_active_blocks
    int occupancy_based = available_sm_count * max_active_blocks;

    // 2. Compute total tile count across all problems
    int total_tiles = group_tile_count(problem_sizes_ptr, problem_count);

    // 3. Single problem: return exact tile count
    if (problem_count == 1) return total_tiles;

    // 4. Otherwise: min(tiles, occupancy_blocks)
    return std::min(total_tiles, occupancy_based);
}
```

The key insight: launching more threadblocks than tiles wastes cycles because idle threadblocks still iterate through problem sizes to discover they have nothing to do. So the grid is capped at `min(total_tiles, SMs × max_blocks_per_SM)`.

---

## 4. Host-Side vs Device-Side Problem Scheduling

CUTLASS supports two scheduling modes, controlled by the `GroupScheduleMode` enum:

### 4.1 `kDeviceOnly` — Fully Device-Side Scheduling

All scheduling decisions happen on the GPU. Each threadblock (via its `ProblemVisitor`) independently determines which problem and tile to work on.

**Algorithm (warp-cooperative search):**

1. Threadblocks are launched with a 1D grid of size `threadblock_count`
2. Each threadblock starts at `tile_idx = blockIdx.x`
3. The warp processes 32 problems at a time:
   - Each of the 32 threads in a warp loads one problem's tile count
   - A warp-level **inclusive prefix sum** computes cumulative tile offsets
   - The warp checks if `tile_idx` falls within the current batch of 32 problems
   - If not, the warp advances to the next batch (group of 32 problems)
4. Within a problem, `threadblock_idx = tile_idx - problem_tile_start` determines the 2D position in the problem's tile grid

**Warp-level prefix sum detail:**
```cpp
CUTLASS_PRAGMA_UNROLL
for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
    int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
    if (lane_idx >= i) {
        problem_ending_tile += val;
    }
}
```

Then `__ballot_sync` finds which problem contains the current tile:
```cpp
int32_t problem_idx_in_group =
    __popc(__ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));
```

**Pros:** No host-device synchronization, no precomputation overhead, simple API.
**Cons:** Each threadblock does O(problems/32) work to find its tile. For many small problems, scheduling overhead can dominate.

### 4.2 `kHostPrecompute` — Host-Computed Schedule

The host precomputes the exact assignment of tiles to threadblocks, stores it in workspace memory, and threadblocks simply index into it.

**Algorithm:**
1. Host calls `ProblemVisitor::host_precompute()` which builds a flat array of `(problem_idx, tile_start)` entries
2. This array is copied to device memory (the "workspace")
3. Each threadblock computes: `iterations_per_block = ceil(tile_count / gridDim.x)`
4. Threadblock `i` processes tiles `[i × iterations_per_block, (i+1) × iterations_per_block)`
5. The schedule is **prefetched into shared memory** in chunks of `kPrefetchTileCount`

```cpp
struct SharedStorage {
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
};
```

**Pros:** Zero scheduling overhead on device — threadblocks just read a precomputed table. Better for many small problems.
**Cons:** Requires host-device synchronization and workspace allocation. Extra memory overhead.

### 4.3 Choosing Between Modes

| Criterion | kDeviceOnly | kHostPrecompute |
|-----------|-------------|-----------------|
| Setup overhead | None | Precomputation + memcpy |
| Device scheduling cost | O(P/32) per threadblock | O(1) per tile |
| Memory overhead | 0 | Workspace array |
| Best for | Few large problems | Many small problems |
| Dynamic problem changes | Easy (update arrays) | Requires re-precompute |

For SAM 3.1's 16 attention heads, `kDeviceOnly` is likely sufficient — 16 problems is a small number, and the warp processes them in a single iteration (16 < 32).

---

## 5. The Kernel Body: How Each Tile Executes

Inside `GemmGrouped::operator()`, the persistent loop structure is:

```cpp
ProblemVisitor problem_visitor(params.problem_visitor,
                                shared_storage.problem_visitor,
                                blockIdx.x);

while (problem_visitor.next_tile()) {
    GemmCoord problem_size  = problem_visitor.problem_size();
    int32_t problem_idx     = problem_visitor.problem_index();
    int32_t threadblock_idx = problem_visitor.threadblock_idx();

    GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

    // Compute 2D tile position within this problem's grid
    GemmCoord threadblock_offset(
        (threadblock_idx / grid_shape.n()) * ThreadblockShape::kM,
        (threadblock_idx % grid_shape.n()) * ThreadblockShape::kN,
        0);

    // Load per-problem pointers from device arrays
    ElementA *ptr_A = params.ptr_A[problem_idx];
    ElementB *ptr_B = params.ptr_B[problem_idx];

    // Construct iterators, run MMA, run epilogue
    // ...

    // Advance to next tile (persistent kernel pattern)
    problem_visitor.advance(gridDim.x);
}
```

Each iteration:
1. **Identifies** which problem this tile belongs to
2. **Loads** the problem-specific pointers and leading dimensions from global memory arrays
3. **Constructs** tile iterators with the correct offsets
4. **Executes** the threadblock-level matrix multiply-accumulate
5. **Runs** the epilogue to write results to the per-problem output buffer

The `advance(gridDim.x)` call implements a **persistent kernel** pattern — threadblocks don't terminate after one tile, they jump to the next unassigned tile.

---

## 6. CUTLASS 3.x: Hopper Grouped GEMM (Modern API)

The CUTLASS 3.x API (targeting Hopper SM90+) provides a fundamentally different grouped GEMM implementation using the **`GroupProblemShape`** abstraction:

```cpp
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
// Each group has its own <M, N, K>
```

Key differences from the CUTLASS 2.x approach:

| Feature | CUTLASS 2.x (`GemmGrouped`) | CUTLASS 3.x (`GroupProblemShape`) |
|---------|---------------------------|----------------------------------|
| Kernel | `GemmGrouped` kernel struct | `GemmUniversal` with `GroupProblemShape` |
| Scheduling | `ProblemVisitor` with warp shuffles | TMA-based, device-modified descriptors |
| TMA support | N/A | On-the-fly TMA descriptor modification |
| Data types | Standard | FP8 (e4m3/e5m2) with fast accumulation |
| Warp specialization | No | Yes (cooperative or pingpong) |

The Hopper example (example 57) uses **TMA (Tensor Memory Accelerator) descriptors** that are modified on-the-fly to move between groups. This eliminates the need for separate pointer-load instructions — the TMA descriptor itself encodes the current group's address and stride.

**Hopper configuration options:**
- **Cooperative:** `TileShape<256,128,128>`, `ClusterShape<1,2,1>` — larger tiles, good for big problems
- **Pingpong:** `TileShape<128,128,128>`, `ClusterShape<2,1,1>` — smaller tiles, better for many small problems

Blackwell (SM100) extends this further with example 75 (`75_blackwell_grouped_gemm`).

---

## 7. Grouped GEMM for Multi-Head Attention

### 7.1 The Attention Computation Pattern

In a standard multi-head attention (MHA) with H heads:

```
For head h in [0, H):
    Q_h = X · W_Q[h]     // (seq_len × head_dim)
    K_h = X · W_K[h]     // (seq_len × head_dim)
    V_h = X · W_V[h]     // (seq_len × head_dim)
    Attention_h = softmax(Q_h · K_h^T / √d) · V_h  // (seq_len × head_dim)
    Output = concat(Attention_0, ..., Attention_{H-1}) · W_O
```

### 7.2 When Heads Have Different Sizes

In SAM 3.1, heads may process different sequence lengths (e.g., due to variable-length prompts, image patches with different resolutions, or sparse attention patterns). This creates a natural grouped GEMM:

- **Group 0:** GEMM with (M₀, N, K) — head 0 with sequence length M₀
- **Group 1:** GEMM with (M₁, N, K) — head 1 with sequence length M₁
- ...
- **Group 15:** GEMM with (M₁₅, N, K) — head 15 with sequence length M₁₅

CUTLASS's example 41 (`41_multi_head_attention`) directly demonstrates this: "attention example with non-fixed sequence length input."

### 7.3 SAM 3.1 Specifics

SAM 3.1 runs 16 attention heads. With grouped GEMM:

1. **Q·Kᵀ computation:** 16 groups, each with Mᵢ (variable sequence length) × d_k (head dimension)
2. **Attention·V computation:** 16 groups, each with Mᵢ × d_v
3. **All 16 heads processed in one kernel launch** — no loop, no per-head kernel overhead

For the Q projection (X → Q across all heads), if all heads share the same head dimension but different sequence lengths, grouped GEMM handles this naturally. The alternative — padding all heads to the maximum sequence length and using batched GEMM — wastes compute on padding tokens.

---

## 8. StreamK Applied to Grouped GEMM

### 8.1 StreamK Background

StreamK is a parallel decomposition strategy for GEMM that addresses **load imbalance** in the K (reduction) dimension. Traditional data-parallel decomposition assigns each threadblock a tile of the output matrix. StreamK allows threadblocks to collaboratively process K-dimension chunks, enabling better load balancing.

CUTLASS provides StreamK via `gemm_universal_streamk_with_broadcast.h` and example 47 (`47_ampere_gemm_universal_streamk`).

### 8.2 StreamK + Grouped GEMM

When combined with grouped GEMM, StreamK addresses a critical issue: **problems with very different K dimensions create severe load imbalance.**

Without StreamK:
- A threadblock assigned to a K=128 problem finishes quickly
- A threadblock assigned to K=8192 runs for 64× longer
- GPU utilization drops because fast-finishing blocks leave SMs idle

With StreamK in grouped GEMM:
- The total K-dimension work across all problems is divided into StreamK chunks
- Threadblocks can steal K-dimension work from problems they weren't initially assigned to
- The "tail" of each problem (last K-chunk) is computed collaboratively

**For SAM 3.1:** If attention heads have significantly different sequence lengths (and thus different K dimensions in Q·Kᵀ), StreamK can smooth out the workload. However, with only 16 heads, the benefit may be modest compared to cases with hundreds or thousands of groups.

---

## 9. Performance: Grouped GEMM vs Individual GEMMs

### 9.1 Overhead Reduction

Running 16 individual GEMM kernel launches incurs:
- **Kernel launch overhead:** ~5-10 μs per launch on modern GPUs → 80-160 μs total
- **CPU-side scheduling:** Problem setup, pointer configuration per launch
- **GPU idle gaps:** Pipeline bubbles between kernel completions and next launches

Grouped GEMM eliminates all of this with a single launch.

### 9.2 CUTLASS Benchmark Evidence

The example 24 (`24_gemm_grouped`) includes explicit comparison against running individual batched GEMMs:

> "Problem sizes are collected and binned to compute the same problem as a series of conventional batched GEMMs (setup for this problem is not timed). This demonstrates the performance enhancement achieved by implementing a specialized grouped GEMM kernel."

Typical results from CUTLASS grouped GEMM benchmarks show:
- **2-5× speedup** over individual GEMM launches for moderate group counts (10-100 groups)
- **Near-parity** with batched GEMM when all problems have the same size (grouped GEMM adds minimal overhead)
- **Diminishing returns** when individual problems are very large (compute dominates over launch overhead)

### 9.3 Memory Access Patterns

Grouped GEMM has a potential disadvantage: **non-contiguous memory access**. Each problem's matrices may be scattered across device memory (different pointer per group). This can cause:
- TLB thrashing if pointers are spread across many memory pages
- Reduced L2 cache reuse between problems
- Extra pointer loads from global memory (one load per problem per pointer)

CUTLASS mitigates this by:
- Sorting problems to co-locate similar-sized problems
- Using persistent kernel pattern to maximize L2 cache temporal locality
- In CUTLASS 3.x, TMA descriptors cache address information

---

## 10. Use Cases in Transformer Architectures

### 10.1 Multi-Head Attention with Variable-Length Sequences

The primary use case. Different heads (or different batch elements within a head) may have different sequence lengths due to:
- **Causal masking** with different prompt lengths
- **Sparse attention** patterns (e.g., local + global attention)
- **Variable-resolution inputs** in vision transformers (SAM 3.1's use case)

### 10.2 Mixture of Experts (MoE)

In MoE models, different experts process different numbers of tokens. Each expert's GEMM has a different M dimension. Grouped GEMM processes all experts in one kernel.

### 10.3 Multi-Query Attention / Grouped-Query Attention

Models like Llama 2 use grouped-query attention (GQA) where K/V heads are shared across multiple Q heads. Grouped GEMM can handle the different effective batch sizes:
- Q projections: more groups (one per Q head)
- K/V projections: fewer groups (one per K/V head group)

### 10.4 Flash Attention Integration

Flash Attention 2/3 and xFormers internally use grouped-GEMM-like patterns when handling variable sequence lengths within a batch. CUTLASS's grouped GEMM provides the building blocks for these implementations.

### 10.5 SAM 3.1 Pipeline

```
Image Encoder → Patch Tokens (variable count per resolution)
                    ↓
Grouped GEMM: Project patches to Q, K, V per head
  - Group 0: head 0, seq_len = f(patch_count_0)
  - Group 1: head 1, seq_len = f(patch_count_1)
  - ...
  - Group 15: head 15, seq_len = f(patch_count_15)
                    ↓
Grouped GEMM: Attention scores (Q·K^T per head)
                    ↓
Grouped GEMM: Attention output (Attn·V per head)
                    ↓
Concatenate heads → Output projection
```

---

## 11. Limitations and Design Constraints

### 11.1 Alignment Requirements

All matrices must satisfy the alignment constraints of the underlying tile size:
- M dimensions must be compatible with `ThreadblockShape::kM`
- N dimensions must be compatible with `ThreadblockShape::kN`
- K dimensions must be compatible with `ThreadblockShape::kK`
- Leading dimensions must satisfy the `kAlignmentA` / `kAlignmentB` / `kAlignmentC` requirements

Misaligned problems require padding, which wastes memory and compute.

### 11.2 Homogeneous Data Types

Within a single grouped GEMM invocation, **all groups must use the same data types, layouts, and tile configurations.** You cannot mix FP16 and BF16 groups, or RowMajor A with ColumnMajor A, in the same call. If mixed types are needed, separate grouped GEMM launches are required.

### 11.3 Shared Tile Shape

All problems share the same `ThreadblockShape`. This means:
- A problem with M=64 wastes most of a 256×128 tile
- Problems much smaller than the tile size are inefficient
- The tile shape must be chosen to accommodate the smallest expected problem

### 11.4 Limited Scheduler Flexibility

The two scheduling modes (`kDeviceOnly`, `kHostPrecompute`) don't support:
- **Priority-based scheduling** (e.g., process small problems first for low latency)
- **Dynamic load balancing** at the threadblock level (no work-stealing)
- **Inter-problem fusion** (each group's epilogue is independent)

### 11.5 Maximum Problem Count

While there's no hard-coded limit, practical constraints exist:
- `kDeviceOnly`: Each threadblock iterates through problems in groups of 32. With 1000+ problems, scheduling overhead grows.
- `kHostPrecompute`: Workspace size scales linearly with problem count × threadblock count.
- Very large problem counts (>10,000) may hit register pressure limits in the scheduling logic.

### 11.6 No Split-K Across Groups

The grouped GEMM kernel does not natively support Split-K (parallel reduction along the K dimension) across groups. Each problem's K iteration is handled by a single threadblock. For very large K dimensions, this limits parallelism. (StreamK partially addresses this, but is a separate mechanism.)

### 11.7 CUTLASS 3.x Requires TMA-Modifiable Descriptors

The Hopper grouped GEMM requires `CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED`. This means:
- Only Hopper (SM90) and newer architectures
- Requires specific CUDA toolkit versions with TMA support
- Not portable to Ampere (SM80/SM86) or older

---

## 12. Summary and Recommendations for SAM 3.1

### Key Takeaways

1. **Grouped GEMM is the right abstraction** for SAM 3.1's 16-head attention with variable sequence lengths — it avoids per-head kernel launch overhead while handling heterogeneous sizes natively.

2. **`kDeviceOnly` scheduling** is appropriate for 16 groups — the warp processes all 16 in a single shuffle iteration, adding negligible overhead.

3. **Sort-by-K** should be applied if heads have significantly different sequence lengths — this groups similar-compute problems together for better load balancing.

4. **CUTLASS 3.x (Hopper+)** offers TMA-based grouped GEMM with on-the-fly descriptor modification, providing the best performance on modern GPUs.

5. **For Ampere targets**, the CUTLASS 2.x `GemmGrouped` API is the path, with careful attention to alignment constraints and tile shape selection.

6. **StreamK** can help if head sizes vary dramatically, but with only 16 heads the benefit is likely modest.

### Recommended Configuration for SAM 3.1

```
Target: Hopper (SM90)
API: CUTLASS 3.x with GroupProblemShape
Schedule: kDeviceOnly (16 heads → trivial scheduling)
Tile: 128×128×128 (pingpong) for moderate per-head sizes
Data types: FP16 or BF16 accumulators, FP8 inputs if available
Sort: Descending by sequence length (proxy for K dimension)
```

---

## References

- NVIDIA CUTLASS repository: https://github.com/NVIDIA/cutlass
- `include/cutlass/gemm/device/base_grouped.h` — BaseGrouped implementation
- `include/cutlass/gemm/device/gemm_grouped.h` — GemmGrouped device API
- `include/cutlass/gemm/kernel/gemm_grouped.h` — Kernel-level GemmGrouped struct
- `include/cutlass/gemm/kernel/grouped_problem_visitor.h` — Device/host scheduling algorithms
- `examples/24_gemm_grouped/` — CUTLASS 2.x grouped GEMM example
- `examples/41_multi_head_attention/` — Multi-head attention with variable sequence lengths
- `examples/57_hopper_grouped_gemm/` — CUTLASS 3.x Hopper grouped GEMM with FP8
- `examples/75_blackwell_grouped_gemm/` — Blackwell SM100 grouped GEMM
- `examples/47_ampere_gemm_universal_streamk/` — StreamK decomposition
