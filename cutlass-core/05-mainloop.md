# CUTLASS Collective Mainloop: The Software Pipelining Engine

## Deep Technical Analysis for SAM 3.1 GEMM Optimization

---

## 1. What the Mainloop Does

The collective mainloop is the **core compute engine** of a CUTLASS GEMM kernel. It orchestrates the entire lifecycle of a single tile's computation: fetching data from global memory, staging it in shared memory, feeding it to the tensor core MMA units, and coordinating with the epilogue for result storage. In CUTLASS 3.x, the mainloop is a `CollectiveMma` struct — a partial template specialization that is uniquely identified by its **dispatch policy** (which encodes architecture, scheduling strategy, and pipeline depth).

The mainloop's `operator()` (or its split `load()` / `mma()` methods in warp-specialized variants) executes the K-dimension iteration loop — the innermost loop that accumulates partial products across the K dimension of the GEMM. Every iteration of this loop:

1. **Waits** on a pipeline barrier to ensure the next shared memory buffer is ready
2. **Issues MMA instructions** (GMMA/UMMA) against the shared memory buffer
3. **Releases** the previous shared memory buffer back to the producer
4. **Issues a new TMA load** for a future K-tile into the next shared memory slot

This load-compute-store overlap is what CUTLASS calls **software pipelining** — and the mainloop is its implementation.

### Key Source Files

| File | Architecture | Strategy |
|------|-------------|----------|
| `sm70_mma_twostage.hpp` | Volta (SM70) | 2-stage, register-resident |
| `sm80_mma_multistage.hpp` | Ampere (SM80) | n-stage cp.async |
| `sm90_mma_tma_gmma_ss.hpp` | Hopper (SM90) | TMA + GMMA, static schedule |
| `sm90_mma_tma_gmma_rs_warpspecialized.hpp` | Hopper (SM90) | TMA + GMMA, register-source A |
| `sm90_mma_multistage_gmma_ss_warpspecialized.hpp` | Hopper (SM90) | cp.async + GMMA, warp-specialized |
| `sm100_mma_warpspecialized.hpp` | Blackwell (SM100) | TMA + UMMA, warp-specialized |

---

## 2. Software Pipelining with Multiple Stages

Software pipelining is the technique of overlapping memory transfers with computation by maintaining multiple in-flight buffers. The **stage count** (template parameter `Stages`) determines how many K-tile buffers reside in shared memory simultaneously.

### How Staging Works

```
Shared Memory Layout for A: (BLK_M, BLK_K, Stages)
Shared Memory Layout for B: (BLK_N, BLK_K, Stages)
```

For `Stages = 4`, there are 4 copies of each operand tile in shared memory, forming a **circular buffer**. At any point in time:

- The **producer** (TMA load engine or cp.async warps) writes to one slot
- The **consumer** (MMA warps) reads from a different slot
- Completed slots are released back to the producer

### Stage Count Selection

| Stages | Pipeline Depth | Use Case |
|--------|---------------|----------|
| 2 | Minimal | Low shared memory pressure, short K |
| 3 | Balanced | Default for most Hopper configs |
| 4 | Deep | Long K dimension, hiding TMA latency |

The `PipelineAsyncMmaStages` parameter in `MainloopSm90TmaGmma` further splits the pipeline between TMA and MMA:

```cpp
constexpr int K_PIPE_MMAS = DispatchPolicy::PipelineAsyncMmaStages;
constexpr int K_PIPE_TMAS = K_PIPE_MAX - K_PIPE_MMAS;
```

This controls how many MMA operations can be in-flight simultaneously before the mainloop must wait (`warpgroup_wait<K_PIPE_MMAS>()`), ensuring the producer doesn't overwrite shared memory still being consumed.

### The Circular Buffer (PipelineState)

```cpp
template<uint32_t Stages_>
struct PipelineState {
    int index_ = 0;      // Current slot in [0, Stages)
    uint32_t phase_ = 0; // Phase bit for barrier signaling (0 or 1)
    uint32_t count_ = 0; // Total number of advances

    void operator++() {
        ++index_;
        ++count_;
        if (index_ == Stages) {
            index_ = 0;
            phase_ ^= 1;  // Flip phase on wrap-around
        }
    }
};
```

The `phase_` bit is critical: it alternates each time the circular buffer wraps, allowing barriers to distinguish "buffer full" from "buffer empty" across wrap-around boundaries. The producer starts with an inverted phase (`make_producer_start_state()` sets `phase_ = 1`), while barriers are initialized to `phase_ = 0`.

---

## 3. TMA-Based Mainloop on Hopper (SM90)

The Hopper TMA (Tensor Memory Accelerator) mainloop (`sm90_mma_tma_gmma_ss.hpp`) represents the state of the art in mainloop design. It uses hardware TMA units for asynchronous global→shared memory transfers, eliminating the need for explicit cp.async instructions.

### Dispatch Policy

```cpp
template<int Stages_, class ClusterShape_, int PipelineAsyncMmaStages_>
struct MainloopSm90TmaGmma {
    constexpr static int Stages = Stages_;
    using ClusterShape = ClusterShape_;
    constexpr static int PipelineAsyncMmaStages = PipelineAsyncMmaStages_;
    using ArchTag = arch::Sm90;
    using Schedule = KernelTma;
};
```

### Pipeline Type

The TMA mainloop uses `PipelineTmaAsync<Stages>`, which maintains two barrier arrays per stage:

- **FullBarrier** (`ClusterTransactionBarrier`): Producer signals that TMA transfer is complete
- **EmptyBarrier** (`ClusterBarrier`): Consumer signals that it's done reading the buffer

### The Mainloop Operator — Anatomy

The `operator()` method implements the complete mainloop:

**Phase 1 — Setup:**
```cpp
// Create shared memory tensors
Tensor sA = make_tensor(make_smem_ptr(storage.smem_A.data()), SmemLayoutA{});
Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.data()), SmemLayoutB{});

// Partition TMA slices for this CTA's position in the cluster
auto block_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
auto block_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);
```

**Phase 2 — Pipeline Initialization:**
```cpp
PipelineParams params;
params.transaction_bytes = TmaTransactionBytes;
params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
params.is_leader = warp_group_thread_idx == 0;
params.num_consumers = NumThreadsPerWarpGroup;

MainloopPipeline pipeline(storage.pipeline_storage, params, ClusterShape{});

PipelineState smem_pipe_read;
PipelineState smem_pipe_release;
PipelineState smem_pipe_write = make_producer_start_state<MainloopPipeline>();
```

**Phase 3 — Prologue TMA Loads:**
```cpp
if (warp_idx == 0 && lane_predicate == 1) {
    int prologue_tma_count = min(K_PIPE_MAX, k_tile_count);
    for (int stage = 0; stage < prologue_tma_count; ++stage) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,stage));
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,stage));
        ++k_tile_iter;
        ++smem_pipe_write;
    }
}
```

**Phase 4 — Pipelined Main Loop:**
```cpp
CUTLASS_PRAGMA_NO_UNROLL
for (; k_tile_count > 0; --k_tile_count) {
    // Consumer: wait for data
    pipeline.consumer_wait(smem_pipe_read);

    // Issue MMA (GMMA instructions via warpgroup)
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA(_,_,_,smem_pipe_read.index()),
                           tCrB(_,_,_,smem_pipe_read.index()), accum);
    warpgroup_commit_batch();

    // Wait for K_PIPE_MMAS outstanding MMAs to drain
    warpgroup_wait<K_PIPE_MMAS>();
    warpgroup_fence_operand(accum);

    // Release the buffer we just finished computing on
    pipeline.consumer_release(smem_pipe_release);

    // Producer: issue next TMA load
    if (warp_idx == 0 && lane_predicate == 1 && k_tile_count_tma > 0) {
        pipeline.producer_acquire(smem_pipe_write);
        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), ...);
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), ...);
    }

    ++smem_pipe_read;
    ++smem_pipe_release;
    ++smem_pipe_write;
}
```

**Phase 5 — Drain:**
```cpp
// Wait for all in-flight GMMAs to complete
warpgroup_wait<0>();

// Cluster synchronization for multi-CTA configurations
if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive();
    cute::cluster_wait();
}
```

---

## 4. The Load → Shared Memory → MMA → Epilogue Pipeline

The mainloop coordinates four distinct hardware units in a data flow pipeline:

```
Global Memory ──TMA──→ Shared Memory ──GMMA/UMMA──→ Register Accumulator ──Epilogue──→ Global Memory
     (HBM)            (SMEM)             (Tensor Cores)        (RF)                  (HBM)
```

### Data Flow Details

1. **TMA Load (Producer):** A single warp (warp 0, elected thread) issues `copy()` calls that are intercepted by the TMA hardware unit. The TMA reads a tile from HBM and writes it to shared memory, arriving on a `ClusterTransactionBarrier` when complete. TMA supports:
   - Multicast: one transfer can signal multiple CTAs in a cluster
   - 128-byte aligned transfers for maximum throughput
   - Hardware predicate support for boundary tiles

2. **Shared Memory Staging:** Data arrives in shared memory at the layout defined by `SmemLayoutA/B`, which is a 3D tensor `(BLK_M, BLK_K, Stages)` or `(BLK_N, BK_K, Stages)`. The layout is optimized for:
   - **TMA box size maximization:** Tiling order is chosen to maximize the TMA transaction size
   - **Swizzle patterns:** Bank conflict avoidance through CuTe's swizzle layouts
   - **GMMA descriptor compatibility:** Layout must be compatible with the MMA atom's shared memory descriptor format

3. **MMA (Consumer):** GMMA (Hopper) or UMMA (Blackwell) instructions source operands directly from shared memory via descriptor iterators (`GMMA::DescriptorIterator` / `UMMA::DescriptorIterator`). The mainloop does NOT copy data to registers before issuing MMA — the tensor cores read shared memory directly. The `warpgroup_arrive()` / `warpgroup_commit_batch()` / `warpgroup_wait<N>()` pattern manages the asynchronous execution of MMA operations.

4. **Epilogue Coordination:** After the mainloop drains all K-tiles (`warpgroup_wait<0>()`), the accumulator registers contain the final partial result. The epilogue then:
   - Reads the accumulator
   - Applies alpha/beta scaling, bias, activation functions
   - Writes results to global memory via TMA store (using `PipelineTmaStore`)

### Shared Storage Layout

```cpp
struct SharedStorage {
    cute::array_aligned<ValTypeA, cosize_v<SmemLayoutA>> smem_A;  // A operand buffer
    cute::array_aligned<ValTypeB, cosize_v<SmemLayoutB>> smem_B;  // B operand buffer
    alignas(16) PipelineStorage pipeline_storage;                  // Barrier arrays
};
```

For a 128×128×64 tile with FP16 and 3 stages:
- smem_A: 128 × 64 × 3 × 2 bytes = 48 KB
- smem_B: 128 × 64 × 3 × 2 bytes = 48 KB
- Pipeline barriers: ~384 bytes (negligible)
- **Total: ~96 KB** (exceeds 48 KB smem on some configs → requires cluster sharing or smaller tiles)

---

## 5. The Role of Pipeline Barriers (PipelineState)

Pipeline barriers are the synchronization mechanism that makes software pipelining correct. CUTLASS provides several barrier types, each designed for specific hardware interaction patterns.

### PipelineTmaAsync Barriers

The TMA pipeline uses **paired barriers** per stage:

```cpp
FullBarrier  full_barrier_[Stages];   // TMA completion signal (transaction barrier)
EmptyBarrier  empty_barrier_[Stages];  // Consumer release signal (regular barrier)
```

**FullBarrier** (`ClusterTransactionBarrier`):
- Initialized with producer arrival count = 1 (single TMA leader thread)
- The TMA hardware atomically arrives on this barrier when a transfer completes
- The consumer waits on this barrier (`consumer_wait()`) before reading
- Supports multicast: a single TMA can signal multiple CTA barriers across the cluster

**EmptyBarrier** (`ClusterBarrier`):
- Initialized with consumer arrival count = number of warpgroups (or cluster-aware count)
- The consumer arrives on this barrier (`consumer_release()`) after finishing computation
- The producer waits on this barrier (`producer_acquire()`) before issuing the next TMA
- Uses optimized multicast signaling across cluster rows/columns

### The Producer/Consumer Dance

```
Producer (TMA warp):                    Consumer (MMA warps):
─────────────────                       ────────────────────
producer_acquire(stage):                consumer_wait(stage):
  └─ empty_barrier[stage].wait(phase)     └─ full_barrier[stage].wait(phase)
  └─ full_barrier.arrive_and_expect_tx
                                        
copy(tma_load, ..., smem[stage])        // Issue GMMA on smem[stage]
  └─ TMA auto-arrives on full_barrier   
                                        consumer_release(stage):
producer_acquire(next_stage):             └─ empty_barrier[stage].arrive(...)
  └─ empty_barrier[next_stage].wait(...)
```

### Cluster-Aware Barrier Initialization

For multi-CTA clusters, barriers are initialized with multicast-aware arrival counts:

```cpp
uint32_t multicast_consumer_arrival_count = 
    (size<0>(cluster_shape) + size<1>(cluster_shape) - 1) * num_consumer_warpgroups_per_cluster;
```

This ensures that the barrier only flips when all relevant CTA peers have arrived, supporting the row/column multicast pattern.

### PipelineState Mechanics

Each `PipelineState` tracks:
- `index_`: Which slot in the circular buffer (0 to Stages-1)
- `phase_`: The barrier phase bit (0 or 1), flips on wrap-around
- `count_`: Total number of advances (for debugging/scheduling)

The phase bit is the key correctness mechanism. Barriers are initialized to expect `phase = 0`. The consumer waits for `phase = 0`, the producer starts at `phase = 1`. After `Stages` advances, both wrap around and the producer is now at `phase = 0` while the barrier expects `phase = 0` — matching the empty state.

---

## 6. Mainloop Traits and How They Customize Behavior

The `CollectiveMma` class is parameterized by a rich set of template arguments that customize every aspect of behavior:

### Dispatch Policy (Schedule Tag)

The dispatch policy is the primary configuration knob. It determines:

```cpp
// Example: MainloopSm90TmaGmma<4, Shape<_2,_1,_1>, 2>
//           Stages=4, ClusterShape=2x1x1, PipelineAsyncMmaStages=2
```

| Policy | Schedule | Data Movement | MMA Source |
|--------|----------|---------------|------------|
| `MainloopSm80CpAsync` | `KernelMultistage` | cp.async | smem |
| `MainloopSm90TmaGmma` | `KernelTma` | TMA | smem (GMMA descriptor) |
| `MainloopSm90TmaGmmaWarpSpecialized` | `KernelTmaWarpSpecialized{Cooperative,Pingpong}` | TMA | smem (GMMA descriptor) |
| `MainloopSm90TmaGmmaRmemAWarpSpecialized` | `KernelTmaWarpSpecialized*` | TMA | A: rmem, B: smem |
| `MainloopSm100TmaUmmaWarpSpecialized` | `KernelTmaWarpSpecialized*` | TMA | smem (UMMA descriptor) |

### Element Type Adaptations

The mainloop handles type conversions transparently:

```cpp
// TMA converts f32 → tf32 when copying to smem
static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, 
                                              uint_bit_t<sizeof_bits_v<ElementA>>>;
```

For FP8 kernels, the mainloop uses `MainloopSm90TmaGmmaWarpSpecializedFP8` which adds fast accumulation support and optional blockwise scaling.

### Layout Optimizations

The shared memory layout tiling is automatically optimized to maximize TMA box size:

```cpp
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{}),
    // Choose tiling order based on stride major mode
    conditional_t<is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
```

### The Collective Builder

The `collective_builder.hpp` automatically selects the appropriate mainloop specialization based on:
- Target architecture (SM80/SM90/SM100)
- Element types (FP16/BF16/FP8/TF32)
- Kernel schedule policy
- Tile shape and cluster shape

This means users rarely need to manually specify the mainloop — the builder deduces the optimal configuration.

---

## 7. Warp Specialization Mainloop (Producer/Consumer Warps)

Warp specialization is a scheduling strategy where different warps within a CTA assume distinct roles:

- **Producer warps:** Issue data loads (TMA or cp.async)
- **Consumer warps:** Issue MMA computations
- **Dedicated warps** may also handle the epilogue

### Two Scheduling Paradigms

**Cooperative Schedule:** All warps participate in both loading and computing. The mainloop serializes load → compute → release within each iteration. This is the `sm90_mma_tma_gmma_ss.hpp` approach — it's simpler but doesn't achieve true overlap.

**Pingpong Schedule:** Warps are divided into groups that alternate between loading and computing. While one group computes, the other loads the next tile. This achieves better occupancy hiding.

**Warp Specialized (Dynamic Schedule):** Warps take on fixed roles with separate code paths:

```cpp
// The load() method — called by producer warps
CUTLASS_DEVICE void load(
    MainloopPipeline pipeline,
    PipelineState smem_pipe_write,
    TensorA const& gA_in, TensorB const& gB_in,
    KTileIterator k_tile_iter, int k_tile_count,
    ...) {
    // Producer: only issues cp.async or TMA loads
    for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);
        copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write;
    }
}

// The mma() method — called by consumer warps
CUTLASS_DEVICE void mma(
    MainloopPipeline pipeline,
    PipelineState smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count, ...) {
    // Consumer: only issues MMA instructions
    for (; k_tile_count > 0; --k_tile_count) {
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
        cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
        warpgroup_commit_batch();
        warpgroup_wait<K_PIPE_MMAS>();
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_read;
    }
}
```

### Register-Source A Mainloop (`sm90_mma_tma_gmma_rs_warpspecialized.hpp`)

A variant where operand A comes from registers instead of shared memory. This is useful when:
- A is reused across multiple B tiles (e.g., in grouped GEMMs)
- WGMMA requires specific k-major layouts that can be achieved via register transpose
- Mixed-precision inputs where A needs type conversion before MMA

This variant asserts:
```cpp
static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
              cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
              "MMA atom must source A from rmem and B operand from smem_desc");
```

---

## 8. Cluster-Level Coordination in Mainloop

Hopper and Blackwell support **CTA clusters** — groups of CTAs that can share data via distributed shared memory and coordinate via cluster-wide barriers.

### TMA Multicast

When using `SM90_TMA_LOAD_MULTICAST`, a single TMA transfer can write to shared memory in multiple CTAs simultaneously:

```cpp
uint16_t mcast_mask_a = 0;
auto block_layout = Layout<ClusterShape>{};
for (int n = 0; n < size<1>(block_layout); ++n) {
    mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
}
```

For a 2×2 cluster, the A operand (loaded along the M dimension) is multicast along the N dimension, so one TMA transfer delivers A to both CTAs in the same row. Similarly, B is multicast along the M dimension.

### Cluster Synchronization

The mainloop uses cluster-wide synchronization primitives:

```cpp
// Barrier initialization requires cluster visibility
if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
} else {
    __syncthreads();
}
```

### Multicast Consumer Arrival

The `consumer_release()` uses optimized multicast signaling. For a cluster of size (M, N), only threads in the same row or column as the target CTA need to signal:

```cpp
CUTLASS_DEVICE void consumer_release(uint32_t stage) {
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_);
}
```

The `dst_blockid_` and `is_signaling_thread_` are pre-computed during pipeline construction to spread the arrival duty evenly across the 128-thread warpgroup.

### Blackwell (SM100) 2SM MMA Cluster Support

Blackwell introduces 2SM MMA instructions where two SMs cooperate on a single large MMA operation. The `PipelineUmmaAsync` pipeline handles this:

```cpp
static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

void producer_commit(uint32_t stage) {
    if constexpr (is_2sm_mma) {
        cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, tmem_sync_mask_);
    } else {
        cutlass::arch::umma_arrive(smem_ptr);
    }
}
```

---

## 9. How the Mainloop Handles K-Dimension Iteration

The K-dimension iteration is the heart of the mainloop. The GEMM C = A × B is decomposed along K into tiles:

```
C(m,n) += Σ_k  A(m,k) × B(k,n)
```

Each K-tile is one iteration of the mainloop.

### K-Tile Counting

```cpp
auto k_tile_iter = ...;   // Iterator over K tiles
int k_tile_count = ...;    // Total K tiles to process
int k_tile_count_tma = k_tile_count;  // Separate counter for TMA producer
```

The producer and consumer maintain **separate counters** because they're offset by the pipeline depth:

- Producer issues loads for `k_tile_count_tma` tiles
- Consumer processes `k_tile_count` tiles
- Initially, `prologue_tma_count = min(Stages, k_tile_count)` loads are issued ahead

### Residue Handling (Edge Cases)

The warp-specialized cp.async mainloop handles K-dimension boundary tiles via predicates:

```cpp
// Shift tensor so residue_k is at origin
Tensor gA = domain_offset(make_coord(0, get<2>(residue_mnk), 0), gA_in);

// First tile: predicate for k residue
for (int k = 0; k < size<2>(tAsA); ++k) {
    if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {
        copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,write_stage));
    } else {
        clear(tAsA(_,_,k,write_stage));  // Zero-fill out-of-bounds
    }
}
```

### K-Dimension Promotion Interval

The `mma_promotion_interval` argument controls how often the accumulator is promoted to higher precision (for FP8 fast accumulation):

```cpp
// In the Arguments struct
uint32_t mma_promotion_interval = 4;
```

Every N K-tiles, the FP32 accumulator is flushed to prevent overflow in FP8 accumulation chains.

---

## 10. Performance Tuning: Stage Count, Tile Size Effects

### Stage Count Impact

| Stages | Smem Usage | Latency Hiding | Register Pressure | Best For |
|--------|-----------|----------------|-------------------|----------|
| 2 | Minimal | Poor | Low | Small K, memory-bound |
| 3 | Moderate | Good | Moderate | Default choice |
| 4 | High | Excellent | Higher | Long K, compute-bound |

**Rule of thumb:** Increase stages until shared memory is exhausted. Each additional stage buffers one more K-tile, hiding more TMA latency. However:
- More stages = less shared memory for tile sizes
- More stages = higher register pressure for pipeline state tracking
- The `PipelineAsyncMmaStages` parameter must be tuned: too many in-flight MMAs can stall the producer

### Tile Size Effects

The tile shape `(TileM, TileN, TileK)` directly interacts with the mainloop:

- **Larger TileK:** Fewer K iterations, better compute-to-load ratio, but more shared memory per stage
- **Larger TileM/TileN:** More work per tile, but more shared memory per stage
- **ClusterShape:** Multiplies effective tile size across CTAs, enables multicast but adds synchronization overhead

For a 128×256×64 tile with FP16 and 3 stages:
```
smem_A: 128 × 64 × 3 × 2 = 48 KB
smem_B: 256 × 64 × 3 × 2 = 96 KB
Total: 144 KB → exceeds single-SM smem
```

Options: reduce stages to 2 (72 KB), reduce tile to 128×128×64 (48 KB), or use cluster sharing.

### Blackwell-Specific Tuning

SM100 introduces `SchedulerPipelineStageCount` and `AccumulatorPipelineStageCount`:

```cpp
template<int Stages, int SchedulerPipelineStageCount, 
         int AccumulatorPipelineStageCount, class ClusterShape>
struct MainloopSm100TmaUmmaWarpSpecialized;
```

- **SchedulerPipelineStageCount:** Stages for the persistent grid scheduler's work distribution
- **AccumulatorPipelineStageCount:** Stages for TMEM accumulator double-buffering (mainloop writes one, epilogue reads another)
- The accumulator uses TMEM (Tensor Memory) on Blackwell, a new 128 KB memory space dedicated to MMA results

### Practical Tuning for SAM 3.1

For SAM 3.1's large GEMMs (e.g., attention projections, MLP layers):

1. **Long K dimension (e.g., 4096+):** Use 3-4 stages. The TMA latency hiding is critical.
2. **Large M/N (e.g., high-res image patches):** Use larger tile M/N with 2 stages to maximize math throughput per tile.
3. **FP8 inference:** Use `MainloopSm90TmaGmmaWarpSpecializedFP8` with fast accumulation. Set `mma_promotion_interval = 4` to balance precision and performance.
4. **Cluster shapes:** For H100, 2×1 or 1×2 clusters often help. For H200/B100, experiment with 2×2 for large tiles.
5. **Warp specialization:** Prefer `KernelTmaWarpSpecializedCooperative` for general use. Switch to `Pingpong` if you see idle warps in profiling.

### Key Performance Indicators

When profiling the mainloop:
- **TMA utilization:** Should be >80% if the pipeline is deep enough
- **GMMA/UMMA pipeline occupancy:** Watch for `warpgroup_wait` stalls — indicates the MMA pipeline is full
- **Shared memory bank conflicts:** Visible as pipeline bubbles between MMA issues
- **Barrier wait time:** High barrier wait time in the producer means the consumer is too slow; vice versa means the producer is bottlenecked

---

## Appendix: Mainloop File Index

### Hopper (SM90) Mainloops

| File | Strategy | Notes |
|------|----------|-------|
| `sm90_mma_tma_gmma_ss.hpp` | TMA+GMMA, shared-source | Static schedule, simplest |
| `sm90_mma_tma_gmma_rs_warpspecialized.hpp` | TMA+GMMA, register-source A | For mixed-precision or layout transposition |
| `sm90_mma_multistage_gmma_ss_warpspecialized.hpp` | cp.async+GMMA | For pre-TMA fallback or specific layout needs |
| `sm90_mma_array_tma_gmma_ss_warpspecialized.hpp` | TMA+GMMA, ptr-array | For grouped/batched GEMMs |
| `sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp` | TMA+GMMA, FP8 | FP8 fast accumulation variant |

### Blackwell (SM100) Mainloops

| File | Strategy | Notes |
|------|----------|-------|
| `sm100_mma_warpspecialized.hpp` | TMA+UMMA | Base Blackwell mainloop with TMEM |
| `sm100_mma_array_warpspecialized.hpp` | TMA+UMMA, ptr-array | Grouped GEMM support |
| `sm100_blockscaled_mma_warpspecialized.hpp` | Block-scaled | For FP4/FP6/FP8 with per-block scales |
| `sm100_sparse_mma_warpspecialized.hpp` | Sparse | 2:4 structured sparsity support |
| `sm100_mma_cpasync_warpspecialized.hpp` | cp.async+UMMA | Fallback without TMA |

### Pipeline Classes

| File | Pipeline Class | Used By |
|------|---------------|---------|
| `pipeline.hpp` | `PipelineAsync` | cp.async mainloops |
| `sm90_pipeline.hpp` | `PipelineTmaAsync` | Hopper TMA mainloops |
| `sm90_pipeline.hpp` | `PipelineTmaStore` | Epilogue TMA stores |
| `sm100_pipeline.hpp` | `PipelineUmmaAsync` | Blackwell UMMA mainloops |
| `sm100_pipeline.hpp` | `PipelineTmaTransformAsync` | Blackwell with transform stage |

---

*This analysis is based on CUTLASS main branch (2026). For SAM 3.1 optimization, focus on the SM90 TMA mainloop variants (Hopper deployment) and SM100 variants (Blackwell deployment). The collective builder will automatically select the appropriate mainloop — the key tuning knobs are Stages, TileShape, ClusterShape, and the kernel schedule policy.*
