# CUTLASS Unit Tests and Testing Infrastructure

> Deep technical analysis for SAM 3.1 kernel validation  
> Source: NVIDIA CUTLASS `test/` directory, main branch (2026)

---

## 1. Test Organization: Three-Tier Hierarchy

CUTLASS organizes its tests in a strict hierarchy under `test/`:

```
test/
├── CMakeLists.txt                  # Top-level: gates unit tests via CUTLASS_ENABLE_GTEST_UNIT_TESTS
├── unit/                           # Google Test-based unit tests
│   ├── CMakeLists.txt              # Infrastructure, test registration, subdirectory enumeration
│   ├── common/                     # Shared headers (cutlass_unit_test.h, filter_architecture.cpp)
│   ├── core/                       # Core type traits, numeric types, layout fundamentals
│   ├── cute/                       # CuTe DSL tensor algebra tests
│   ├── gemm/                       # GEMM tests (thread → warp → threadblock → device)
│   │   ├── thread/                 # Thread-level MMA tests
│   │   ├── warp/                   # Warp-level MMA tests
│   │   ├── threadblock/            # Threadblock-level tile tests
│   │   └── device/                 # Full device-wide GEMM tests (~300+ executables)
│   ├── conv/                       # Convolution tests
│   ├── epilogue/                   # Epilogue fusion tests
│   ├── layout/                     # Layout and stride tests
│   ├── transform/                  # Data transformation tests
│   ├── reduction/                  # Reduction operation tests
│   ├── pipeline/                   # Async pipeline (TMA) tests
│   ├── substrate/                  # Low-level substrate tests
│   ├── cluster_launch/             # Cluster launch (sm90+) tests
│   ├── nvrtc/                      # NVRTC runtime compilation tests (conditional)
│   └── util/                       # Utility function tests
├── self_contained_includes/        # Compile-time check: each header is self-contained
├── python/                         # Python CuTe DSL tests
└── examples/CuTeDSL/              # CuTe DSL example tests
```

### Key Design Principles

1. **Google Test (gtest) framework** — all unit tests use `TEST()` / `EXPECT_TRUE()` macros from gtest.
2. **Architecture gating** — tests compile only for the target SM architecture via `#if defined(CUTLASS_ARCH_MMA_SMXX_SUPPORTED)` guards.
3. **Test levels** — Three levels (L0, L1, L2) controlled by `CUTLASS_TEST_LEVEL`:
   - **L0**: Always run. Core functionality.
   - **L1**: Run when `CUTLASS_TEST_LEVEL >= 1`. Extended coverage.
   - **L2**: Run when `CUTLASS_TEST_LEVEL >= 2`. Exhaustive/exotic combinations.

   ```cpp
   #define CUTLASS_TEST_L0(NAME_STATIC, NAME_DYNAMIC, ...) CUTLASS_TEST_LEVEL_ACTIVE(0, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
   #define CUTLASS_TEST_L1(NAME_STATIC, NAME_DYNAMIC, ...) // Active or disabled based on level
   #define CUTLASS_TEST_L2(NAME_STATIC, NAME_DYNAMIC, ...) // Active only at highest level
   ```

4. **Separate from examples** — The `test/` tree is distinct from `examples/`. Tests validate correctness; examples demonstrate usage.

---

## 2. GEMM Test Patterns: What Combinations Are Tested

The GEMM device tests (`test/unit/gemm/device/`) are the most extensive part of the test suite, with hundreds of test executables organized by architecture and data type.

### Naming Convention

Test files follow the pattern:
```
gemm_{dtypeA}{layoutA}_{dtypeB}{layoutB}_{dtypeC}{layoutC}_{math_mode}_{smXX}.cu
```

For example: `gemm_f16n_f16t_f16t_tensor_op_f16_sm75.cu` = f16 input A (ColumnMajor), f16 input B (RowMajor), f16 output (RowMajor), tensor op, f16 accumulation, targeting SM75.

SM90+ files use a newer convention: `sm90_gemm_f16_f16_f16_tensor_op.cu` — layout is specified inside the test via template parameters.

### What's Combinatorially Tested

| Dimension | Tested Values |
|-----------|--------------|
| **Data types (A/B/C)** | f16, bf16, f32, tf32, f64, cf32, cf64, s8, u8, s4, b1, f8 (e4m3/e5m2) |
| **Layouts** | ColumnMajor (N), RowMajor (T), all 4 combos (NN, NT, TN, TT) |
| **Accumulator types** | f16, f32, s32, f64 — often wider than inputs |
| **Math modes** | SIMT, TensorOp, WMMA |
| **Tile shapes** | 256×128×32, 128×256×32, 128×128×32, 64×128×32, 64×64×32, 32×256×32, etc. |
| **Warp shapes** | 64×64×32, 32×64×32, 32×32×32, etc. |
| **Instruction shapes** | 16×8×8 (sm75), 16×8×16 (sm80), MMA atom shapes (sm90+) |
| **Stages (pipeline depth)** | 2, 3, 4 |
| **Split-K** | Serial and parallel split-K variants |
| **Alignment** | Full alignment and sub-alignment (alignment-x) tests |

### SM90+ Specific Test Categories

For Hopper (sm90), the test suite covers:
- **GMMA (Tensor Core MMA)** with `CollectiveBuilder` API
- **Warp-specialized kernels** (pingpong and cooperative schedules)
- **Cluster multicast** — testing multi-CTA cluster communication
- **GMMA-RS** — register-sharing MMA patterns
- **Stream-K** — cooperative stream-K scheduler tests
- **Group GEMM** — variable problem sizes per group
- **Pointer arrays** — batched GEMM via pointer arrays
- **Blockwise GEMM** — block-scaled FP8 operations
- **Sparse GEMM** — 2:4 structured sparsity
- **Epilogue fusions** — tensor broadcast, bias elementwise, row broadcast, reductions, DAG fusions, aux load/store

### SM100 Specific Test Categories

For Blackwell (sm100), tests are organized in subdirectories:
- `sm100_tensorop_gemm/` — f16×f16, f8×f8, s8×s8 with standard and narrow MMA variants
- `sm100_blockscaled_tensorop_gemm/` — block-scaled operations
- `sm100_sparse_tensorop_gemm/` — sparse GEMM
- `sm100_blockscaled_sparse_tensorop_gemm/` — combined block-scale + sparse

---

## 3. Correctness Validation: Reference Comparison

CUTLASS validates kernel correctness by comparing GPU results against a CPU reference implementation.

### The Testbed Pattern

The core validation pattern (from `test/unit/gemm/device/testbed.h`):

```cpp
template <typename Gemm, bool Relu = false>
struct Testbed {
    // Host tensors for inputs/outputs
    cutlass::HostTensor<ElementA, LayoutA> tensor_A;
    cutlass::HostTensor<ElementB, LayoutB> tensor_B;
    cutlass::HostTensor<ElementC, LayoutC> tensor_C;  // bias
    cutlass::HostTensor<ElementC, LayoutC> tensor_D;  // GPU output
    cutlass::HostTensor<ElementC, LayoutC> reference_D; // CPU reference

    bool verify(GemmCoord problem_size, ElementCompute alpha, ElementCompute beta) {
        // 1. Run CPU reference GEMM
        cutlass::reference::host::Gemm<
            ElementA, LayoutA, ElementB, LayoutB,
            ElementC, LayoutC, ElementCompute, ElementAccumulator>
            reference_gemm;

        reference_gemm(problem_size, alpha, tensor_A.host_ref(),
                       tensor_B.host_ref(), beta, reference_D.host_ref(),
                       ElementAccumulator(0));

        // 2. Sync GPU result back to host
        tensor_D.sync_host();

        // 3. Compare element-wise
        bool passed = cutlass::reference::host::TensorEquals(
            reference_D.host_view(), tensor_D.host_view());

        // 4. On failure: dump matrices to file for debugging
        if (!passed) {
            std::ofstream file("error_Gemm_device_...");
            file << "A =\n" << tensor_A.host_view()
                 << "\nB =\n" << tensor_B.host_view()
                 << "\nReference =\n" << reference_D.host_view()
                 << "\nComputed =\n" << tensor_D.host_view();
        }
        return passed;
    }
};
```

### Key Validation Details

1. **Initialization** — Tensors filled with random uniform distributions, range-tuned to data type:
   - 1-bit inputs: range [0, 2)
   - ≤8-bit inputs: range [-1, 1)
   - 16-bit outputs: range [-5, 5)
   - 32-bit outputs: range [-8, 8)

2. **Corner case seeding** — Upper-left corner elements explicitly set to 1 to avoid all-zero random initialization.

3. **Norm checks** — `TensorNorm` verifies that both reference and computed tensors have non-zero norms (catches silent zeros).

4. **Reference GEMM** — `cutlass::reference::host::Gemm` implements a straightforward triple-loop matmul on the CPU with the same accumulator type. This is the "golden reference."

5. **Problem sizes tested** — `TestAllGemm<Gemm>()` tests multiple problem sizes:
   - Small (below tile size): 11×13×17
   - Exactly one tile: matching the threadblock shape
   - Multi-tile: multiples of tile size
   - Odd sizes: prime numbers to test boundary handling
   - Configurable via `CUTLASS_UNIT_TEST_PROBLEM_COUNT` environment variable

6. **Alpha/Beta combinations** — Tests both α=1, β=0 (pure matmul) and α=2, β=3 (general linear combination).

---

## 4. Architecture-Specific Test Coverage

### Compile-Time Architecture Gating

Tests are guarded by preprocessor macros:
```cpp
#if defined(CUTLASS_ARCH_MMA_SM70_SUPPORTED)
  // Volta tensor op tests
#endif

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)
  // Turing tensor op tests
#endif

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
  // Ampere tensor op tests
#endif

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // Hopper GMMA tests (requires CUDA 12.0+)
#endif
```

The `filter_architecture.cpp` utility in `test/unit/common/` filters tests at runtime to match the actual GPU present.

### Architecture Coverage Matrix

| Architecture | SM | Math Mode | Data Types Tested | Special Features |
|-------------|-----|-----------|------------------|-----------------|
| **Volta** | sm70 | TensorOp | f16 (Volta MMA) | Split-K |
| **Turing** | sm75 | TensorOp | f16, s8, s4, b1 | All layout combos |
| **Ampere** | sm80 | TensorOp | f16, bf16, tf32, f64, cf32, cf64, s8, s4, b1 | Mixed input types, sparse |
| **Ada** | sm89 | TensorOp | f8 (e4m3/e5m2) | FP8 support |
| **Hopper** | sm90 | GMMA | f16, bf16, f32, tf32, f8, s8 | Warp-specialized, TMA, clusters, stream-K, group GEMM, epilogue fusion, sparse |
| **Blackwell** | sm100 | TensorOp | f16, f8, s8 | Block-scaled, narrow MMA, sparse+blockscaled |
| **sm120** | sm120 | TensorOp | f16, f8, s8 | Block-scaled, sparse |

### CMake-Level Architecture Filtering

The CMake build system restricts which test executables are compiled:
```cmake
# SM100 tests only compile when targeting sm100a
if (CUTLASS_NVCC_ARCHS MATCHES 100a)
  add_subdirectory(sm100_tensorop_gemm)
endif()
```

Sparse tests may be skipped on old compilers:
```cmake
# Sparse kernels trigger an ICE in gcc 7.5
if (NOT (CUTLASS_GNU_HOST_COMPILE AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0))
  add_subdirectory(sm90_sparse...)
endif()
```

---

## 5. Test Infrastructure: How to Add New Test Cases

### Step-by-Step: Adding a New GEMM Test

**1. Create the test file** in `test/unit/gemm/device/`:

```cpp
// my_new_gemm_test.cu
#include "../../common/cutlass_unit_test.h"
#include "cutlass/gemm/device/gemm.h"
#include "testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

TEST(SM80_Device_Gemm_f16_f16_f32_MyConfig, 128x128x32) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float,  // accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,   // Threadblock
        cutlass::gemm::GemmShape<64, 64, 32>,      // Warp
        cutlass::gemm::GemmShape<16, 8, 16>,       // Instruction
        cutlass::epilogue::thread::LinearCombination<float, 8, float, float>
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

#endif
```

**2. Register in CMakeLists.txt** — Add to an existing executable or create a new one:

```cmake
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_unit_gemm_device_my_tests
  BATCH_SOURCES ON
  BATCH_SIZE 4
  my_new_gemm_test.cu
)
```

The `cutlass_test_unit_gemm_device_add_executable` function:
- Calls `cutlass_test_unit_add_executable` (defined in `test/unit/CMakeLists.txt`)
- Auto-registers with the `test_unit_gemm_device` target
- Sets up gtest XML output
- Supports `BATCH_SOURCES` for compile-time parallelism (groups multiple .cu files into one executable)

**3. For SM90+ 3.x API tests**, use `gemm_testbed_3x.hpp` and `CollectiveBuilder`:

```cpp
#include "gemm_testbed_3x.hpp"

TEST(SM90_Device_Gemm_f16_f16_f16_MyTest, 64x128x64) {
    using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        Shape<_64,_128,_64>, Shape<_1,_1,_1>,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        Shape<_64,_128,_64>, Shape<_1,_1,_1>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float, float,
        cutlass::half_t, LayoutC, 8,
        cutlass::half_t, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>, CollectiveOp, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}
```

### Key Infrastructure Functions

| Function | Purpose |
|----------|---------|
| `cutlass_test_unit_add_executable()` | Creates a test executable, links gtest + CUTLASS infra, registers with CTest |
| `cutlass_test_unit_add_executable_split_file()` | Splits multi-TEST files to reduce peak compiler memory |
| `cutlass_test_unit_gemm_device_add_executable()` | GEMM-specific wrapper that also registers with gemm_device target |
| `TestAllGemm<Gemm>()` | Runs GEMM across multiple problem sizes, alpha/beta combos |
| `TestAll<Gemm>()` | 3.x API test entry point |
| `CutlassUnitTestProblemCount()` | Reads `CUTLASS_UNIT_TEST_PROBLEM_COUNT` env var to control test count |

### Result Caching

Tests can use cached results for faster iteration:
```
test/unit/data/hashes/cached_results_cutlass_test_unit_gemm_device_simt.txt
```
If a cached result file exists for a test, CMake passes it as `RESULT_CACHE_FILE` to the test runner.

---

## 6. Benchmarking Patterns in the Test Suite

CUTLASS's unit tests are primarily correctness tests, not benchmarks. However, there are important performance-related patterns:

### Performance-Adjacent Testing

1. **Multiple tile shape configurations** — Each test file tests 6-12 different threadblock/warp shape combinations, which implicitly exercises different performance characteristics.

2. **Alignment testing** — `sm90_gemm_f16_f16_f16_alignx_tensor_op_f32.cu` tests sub-alignment scenarios (alignment-x), which affects memory coalescing and performance.

3. **Schedule variant testing** — SM90 tests cover all scheduling strategies:
   - `KernelScheduleAuto`
   - Warp-specialized cooperative
   - Warp-specialized pingpong
   - Cluster warpspecialized cooperative
   - Cluster warpspecialized pingpong

### Separate Benchmark Suite

CUTLASS maintains a separate `tools/profiler/` directory for performance benchmarking, which is distinct from the test suite. The profiler:
- Measures GFLOPS across problem sizes
- Compares against cuBLAS baselines
- Generates performance reports

---

## 7. Using CUTLASS Tests as a Template for Custom Kernel Testing

For SAM 3.1 custom CUTLASS kernels, follow this pattern:

### Template: Custom Kernel Test

```cpp
// test_sam31_gemm.cu
#include "cutlass_unit_test.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// Include your custom collective ops or use standard builders
#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Test 1: Basic correctness with standard tile shapes
TEST(SAM31_Gemm, Basic_64x128x64) {
    // Define your kernel using CollectiveBuilder or manual collective ops
    using CollectiveOp = /* your custom or builder-generated collective */;
    using EpilogueOp = /* your custom or builder-generated epilogue */;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveOp, EpilogueOp>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

// Test 2: Edge cases — small M, large N, large K
TEST(SAM31_Gemm, SmallM_1x1024x4096) { /* ... */ }

// Test 3: Odd sizes (prime numbers)
TEST(SAM31_Gemm, OddSizes_73x173x251) { /* ... */ }

// Test 4: Different alpha/beta
// TestAllGemm tests alpha=1,beta=0 and alpha=2,beta=3 automatically

// Test 5: Alignment boundaries
TEST(SAM31_Gemm, SubAligned_63x127x63) { /* ... */ }

#endif
```

### CMake Integration

```cmake
# In your project's CMakeLists.txt
cutlass_test_unit_gemm_device_add_executable(
  cutlass_test_sam31_gemm
  BATCH_SOURCES ON
  BATCH_SIZE 2
  test_sam31_gemm.cu
)
```

### Best Practices for Custom Kernel Tests

1. **Always test across multiple problem sizes** — use `TestAllGemm` which tests small, tile-aligned, multi-tile, and odd sizes.
2. **Test all layout combinations** your kernel supports (NN, NT, TN, TT).
3. **Test boundary alignments** — add `alignx` tests for sub-vector-width alignments.
4. **Test epilogue fusions** if your kernel has custom epilogues.
5. **Use the 3.x API** for SM90+ — it's the modern path and better tested.
6. **Match your tile shapes** — test at least the tile shapes you plan to deploy.

---

## 8. Continuous Integration Setup

### GitHub Actions

CUTLASS's CI is primarily managed through NVIDIA's Blossom CI system (`.github/workflows/blossom-ci.yml`), not standard GitHub Actions runners, because:

1. **GPU hardware is required** — Tests need actual NVIDIA GPUs (sm70 through sm100).
2. **CUDA toolkit dependencies** — Specific CUDA versions per architecture.
3. **Compilation time** — Full test suite compilation can take hours.

### CI Workflow Structure

```
blossom-ci.yml          → NVIDIA internal CI (triggers on PR)
auto-label-issues.yml   → Auto-labels issues by area
labeler.yml             → PR label management
stale.yml               → Stale issue management
```

### What CI Tests

NVIDIA's internal CI runs:
1. **Compilation tests** — All test executables compile for each supported architecture.
2. **Runtime tests** — Tests execute on actual GPU hardware.
3. **Self-contained include checks** — Each header compiles independently.
4. **Cross-architecture validation** — Same test logic, different SM targets.

### Local CI Simulation

Developers can simulate CI locally:
```bash
# Build all tests for a specific architecture
cmake -B build -DCUTLASS_NVCC_ARCHS="90a" -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=ON
cmake --build build -j$(nproc)

# Run all unit tests
cd build && ctest -j$(nproc) --output-on-failure

# Run only GEMM device tests
cd build && ctest -R "gemm_device" -j4 --output-on-failure

# Run a specific test
./bin/cutlass_test_unit_gemm_device_tensorop_sm90 --gtest_filter="*f16_f16_f16*"
```

### Test Sharding

`test/utils/test_sharding.py` supports splitting test suites across multiple GPU machines for faster CI turnaround.

---

## 9. Common Test Failure Patterns and Debugging

### Failure Pattern 1: Numeric Mismatch (Most Common)

**Symptom**: `reference_D does not equal tensor_D`

**Root causes**:
- Accumulator precision too low for problem size (e.g., f16 accumulator with large K)
- TMA or shared memory staging bug
- Race condition in async pipeline
- Epilogue order-of-operations difference vs reference

**Debug approach**:
- The testbed automatically dumps `A`, `B`, `C`, reference, and computed matrices to a file named `error_Gemm_device_{M}x{N}x{K}_{TB}_{Warp}.txt`
- Compare element-by-element to find the divergence pattern
- Check if errors are systematic (all elements off) or scattered (race condition)

### Failure Pattern 2: Compilation Failure

**Symptom**: `error: identifier "..." is undefined` or `static_assert` failure

**Root causes**:
- Targeting wrong SM architecture
- Missing `#if defined(CUTLASS_ARCH_MMA_SMXX_SUPPORTED)` guard
- CUDA toolkit version too old for the architecture (SM90 needs CUDA 12.0+)

**Debug approach**:
```bash
# Check what architectures your build targets
cmake -B build -DCUTLASS_NVCC_ARCHS="90a" -LA | grep NVCC_ARCHS

# Compile a single test to isolate the error
nvcc -arch=sm_90 -I./include test_file.cu
```

### Failure Pattern 3: Silent Zero Output

**Symptom**: `tensor_D has nonpositive norm`

**Root causes**:
- Grid launch misconfiguration (block count too low)
- Shared memory allocation exceeded
- Problem size not handled (M=0 or N=0 edge case)

**Debug approach**:
- Check `cudaGetLastError()` after kernel launch
- Use `compute-sanitizer` for memory access violations
- Verify shared memory usage: `--ptxas-options=-v`

### Failure Pattern 4: Test Timeout

**Symptom**: Test hangs or exceeds time limit

**Root causes**:
- Infinite loop in kernel (rare)
- Deadlock in cluster/async pipeline
- Host-side reference GEMM too slow for large problem sizes

**Debug approach**:
- Use `compute-sanitizer --tool=synccheck` for deadlock detection
- Reduce `CUTLASS_UNIT_TEST_PROBLEM_COUNT` to test fewer sizes
- Check if the test is running a very large K dimension

### Failure Pattern 5: Architecture-Specific

**Symptom**: Test passes on sm80, fails on sm90

**Root causes**:
- TMA descriptor setup differs between architectures
- GMMA instruction behavior differences
- Cluster configuration issues

**Debug approach**:
- Run the test with `CUTLASS_TRACE=1` for kernel trace output
- Compare PTX output for both architectures
- Use Nsight Compute to profile the failing kernel

### Useful Debugging Commands

```bash
# Run with compute-sanitizer
compute-sanitizer ./bin/cutlass_test_unit_gemm_device_tensorop_sm90

# Run with specific problem count
CUTLASS_UNIT_TEST_PROBLEM_COUNT=1 ./bin/cutlass_test_unit_gemm_device_tensorop_sm90

# Run with verbose trace
CUTLASS_TRACE=1 ./bin/cutlass_test_unit_gemm_device_tensorop_sm90

# Run only specific test cases
./bin/cutlass_test_unit_gemm_device_tensorop_sm90 --gtest_filter="*f16*64x128*"

# List all test cases without running
./bin/cutlass_test_unit_gemm_device_tensorop_sm90 --gtest_list_tests
```

---

## 10. Benchmarking CUTLASS Kernels vs cuBLAS Baselines

### CUTLASS Profiler (Primary Benchmarking Tool)

The CUTLASS profiler (`tools/profiler/`) is the official benchmarking tool:

```bash
# Build the profiler
cmake -B build -DCUTLASS_NVCC_ARCHS="90a" -DCUTLASS_ENABLE_PROFILER=ON
cmake --build build --target cutlass_profiler -j$(nproc)

# Benchmark a GEMM kernel
./build/tools/profiler/cutlass_profiler \
  --m=4096 --n=4096 --k=4096 \
  --A=f16:column --B=f16:row --C=f16:column \
  --op_class=tensorop \
  --verification=true

# Compare against cuBLAS
./build/tools/profiler/cutlass_profiler \
  --m=4096 --n=4096 --k=4096 \
  --A=f16:column --B=f16:row --C=f16:column \
  --providers=cublas,cutlass
```

### cuBLAS Integration in Tests

The test infrastructure links cuBLAS when `CUTLASS_ENABLE_CUBLAS` is set:
```cmake
target_link_libraries(cutlass_test_unit_infra
  PUBLIC
  $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
)
```

### Writing a Custom Benchmark for SAM 3.1

```cpp
// benchmark_sam31.cpp
#include <cuda_runtime.h>
#include <chrono>
#include "cutlass/gemm/device/gemm_universal_adapter.h"
// ... your kernel includes

void benchmark_gemm(int M, int N, int K, int iterations = 100) {
    // Allocate and initialize
    // Warmup
    for (int i = 0; i < 10; i++) { /* launch kernel */ }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        // Launch CUTLASS kernel
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = 2.0 * M * N * K * iterations / (ms / 1000.0) / 1e9;
    printf("M=%d N=%d K=%d: %.2f GFLOPS (%.3f ms/iter)\n",
           M, N, K, gflops, ms / iterations);

    // Repeat with cuBLAS for comparison
    // ...
}
```

### Key Benchmarking Considerations

1. **Problem sizes matter** — cuBLAS is highly tuned for common sizes (powers of 2, multiples of 64). CUTLASS advantage appears at custom tile shapes and fused epilogues.
2. **Warmup iterations** — Always discard the first few launches (cold cache, JIT compilation).
3. **Synchronization** — Use `cudaEventSynchronize` or `cudaDeviceSynchronize` for accurate timing.
4. **Memory bandwidth** — For memory-bound kernels (small K), both CUTLASS and cuBLAS are bandwidth-limited; GFLOPS comparison is less meaningful.
5. **Epilogue fusion** — CUTLASS's real advantage is fused epilogues (bias + ReLU + scaling in one pass). cuBLAS requires separate kernel launches for post-processing.

---

## Summary: What This Means for SAM 3.1

For validating SAM 3.1 CUTLASS kernels, follow this testing checklist:

- [ ] **Create device-level tests** for each kernel configuration using the `testbed.h` pattern
- [ ] **Test all target architectures** (likely sm90 or sm100) with proper `#if` guards
- [ ] **Test multiple tile shapes** — at least the shapes you plan to deploy, plus one large and one small
- [ ] **Test layout combinations** — all layouts your kernel supports
- [ ] **Test edge cases** — small M/N/K, odd sizes (primes), max alignment, sub-alignment
- [ ] **Test alpha/beta variations** — at least α=1/β=0 and α=2/β=3
- [ ] **Test epilogue fusions** if applicable (bias, ReLU, scaling)
- [ ] **Integrate with CMake** using `cutlass_test_unit_gemm_device_add_executable()`
- [ ] **Benchmark against cuBLAS** using the profiler or custom benchmark harness
- [ ] **Run compute-sanitizer** on all tests before considering them production-ready

The CUTLASS test suite is an engineering marvel in its own right — hundreds of executables covering every architecture, data type, layout, and tile shape combination. Use it as both a correctness oracle and a template for your own kernel validation infrastructure.
