# CUDA Compiler Flags for Maximum SAM 3.1 Performance

## Essential Flags

### Architecture Targeting
```
--gpu-architecture=sm_100a    # Blackwell SM100 (data center)
--gpu-architecture=sm_120a    # Blackwell SM120 (consumer RTX 5060)
--gpu-architecture=sm_90a     # Hopper (fallback if SM120 not supported)
```
The 'a' suffix enables architecture-specific features (TMA, WGMMA, etc).

### Optimization Level
```
-O3                           # Maximum optimization
--use_fast_math               # Approximate math (sin, cos, exp, tanh)
                              # +5-10% speed, slight accuracy loss
                              # GELU uses tanh — benefit applies!
```

### Register Control
```
-Xptxas --maxrregcount=128   # Limit registers per thread
                              # More threads per SM = higher occupancy
                              # But too few = register spilling
                              # Sweet spot: 64-128 for GEMM, 32-64 for simple kernels
```

### Compilation Speed
```
--split-compile=32            # Parallel compilation (32 threads)
-ftz=true                     # Flush denormals to zero (free performance)
-fmad=true                    # Use FMA instructions (default, but explicit)
```

## CUTLASS-Specific Flags
```cmake
target_compile_options(sam31 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_120a>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=32>
)
```

## Impact Estimation

| Flag | Performance Gain | Risk |
|------|-----------------|------|
| -O3 vs -O2 | +10-15% | None |
| --use_fast_math | +5-10% | GELU accuracy slightly changes |
| --ftz=true | +2-5% | Denormals become zero |
| --maxrregcount=128 | +5-15% | May cause spilling if too low |
| --split-compile=32 | Build speed only | None |

**Combined: 20-40% faster vs default flags**

