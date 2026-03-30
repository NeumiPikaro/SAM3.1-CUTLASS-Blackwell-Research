# Triton Compiler for SAM 3.1 Optimization

## What is Triton
OpenAI's Python-based GPU kernel compiler. Write attention kernels in Python, auto-tune tile sizes.

## Key Features
- Block-level SPMD programming model
- Auto-tuning: explores tile sizes, pipeline stages automatically
- JIT compilation: seconds, not minutes
- Flash attention in Triton: `triton.ops.flash_attention`

## Triton vs CUTLASS
| Aspect | Triton | CUTLASS |
|--------|--------|---------|
| Language | Python | C++ templates |
| Auto-tune | Built-in | Manual |
| Performance ceiling | Good (80-90% of peak) | Best (90%+ of peak) |
| Blackwell support | Catching up | Day 1 |
| Complexity | Low | High |
| Custom fusion | Easy | EVT system |

## SAM 3.1 Application
- Triton for rapid prototyping: auto-tune attention kernels in hours
- CUTLASS for production: hand-tuned for maximum performance
- Strategy: prototype in Triton, port winning configs to CUTLASS

## Triton on Blackwell
Status: SM100 support is experimental. SM120 (RTX 5060) may not be fully supported yet.

