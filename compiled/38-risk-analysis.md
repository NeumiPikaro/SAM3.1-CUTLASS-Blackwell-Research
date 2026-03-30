# Risk Analysis & Mitigation

## 1. Technical Risks

### Risk: RTX 5060 specs differ from estimates
- **Impact:** Tile sizes and occupancy calculations may be wrong
- **Mitigation:** Design configs with tunable parameters; auto-tune on real hardware
- **Probability:** Medium (specs may shift before launch)

### Risk: FP8 accuracy unacceptable for segmentation
- **Impact:** Can't use FP8 quantization
- **Mitigation:** Per-channel quantization + smooth quantization; fall back to BF16
- **Probability:** Low (FP8 works well for transformers in practice)

### Risk: CUTLASS SM100 support incomplete for some ops
- **Impact:** Some kernels can't be optimized
- **Mitigation:** Use SM90 kernels as fallback (run on SM100 with compatibility)
- **Probability:** Low (CUTLASS 4.4 is mature for SM100)

### Risk: Custom EVT for RoPE is too complex
- **Impact:** Can't fuse RoPE into epilogue
- **Mitigation:** Use separate RoPE kernel (small overhead, ~1% of total)
- **Probability:** Medium (custom EVT nodes are advanced)

## 2. Performance Risks

### Risk: Can't reach 390ms target
- **Impact:** Performance not competitive with TensorRT on HF transformers version
- **Mitigation:** Target is conservative; each tier provides independent speedup
- **Probability:** Medium (estimates have ~20% uncertainty)

### Risk: Attention is bottleneck (not GEMMs)
- **Impact:** GEMM optimization doesn't help as much
- **Mitigation:** Flash Attention addresses this directly
- **Probability:** Low (our profiling shows GEMMs dominate)

## 3. Project Risks

### Risk: 12-week timeline too aggressive
- **Impact:** Some optimization tiers not completed
- **Mitigation:** Each tier delivers independently useful results
- **Probability:** Medium (depends on team size)

### Risk: CUTLASS learning curve delays development
- **Impact:** Slower initial progress
- **Mitigation:** CuTe DSL enables rapid prototyping; extensive examples available
- **Probability:** Medium (CUTLASS has steep curve)

## 4. Mitigation Priority Matrix

| Risk | Impact | Probability | Priority |
|------|--------|-------------|----------|
| RTX 5060 specs wrong | High | Medium | Critical |
| FP8 accuracy loss | Medium | Low | High |
| CUTLASS SM100 gaps | Medium | Low | Medium |
| RoPE EVT complexity | Low | Medium | Low |
| Can't reach 390ms | Medium | Medium | High |
| Timeline overrun | Medium | Medium | Medium |

