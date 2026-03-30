# CUDA Graphs & Kernel Launch Optimization

## 1. The Kernel Launch Problem

Each CUDA kernel launch incurs:
- CPU overhead: 5-15μs (driver processing)
- GPU scheduling: 1-3μs (command processor)
- **Total: 6-18μs per launch**

### SAM 3.1 Launch Count
```
ViT: 32 blocks × (8 GEMMs + 4 non-GEMM ops) = 384 launches
CLIP: 24 layers × 12 ops = 288 launches  
DETR: 12 layers × 15 ops = 180 launches
FPN: ~10 launches
Seg: ~5 launches
Total: ~867 kernel launches
```

### Launch Overhead
```
867 × 12μs (average) = 10.4ms of pure launch overhead
That's 0.4% of T4's 2897ms — not huge but free to eliminate
```

## 2. CUDA Graph Capture

```python
# Capture entire forward pass
import torch

# Warmup (required before capture)
for _ in range(3):
    output = model(image, prompt)
torch.cuda.synchronize()

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(image, prompt)

# Replay — zero launch overhead
g.replay()
```

### How It Works
1. During capture: record all kernel launches, memory ops, dependencies
2. Graph stores: kernel arguments, launch configs, memory addresses
3. Replay: GPU executes entire graph from command processor — no CPU involvement
4. Overhead: single graph launch (~2μs) instead of 867 individual launches

### Limitations
- **Static shapes:** graph captured for specific input sizes
- **No data-dependent control flow:** all branches must be captured
- **Memory addresses fixed:** can't allocate during replay

### SAM 3.1 Compatibility
- SAM 3.1 forward pass has no data-dependent branches ✓
- Fixed resolution per graph (different graph per resolution) ✓
- All memory allocated before capture ✓

## 3. Graph Memory Management

```python
# Pre-allocate all memory
static_image = torch.zeros((1, 3, 1024, 1024), device='cuda', dtype=torch.bfloat16)
static_input_ids = torch.zeros((1, 77), device='cuda', dtype=torch.long)

# Capture with static inputs
with torch.cuda.graph(g):
    static_output = model(static_image, static_input_ids)

# Copy real data into static inputs, then replay
static_image.copy_(real_image)
static_input_ids.copy_(real_input_ids)
g.replay()
```

## 4. Multi-Graph Strategy

For different resolutions:
```python
graphs = {}
def get_or_create_graph(resolution):
    if resolution not in graphs:
        graphs[resolution] = capture_graph(resolution)
    return graphs[resolution]

# Usage
output = get_or_create_graph(1024).replay()
output = get_or_create_graph(512).replay()  # Different graph
```

## 5. Graph + PDL Combination

CUDA graphs eliminate CPU→GPU launch overhead.
PDL eliminates GPU→CPU→GPU round-trip between dependent kernels.

Together:
```
Without: Kernel A → (6μs CPU) → Kernel B → (6μs CPU) → Kernel C
With graph: [A, B, C] → (2μs single launch)
With PDL: A → (0μs) → B → (0μs) → C (within graph)
```

## 6. Expected Savings

```
Launch overhead elimination: 10.4ms → 0.2ms = 10.2ms saved
On T4 (2897ms): 0.4% speedup — minor
On optimized 5060 (390ms): 2.6% speedup — more significant

For batch inference (4 images):
  4 × 867 launches = 3468 launches × 12μs = 41.6ms overhead
  Graph: 2μs → 41.6ms saved — 10% of 390ms target!
```

