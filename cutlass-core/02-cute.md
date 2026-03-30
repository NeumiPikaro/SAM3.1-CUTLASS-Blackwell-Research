# CuTe: CUDA Tensor Templates — Deep Technical Analysis

## 1. Introduction and Motivation

CuTe (CUDA Tensor Templates) is a C++ template library within NVIDIA's CUTLASS project that provides a composable, mathematically-grounded abstraction for describing multi-dimensional data layouts on GPUs. Introduced in CUTLASS 3.x, CuTe completely replaces the older `cutlass::TensorRef` and `cutlass::layout::*` classes that dominated CUTLASS 2.x.

### Why CuTe Was Created

The CUTLASS 2.x layout system suffered from several architectural problems:

1. **Rigid layout taxonomy**: Each layout (ColumnMajor, RowMajor, TensorOpMultiplicand, etc.) was a separate class with its own coordinate-to-offset logic. Adding new layout types required writing boilerplate `operator()`, `stride()`, and `capacity()` implementations.

2. **Composition was implicit and fragile**: Tiling a shared-memory tensor for register-level MMA required manual index computation. There was no algebraic way to express "reshape this 2D layout into a 4D tiled layout."

3. **Swizzle was ad-hoc**: The smem swizzle mechanism in CUTLASS 2.x was baked into specific GEMM kernel templates with hardcoded bit manipulations, making it difficult to reason about or generalize.

4. **Code was opaque**: Reading a CUTLASS 2.x GEMM kernel required mentally tracking pointer offsets through layers of template specializations. The data movement logic was inseparable from the compute logic.

CuTe addresses all of these problems by providing a single mathematical object — the **Layout** — and building all higher-level abstractions (tiling, partitioning, swizzling, tensor views) as composable operations on layouts.

---

## 2. The Layout Concept: Shape, Stride, Coordinate → Address Mapping

### 2.1 The Layout Template

The core type in CuTe is `cute::Layout<Shape, Stride>`:

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape>>
struct Layout
    : private cute::tuple<Shape, Stride>   // EBO for static layouts
{
    CUTE_HOST_DEVICE constexpr
    Layout(Shape const& shape = {}, Stride const& stride = {})
        : cute::tuple<Shape, Stride>(shape, stride)
    {}
    
    static constexpr int rank = rank_v<Shape>;
    
    // Map a coordinate to a linear index
    template <class Coord>
    CUTE_HOST_DEVICE constexpr
    auto operator()(Coord const& coord) const {
        return crd2idx(coord, shape(), stride());
    }
};
```

A `Layout` is a pure function from **logical coordinates** to **linear offsets**. It stores two things:

- **Shape**: An `IntTuple` — an integer or nested `cute::tuple` of integers — defining the domain of the function. For example, `Shape = cute::tuple<Int<64>, Int<64>>` defines a 64×64 2D grid.
- **Stride**: An `IntTuple` congruent to Shape, defining how each dimension contributes to the linear offset.

### 2.2 The Coordinate-to-Index Mapping

The mapping is implemented by `crd2idx(coord, shape, stride)`, defined in `stride.hpp`:

```cpp
// [coord, shape, and stride are all integers => step forward by stride]
op(c, s, d)             => c * d

// [coord is integer, shape and stride are tuple => divmod coord for each mode]
op(c, (s,S), (d,D))     => op(c % prod(s), s, d) + op(c / prod(s), (S), (D))

// [coord, shape, and stride are all tuples => consider each mode independently]
op((c,C), (s,S), (d,D)) => op(c, s, d) + op((C), (S), (D))
```

This recursive definition handles:
- **Scalar × Scalar × Scalar**: The base case. `crd2idx(c, s, d) = c * d`.
- **Scalar × Tuple × Tuple**: The coordinate is split via divmod against the product of the shape's first element, then recursed. This handles hierarchical shapes.
- **Tuple × Tuple × Tuple**: Each element of the coordinate tuple is independently mapped with the corresponding shape/stride pair, then results are summed.

For a concrete example, consider a column-major 2D layout:

```cpp
// Column-major: M=4, N=8
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutLeft{});
// shape  = (4, 8)
// stride = (1, 4)   // column-major: stride of dim-0 is 1, dim-1 is product of dim-0 sizes

// crd2idx(2, 3, shape, stride) = 2*1 + 3*4 = 14
layout(make_coord(2, 3));  // => 14
```

Row-major:

```cpp
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
// shape  = (4, 8)
// stride = (8, 1)   // row-major: stride of dim-1 is 1, dim-0 is product of dim-1 sizes

layout(make_coord(2, 3));  // => 2*8 + 3*1 = 19
```

### 2.3 Hierarchical Shapes and Strides

The critical insight is that Shape and Stride can be **nested tuples**. A layout with shape `((2,2), (4,2))` and stride `((1,8), (2,16))` describes a hierarchical coordinate system where each dimension is subdivided. The `crd2idx` recursion handles this naturally through its tuple-tuple-tuple branch.

This is the foundation for tiling: when CuTe tiles a layout, it produces a new layout whose shape is a nested tuple like `((TileM, TileN), (RestM, RestN))`.

### 2.4 Static vs Dynamic Values

CuTe aggressively uses compile-time values via `cute::Int<N>`, a type-level integer. When Shape and Stride are composed entirely of `Int<>` values, the entire coordinate-to-offset computation is resolved at compile time — no runtime arithmetic is emitted for constant-layout operations. Dynamic values (regular `int`) are used when dimensions are determined at runtime.

```cpp
// Fully static — resolved at compile time
Layout<Shape<Int<4>, Int<8>>, Stride<Int<1>, Int<4>>> static_layout;

// Mixed — some dimensions are dynamic
Layout<Shape<int, Int<8>>, Stride<Int<1>, int>> mixed_layout;
```

The default stride (when none is provided) is `LayoutLeft::Apply<Shape>`, which computes compact column-major strides.

---

## 3. Tensor Abstraction: Owning vs Non-Owning

### 3.1 The Tensor Template

```cpp
template <class Engine, class Layout>
struct Tensor
{
    using iterator     = typename Engine::iterator;
    using value_type   = typename Engine::value_type;
    using element_type = typename Engine::element_type;
    
    cute::tuple<layout_type, engine_type> rep_;
    
    // Indexing: layout computes offset, engine dereferences
    template <class Coord>
    CUTE_HOST_DEVICE constexpr
    decltype(auto) operator[](Coord const& coord) {
        return data()[layout()(coord)];
    }
    
    // Slicing: returns a new non-owning Tensor with a sub-layout
    template <class Coord>
    CUTE_HOST_DEVICE constexpr
    decltype(auto) operator()(Coord const& coord) {
        if constexpr (has_underscore<Coord>::value) {
            auto [sliced_layout, offset] = slice_and_offset(coord, layout());
            return make_tensor(data() + offset, sliced_layout);
        } else {
            return data()[layout()(coord)];
        }
    }
};
```

A `Tensor<Engine, Layout>` combines:
- **Engine**: The data storage mechanism (iterator or owning container)
- **Layout**: The coordinate-to-offset mapping

### 3.2 Engine Types

CuTe defines several engine types:

```cpp
// Owning: stores data inline (stack allocation)
template <class T, size_t N>
struct ArrayEngine {
    using Storage = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                         array_aligned<T,N>,
                                         array_subbyte<T,N>>::type;
    Storage storage_;
    CUTE_HOST_DEVICE constexpr auto begin() { return storage_.begin(); }
};

// Non-owning: wraps an existing iterator/pointer
template <class Iterator>
struct ViewEngine {
    iterator storage_;
    CUTE_HOST_DEVICE constexpr iterator& begin() { return storage_; }
};
```

The `make_tensor` factory function dispatches based on whether its first argument is an iterator (→ `ViewEngine`, non-owning) or a type/size specification (→ `ArrayEngine`, owning):

```cpp
// Non-owning tensor from a raw pointer
float* ptr = ...;
auto tensor = make_tensor(ptr, make_layout(make_shape(64, 64), LayoutLeft{}));

// Owning tensor (e.g., for register fragments)
auto tensor = make_tensor<float>(make_shape(Int<16>{}, Int<16>{}));
```

### 3.3 The Underscore (Slice) Operator

The `operator()` overload with `_` (underscore) is a key feature. When a coordinate contains `_` wildcard entries, the tensor returns a **sub-tensor** (a view), not a scalar element:

```cpp
auto tile = tensor(make_coord(_, _), make_coord(0, _));
// Returns a Tensor viewing a 2D slice of the original
```

This is how CuTe implements the "partition" concept — a thread's view of a tile is obtained by slicing with thread indices and wildcards.

---

## 4. Mathematical Foundation of Layout

A CuTe Layout is a **mathematical function** f: D → C where:
- **Domain D** is the set of all valid coordinates, defined by the Shape
- **Codomain C** is a subset of ℕ (the linear index space), whose size is `cosize(layout)`
- **The function** maps each coordinate c ∈ D to f(c) = crd2idx(c, shape, stride) ∈ C

Key properties:

| Property | Definition | CuTe Function |
|----------|-----------|---------------|
| Domain size | \|D\| = ∏ᵢ shapeᵢ | `size(layout)` |
| Codomain size | \|C\| ≥ \|D\| | `cosize(layout)` |
| Injectivity | f is 1-1 iff layout is "valid" | Checked by `is_valid_layout` |
| Compactness | f is bijective onto [0, \|D\|) | `size(layout) == cosize(layout)` |
| Rank | Number of modes | `Layout::rank` |
| Depth | Nesting depth of shape | `depth(layout)` |

A layout is **compact** when `size == cosize`, meaning every index in [0, size) maps to exactly one coordinate. A compact layout can be iterated linearly without gaps.

The `LayoutLeft` and `LayoutRight` tags produce compact layouts with column-major and row-major stride ordering, respectively:

```cpp
compact_major<LayoutLeft>(Shape<4,8>)   // => Stride<1, 4>
compact_major<LayoutRight>(Shape<4,8>)  // => Stride<8, 1>
```

---

## 5. Composition and Complement Operations

### 5.1 Layout Composition

Composition is CuTe's most powerful operation. Given layouts A and B:

```
composition(A, B) produces a layout C such that C(x) = A(B(x))
```

When B is a single layout and `size(B) == size(A)` (divisibility condition), CuTe computes the composition directly:

```cpp
template <class LayoutA, class LayoutB>
CUTE_HOST_DEVICE constexpr
auto composition(LayoutA const& a, LayoutB const& b);
```

The mathematical result: if `A = Layout<ShapeA, StrideA>` and `B = Layout<ShapeB, StrideB>`, then:
- `ShapeC = ShapeB` (the composed layout's domain is B's domain)
- `StrideC` is computed by transforming StrideA through B's inverse mapping

### 5.2 ComposedLayout (Non-Trivial Composition)

When the divisibility condition is violated (A's codomain is not evenly divisible by B's domain), CuTe falls back to a `ComposedLayout<A, Offset, B>`:

```cpp
template <class LayoutA, class Offset, class LayoutB>
struct ComposedLayout : private cute::tuple<LayoutA, Offset, LayoutB>
{
    // Mapping: (A ∘ O ∘ B)(c) = A(O + B(c))
    template <class Coord>
    CUTE_HOST_DEVICE constexpr
    auto operator()(Coord const& coord) const {
        return layout_a()(offset() + layout_b()(coord));
    }
    
    // stride() is DELETED — composed layouts don't have meaningful strides
    CUTE_HOST_DEVICE constexpr
    auto stride() const = delete;
};
```

`ComposedLayout` is essential for swizzle layouts and other non-linear transforms where a simple stride-based representation is insufficient.

### 5.3 The Complement

The complement operation finds a layout B such that `composition(A, B)` produces a specific desired result. Given a layout with shape S and a target tile shape T, the **logical divide** produces:

```
Layout divided = logical_divide(layout, tile_shape)
// divided has shape ((T₀, T₁, ...), (R₀, R₁, ...)) where Rᵢ = Sᵢ / Tᵢ
// The inner tuple is the "tile" dimension, outer is the "rest" dimension
```

The **tiled_divide** operation is used everywhere in CUTLASS 3.x to partition tensors:

```cpp
// Partition a 128×128 matrix into 32×32 tiles
auto smem_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));
auto tiled = smem_layout.tile(make_shape(Int<32>{}, Int<32>{}));
// tiled has shape ((32, 32), (4, 4)) — a 4×4 grid of 32×32 tiles
```

---

## 6. Swizzle Layouts for Bank Conflict Avoidance

Shared memory on NVIDIA GPUs is organized into 32 banks. When multiple threads in a warp access different addresses that map to the same bank, a bank conflict occurs, serializing the accesses. CuTe provides `Swizzle<B, M, S>` as a composable layout primitive to avoid these conflicts.

### 6.1 The Swizzle Functor

```cpp
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
    static constexpr int num_bits = BBits;
    static constexpr int num_base = MBase;
    static constexpr int num_shft = SShift;
    
    using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
    using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0, num_shft))>;
    using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0, num_shft))>;
    using msk_sft = cute::constant<int, num_shft>;
    
    static constexpr uint32_t swizzle_code = uint32_t(yyy_msk::value | zzz_msk::value);
    
    template <class Offset>
    CUTE_HOST_DEVICE constexpr static
    auto apply(Offset const& offset) {
        return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});  // ZZZ ^= YYY
    }
};
```

The swizzle is a **bitwise XOR** on the linear address. Given an address of the form:

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
```

Where:
- `MBase` = number of least-significant bits to keep constant (typically 2 for 4-byte alignment)
- `BBits` = number of bits in the mask
- `SShift` = distance to shift the YYY mask

The result XORs ZZZ with YYY:

```
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx  where AA = ZZ xor YY
```

### 6.2 How Swizzle Avoids Bank Conflicts

The GPU shared memory has 32 banks, and bank assignment is `address % 32` (for 4-byte words). For a column-major tile where consecutive threads read consecutive rows, the stride between rows can cause all threads to hit the same banks.

By XORing bits of the address, the swizzle permutes which bank each address maps to. If thread `t` accesses address `a`, and thread `t+1` accesses address `a + stride`, the XOR transformation can ensure `(a ^ swizzle(a)) % 32 ≠ ((a + stride) ^ swizzle(a + stride)) % 32`, distributing accesses across banks.

The swizzle parameters are chosen to match the specific access pattern. For example, `Swizzle<3, 4, 3>` is common for Ampere's 128-byte bank-aligned shared memory.

### 6.3 Swizzle as a Layout

CuTe implements swizzle as a `ComposedLayout<Swizzle<B,M,S>, Offset, LayoutB>`:

```cpp
template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr
auto make_layout(Swizzle<B,M,S> const& sxor) {
    return composition(sxor, Layout<Int<M+B+abs(S)>, Int<1>>{});
}
```

The `swizzle_layout.hpp` specialization provides `operator()` that applies the swizzle to the index computed by the inner layout:

```cpp
// For ComposedLayout<Swizzle<B,M,S>, Offset, LayoutB>:
// operator()(coord) = Swizzle<B,M,S>::apply(Offset + layout_b()(coord))
```

Note that `stride()` is **deleted** for swizzle layouts because the swizzle is non-linear — there is no single stride vector that can represent the address mapping.

### 6.4 Pointer Swizzle (Position-Dependent)

There are two distinct swizzle mechanisms:

1. **Swizzle Layout** (`swizzle_layout.hpp`): Position-independent swizzle applied to layout offsets. Used for `cute::Tensor` objects.

2. **Swizzle Pointer** (`pointer_swizzle.hpp`): Position-dependent swizzle applied to the shared memory pointer itself. This is what the hardware actually implements.

```cpp
template <class SwizzleFn, class Iterator>
struct swizzle_ptr : iter_adaptor<Iterator, swizzle_ptr<SwizzleFn, Iterator>>
{
    CUTE_HOST_DEVICE constexpr
    reference operator[](Int const& i) const {
        return *apply_swizzle(this->get() + i);
    }
    
    template <class T>
    CUTE_HOST_DEVICE constexpr static
    T* apply_swizzle(T* ptr) {
        return reinterpret_cast<T*>(SwizzleFn::apply(reinterpret_cast<uintptr_t>(ptr)));
    }
};
```

The key distinction: `Swizzle<0, M, S>` (zero bits) immediately decays to the raw pointer, which is used for smem layouts that don't need swizzling.

---

## 7. TiledMMA Integration with CuTe

### 7.1 The Atom/MMA Architecture

In CUTLASS 3.x, MMA (Matrix Multiply-Accumulate) operations are described as **atoms** — objects that define:
- The instruction to use (e.g., `SM80_16x8x16_F16F16F16F16_TN`)
- The **layout of registers** that the atom expects (how A, B, C tiles are partitioned among threads)

A `TiledMMA` composes an MMA atom with a tiling pattern:

```cpp
// The MMA atom knows the register layout for its operands
using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;

// TiledMMA applies a thread-level and value-level tiling
using TiledMMA = TiledMMA<
    MMA_Atom,
    Layout<Shape<_4,_8,_1>>,     // Thread arrangement (32 threads)
    Layout<Shape<_1,_2,_1>>       // Value arrangement (per-thread values)
>;
```

### 7.2 Partitioning with CuTe Layouts

The `TiledMMA` uses CuTe's layout operations to partition shared-memory tensors into per-thread, per-MMA-instruction fragments:

```cpp
// Partition the A matrix for the tiled MMA
auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
auto tAgA = thr_mma.partition_A(gmem_tensor);   // Thread's view of global memory A
auto tAsA = thr_mma.partition_A(smem_tensor);    // Thread's view of shared memory A
auto tArA = thr_mma.make_fragment_A(tAgA);       // Register fragment matching tAgA's layout
```

Each `partition_*` call is a `composition` / `tile` operation on the tensor's layout. The thread slice produces a layout that maps (thr_idx, val_idx, ...) → original tensor coordinate. This is expressed entirely in CuTe's layout algebra.

### 7.3 Register Fragment Layout

The register fragment for an MMA is an owning `Tensor` whose layout exactly matches the partitioned view:

```cpp
// tArA is a Tensor<ArrayEngine<half_t, ...>, Layout<...>>
// Its layout describes which register holds which element of the MMA tile

// The copy from smem to registers is just:
copy(tAsA, tArA);
```

The `copy` algorithm uses the layouts of both tensors to emit the appropriate `ldmatrix` or `cp.async` instructions.

---

## 8. cute::Tensor vs cutlass::TensorRef

| Aspect | `cutlass::TensorRef` (2.x) | `cute::Tensor` (3.x) |
|--------|---------------------------|----------------------|
| Layout | Pointer + stride pointer + extent | Separated `Layout<Shape,Stride>` object |
| Type safety | Strides are runtime `int*` | Strides can be `Int<N>` (compile-time) |
| Tiling | Manual index computation | `layout().tile(shape)` → automatic |
| Swizzle | Baked into kernel template | Composable `Swizzle<B,M,S>` layout |
| Slicing | No native support | `operator()` with `_` returns sub-tensor |
| Ownership | Always non-owning | Owning (`ArrayEngine`) or non-owning (`ViewEngine`) |
| Compile-time | Minimal | Aggressive use of `Int<>` for zero-cost abstraction |
| Expressiveness | `TensorRef<T, LayoutTag>` | `Tensor<Engine, Layout>` where Layout is a composable algebra |

The key philosophical difference: `TensorRef` bundles pointer + layout metadata together in an opaque handle. `Tensor` is a clean separation of **data** (Engine) and **coordinate mapping** (Layout), where the Layout is a first-class mathematical object that can be composed, tiled, sliced, and reasoned about independently.

### Migration Pattern

```cpp
// CUTLASS 2.x
TensorRef<ElementA, LayoutA> tensor_ref_A(ptr_A, lda);
int offset = tensor_ref_A.offset(make_Coord(m, n));
ElementA val = tensor_ref_A.at(make_Coord(m, n));

// CUTLASS 3.x / CuTe
auto tensor_A = make_tensor(ptr_A, make_layout(make_shape(M, N), make_stride(lda, Int<1>{})));
auto val = tensor_A(make_coord(m, n));

// Tiling — impossible in 2.x without manual math
auto tiled_A = tile(tensor_A, make_shape(Int<32>{}, Int<32>{}));
// tiled_A has shape ((32, 32), (M/32, N/32))
```

---

## 9. CuTe's Role in CUTLASS Readability and Maintainability

### 9.1 Separation of Concerns

In CUTLASS 2.x, the GEMM kernel's `operator()` contained interleaved:
- Tiling logic
- Pointer arithmetic
- Swizzle computation
- MMA instruction dispatch
- Mainloop body

In CUTLASS 3.x, these are cleanly separated:

```
┌─────────────────────────────────────────┐
│ Kernel (coordination)                   │
│  ├── Tensor creation (Engine + Layout)  │
│  ├── Tiling (Layout operations)         │
│  ├── Partitioning (Layout composition)  │
│  ├── Copy (Algorithm on Tensors)        │
│  └── MMA (TiledMMA on Tensors)          │
└─────────────────────────────────────────┘
```

Each layer operates on `cute::Tensor` objects. The layout algebra means that tiling, partitioning, and swizzling are **declarative** — you describe *what* layout you want, not *how* to compute each offset.

### 9.2 Self-Documenting Code

CuTe layouts are readable because they encode the mathematical structure:

```cpp
// This is a 128×128 smem tensor, tiled into 64×64 tiles, with 16-byte swizzle
auto smem = make_tensor(smem_ptr,
    composition(Swizzle<3,4,3>{},
        make_layout(make_shape(Int<128>{}, Int<128>{}),
                    make_stride(Int<128>{}, Int<1>{}))));
```

The shape hierarchy tells you the data structure. The swizzle tells you the bank conflict strategy. The strides tell you the memory ordering. None of this requires reading 500 lines of template specializations.

### 9.3 Composability Enables Generality

Because everything is a layout or an operation on layouts, CuTe naturally supports:
- New data types (FP8, INT4) via the Engine abstraction
- New architectures via different MMA atoms
- New operations (convolution, attention) via different tiling patterns
- New memory hierarchies (async cp, TMA) via different copy algorithms

All of these share the same layout algebra.

---

## 10. Practical Examples of Layout Manipulation

### 10.1 Constructing Common Layouts

```cpp
// Row-major M×N
auto row_major = make_layout(make_shape(M, N), LayoutRight{});
// shape = (M, N), stride = (N, 1)

// Column-major M×N
auto col_major = make_layout(make_shape(M, N), LayoutLeft{});
// shape = (M, N), stride = (1, M)

// Custom stride: each row separated by `lda` elements
auto custom = make_layout(make_shape(M, N), make_stride(lda, Int<1>{}));

// Static 4D layout
auto layout_4d = make_layout(
    make_shape(Int<4>{}, Int<8>{}, Int<2>{}, Int<16>{}),
    make_stride(Int<1>{}, Int<4>{}, Int<32>{}, Int<64>{})
);
```

### 10.2 Tiling a Tensor

```cpp
// Start with a 128×256 matrix in global memory
auto gA = make_tensor(gmem_ptr,
    make_layout(make_shape(Int<128>{}, Int<256>{}), LayoutRight{}));

// Tile it into 32×64 blocks
auto tiled_gA = gA.tile(make_shape(Int<32>{}, Int<64>{}));
// tiled_gA has shape ((32, 64), (4, 4))
// Index as: tiled_gA(make_coord(m_local, n_local), make_coord(m_tile, n_tile))

// Access element in tile (1, 2), local coordinate (3, 5):
auto elem = tiled_gA(make_coord(3, 5), make_coord(1, 2));
// Same as gA(make_coord(3 + 1*32, 5 + 2*64)) = gA(make_coord(35, 133))
```

### 10.3 Thread Partitioning

```cpp
// A 64×64 tile partitioned among 32 threads
auto tile = make_tensor(smem_ptr,
    make_layout(make_shape(Int<64>{}, Int<64>{}), LayoutLeft{}));

// Each thread gets a 8×8 sub-tile (64 elements)
// Group threads as 8×4
auto thread_layout = make_layout(make_shape(Int<8>{}, Int<4>{}));
// Partition the tile by thread layout
auto partitioned = zipped_divide(tile, thread_layout);
// partitioned has shape ((8, 4), (8, 8), (1, 16))
//   (thread_row, thread_col, element_row, element_col, ...)
```

### 10.4 Swizzled Shared Memory Layout

```cpp
// 128×64 smem tensor with 128-byte bank swizzle
auto smem_layout = make_layout(
    make_shape(Int<128>{}, Int<64>{}),
    make_stride(Int<1>{}, Int<128>{})
);

// Apply Swizzle<3,4,3>: 3-bit XOR mask at bit position 4, shifted by 3
// This permutes addresses within a 128-byte window to avoid bank conflicts
// on 8×8 MMA tile loads
auto swizzled_smem = composition(
    Swizzle<3, 4, 3>{},
    Int<0>{},  // offset
    smem_layout
);
// swizzled_smem is a ComposedLayout — can't query stride(), but operator() works
```

### 10.5 Complement for Register Tiling

```cpp
// Global tensor: 256×256
auto gC = make_tensor(gmem_ptr,
    make_layout(make_shape(Int<256>{}, Int<256>{}), LayoutLeft{}));

// Block tile: 128×128
auto block_tile = make_shape(Int<128>{}, Int<128>{});
auto tiled_gC = gC.tile(block_tile);
// shape = ((128, 128), (2, 2))

// Within block, partition for 4 warps × 8 values per warp
auto warp_layout = make_shape(Int<4>{}, Int<1>{});
auto val_layout  = make_shape(Int<32>{}, Int<128>{});

// Full partitioning stack
auto thr_tile = tiled_divide(tiled_gC, make_tile(warp_layout, val_layout));
// This produces a deeply nested shape that describes:
// (block_tile, warp_index, value_index) → linear offset
```

### 10.6 Index-to-Coordinate (Inverse Mapping)

```cpp
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutLeft{});

// Given linear index 14, find the hierarchical coordinate
auto hier = layout.get_hier_coord(14);
// hier = (2, 3)  — in shape (4, 8), index 14 is at (2, 3)

// Flat coordinate (depth == 1)
auto flat = layout.get_flat_coord(14);
// flat = (2, 3)

// 1D coordinate (generalized column-major)
auto one_d = layout.get_1d_coord(14);
// one_d = 14 (identity for compact layouts)
```

---

## Summary

CuTe represents a paradigm shift in GPU kernel programming. By treating data layouts as **first-class mathematical functions** rather than ad-hoc pointer arithmetic, it enables:

1. **Algebraic composition**: Tiling, partitioning, and swizzling are layout operations that compose predictably.
2. **Zero-cost abstraction**: Compile-time `Int<>` values eliminate runtime overhead for static shapes.
3. **Clean separation**: Data (Engine) is independent from coordinate mapping (Layout), enabling both owning and non-owning tensors.
4. **Readability**: The shape hierarchy and stride information are visible in the type system, making code self-documenting.
5. **Generality**: New architectures, data types, and operations are added by defining new layouts and atoms — the algebraic framework remains unchanged.

The mathematical foundation — a Layout as a function from coordinates to offsets, with composition as function composition — provides a rigorous basis that ensures correctness when building complex multi-level tiled, swizzled, partitioned tensor views for high-performance GPU computation.

---

**Sources**: NVIDIA CUTLASS repository — `include/cute/layout.hpp`, `include/cute/layout_composed.hpp`, `include/cute/tensor_impl.hpp`, `include/cute/swizzle.hpp`, `include/cute/swizzle_layout.hpp`, `include/cute/pointer_swizzle.hpp`, `include/cute/stride.hpp`, `include/cute/int_tuple.hpp`.
