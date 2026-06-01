# lau-cudaclaw-bridge

> Bridge connecting the **lau-\*** math library ecosystem (77+ crates) to the **CUDAclaw** GPU dispatch system.

Part of the **PLATO/LAU ecosystem** — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

---

## What This Does

`lau-cudaclaw-bridge` is the glue layer between the `lau-*` mathematical crates and CUDAclaw's GPU command queue. It provides:

- **Adapter traits** (`GpuDispatch`) so any `lau-*` crate can dispatch GPU work through CUDAclaw
- **Unified memory buffers** (`UnifiedBuffer<T>`) that abstract over GPU and CPU storage
- **Tensor bridges** converting `nalgebra` matrices to/from unified memory
- **Math operation dispatch** (matmul, FFT, reduce, elementwise, transpose) with CPU fallback
- **PDE grid operations** (Laplacian stencil, explicit Euler heat equation)
- **Neural network forward passes** (multi-layer, batched, ReLU)
- **Graph algorithms** (BFS, shortest paths, PageRank)
- **Benchmark harness** comparing CPU vs GPU execution times

When a CUDA-capable GPU is available, operations are dispatched through CUDAclaw's volatile lock-free command queue (~50–100 ns dispatch latency). When no GPU is present, everything falls back to CPU implementations — transparently, with no code changes.

---

## The Key Idea

The crate is built on a single design principle: **every GPU operation must degrade gracefully to CPU**. This means:

1. **`GpuDispatch` trait** — implement this to get GPU acceleration; the trait provides a `DispatchContext` that tells you whether a GPU is available.
2. **`UnifiedBuffer<T>`** — a shaped, typed buffer that lives in host memory on CPU-only systems and in CUDA unified memory when a GPU is present.
3. **Bridge structs** (`MathBridge`, `PdeBridge`, `NnBridge`, `GraphBridge`) — each encapsulates a domain (linear algebra, PDEs, neural nets, graph algorithms) and dispatches through the same `GpuDispatch` pipeline.

Everything is `serde`-serializable for snapshot testing and distributed computation scenarios.

---

## Install

```bash
cargo add lau-cudaclaw-bridge
```

### Dependencies

| Crate | Version | Why |
|---|---|---|
| `serde` | 1 | Serialization of all public types |
| `nalgebra` | 0.33 | Matrix ↔ unified buffer conversion |
| `serde_json` | 1 | *(dev-only)* test serialization round-trips |

No CUDA toolkit is required at compile time — the crate compiles on any platform and probes for GPU availability at runtime.

---

## Quick Start

### GPU availability probe

```rust
use lau_cudaclaw_bridge::GpuAvailability;

let avail = GpuAvailability::probe();
println!("GPU: {} ({:?})", avail.gpu_available, avail.device_name);
```

### Math operations with CPU fallback

```rust
use lau_cudaclaw_bridge::{MathBridge, MathOp, ReduceOp};

let bridge = MathBridge::new();
let result = bridge.dispatch_op(
    &MathOp::Reduce { n: 4, op: ReduceOp::Sum },
    &[1.0, 2.0, 3.0, 4.0],
)?;
assert_eq!(result[0], 10.0);
```

### Matrix ↔ unified buffer round-trip

```rust
use lau_cudaclaw_bridge::TensorBridge;
use nalgebra::DMatrix;

let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let buf = TensorBridge::matrix_to_unified(&m);   // row-major on GPU
let m2  = TensorBridge::unified_to_matrix(&buf)?; // back to nalgebra (col-major)
assert_eq!(m, m2);
```

### PDE heat equation simulation

```rust
use lau_cudaclaw_bridge::{PdeGrid, PdeBridge};

let mut grid = PdeGrid::new(64, 64, 0.01, 0.01);
grid.data[32 * 64 + 32] = 100.0; // heat source at center
grid.set_boundary(0.0);           // Dirichlet BC

let bridge = PdeBridge::new();
for _ in 0..1000 {
    grid = bridge.dispatch_heat_step(&grid, 0.5, 0.001);
}
```

### Neural network forward pass

```rust
use lau_cudaclaw_bridge::{NnBridge, NnLayer};

let bridge = NnBridge::new();
let layers = vec![NnLayer::new(3, 16), NnLayer::new(16, 2)];
let batch = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
let outputs = bridge.forward_batch(&layers, &batch)?;
```

### Graph algorithms

```rust
use lau_cudaclaw_bridge::{Graph, GraphBridge};

let mut g = Graph::new(100);
g.add_edge(0, 1);
g.add_edge(1, 2);

let bridge = GraphBridge::new();
let pr = bridge.dispatch_pagerank(&g, 20, 0.85);
let distances = bridge.dispatch_shortest_paths(&g, 0);
```

---

## API Reference

### Error Handling

```rust
pub enum BridgeError {
    NoGpuAvailable,
    DispatchFailed(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    MemoryAllocationFailed(String),
    KernelError(String),
    Timeout,
    Other(String),
}
pub type Result<T> = std::result::Result<T, BridgeError>;
```

### GPU Probe

```rust
pub struct GpuAvailability {
    pub gpu_available: bool,
    pub device_count: u32,
    pub device_name: Option<String>,
}
impl GpuAvailability {
    pub fn probe() -> Self;
}
```

### Unified Buffer

```rust
pub struct UnifiedBuffer<T> { /* private */ }
impl<T: Clone> UnifiedBuffer<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self>;
    pub fn from_flat(data: Vec<T>) -> Self;
    pub fn data(&self) -> &[T];
    pub fn data_mut(&mut self) -> &mut Vec<T>;
    pub fn shape(&self) -> &[usize];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn reshape(self, new_shape: Vec<usize>) -> Result<Self>;
}
```

### GpuDispatch Trait

```rust
pub trait GpuDispatch: Send + Sync {
    fn label(&self) -> &str;
    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult>;
    fn uses_gpu(&self) -> bool { false }
    fn estimated_flops(&self) -> u64 { 0 }
}
```

### Tensor Bridge

```rust
pub struct TensorBridge;
impl TensorBridge {
    pub fn to_unified(data: Vec<f64>, shape: Vec<usize>) -> Result<UnifiedBuffer<f64>>;
    pub fn from_unified(buf: &UnifiedBuffer<f64>) -> (Vec<f64>, Vec<usize>);
    pub fn matrix_to_unified(m: &DMatrix<f64>) -> UnifiedBuffer<f64>;
    pub fn unified_to_matrix(buf: &UnifiedBuffer<f64>) -> Result<DMatrix<f64>>;
}
```

### Math Operations

```rust
pub enum MathOp {
    MatMul { a_shape: (usize, usize), b_shape: (usize, usize) },
    Fft { n: usize },
    Reduce { n: usize, op: ReduceOp },
    ElementWise { n: usize, op: ElementOp },
    Transpose { rows: usize, cols: usize },
}
pub enum ReduceOp { Sum, Max, Min, Product }
pub enum ElementOp { Add, Sub, Mul, Div, Exp, Log, Sqrt }

pub struct MathBridge;
impl MathBridge {
    pub fn new() -> Self;
    pub fn dispatch_op(&self, op: &MathOp, input: &[f64]) -> Result<Vec<f64>>;
}
```

### PDE Grid

```rust
pub struct PdeGrid {
    pub nx: usize, pub ny: usize,
    pub dx: f64, pub dy: f64,
    pub data: Vec<f64>,
}
impl PdeGrid {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self;
    pub fn laplacian(&self) -> PdeGrid;
    pub fn heat_step(&self, alpha: f64, dt: f64) -> PdeGrid;
    pub fn set_boundary(&mut self, value: f64);
}

pub struct PdeBridge;
impl PdeBridge {
    pub fn new() -> Self;
    pub fn dispatch_laplacian(&self, grid: &PdeGrid) -> PdeGrid;
    pub fn dispatch_heat_step(&self, grid: &PdeGrid, alpha: f64, dt: f64) -> PdeGrid;
}
```

### Neural Networks

```rust
pub struct NnLayer {
    pub input_dim: usize, pub output_dim: usize,
    pub weights: Vec<f64>, pub bias: Vec<f64>,
}
impl NnLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self; // Xavier-ish init
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>>; // ReLU activation
}

pub struct NnBridge;
impl NnBridge {
    pub fn new() -> Self;
    pub fn forward_batch(&self, layers: &[NnLayer], batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>>;
}
```

### Graph Algorithms

```rust
pub struct Graph {
    pub node_count: usize,
    pub adjacency: Vec<Vec<usize>>,
}
impl Graph {
    pub fn new(node_count: usize) -> Self;
    pub fn add_edge(&mut self, a: usize, b: usize);        // undirected
    pub fn add_directed_edge(&mut self, from: usize, to: usize);
    pub fn bfs(&self, start: usize) -> Vec<usize>;
    pub fn shortest_paths(&self, start: usize) -> Vec<Option<usize>>;
    pub fn edge_count(&self) -> usize;
}

pub struct GraphBridge;
impl GraphBridge {
    pub fn new() -> Self;
    pub fn dispatch_bfs(&self, graph: &Graph, start: usize) -> Vec<usize>;
    pub fn dispatch_shortest_paths(&self, graph: &Graph, start: usize) -> Vec<Option<usize>>;
    pub fn dispatch_pagerank(&self, graph: &Graph, iterations: usize, damping: f64) -> Vec<f64>;
}
```

### Benchmark Harness

```rust
pub struct BenchResult {
    pub label: String,
    pub cpu_elapsed_ns: u64,
    pub gpu_elapsed_ns: Option<u64>,
    pub speedup: Option<f64>,
    pub flops_estimated: u64,
}

pub struct BenchmarkHarness { pub warmup_iters: usize, pub bench_iters: usize }
impl BenchmarkHarness {
    pub fn new(warmup_iters: usize, bench_iters: usize) -> Self;
    pub fn bench_cpu<F>(&self, label: &str, flops: u64, f: F) -> BenchResult;
    pub fn bench_cpu_vs_gpu<Fcpu, Fgpu>(...) -> BenchResult;
}
```

### Dispatch Adapters

```rust
pub struct MathDispatch  { pub op: MathOp, pub input: Vec<f64> }  // impl GpuDispatch
pub struct PdeDispatch   { pub grid: PdeGrid, pub op: PdeOp }     // impl GpuDispatch
pub struct NnDispatch    { pub layers: Vec<NnLayer>, pub batch: Vec<Vec<f64>> } // impl GpuDispatch
pub struct GraphDispatch { pub graph: Graph, pub op: GraphOp }    // impl GpuDispatch

pub enum PdeOp  { Laplacian, HeatStep { alpha: f64, dt: f64 } }
pub enum GraphOp { Bfs { start: usize }, ShortestPaths { start: usize }, PageRank { iterations: usize, damping: f64 } }
```

---

## How It Works

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  lau-* crate │────▶│ GpuDispatch  │────▶│  CUDAclaw   │
│  (your code) │     │  trait impl  │     │  GPU queue  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │ no GPU?
                           ▼
                    ┌──────────────┐
                    │  CPU fallback │
                    └──────────────┘
```

1. **Your code** calls a bridge method (`dispatch_op`, `dispatch_laplacian`, etc.)
2. The bridge **probes** `GpuAvailability::probe()` to see if CUDAclaw is present
3. If GPU: serializes inputs into a `UnifiedBuffer`, enqueues a CUDAclaw command, awaits result
4. If CPU: executes the equivalent algorithm directly in Rust (same API, same correctness guarantees)

### GpuDispatch Pipeline

Every domain has a corresponding `*Dispatch` struct that implements `GpuDispatch`:

| Struct | Domain | `estimated_flops()` |
|---|---|---|
| `MathDispatch` | Linear algebra, FFT, reductions | matmul: O(mkn), FFT: O(n²), etc. |
| `PdeDispatch` | PDE grid stencils | proportional to grid points |
| `NnDispatch` | Neural net forward passes | O(batch × Σ in×out per layer) |
| `GraphDispatch` | BFS, shortest paths, PageRank | O(V + E) for traversal, O(iter × E) for PageRank |

The `DispatchContext` carries `gpu_available`, `device_id`, `stream_id`, and `timeout_ms`.

### CPU Fallback Implementations

| Operation | Algorithm |
|---|---|
| Matrix multiplication | Triple-loop O(n³) naive multiplication |
| FFT | Naive DFT O(n²) — suitable for testing, not production |
| Reduce | Single-pass scan |
| Element-wise | Zip-map over split input |
| Transpose | Index remapping |
| Laplacian | 5-point finite difference stencil |
| Heat equation | Explicit Euler: u(t+dt) = u(t) + α·dt·∇²u |
| Neural net | Dense layer + ReLU, batched sequential |
| BFS | Standard queue-based breadth-first search |
| PageRank | Power iteration with damping factor |

---

## The Math

### 5-Point Laplacian Stencil

For a 2D scalar field u on a uniform grid with spacing Δx, Δy, the discrete Laplacian at interior point (i, j) is:

$$\nabla^2 u_{i,j} = \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}$$

This is a second-order central difference approximation with truncation error O(Δx² + Δy²).

### Explicit Euler Heat Equation

The heat equation ∂u/∂t = α∇²u is discretized in time as:

$$u^{n+1}_{i,j} = u^n_{i,j} + \alpha \cdot \Delta t \cdot \nabla^2 u^n_{i,j}$$

**Stability condition** (CFL): α·Δt·(1/Δx² + 1/Δy²) ≤ 0.5

### Discrete Fourier Transform

The CPU fallback implements the naive DFT (not FFT):

$$X[k] = \sum_{t=0}^{N-1} x[t] \cdot e^{-i 2\pi k t / N}$$

Input/output are interleaved real/imaginary pairs: `[re₀, im₀, re₁, im₁, …]`.

### PageRank

The PageRank vector **r** is computed via power iteration:

$$\mathbf{r}^{(t+1)} = \frac{1-d}{N} \mathbf{1} + d \cdot M \cdot \mathbf{r}^{(t)}$$

where d is the damping factor (typically 0.85), N is the number of nodes, and M is the column-stochastic adjacency matrix. Dangling nodes distribute their rank uniformly.

### Xavier Weight Initialization

Neural network layers use Xavier/Glorot initialization:

$$W_{ij} \sim \text{Uniform}\left(-\frac{\sqrt{2}}{\sqrt{n_{\text{in}}}}, +\frac{\sqrt{2}}{\sqrt{n_{\text{in}}}}\right)$$

where n_in is the input dimension of the layer.

---

## Testing

**69 tests** covering:

- **Error display** — `BridgeError` formatting for all variants
- **GPU availability** — default (no GPU) and simulated (`LAU_CUDACLAWSIM_GPU=1`)
- **UnifiedBuffer** — construction, shape validation, reshape, mutation, emptiness
- **TensorBridge** — `nalgebra` ↔ unified buffer round-trips, dimension errors
- **MathBridge** — matmul (identity), reduce (sum/max/min/product), elementwise (add/mul), transpose, FFT DC component
- **PDE grid** — flat field Laplacian (zero), spike Laplacian (negative), heat step shape preservation, Dirichlet boundary
- **Neural network** — forward pass shape/value checks, ReLU clipping, batch processing, empty batch
- **Graph** — edge addition (directed/undirected), BFS connectivity, shortest paths, edge count, PageRank convergence
- **Dispatch adapters** — `GpuDispatch` impls for all four domains, FLOP estimation, GPU flag
- **Benchmark harness** — CPU timing, CPU-vs-GPU comparison, FLOP recording
- **Serde round-trips** — JSON serialization for `MathOp`, `UnifiedBuffer`, `Graph`, `PdeGrid`, `NnLayer`

Run:

```bash
cargo test
```

---

## License

MIT
