# lau-cudaclaw-bridge

> Bridge connecting lau-* math ecosystem to CUDAclaw GPU dispatch

## What This Does

Bridge connecting lau-* math ecosystem to CUDAclaw GPU dispatch. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-cudaclaw-bridge
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_cudaclaw_bridge::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub enum BridgeError 
pub struct GpuAvailability 
    pub fn probe() -> Self 
pub struct UnifiedBuffer<T> 
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self> 
    pub fn from_flat(data: Vec<T>) -> Self 
    pub fn data(&self) -> &[T] 
    pub fn data_mut(&mut self) -> &mut Vec<T> 
    pub fn shape(&self) -> &[usize] 
    pub fn len(&self) -> usize 
    pub fn is_empty(&self) -> bool 
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Result<Self> 
pub trait GpuDispatch: Send + Sync 
pub struct DispatchContext 
pub struct DispatchResult 
pub struct TensorBridge;
    pub fn to_unified(data: Vec<f64>, shape: Vec<usize>) -> Result<UnifiedBuffer<f64>> 
    pub fn from_unified(buf: &UnifiedBuffer<f64>) -> (Vec<f64>, Vec<usize>) 
    pub fn matrix_to_unified(m: &nalgebra::DMatrix<f64>) -> UnifiedBuffer<f64> 
    pub fn unified_to_matrix(buf: &UnifiedBuffer<f64>) -> Result<nalgebra::DMatrix<f64>> 
pub enum MathOp 
pub enum ReduceOp 
pub enum ElementOp 
pub struct MathBridge 
    pub fn new() -> Self 
    pub fn dispatch_op(&self, op: &MathOp, input: &[f64]) -> Result<Vec<f64>> 
pub struct PdeGrid 
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self 
    pub fn laplacian(&self) -> PdeGrid 
    pub fn heat_step(&self, alpha: f64, dt: f64) -> PdeGrid 
    pub fn set_boundary(&mut self, value: f64) 
pub struct PdeBridge 
    pub fn new() -> Self 
    pub fn dispatch_laplacian(&self, grid: &PdeGrid) -> PdeGrid 
    pub fn dispatch_heat_step(&self, grid: &PdeGrid, alpha: f64, dt: f64) -> PdeGrid 
pub struct NnLayer 
    pub fn new(input_dim: usize, output_dim: usize) -> Self 
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> 
pub struct NnBridge 
    pub fn new() -> Self 
    pub fn forward_batch(&self, layers: &[NnLayer], batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> 
pub struct Graph 
    pub fn new(node_count: usize) -> Self 
    pub fn add_edge(&mut self, a: usize, b: usize) 
    pub fn add_directed_edge(&mut self, from: usize, to: usize) 
    pub fn bfs(&self, start: usize) -> Vec<usize> 
    pub fn shortest_paths(&self, start: usize) -> Vec<Option<usize>> 
    pub fn edge_count(&self) -> usize 
pub struct GraphBridge 
    pub fn new() -> Self 
    pub fn dispatch_bfs(&self, graph: &Graph, start: usize) -> Vec<usize> 
    pub fn dispatch_shortest_paths(&self, graph: &Graph, start: usize) -> Vec<Option<usize>> 
    pub fn dispatch_pagerank(&self, graph: &Graph, iterations: usize, damping: f64) -> Vec<f64> 
pub struct BenchResult 
pub struct BenchmarkHarness 
    pub fn new(warmup_iters: usize, bench_iters: usize) -> Self 
    pub fn bench_cpu<F>(&self, label: &str, flops: u64, mut f: F) -> BenchResult
    pub fn bench_cpu_vs_gpu<Fcpu, Fgpu>(
pub struct MathDispatch 
    pub fn new(op: MathOp, input: Vec<f64>) -> Self 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**69 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
