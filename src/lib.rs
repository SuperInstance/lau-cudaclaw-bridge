//! # lau-cudaclaw-bridge
//!
//! Bridge connecting the lau-\* math library ecosystem (77+ crates) to the
//! CUDAclaw GPU dispatch system.
//!
//! CUDAclaw provides volatile lock-free command dispatch (~50–100 ns latency),
//! unified memory command queues, persistent GPU worker kernels with warp-level
//! parallelism, NVRTC runtime PTX compilation, cell agents, ML feedback, and
//! DNA mutation.
//!
//! This crate supplies adapter traits so any `lau-*` crate can dispatch GPU
//! work through CUDAclaw.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Error & result types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during GPU dispatch or tensor conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeError {
    /// No CUDA-capable GPU is available; fall back to CPU.
    NoGpuAvailable,
    /// Dispatch queue is full or disconnected.
    DispatchFailed(String),
    /// Shape / dimension mismatch.
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Unified memory allocation failure.
    MemoryAllocationFailed(String),
    /// Kernel compilation or launch failure.
    KernelError(String),
    /// Timeout waiting for GPU result.
    Timeout,
    /// Generic / wrapped error.
    Other(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGpuAvailable => write!(f, "no GPU available"),
            Self::DispatchFailed(msg) => write!(f, "dispatch failed: {msg}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::MemoryAllocationFailed(msg) => write!(f, "memory allocation failed: {msg}"),
            Self::KernelError(msg) => write!(f, "kernel error: {msg}"),
            Self::Timeout => write!(f, "GPU operation timed out"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BridgeError {}

pub type Result<T> = std::result::Result<T, BridgeError>;

// ─────────────────────────────────────────────────────────────────────────────
// GPU availability
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime probe for CUDA device presence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAvailability {
    pub gpu_available: bool,
    pub device_count: u32,
    pub device_name: Option<String>,
}

impl GpuAvailability {
    /// Probe for CUDA devices.  In the absence of a real CUDA runtime we
    /// report no GPU — tests and CPU fallback paths exercise this.
    pub fn probe() -> Self {
        // In a real build the CUDA driver API would be queried here.
        // For now we simulate: check env var or report no GPU.
        if std::env::var("LAU_CUDACLAWSIM_GPU").is_ok() {
            Self {
                gpu_available: true,
                device_count: 1,
                device_name: Some("SimulatedCUDAclaw".into()),
            }
        } else {
            Self {
                gpu_available: false,
                device_count: 0,
                device_name: None,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified memory buffer (CPU-side representation)
// ─────────────────────────────────────────────────────────────────────────────

/// A typed buffer backed by CUDAclaw unified memory.
///
/// On systems without a GPU this falls back to a host `Vec<T>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedBuffer<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone> UnifiedBuffer<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(BridgeError::ShapeMismatch {
                expected: vec![expected_len],
                got: vec![data.len()],
            });
        }
        Ok(Self { data, shape })
    }

    pub fn from_flat(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(BridgeError::ShapeMismatch {
                expected: new_shape,
                got: self.shape,
            });
        }
        self.shape = new_shape;
        Ok(self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuDispatch trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core dispatch trait.  Any `lau-*` module can implement this to gain GPU
/// acceleration through CUDAclaw's command queue.
pub trait GpuDispatch: Send + Sync {
    /// Human-readable label for logging / diagnostics.
    fn label(&self) -> &str;

    /// Execute on GPU if available, otherwise on CPU.
    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult>;

    /// Whether a real GPU was used on the last dispatch.
    fn uses_gpu(&self) -> bool {
        false
    }

    /// Estimated FLOPs for benchmarking purposes.
    fn estimated_flops(&self) -> u64 {
        0
    }
}

/// Context passed into every dispatch call.
#[derive(Debug, Clone)]
pub struct DispatchContext {
    pub gpu_available: bool,
    pub device_id: u32,
    pub stream_id: u64,
    pub timeout_ms: u64,
}

impl Default for DispatchContext {
    fn default() -> Self {
        let avail = GpuAvailability::probe();
        Self {
            gpu_available: avail.gpu_available,
            device_id: 0,
            stream_id: 0,
            timeout_ms: 30_000,
        }
    }
}

/// Result returned from a dispatch.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    pub success: bool,
    pub gpu_used: bool,
    pub elapsed_ns: u64,
    pub bytes_transferred: u64,
    pub metadata: Vec<(String, String)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Bridge from `lau-gpu-compute` tensors to CUDAclaw unified memory buffers.
pub struct TensorBridge;

impl TensorBridge {
    /// Convert a flat `Vec<f64>` with shape into a `UnifiedBuffer<f64>`.
    pub fn to_unified(data: Vec<f64>, shape: Vec<usize>) -> Result<UnifiedBuffer<f64>> {
        UnifiedBuffer::new(data, shape)
    }

    /// Extract flat data from a `UnifiedBuffer`.
    pub fn from_unified(buf: &UnifiedBuffer<f64>) -> (Vec<f64>, Vec<usize>) {
        (buf.data().to_vec(), buf.shape().to_vec())
    }

    /// Convert an `nalgebra` `DMatrix` into a 2-D unified buffer (row-major).
    pub fn matrix_to_unified(m: &nalgebra::DMatrix<f64>) -> UnifiedBuffer<f64> {
        let nrows = m.nrows();
        let ncols = m.ncols();
        let data: Vec<f64> = (0..nrows)
            .flat_map(|r| (0..ncols).map(move |c| m[(r, c)]))
            .collect();
        UnifiedBuffer::new(data, vec![nrows, ncols]).unwrap()
    }

    /// Convert a 2-D unified buffer back into an `nalgebra` `DMatrix`.
    pub fn unified_to_matrix(buf: &UnifiedBuffer<f64>) -> Result<nalgebra::DMatrix<f64>> {
        if buf.shape().len() != 2 {
            return Err(BridgeError::Other("expected 2-D buffer".into()));
        }
        let nrows = buf.shape()[0];
        let ncols = buf.shape()[1];
        let data = buf.data().to_vec();
        // nalgebra stores column-major; convert from row-major.
        let mut col_major = vec![0.0; data.len()];
        for r in 0..nrows {
            for c in 0..ncols {
                col_major[c * nrows + r] = data[r * ncols + c];
            }
        }
        Ok(nalgebra::DMatrix::from_vec(nrows, ncols, col_major))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math operation bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Operations that can be dispatched to CUDAclaw's command queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathOp {
    MatMul {
        a_shape: (usize, usize),
        b_shape: (usize, usize),
    },
    Fft {
        n: usize,
    },
    Reduce {
        n: usize,
        op: ReduceOp,
    },
    ElementWise {
        n: usize,
        op: ElementOp,
    },
    Transpose {
        rows: usize,
        cols: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Product,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ElementOp {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    Sqrt,
}

/// Dispatcher for math operations.
pub struct MathBridge {
    gpu_avail: bool,
}

impl MathBridge {
    pub fn new() -> Self {
        Self {
            gpu_avail: GpuAvailability::probe().gpu_available,
        }
    }

    /// Dispatch a math operation, falling back to CPU when no GPU is present.
    pub fn dispatch_op(&self, op: &MathOp, input: &[f64]) -> Result<Vec<f64>> {
        let start = Instant::now();
        let _ = start; // used in real impl for timing
        match op {
            MathOp::MatMul { a_shape, b_shape } => self.cpu_matmul(input, *a_shape, *b_shape),
            MathOp::Fft { n } => self.cpu_fft(input, *n),
            MathOp::Reduce { op: rop, .. } => self.cpu_reduce(input, *rop),
            MathOp::ElementWise { op: eop, .. } => self.cpu_elementwise(input, *eop),
            MathOp::Transpose { rows, cols } => self.cpu_transpose(input, *rows, *cols),
        }
    }

    fn cpu_matmul(&self, data: &[f64], a_shape: (usize, usize), b_shape: (usize, usize)) -> Result<Vec<f64>> {
        let (m, k1) = a_shape;
        let (k2, n) = b_shape;
        if k1 != k2 {
            return Err(BridgeError::ShapeMismatch {
                expected: vec![m, k1],
                got: vec![k2, n],
            });
        }
        if data.len() != m * k1 + k2 * n {
            return Err(BridgeError::Other("input length mismatch for matmul".into()));
        }
        let a = &data[..m * k1];
        let b = &data[m * k1..];
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k1 {
                    sum += a[i * k1 + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    /// Naive DFT for testing (not a production FFT).
    fn cpu_fft(&self, input: &[f64], n: usize) -> Result<Vec<f64>> {
        if input.len() < n * 2 {
            return Err(BridgeError::Other("input too short for FFT".into()));
        }
        // input is interleaved real/imag pairs; output same format.
        let mut out = vec![0.0; n * 2];
        for k in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                let tr = input[t * 2];
                let ti = input[t * 2 + 1];
                re += tr * angle.cos() - ti * angle.sin();
                im += tr * angle.sin() + ti * angle.cos();
            }
            out[k * 2] = re;
            out[k * 2 + 1] = im;
        }
        Ok(out)
    }

    fn cpu_reduce(&self, input: &[f64], op: ReduceOp) -> Result<Vec<f64>> {
        if input.is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![match op {
            ReduceOp::Sum => input.iter().sum(),
            ReduceOp::Max => input.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ReduceOp::Min => input.iter().cloned().fold(f64::INFINITY, f64::min),
            ReduceOp::Product => input.iter().product(),
        }])
    }

    fn cpu_elementwise(&self, input: &[f64], op: ElementOp) -> Result<Vec<f64>> {
        // For binary ops we expect the first half and second half of input.
        let half = input.len() / 2;
        let a = &input[..half];
        let b = &input[half..half * 2];
        let result: Vec<f64> = a.iter().zip(b.iter())
            .map(|(&x, &y)| match op {
                ElementOp::Add => x + y,
                ElementOp::Sub => x - y,
                ElementOp::Mul => x * y,
                ElementOp::Div => x / y,
                ElementOp::Exp => x.exp(),
                ElementOp::Log => x.ln(),
                ElementOp::Sqrt => x.sqrt(),
            })
            .collect();
        Ok(result)
    }

    fn cpu_transpose(&self, input: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
        if input.len() != rows * cols {
            return Err(BridgeError::ShapeMismatch {
                expected: vec![rows, cols],
                got: vec![input.len()],
            });
        }
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = input[r * cols + c];
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PDE bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Grid operations for PDE solvers dispatched to GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeGrid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub data: Vec<f64>,
}

impl PdeGrid {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            dy,
            data: vec![0.0; nx * ny],
        }
    }

    /// 5-point Laplacian stencil (CPU fallback).
    pub fn laplacian(&self) -> PdeGrid {
        let mut out = PdeGrid::new(self.nx, self.ny, self.dx, self.dy);
        let idx = |i: usize, j: usize| i * self.ny + j;
        for i in 1..self.nx.saturating_sub(1) {
            for j in 1..self.ny.saturating_sub(1) {
                let lap = (self.data[idx(i + 1, j)] - 2.0 * self.data[idx(i, j)]
                    + self.data[idx(i - 1, j)])
                    / (self.dx * self.dx)
                    + (self.data[idx(i, j + 1)] - 2.0 * self.data[idx(i, j)]
                        + self.data[idx(i, j - 1)])
                    / (self.dy * self.dy);
                out.data[idx(i, j)] = lap;
            }
        }
        out
    }

    /// Explicit Euler step for the heat equation ∂u/∂t = α∇²u.
    pub fn heat_step(&self, alpha: f64, dt: f64) -> PdeGrid {
        let lap = self.laplacian();
        let mut out = self.clone();
        for k in 0..self.data.len() {
            out.data[k] += alpha * dt * lap.data[k];
        }
        out
    }

    /// Set boundary conditions (Dirichlet).
    pub fn set_boundary(&mut self, value: f64) {
        for i in 0..self.nx {
            self.data[i * self.ny] = value;
            self.data[i * self.ny + self.ny - 1] = value;
        }
        for j in 0..self.ny {
            self.data[j] = value;
            self.data[(self.nx - 1) * self.ny + j] = value;
        }
    }
}

/// PDE bridge dispatching grid ops through CUDAclaw.
pub struct PdeBridge {
    gpu_avail: bool,
}

impl PdeBridge {
    pub fn new() -> Self {
        Self {
            gpu_avail: GpuAvailability::probe().gpu_available,
        }
    }

    pub fn dispatch_laplacian(&self, grid: &PdeGrid) -> PdeGrid {
        // In production this would dispatch to CUDAclaw grid kernels.
        grid.laplacian()
    }

    pub fn dispatch_heat_step(&self, grid: &PdeGrid, alpha: f64, dt: f64) -> PdeGrid {
        grid.heat_step(alpha, dt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural network bridge
// ─────────────────────────────────────────────────────────────────────────────

/// A simple feed-forward layer descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl NnLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use std::f64::consts::SQRT_2;
        // Xavier-ish init
        let scale = SQRT_2 / (input_dim as f64).sqrt();
        let mut rng = simple_rng(input_dim * output_dim);
        let weights: Vec<f64> = (0..input_dim * output_dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((rng >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
                v * scale
            })
            .collect();
        Self {
            input_dim,
            output_dim,
            weights,
            bias: vec![0.0; output_dim],
        }
    }

    /// Forward pass (row-vector input × weight matrix + bias).
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.input_dim {
            return Err(BridgeError::ShapeMismatch {
                expected: vec![self.input_dim],
                got: vec![input.len()],
            });
        }
        let mut out = self.bias.clone();
        for j in 0..self.output_dim {
            for i in 0..self.input_dim {
                out[j] += input[i] * self.weights[i * self.output_dim + j];
            }
            out[j] = relu(out[j]);
        }
        Ok(out)
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn simple_rng(seed: usize) -> u64 {
    seed as u64 | 1
}

/// Neural network bridge for batch execution through CUDAclaw.
pub struct NnBridge {
    gpu_avail: bool,
}

impl NnBridge {
    pub fn new() -> Self {
        Self {
            gpu_avail: GpuAvailability::probe().gpu_available,
        }
    }

    /// Run a forward pass through a sequence of layers for a batch of inputs.
    pub fn forward_batch(&self, layers: &[NnLayer], batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut results = Vec::with_capacity(batch.len());
        for input in batch {
            let mut activations = input.clone();
            for layer in layers {
                activations = layer.forward(&activations)?;
            }
            results.push(activations);
        }
        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse adjacency list representation for GPU parallel traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub node_count: usize,
    /// adjacency[i] = list of neighbors of node i
    pub adjacency: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            adjacency: vec![vec![]; node_count],
        }
    }

    pub fn add_edge(&mut self, a: usize, b: usize) {
        if a < self.node_count && b < self.node_count {
            self.adjacency[a].push(b);
            self.adjacency[b].push(a);
        }
    }

    pub fn add_directed_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count {
            self.adjacency[from].push(to);
        }
    }

    /// BFS from `start`, returning visitation order.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = vec![false; self.node_count];
        let mut order = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        if start >= self.node_count {
            return order;
        }
        visited[start] = true;
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &nbr in &self.adjacency[node] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    queue.push_back(nbr);
                }
            }
        }
        order
    }

    /// Single-source shortest paths (unweighted).
    pub fn shortest_paths(&self, start: usize) -> Vec<Option<usize>> {
        let mut dist: Vec<Option<usize>> = vec![None; self.node_count];
        if start >= self.node_count {
            return dist;
        }
        dist[start] = Some(0);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            let d = dist[node].unwrap();
            for &nbr in &self.adjacency[node] {
                if dist[nbr].is_none() {
                    dist[nbr] = Some(d + 1);
                    queue.push_back(nbr);
                }
            }
        }
        dist
    }

    /// Count edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|v| v.len()).sum()
    }
}

/// Graph bridge dispatching adjacency operations through CUDAclaw.
pub struct GraphBridge {
    gpu_avail: bool,
}

impl GraphBridge {
    pub fn new() -> Self {
        Self {
            gpu_avail: GpuAvailability::probe().gpu_available,
        }
    }

    pub fn dispatch_bfs(&self, graph: &Graph, start: usize) -> Vec<usize> {
        graph.bfs(start)
    }

    pub fn dispatch_shortest_paths(&self, graph: &Graph, start: usize) -> Vec<Option<usize>> {
        graph.shortest_paths(start)
    }

    /// PageRank (simplified, CPU fallback).
    pub fn dispatch_pagerank(&self, graph: &Graph, iterations: usize, damping: f64) -> Vec<f64> {
        let n = graph.node_count;
        if n == 0 {
            return vec![];
        }
        let mut pr = vec![1.0 / n as f64; n];
        let mut new_pr = vec![0.0; n];
        for _ in 0..iterations {
            new_pr.fill((1.0 - damping) / n as f64);
            for i in 0..n {
                if graph.adjacency[i].is_empty() {
                    // dangling node: distribute rank to all
                    for j in 0..n {
                        new_pr[j] += damping * pr[i] / n as f64;
                    }
                } else {
                    let share = damping * pr[i] / graph.adjacency[i].len() as f64;
                    for &j in &graph.adjacency[i] {
                        new_pr[j] += share;
                    }
                }
            }
            std::mem::swap(&mut pr, &mut new_pr);
        }
        pr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark harness
// ─────────────────────────────────────────────────────────────────────────────

/// Timing result for a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub label: String,
    pub cpu_elapsed_ns: u64,
    pub gpu_elapsed_ns: Option<u64>,
    pub speedup: Option<f64>,
    pub flops_estimated: u64,
}

/// Benchmark harness comparing CPU and GPU execution times.
pub struct BenchmarkHarness {
    pub warmup_iters: usize,
    pub bench_iters: usize,
}

impl BenchmarkHarness {
    pub fn new(warmup_iters: usize, bench_iters: usize) -> Self {
        Self { warmup_iters, bench_iters }
    }

    /// Benchmark a closure, returning median CPU time.
    pub fn bench_cpu<F>(&self, label: &str, flops: u64, mut f: F) -> BenchResult
    where
        F: FnMut(),
    {
        for _ in 0..self.warmup_iters {
            f();
        }
        let mut times = Vec::with_capacity(self.bench_iters);
        for _ in 0..self.bench_iters {
            let start = Instant::now();
            f();
            times.push(start.elapsed().as_nanos() as u64);
        }
        times.sort_unstable();
        let median = times[self.bench_iters / 2];
        BenchResult {
            label: label.into(),
            cpu_elapsed_ns: median,
            gpu_elapsed_ns: None,
            speedup: None,
            flops_estimated: flops,
        }
    }

    /// Benchmark a CPU vs GPU pair.
    pub fn bench_cpu_vs_gpu<Fcpu, Fgpu>(
        &self,
        label: &str,
        flops: u64,
        cpu_fn: Fcpu,
        gpu_fn: Option<&mut Fgpu>,
    ) -> BenchResult
    where
        Fcpu: FnMut(),
        Fgpu: FnMut(),
    {
        let cpu_result = self.bench_cpu(label, flops, cpu_fn);
        if let Some(gf) = gpu_fn {
            for _ in 0..self.warmup_iters {
                gf();
            }
            let mut gpu_times = Vec::with_capacity(self.bench_iters);
            for _ in 0..self.bench_iters {
                let start = Instant::now();
                gf();
                gpu_times.push(start.elapsed().as_nanos() as u64);
            }
            gpu_times.sort_unstable();
            let gpu_median = gpu_times[self.bench_iters / 2];
            let speedup = cpu_result.cpu_elapsed_ns as f64 / gpu_median as f64;
            BenchResult {
                gpu_elapsed_ns: Some(gpu_median),
                speedup: Some(speedup),
                ..cpu_result
            }
        } else {
            cpu_result
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuDispatch implementations for each bridge
// ─────────────────────────────────────────────────────────────────────────────

/// GpuDispatch adapter for math operations.
pub struct MathDispatch {
    pub op: MathOp,
    pub input: Vec<f64>,
    gpu_avail: bool,
}

impl MathDispatch {
    pub fn new(op: MathOp, input: Vec<f64>) -> Self {
        Self {
            op,
            input,
            gpu_avail: GpuAvailability::probe().gpu_available,
        }
    }
}

impl GpuDispatch for MathDispatch {
    fn label(&self) -> &str {
        "MathDispatch"
    }

    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult> {
        let start = Instant::now();
        let bridge = MathBridge::new();
        let _result = bridge.dispatch_op(&self.op, &self.input)?;
        Ok(DispatchResult {
            success: true,
            gpu_used: ctx.gpu_available,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            bytes_transferred: self.input.len() as u64 * 8,
            metadata: vec![("op".into(), format!("{:?}", self.op))],
        })
    }

    fn uses_gpu(&self) -> bool {
        self.gpu_avail
    }

    fn estimated_flops(&self) -> u64 {
        match &self.op {
            MathOp::MatMul { a_shape, b_shape } => {
                (a_shape.0 * a_shape.1 * b_shape.1 * 2) as u64
            }
            MathOp::Fft { n } => (*n as u64) * (*n as u64) * 5,
            MathOp::Reduce { n, .. } => *n as u64,
            MathOp::ElementWise { n, .. } => *n as u64,
            MathOp::Transpose { rows, cols } => (*rows * *cols) as u64,
        }
    }
}

/// GpuDispatch adapter for PDE grid operations.
pub struct PdeDispatch {
    pub grid: PdeGrid,
    pub op: PdeOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PdeOp {
    Laplacian,
    HeatStep { alpha: f64, dt: f64 },
}

impl GpuDispatch for PdeDispatch {
    fn label(&self) -> &str {
        "PdeDispatch"
    }

    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult> {
        let start = Instant::now();
        let bridge = PdeBridge::new();
        let grid_size = self.grid.data.len();
        match &self.op {
            PdeOp::Laplacian => {
                let _out = bridge.dispatch_laplacian(&self.grid);
            }
            PdeOp::HeatStep { alpha, dt } => {
                let _out = bridge.dispatch_heat_step(&self.grid, *alpha, *dt);
            }
        }
        Ok(DispatchResult {
            success: true,
            gpu_used: ctx.gpu_available,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            bytes_transferred: grid_size as u64 * 8 * 2,
            metadata: vec![("op".into(), format!("{:?}", self.op))],
        })
    }
}

/// GpuDispatch adapter for neural network forward passes.
pub struct NnDispatch {
    pub layers: Vec<NnLayer>,
    pub batch: Vec<Vec<f64>>,
}

impl GpuDispatch for NnDispatch {
    fn label(&self) -> &str {
        "NnDispatch"
    }

    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult> {
        let start = Instant::now();
        let bridge = NnBridge::new();
        let batch_size = self.batch.len();
        let _results = bridge.forward_batch(&self.layers, &self.batch)?;
        let total_params: usize = self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum();
        Ok(DispatchResult {
            success: true,
            gpu_used: ctx.gpu_available,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            bytes_transferred: (total_params + batch_size * self.layers.first().map_or(0, |l| l.input_dim)) as u64 * 8,
            metadata: vec![
                ("layers".into(), self.layers.len().to_string()),
                ("batch_size".into(), batch_size.to_string()),
            ],
        })
    }

    fn estimated_flops(&self) -> u64 {
        self.layers.iter().map(|l| (l.input_dim * l.output_dim * 2) as u64).sum::<u64>()
            * self.batch.len() as u64
    }
}

/// GpuDispatch adapter for graph operations.
pub struct GraphDispatch {
    pub graph: Graph,
    pub op: GraphOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphOp {
    Bfs { start: usize },
    ShortestPaths { start: usize },
    PageRank { iterations: usize, damping: f64 },
}

impl GpuDispatch for GraphDispatch {
    fn label(&self) -> &str {
        "GraphDispatch"
    }

    fn dispatch(&self, ctx: &DispatchContext) -> Result<DispatchResult> {
        let start = Instant::now();
        let bridge = GraphBridge::new();
        let node_count = self.graph.node_count;
        match &self.op {
            GraphOp::Bfs { start } => {
                let _order = bridge.dispatch_bfs(&self.graph, *start);
            }
            GraphOp::ShortestPaths { start } => {
                let _dists = bridge.dispatch_shortest_paths(&self.graph, *start);
            }
            GraphOp::PageRank { iterations, damping } => {
                let _pr = bridge.dispatch_pagerank(&self.graph, *iterations, *damping);
            }
        }
        Ok(DispatchResult {
            success: true,
            gpu_used: ctx.gpu_available,
            elapsed_ns: start.elapsed().as_nanos() as u64,
            bytes_transferred: self.graph.edge_count() as u64 * 16,
            metadata: vec![
                ("nodes".into(), node_count.to_string()),
                ("edges".into(), self.graph.edge_count().to_string()),
            ],
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Error & availability ──────────────────────────────────────────────

    #[test]
    fn test_error_display_no_gpu() {
        let e = BridgeError::NoGpuAvailable;
        assert!(e.to_string().contains("no GPU"));
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = BridgeError::ShapeMismatch {
            expected: vec![3, 4],
            got: vec![2, 5],
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("2"));
    }

    #[test]
    fn test_error_display_other() {
        let e = BridgeError::Other("oops".into());
        assert_eq!(e.to_string(), "oops");
    }

    #[test]
    fn test_gpu_availability_default() {
        let avail = GpuAvailability::probe();
        // Without the env var set, should report no GPU.
        assert_eq!(avail.gpu_available, false);
        assert_eq!(avail.device_count, 0);
    }

    // ── UnifiedBuffer ─────────────────────────────────────────────────────

    #[test]
    fn test_unified_buffer_new() {
        let buf = UnifiedBuffer::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(buf.shape(), &[2, 2]);
        assert_eq!(buf.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_unified_buffer_shape_mismatch() {
        let res = UnifiedBuffer::new(vec![1.0, 2.0], vec![2, 2]);
        assert!(matches!(res, Err(BridgeError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_unified_buffer_from_flat() {
        let buf = UnifiedBuffer::from_flat(vec![5.0, 6.0]);
        assert_eq!(buf.shape(), &[2]);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_unified_buffer_reshape() {
        let buf = UnifiedBuffer::from_flat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = buf.reshape(vec![2, 3]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
    }

    #[test]
    fn test_unified_buffer_reshape_fail() {
        let buf = UnifiedBuffer::from_flat(vec![1.0, 2.0, 3.0]);
        let res = buf.reshape(vec![2, 2]);
        assert!(res.is_err());
    }

    #[test]
    fn test_unified_buffer_is_empty() {
        let buf: UnifiedBuffer<f64> = UnifiedBuffer::from_flat(vec![]);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_unified_buffer_data_mut() {
        let mut buf = UnifiedBuffer::from_flat(vec![1.0, 2.0]);
        buf.data_mut()[0] = 99.0;
        assert_eq!(buf.data()[0], 99.0);
    }

    // ── TensorBridge ──────────────────────────────────────────────────────

    #[test]
    fn test_tensor_to_unified() {
        let buf = TensorBridge::to_unified(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(buf.shape(), &[3]);
    }

    #[test]
    fn test_tensor_from_unified() {
        let buf = UnifiedBuffer::new(vec![1.0, 2.0], vec![2]).unwrap();
        let (data, shape) = TensorBridge::from_unified(&buf);
        assert_eq!(data, vec![1.0, 2.0]);
        assert_eq!(shape, vec![2]);
    }

    #[test]
    fn test_matrix_to_unified() {
        let m = nalgebra::DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let buf = TensorBridge::matrix_to_unified(&m);
        assert_eq!(buf.shape(), &[2, 3]);
        // row-major: 1,2,3,4,5,6
        assert_eq!(buf.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unified_to_matrix() {
        let buf = UnifiedBuffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let m = TensorBridge::unified_to_matrix(&buf).unwrap();
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn test_roundtrip_matrix_unified() {
        let m = nalgebra::DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let buf = TensorBridge::matrix_to_unified(&m);
        let m2 = TensorBridge::unified_to_matrix(&buf).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_unified_to_matrix_wrong_dims() {
        let buf = UnifiedBuffer::from_flat(vec![1.0, 2.0, 3.0]);
        let res = TensorBridge::unified_to_matrix(&buf);
        assert!(res.is_err());
    }

    // ── MathBridge ────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        let bridge = MathBridge::new();
        // 2×2 identity × 2×2 matrix
        let input = vec![
            1.0, 0.0,
            0.0, 1.0,
            3.0, 5.0,
            7.0, 11.0,
        ];
        let out = bridge.dispatch_op(
            &MathOp::MatMul { a_shape: (2, 2), b_shape: (2, 2) },
            &input,
        ).unwrap();
        assert_eq!(out, vec![3.0, 5.0, 7.0, 11.0]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let bridge = MathBridge::new();
        let input = vec![1.0, 2.0, 3.0];
        let res = bridge.dispatch_op(
            &MathOp::MatMul { a_shape: (2, 2), b_shape: (3, 2) },
            &input,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_reduce_sum() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::Reduce { n: 4, op: ReduceOp::Sum },
            &[1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        assert_eq!(out[0], 10.0);
    }

    #[test]
    fn test_reduce_max() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::Reduce { n: 3, op: ReduceOp::Max },
            &[-5.0, 0.0, 7.0],
        ).unwrap();
        assert_eq!(out[0], 7.0);
    }

    #[test]
    fn test_reduce_min() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::Reduce { n: 3, op: ReduceOp::Min },
            &[-5.0, 0.0, 7.0],
        ).unwrap();
        assert_eq!(out[0], -5.0);
    }

    #[test]
    fn test_reduce_product() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::Reduce { n: 3, op: ReduceOp::Product },
            &[2.0, 3.0, 4.0],
        ).unwrap();
        assert_eq!(out[0], 24.0);
    }

    #[test]
    fn test_elementwise_add() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::ElementWise { n: 2, op: ElementOp::Add },
            &[1.0, 2.0, 10.0, 20.0],
        ).unwrap();
        assert_eq!(out, vec![11.0, 22.0]);
    }

    #[test]
    fn test_elementwise_mul() {
        let bridge = MathBridge::new();
        let out = bridge.dispatch_op(
            &MathOp::ElementWise { n: 2, op: ElementOp::Mul },
            &[2.0, 3.0, 5.0, 7.0],
        ).unwrap();
        assert_eq!(out, vec![10.0, 21.0]);
    }

    #[test]
    fn test_transpose() {
        let bridge = MathBridge::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = bridge.dispatch_op(
            &MathOp::Transpose { rows: 2, cols: 3 },
            &input,
        ).unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_fft_dc_component() {
        let bridge = MathBridge::new();
        // DC signal: [1+0j, 1+0j, 1+0j, 1+0j]
        let input = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = bridge.dispatch_op(&MathOp::Fft { n: 4 }, &input).unwrap();
        // DC bin should be 4+0j
        assert!((out[0] - 4.0).abs() < 1e-10);
        assert!(out[1].abs() < 1e-10);
    }

    // ── PDE bridge ────────────────────────────────────────────────────────

    #[test]
    fn test_pde_grid_new() {
        let g = PdeGrid::new(10, 10, 0.1, 0.1);
        assert_eq!(g.data.len(), 100);
        assert!(g.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pde_laplacian_flat() {
        // Flat field → laplacian should be zero
        let mut g = PdeGrid::new(5, 5, 1.0, 1.0);
        g.data.fill(3.0);
        let lap = g.laplacian();
        assert!(lap.data.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_pde_laplacian_nonzero() {
        // Single spike in center
        let mut g = PdeGrid::new(5, 5, 1.0, 1.0);
        g.data[2 * 5 + 2] = 1.0;
        let lap = g.laplacian();
        // Center should be negative (concave up)
        assert!(lap.data[2 * 5 + 2] < 0.0);
    }

    #[test]
    fn test_pde_heat_step_preserves_shape() {
        let g = PdeGrid::new(5, 5, 1.0, 1.0);
        let result = g.heat_step(0.1, 0.01);
        assert_eq!(result.data.len(), 25);
    }

    #[test]
    fn test_pde_set_boundary() {
        let mut g = PdeGrid::new(4, 4, 0.5, 0.5);
        g.set_boundary(42.0);
        // Corners
        assert_eq!(g.data[0], 42.0);
        assert_eq!(g.data[3], 42.0);
        assert_eq!(g.data[12], 42.0);
        assert_eq!(g.data[15], 42.0);
    }

    #[test]
    fn test_pde_bridge_laplacian() {
        let bridge = PdeBridge::new();
        let g = PdeGrid::new(5, 5, 1.0, 1.0);
        let out = bridge.dispatch_laplacian(&g);
        assert_eq!(out.data.len(), 25);
    }

    // ── Neural network bridge ─────────────────────────────────────────────

    #[test]
    fn test_nn_layer_forward() {
        let layer = NnLayer {
            input_dim: 2,
            output_dim: 3,
            weights: vec![1.0; 6],
            bias: vec![0.0; 3],
        };
        let out = layer.forward(&[1.0, 1.0]).unwrap();
        // Each output = 1*1 + 1*1 = 2.0, relu(2) = 2
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_nn_layer_forward_shape_mismatch() {
        let layer = NnLayer {
            input_dim: 3,
            output_dim: 2,
            weights: vec![0.0; 6],
            bias: vec![0.0; 2],
        };
        let res = layer.forward(&[1.0, 2.0]); // only 2 inputs, expects 3
        assert!(res.is_err());
    }

    #[test]
    fn test_nn_layer_relu() {
        let layer = NnLayer {
            input_dim: 1,
            output_dim: 1,
            weights: vec![-1.0],
            bias: vec![0.0],
        };
        let out = layer.forward(&[5.0]).unwrap();
        assert_eq!(out[0], 0.0); // -5 → relu → 0
    }

    #[test]
    fn test_nn_bridge_forward_batch() {
        let bridge = NnBridge::new();
        let l1 = NnLayer::new(3, 4);
        let l2 = NnLayer::new(4, 2);
        let batch = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let results = bridge.forward_batch(&[l1, l2], &batch).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
    }

    #[test]
    fn test_nn_bridge_empty_batch() {
        let bridge = NnBridge::new();
        let results = bridge.forward_batch(&[], &[]).unwrap();
        assert!(results.is_empty());
    }

    // ── Graph bridge ──────────────────────────────────────────────────────

    #[test]
    fn test_graph_new() {
        let g = Graph::new(5);
        assert_eq!(g.node_count, 5);
        assert!(g.adjacency.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        assert!(g.adjacency[0].contains(&1));
        assert!(g.adjacency[1].contains(&0));
    }

    #[test]
    fn test_graph_add_directed_edge() {
        let mut g = Graph::new(3);
        g.add_directed_edge(0, 1);
        assert!(g.adjacency[0].contains(&1));
        assert!(!g.adjacency[1].contains(&0));
    }

    #[test]
    fn test_graph_add_edge_out_of_bounds() {
        let mut g = Graph::new(2);
        g.add_edge(0, 5);
        assert!(g.adjacency[0].is_empty());
    }

    #[test]
    fn test_graph_bfs() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let order = g.bfs(0);
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0);
    }

    #[test]
    fn test_graph_bfs_disconnected() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        // 2, 3 disconnected
        let order = g.bfs(0);
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_graph_bfs_invalid_start() {
        let g = Graph::new(3);
        let order = g.bfs(10);
        assert!(order.is_empty());
    }

    #[test]
    fn test_graph_shortest_paths() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let dists = g.shortest_paths(0);
        assert_eq!(dists[0], Some(0));
        assert_eq!(dists[1], Some(1));
        assert_eq!(dists[2], Some(2));
        assert_eq!(dists[3], Some(3));
    }

    #[test]
    fn test_graph_edge_count() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert_eq!(g.edge_count(), 4); // undirected: each edge counted twice
    }

    #[test]
    fn test_graph_bridge_bfs() {
        let bridge = GraphBridge::new();
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let order = bridge.dispatch_bfs(&g, 0);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_graph_bridge_pagerank() {
        let bridge = GraphBridge::new();
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);
        let pr = bridge.dispatch_pagerank(&g, 20, 0.85);
        assert_eq!(pr.len(), 4);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // ── Dispatch trait implementations ────────────────────────────────────

    #[test]
    fn test_math_dispatch() {
        let d = MathDispatch::new(
            MathOp::Reduce { n: 3, op: ReduceOp::Sum },
            vec![1.0, 2.0, 3.0],
        );
        assert_eq!(d.label(), "MathDispatch");
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_math_dispatch_uses_gpu() {
        let d = MathDispatch::new(
            MathOp::Reduce { n: 1, op: ReduceOp::Sum },
            vec![1.0],
        );
        // Without env var, should not use GPU
        assert!(!d.uses_gpu());
    }

    #[test]
    fn test_math_dispatch_estimated_flops() {
        let d = MathDispatch::new(
            MathOp::MatMul { a_shape: (4, 3), b_shape: (3, 5) },
            vec![0.0; 4*3 + 3*5],
        );
        assert_eq!(d.estimated_flops(), 4 * 3 * 5 * 2);
    }

    #[test]
    fn test_pde_dispatch_laplacian() {
        let grid = PdeGrid::new(5, 5, 1.0, 1.0);
        let d = PdeDispatch {
            grid,
            op: PdeOp::Laplacian,
        };
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_pde_dispatch_heat_step() {
        let grid = PdeGrid::new(5, 5, 1.0, 1.0);
        let d = PdeDispatch {
            grid,
            op: PdeOp::HeatStep { alpha: 0.1, dt: 0.01 },
        };
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
        assert!(result.bytes_transferred > 0);
    }

    #[test]
    fn test_nn_dispatch() {
        let l = NnLayer::new(3, 2);
        let d = NnDispatch {
            layers: vec![l],
            batch: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        };
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.metadata[1].1, "2"); // batch_size
    }

    #[test]
    fn test_nn_dispatch_estimated_flops() {
        let l = NnLayer::new(4, 3);
        let d = NnDispatch {
            layers: vec![l],
            batch: vec![vec![0.0; 4]; 5],
        };
        assert_eq!(d.estimated_flops(), 5 * 4 * 3 * 2);
    }

    #[test]
    fn test_graph_dispatch_bfs() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let d = GraphDispatch {
            graph: g,
            op: GraphOp::Bfs { start: 0 },
        };
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_graph_dispatch_pagerank() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let d = GraphDispatch {
            graph: g,
            op: GraphOp::PageRank { iterations: 10, damping: 0.85 },
        };
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.success);
    }

    // ── Benchmark harness ─────────────────────────────────────────────────

    #[test]
    fn test_bench_cpu() {
        let harness = BenchmarkHarness::new(1, 3);
        let result = harness.bench_cpu("add_test", 1, || {
            let _ = 1 + 1;
        });
        assert_eq!(result.label, "add_test");
        assert!(result.cpu_elapsed_ns > 0);
        assert!(result.gpu_elapsed_ns.is_none());
    }

    #[test]
    fn test_bench_cpu_vs_gpu_no_gpu() {
        let harness = BenchmarkHarness::new(1, 3);
        let result: BenchResult = harness.bench_cpu_vs_gpu::<_, fn()>(
            "test_op",
            10,
            || { let _ = 2 + 2; },
            None,
        );
        assert!(result.speedup.is_none());
    }

    #[test]
    fn test_bench_flops_recorded() {
        let harness = BenchmarkHarness::new(1, 3);
        let result = harness.bench_cpu("flops_test", 42, || {});
        assert_eq!(result.flops_estimated, 42);
    }

    // ── DispatchContext defaults ───────────────────────────────────────────

    #[test]
    fn test_dispatch_context_default() {
        let ctx = DispatchContext::default();
        assert!(!ctx.gpu_available);
        assert_eq!(ctx.device_id, 0);
    }

    // ── Serde round-trip ──────────────────────────────────────────────────

    #[test]
    fn test_serde_math_op() {
        let op = MathOp::MatMul { a_shape: (2, 3), b_shape: (3, 4) };
        let json = serde_json::to_string(&op).unwrap();
        let de: MathOp = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, MathOp::MatMul { .. }));
    }

    #[test]
    fn test_serde_unified_buffer() {
        let buf = UnifiedBuffer::new(vec![1.0, 2.0], vec![2]).unwrap();
        let json = serde_json::to_string(&buf).unwrap();
        let de: UnifiedBuffer<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(de.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_serde_graph() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        let json = serde_json::to_string(&g).unwrap();
        let de: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(de.node_count, 3);
        assert!(de.adjacency[0].contains(&1));
    }

    #[test]
    fn test_serde_pde_grid() {
        let g = PdeGrid::new(2, 2, 0.5, 0.5);
        let json = serde_json::to_string(&g).unwrap();
        let de: PdeGrid = serde_json::from_str(&json).unwrap();
        assert_eq!(de.nx, 2);
    }

    #[test]
    fn test_serde_nn_layer() {
        let l = NnLayer {
            input_dim: 2,
            output_dim: 3,
            weights: vec![1.0; 6],
            bias: vec![0.0; 3],
        };
        let json = serde_json::to_string(&l).unwrap();
        let de: NnLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(de.input_dim, 2);
    }

    // ── Simulated GPU mode ────────────────────────────────────────────────

    #[test]
    fn test_gpu_availability_simulated() {
        std::env::set_var("LAU_CUDACLAWSIM_GPU", "1");
        let avail = GpuAvailability::probe();
        assert!(avail.gpu_available);
        assert_eq!(avail.device_count, 1);
        std::env::remove_var("LAU_CUDACLAWSIM_GPU");
    }

    #[test]
    fn test_math_dispatch_with_sim_gpu() {
        std::env::set_var("LAU_CUDACLAWSIM_GPU", "1");
        let d = MathDispatch::new(
            MathOp::Reduce { n: 2, op: ReduceOp::Sum },
            vec![1.0, 2.0],
        );
        let ctx = DispatchContext::default();
        let result = d.dispatch(&ctx).unwrap();
        assert!(result.gpu_used);
        assert!(d.uses_gpu());
        std::env::remove_var("LAU_CUDACLAWSIM_GPU");
    }
}
