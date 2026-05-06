# Tensors from Scratch

> Phase 2 — Optimization | Kata 2.5

---

## Concept & Intuition

### What problem are we solving?

A tensor is a multi-dimensional array — the fundamental data structure of modern machine learning. A scalar is a 0D tensor, a vector is a 1D tensor, a matrix is a 2D tensor, and higher-dimensional arrays are higher-order tensors. In frameworks like PyTorch, tensors are the primary way data flows through models, and operations on tensors are automatically differentiated for gradient computation.

Building a tensor type from scratch in Rust teaches us what tensors really are under the hood: a contiguous block of floating-point numbers plus shape metadata that tells us how to index into that flat array. A 2x3 matrix stored as [1,2,3,4,5,6] uses shape (2,3) and strides (3,1) to map the 2D index [i,j] to the flat index i*3+j. Understanding this layout is crucial for writing efficient numerical code.

By implementing basic tensor operations (creation, reshaping, element-wise ops, matrix multiplication, broadcasting), we gain deep insight into the computational foundations of deep learning. Every neural network forward pass is just a sequence of tensor operations.

### Why naive approaches fail

Using nested vectors (Vec<Vec<f64>>) for matrices is convenient but inefficient — data is scattered across the heap, causing cache misses. Real tensor libraries use a single contiguous allocation with stride-based indexing. This is not just an optimization; it changes what operations are possible (views, transposes without copying, broadcasting).

### Mental models

- **Tensor = flat data + shape**: A tensor is just a `Vec<f64>` plus a `Vec<usize>` for the shape. All the magic is in how we interpret the flat data using the shape.
- **Strides map indices to offsets**: For a shape [2,3,4], element [i,j,k] is at flat index i*12 + j*4 + k. The strides [12,4,1] encode this mapping.
- **Broadcasting stretches dimensions**: When two tensors have different shapes, broadcasting "virtually" repeats the smaller one to match the larger one, without copying data.

### Visual explanations

```
  Tensor storage:

  Shape: [2, 3]    Flat data: [1, 2, 3, 4, 5, 6]
  Strides: [3, 1]

  Logical view:       Index mapping:
  ┌───┬───┬───┐       [0,0]→0  [0,1]→1  [0,2]→2
  │ 1 │ 2 │ 3 │       [1,0]→3  [1,1]→4  [1,2]→5
  ├───┼───┼───┤
  │ 4 │ 5 │ 6 │       flat_idx = i * stride[0] + j * stride[1]
  └───┴───┴───┘                = i * 3 + j * 1
```

---

## Hands-on Exploration

1. Build a Tensor struct with shape, strides, and flat data storage.
2. Implement indexing, element-wise operations, and matrix multiplication.
3. Implement reshape and transpose operations using stride manipulation.

---

## Live Code

```rust
/// A simple tensor: contiguous data + shape + strides.
#[derive(Clone, Debug)]
struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from data and shape.
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Data length must match shape product");

        let strides = Self::compute_strides(&shape);
        Tensor { data, shape, strides }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Create a tensor filled with zeros.
    fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    /// Create a tensor filled with a value.
    fn full(shape: Vec<usize>, value: f64) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![value; size], shape)
    }

    /// Number of dimensions.
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get element at multi-dimensional index.
    fn get(&self, indices: &[usize]) -> f64 {
        let flat_idx: usize = indices.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum();
        self.data[flat_idx]
    }

    /// Set element at multi-dimensional index.
    fn set(&mut self, indices: &[usize], value: f64) {
        let flat_idx: usize = indices.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum();
        self.data[flat_idx] = value;
    }

    /// Element-wise addition.
    fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise multiplication (Hadamard product).
    fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for multiplication");
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Scalar multiplication.
    fn scale(&self, scalar: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Matrix multiplication (2D tensors only).
    fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = Tensor::zeros(vec![m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.get(&[i, p]) * other.get(&[p, j]);
                }
                result.set(&[i, j], sum);
            }
        }
        result
    }

    /// Transpose (2D tensors only).
    fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = Tensor::zeros(vec![n, m]);
        for i in 0..m {
            for j in 0..n {
                result.set(&[j, i], self.get(&[i, j]));
            }
        }
        result
    }

    /// Reshape (returns new tensor with same data, different shape).
    fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_size, "Total elements must match");
        Tensor::new(self.data.clone(), new_shape)
    }

    /// Sum all elements.
    fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Apply a function element-wise.
    fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Pretty print.
    fn print(&self, name: &str) {
        println!("  {} (shape={:?}):", name, self.shape);
        if self.ndim() == 1 {
            let vals: Vec<String> = self.data.iter().map(|x| format!("{:.2}", x)).collect();
            println!("    [{}]", vals.join(", "));
        } else if self.ndim() == 2 {
            for i in 0..self.shape[0] {
                let row: Vec<String> = (0..self.shape[1])
                    .map(|j| format!("{:>7.2}", self.get(&[i, j])))
                    .collect();
                println!("    [{}]", row.join(", "));
            }
        } else {
            println!("    {:?}", &self.data[..self.data.len().min(20)]);
        }
        println!();
    }
}

fn main() {
    println!("=== Tensors from Scratch ===\n");

    // === Create tensors ===
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    a.print("A (2x3 matrix)");

    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3]);
    b.print("B (2x3 matrix)");

    // === Element-wise operations ===
    println!("=== Element-wise Operations ===\n");

    let c = a.add(&b);
    c.print("A + B");

    let d = a.mul(&b);
    d.print("A * B (Hadamard product)");

    let e = a.scale(2.0);
    e.print("A * 2.0");

    // === Matrix multiplication ===
    println!("=== Matrix Multiplication ===\n");

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let y = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    x.print("X (2x3)");
    y.print("Y (3x2)");

    let z = x.matmul(&y);
    z.print("X @ Y (2x2)");

    // Verify: [1,2,3] . [7,9,11] = 7+18+33 = 58
    println!("  Verification: [1,2,3] . [7,9,11] = 1*7 + 2*9 + 3*11 = {}\n",
        1.0*7.0 + 2.0*9.0 + 3.0*11.0);

    // === Transpose ===
    println!("=== Transpose ===\n");
    let xt = x.transpose();
    xt.print("X^T (3x2)");

    // === Reshape ===
    println!("=== Reshape ===\n");
    let flat = Tensor::new((1..=12).map(|i| i as f64).collect(), vec![12]);
    flat.print("flat (12,)");

    let reshaped = flat.reshape(vec![3, 4]);
    reshaped.print("reshaped (3x4)");

    let reshaped2 = flat.reshape(vec![2, 2, 3]);
    println!("  reshaped to (2,2,3): shape={:?}, data={:?}\n", reshaped2.shape, reshaped2.data);

    // === Element-wise functions ===
    println!("=== Element-wise Functions ===\n");

    let v = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    v.print("v");

    let relu = v.map(|x| if x > 0.0 { x } else { 0.0 });
    relu.print("ReLU(v)");

    let sigmoid = v.map(|x| 1.0 / (1.0 + (-x).exp()));
    sigmoid.print("sigmoid(v)");

    let squared = v.map(|x| x * x);
    squared.print("v²");

    // === Practical example: linear layer forward pass ===
    println!("=== Practical: Linear Layer Forward Pass ===\n");
    println!("  y = X @ W + b  (batch of 3 inputs, 4 features → 2 outputs)\n");

    let inputs = Tensor::new(vec![
        1.0, 0.5, -1.0, 2.0,
        -0.5, 1.0, 0.0, -1.0,
        0.0, 0.0, 1.0, 1.0,
    ], vec![3, 4]);

    let weights = Tensor::new(vec![
        0.1, -0.2,
        0.3, 0.4,
        -0.5, 0.1,
        0.2, -0.3,
    ], vec![4, 2]);

    let bias = Tensor::new(vec![0.1, -0.1], vec![1, 2]);

    inputs.print("Input X (3x4)");
    weights.print("Weights W (4x2)");
    bias.print("Bias b (1x2)");

    let output = inputs.matmul(&weights);
    // Manual broadcast add (add bias to each row)
    let mut final_output = output.clone();
    for i in 0..final_output.shape[0] {
        for j in 0..final_output.shape[1] {
            let val = final_output.get(&[i, j]) + bias.get(&[0, j]);
            final_output.set(&[i, j], val);
        }
    }
    final_output.print("Output y = X@W + b (3x2)");

    println!("  This is exactly what a linear (fully-connected) neural network layer does.");
    println!("  Every forward pass is just tensor operations: matmul, add, activation.\n");

    // === Storage info ===
    println!("=== Tensor Internals ===\n");
    println!("  Tensor A:");
    println!("    shape:   {:?}", a.shape);
    println!("    strides: {:?}", a.strides);
    println!("    ndim:    {}", a.ndim());
    println!("    numel:   {}", a.numel());
    println!("    data:    {:?}", a.data);

    println!();
    println!("Key insight: Tensors are flat arrays with shape metadata.");
    println!("All of deep learning is tensor operations: matmul, add, activation functions.");
}
```

---

## Key Takeaways

- A tensor is a multi-dimensional array stored as contiguous flat data plus shape and stride metadata.
- All neural network computations reduce to tensor operations: matrix multiplication, element-wise addition, and activation functions.
- Understanding tensor storage (strides, reshaping, transposing) is essential for writing efficient numerical code and debugging shape errors.
- A linear layer forward pass is simply y = X @ W + b — matrix multiplication followed by bias addition.
