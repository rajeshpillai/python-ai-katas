# Convolution Intuition

> Phase 5 — CNN | Kata 5.2

---

## Concept & Intuition

### What problem are we solving?

Convolution is the core operation that makes neural networks effective on spatial data. Instead of connecting every input to every output (as dense layers do), convolution slides a small kernel across the input, computing a weighted sum at each position. This single idea solves the two fundamental problems we identified: it enforces local connectivity (each output depends only on a small neighborhood) and parameter sharing (the same kernel weights are reused at every position).

The mathematical definition is straightforward: given a 2D input and a small kernel, the output at position (i, j) is the sum of element-wise products between the kernel and the input patch centered at (i, j). This is a dot product between the kernel and a local region, repeated across the entire input. The output is called a feature map, and its values indicate how strongly the local pattern matches the kernel at each position.

What makes this powerful is that a single kernel with, say, 9 parameters (3x3) can scan an entire image of any size. If the kernel detects a vertical edge, it will find vertical edges everywhere in the image, automatically achieving translational equivariance. The network learns what patterns to look for (the kernel values) while the sliding mechanism handles where to look.

### Why naive approaches fail

Without convolution, you might try to hand-engineer features: compute edge counts, color histograms, or texture statistics. This works for simple tasks but requires domain expertise for each new problem and fails to capture the hierarchical, compositional nature of visual features. A face is made of eyes made of edges, and this hierarchy must be learned, not specified.

Another naive approach is to use dense networks with data augmentation (training with shifted, rotated copies). While this helps, it cannot fundamentally overcome the parameter explosion, and the network still treats each augmented version as a separate pattern rather than understanding the underlying spatial invariance.

### Mental models

- **Rubber stamp**: A convolution kernel is like a rubber stamp that you press onto every part of the image. The ink intensity at each position tells you how well that region matches the stamp's pattern.
- **Sliding window search**: Like running a small template across a security camera feed to find a specific pattern, the kernel acts as a detector that reports a match score at every location.
- **Shared ears**: If a dense network needs separate "ear detectors" for every possible ear position, a convolutional network has one ear detector that it slides everywhere.

### Visual explanations

```
  Input (5x5):        Kernel (3x3):       Output (3x3):

  [1  0  1  0  1]     [1  0  1]     Slide kernel across input:
  [0  1  0  1  0]     [0  1  0]
  [1  0  1  0  1]     [1  0  1]     pos(0,0): 1*1+0*0+1*1+0*0+1*1+0*0+1*1+0*0+1*1 = 5
  [0  1  0  1  0]                   pos(0,1): 0*1+1*0+0*1+1*0+0*1+1*0+0*1+1*0+0*1 = 0
  [1  0  1  0  1]                   ...

  Result:              The high values (5) indicate where the
  [5  0  5]            checkerboard pattern matches the kernel.
  [0  5  0]            The low values (0) indicate no match.
  [5  0  5]            Same kernel, reused at every position!

  Convolution operation step by step:

  Step 1: overlay kernel    Step 2: multiply        Step 3: sum
  at position (0,0)         element-wise            all products

  [1  0  1] . [1  0  1]    [1  0  1]               1+0+1+0+1+0
  [0  1  0]   [0  1  0]  = [0  1  0]             + 1+0+1 = 5
  [1  0  1]   [1  0  1]    [1  0  1]
```

---

## Hands-on Exploration

1. Implement 2D convolution from scratch on a small grid.
2. Apply an identity kernel and verify the output matches the input center.
3. Apply an edge-detection kernel and observe how it highlights boundaries.
4. Experiment with different kernel values to build intuition for what each detects.

---

## Live Code

```rust
fn main() {
    println!("=== Convolution Intuition ===\n");

    // 7x7 input image with a bright square in the center
    let image: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    println!("Input image (7x7, bright square in center):");
    print_grid(&image);

    // Kernel 1: Identity (center pixel only)
    let identity_kernel = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];

    let result = convolve2d(&image, &identity_kernel);
    println!("After IDENTITY kernel (should preserve the image):");
    print_grid(&result);

    // Kernel 2: Horizontal edge detector
    let h_edge_kernel = vec![
        vec![-1.0, -1.0, -1.0],
        vec![ 0.0,  0.0,  0.0],
        vec![ 1.0,  1.0,  1.0],
    ];

    let result = convolve2d(&image, &h_edge_kernel);
    println!("After HORIZONTAL EDGE kernel:");
    print_grid(&result);

    // Kernel 3: Vertical edge detector
    let v_edge_kernel = vec![
        vec![-1.0, 0.0, 1.0],
        vec![-1.0, 0.0, 1.0],
        vec![-1.0, 0.0, 1.0],
    ];

    let result = convolve2d(&image, &v_edge_kernel);
    println!("After VERTICAL EDGE kernel:");
    print_grid(&result);

    // Kernel 4: Sharpen
    let sharpen_kernel = vec![
        vec![ 0.0, -1.0,  0.0],
        vec![-1.0,  5.0, -1.0],
        vec![ 0.0, -1.0,  0.0],
    ];

    let result = convolve2d(&image, &sharpen_kernel);
    println!("After SHARPEN kernel:");
    print_grid(&result);

    // Kernel 5: Blur (average)
    let blur_kernel = vec![
        vec![1.0/9.0, 1.0/9.0, 1.0/9.0],
        vec![1.0/9.0, 1.0/9.0, 1.0/9.0],
        vec![1.0/9.0, 1.0/9.0, 1.0/9.0],
    ];

    let result = convolve2d(&image, &blur_kernel);
    println!("After BLUR (averaging) kernel:");
    print_grid(&result);

    // Demonstrate parameter efficiency
    println!("=== Parameter Efficiency ===");
    println!("3x3 kernel parameters: 9");
    println!("Applied to 7x7 image: produces 5x5 output");
    println!("Same 9 parameters used at all 25 output positions!");
    println!(
        "Dense equivalent: {} -> {} would need {} parameters",
        7 * 7,
        5 * 5,
        7 * 7 * 5 * 5
    );
}

fn convolve2d(input: &[Vec<f64>], kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let in_h = input.len();
    let in_w = input[0].len();
    let k_h = kernel.len();
    let k_w = kernel[0].len();
    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;

    let mut output = vec![vec![0.0; out_w]; out_h];

    for i in 0..out_h {
        for j in 0..out_w {
            let mut sum = 0.0;
            for ki in 0..k_h {
                for kj in 0..k_w {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }

    output
}

fn print_grid(grid: &[Vec<f64>]) {
    for row in grid {
        let cells: Vec<String> = row
            .iter()
            .map(|v| format!("{:6.2}", v))
            .collect();
        println!("  [{}]", cells.join(" "));
    }
    println!();
}
```

---

## Key Takeaways

- Convolution slides a small kernel across the input, computing local dot products to produce a feature map.
- Parameter sharing means the same kernel weights detect the same pattern at every spatial position, giving translational equivariance for free.
- A 3x3 kernel uses only 9 parameters regardless of input size, versus thousands or millions for a dense equivalent.
- Different kernel values detect different features: edges, blurs, sharpening, and more complex patterns when learned from data.
