# Pooling Effects

> Phase 5 — CNN | Kata 5.5

---

## Concept & Intuition

### What problem are we solving?

After convolution, feature maps can be large and contain redundant spatial detail. Pooling is a downsampling operation that reduces the spatial dimensions of feature maps while preserving the most important information. The two most common variants are max pooling (taking the maximum value in each window) and average pooling (taking the mean). A 2x2 max pool with stride 2, for example, reduces each spatial dimension by half, cutting the total number of values by 75%.

Pooling serves three critical purposes. First, it reduces computational cost by shrinking feature maps before subsequent convolutional layers process them. Second, it increases the effective receptive field: after a 2x2 pool, each cell in the next convolution "sees" twice as much of the original input. Third, and most subtly, pooling introduces a degree of translation invariance. If a feature shifts by one pixel, the max pool output often remains the same because the maximum in a 2x2 window is robust to small shifts.

However, pooling is a lossy operation. It discards spatial precision, which matters for tasks requiring fine-grained localization (like semantic segmentation or object detection). Modern architectures sometimes replace pooling with strided convolutions, which learn what information to discard rather than using a fixed rule. Understanding pooling's trade-offs helps us choose the right approach for each task.

### Why naive approaches fail

Simply reducing image resolution before feeding it to the network (naive downsampling) throws away information before the network has had a chance to extract features. Pooling, by contrast, operates on feature maps after convolution, preserving the detected patterns while reducing spatial redundancy.

Using no pooling at all keeps full spatial resolution but makes deeper layers prohibitively expensive. A 224x224 image with 256 channels has 12.8 million values per layer. Without pooling, stacking 10 such layers requires enormous memory and computation, and the small 3x3 kernels would need many more layers to achieve a sufficient receptive field.

### Mental models

- **Headline extraction**: Pooling is like reading a newspaper by only looking at headlines. You lose detail but capture the most important information (max pooling) or get the overall tone (average pooling).
- **Zoom out**: Each pooling step is like taking a step back from a painting. You lose fine brushstrokes but see larger patterns and compositions.
- **Robustness through coarsening**: A slight shift in the input might move a feature from one pixel to an adjacent one, but max pooling over a 2x2 window captures it either way.

### Visual explanations

```
  Max Pooling (2x2, stride 2):

  Input (4x4):              Output (2x2):
  [ 1  3 | 2  1 ]           [ 3   2 ]   (max of each 2x2 block)
  [ 0  2 | 1  0 ]           [ 4   5 ]
  -------+-------
  [ 4  1 | 5  2 ]
  [ 1  0 | 3  1 ]

  Average Pooling (2x2, stride 2):

  Input (4x4):              Output (2x2):
  [ 1  3 | 2  1 ]           [1.5  1.0]   (mean of each 2x2 block)
  [ 0  2 | 1  0 ]           [1.5  2.75]
  -------+-------
  [ 4  1 | 5  2 ]
  [ 1  0 | 3  1 ]

  Translation invariance from max pool:

  Original:     Shifted 1px:    Max pool of both:
  [0 5 0 0]     [0 0 5 0]      [5  0]  (same!)
  [0 0 0 0]     [0 0 0 0]      [0  0]  (same!)
  [0 0 0 0]     [0 0 0 0]
  [0 0 0 0]     [0 0 0 0]
```

---

## Hands-on Exploration

1. Implement max pooling and average pooling from scratch.
2. Apply them to a feature map and observe the spatial reduction.
3. Demonstrate translation invariance by shifting a pattern and showing pooled outputs match.
4. Compare the information loss of max pooling vs average pooling.

---

## Live Code

```rust
fn main() {
    println!("=== Pooling Effects ===\n");

    // Create an 8x8 feature map with a bright spot and some texture
    let mut fmap = vec![vec![0.0_f64; 8]; 8];

    // Bright spot at (1,1)-(2,2)
    fmap[1][1] = 9.0;
    fmap[1][2] = 7.0;
    fmap[2][1] = 6.0;
    fmap[2][2] = 8.0;

    // Some texture in bottom-right
    fmap[5][5] = 3.0;
    fmap[5][6] = 4.0;
    fmap[6][5] = 5.0;
    fmap[6][6] = 2.0;

    // Scattered values
    fmap[0][6] = 1.0;
    fmap[3][4] = 2.0;
    fmap[7][0] = 1.0;

    println!("Original feature map (8x8):");
    print_grid(&fmap);

    // Max pooling 2x2
    let max_pooled = pool2d(&fmap, 2, PoolType::Max);
    println!("After MAX pooling (2x2, stride 2) -> 4x4:");
    print_grid(&max_pooled);

    // Average pooling 2x2
    let avg_pooled = pool2d(&fmap, 2, PoolType::Average);
    println!("After AVERAGE pooling (2x2, stride 2) -> 4x4:");
    print_grid(&avg_pooled);

    // Demonstrate translation invariance
    println!("=== Translation Invariance Demo ===\n");

    let mut original = vec![vec![0.0_f64; 6]; 6];
    original[1][1] = 5.0;
    original[1][2] = 3.0;
    original[2][1] = 4.0;
    original[2][2] = 6.0;

    let mut shifted = vec![vec![0.0_f64; 6]; 6];
    shifted[1][2] = 5.0;
    shifted[1][3] = 3.0;
    shifted[2][2] = 4.0;
    shifted[2][3] = 6.0;

    println!("Original pattern:");
    print_grid(&original);

    println!("Pattern shifted 1 pixel right:");
    print_grid(&shifted);

    let orig_pooled = pool2d(&original, 2, PoolType::Max);
    let shift_pooled = pool2d(&shifted, 2, PoolType::Max);

    println!("Max pooled ORIGINAL:");
    print_grid(&orig_pooled);

    println!("Max pooled SHIFTED:");
    print_grid(&shift_pooled);

    let diff: f64 = orig_pooled
        .iter()
        .flatten()
        .zip(shift_pooled.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!(
        "Total absolute difference after pooling: {:.1} (small shift partially absorbed)\n",
        diff
    );

    // Information loss analysis
    println!("=== Information Loss Analysis ===\n");

    // Reconstruct from pooled (upsample) and measure error
    let upsampled_max = upsample(&max_pooled, 2);
    let upsampled_avg = upsample(&avg_pooled, 2);

    let mse_max = mse(&fmap, &upsampled_max);
    let mse_avg = mse(&fmap, &upsampled_avg);

    println!("MSE after max pool + nearest-neighbor upsample:  {:.3}", mse_max);
    println!("MSE after avg pool + nearest-neighbor upsample:  {:.3}", mse_avg);
    println!();
    println!("Max pooling preserves strong activations but loses spatial precision.");
    println!("Average pooling preserves overall energy but dilutes strong signals.");

    // Computational savings
    println!("\n=== Computational Savings ===\n");
    let sizes = vec![
        (224, "224x224"),
        (112, "112x112 (after 1 pool)"),
        (56, "56x56 (after 2 pools)"),
        (28, "28x28 (after 3 pools)"),
        (14, "14x14 (after 4 pools)"),
        (7, "7x7 (after 5 pools)"),
    ];
    let channels = 64;

    for (size, label) in &sizes {
        let values = size * size * channels;
        let conv_ops = size * size * channels * 3 * 3 * channels;
        println!(
            "{:<30} values: {:>10}  conv3x3 ops: {:>14}",
            label, values, conv_ops
        );
    }
    println!("\nEach 2x2 pool reduces computation by ~4x for the next layer.");
}

enum PoolType {
    Max,
    Average,
}

fn pool2d(input: &[Vec<f64>], pool_size: usize, pool_type: PoolType) -> Vec<Vec<f64>> {
    let in_h = input.len();
    let in_w = input[0].len();
    let out_h = in_h / pool_size;
    let out_w = in_w / pool_size;

    let mut output = vec![vec![0.0; out_w]; out_h];

    for i in 0..out_h {
        for j in 0..out_w {
            let mut values = Vec::new();
            for pi in 0..pool_size {
                for pj in 0..pool_size {
                    values.push(input[i * pool_size + pi][j * pool_size + pj]);
                }
            }
            output[i][j] = match pool_type {
                PoolType::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                PoolType::Average => values.iter().sum::<f64>() / values.len() as f64,
            };
        }
    }

    output
}

fn upsample(input: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
    let in_h = input.len();
    let in_w = input[0].len();
    let out_h = in_h * factor;
    let out_w = in_w * factor;

    let mut output = vec![vec![0.0; out_w]; out_h];
    for i in 0..out_h {
        for j in 0..out_w {
            output[i][j] = input[i / factor][j / factor];
        }
    }
    output
}

fn mse(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len() * a[0].len();
    let sum: f64 = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    sum / n as f64
}

fn print_grid(grid: &[Vec<f64>]) {
    for row in grid {
        let cells: Vec<String> = row.iter().map(|v| format!("{:5.1}", v)).collect();
        println!("  [{}]", cells.join(" "));
    }
    println!();
}
```

---

## Key Takeaways

- Pooling downsamples feature maps, reducing spatial dimensions while preserving the most salient information (max) or overall distribution (average).
- Max pooling provides partial translation invariance because small shifts often do not change the maximum within a pooling window.
- Each 2x2 pooling layer reduces computation by approximately 4x for subsequent layers, making deep networks tractable.
- Pooling is a lossy operation that trades spatial precision for computational efficiency and larger receptive fields; modern architectures sometimes replace it with strided convolutions for learnable downsampling.
