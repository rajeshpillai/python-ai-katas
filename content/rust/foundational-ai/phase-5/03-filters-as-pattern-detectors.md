# Filters as Pattern Detectors

> Phase 5 — CNN | Kata 5.3

---

## Concept & Intuition

### What problem are we solving?

In the previous kata we saw that different kernel values detect different features. But in a real CNN, we do not hand-design kernels. The network learns them from data through backpropagation. Each learned kernel becomes a pattern detector tuned to features useful for the task at hand. The first convolutional layer typically learns to detect simple patterns like edges at various orientations, color gradients, and spots. Deeper layers compose these into increasingly abstract detectors: corners, textures, object parts, and eventually entire objects.

The key insight is that a single convolutional layer applies multiple filters in parallel. If we have 16 filters, we get 16 output feature maps, each highlighting a different pattern across the entire image. This is like having 16 different specialists simultaneously scanning the same scene: one looking for horizontal edges, another for vertical edges, another for diagonal lines, and so on. The collection of all these feature maps forms a rich, multi-channel representation of the input.

Understanding what filters detect helps us reason about network behavior, diagnose failures, and appreciate why depth matters. Early filters detect local patterns; stacking layers allows the network to detect patterns of patterns, building a hierarchy from pixels to parts to objects.

### Why naive approaches fail

Hand-crafting filters (Sobel, Gabor, etc.) works for classical computer vision but is limited to known, low-level features. You cannot hand-design a filter that detects "cat ear" because such features are complex compositions of simpler patterns. Attempting to enumerate all useful filters manually is combinatorially infeasible and domain-specific.

Using random filters provides surprisingly useful features (this is the basis of random projection methods), but they are not optimized for any particular task. Learned filters consistently outperform random or hand-crafted ones because gradient descent tunes them to extract exactly the information the classification task requires.

### Mental models

- **Team of specialists**: Each filter is a specialist with a unique focus. One specialist looks for horizontal lines, another for color transitions. Together, they provide a comprehensive analysis of each image region.
- **Bank of stencils**: Imagine pressing different shaped stencils against the image. Where a stencil matches well, you get a strong response. The CNN learns which stencil shapes are most useful for distinguishing between classes.
- **Hierarchical decomposition**: First-layer filters are letters; second-layer combinations are words; third-layer combinations are sentences. Each level builds meaning from the level below.

### Visual explanations

```
  Multiple filters applied to the same input:

  Input Image        Filter 1 (horiz)   Filter 2 (vert)    Filter 3 (diag)
  +---------+        [−1 −1 −1]         [−1  0  1]         [−1 −1  0]
  | /--\    |        [ 0  0  0]         [−1  0  1]         [−1  0  1]
  ||    |   |        [ 1  1  1]         [−1  0  1]         [ 0  1  1]
  ||    |   |            |                   |                   |
  | \--/    |            v                   v                   v
  +---------+        Feature Map 1      Feature Map 2      Feature Map 3
                     (horiz edges)      (vert edges)       (diag edges)
                     +---------+        +---------+        +---------+
                     |  ****   |        | *    *  |        | *       |
                     |         |        | *    *  |        |    *    |
                     |  ****   |        | *    *  |        |       *|
                     +---------+        +---------+        +---------+

  Stacking layers:

  Layer 1 filters     Layer 2 filters      Layer 3 filters
  (edges)          -> (corners, curves) -> (object parts)
  [/ | \ —]          [L  T  arc  ...]     [eye  ear  ...]
```

---

## Hands-on Exploration

1. Create a synthetic image containing multiple pattern types (horizontal lines, vertical lines, a diagonal).
2. Apply several hand-crafted filters and observe which parts of the image each filter responds to.
3. Measure the response strength (activation magnitude) for matched vs unmatched regions.
4. Simulate how stacking two convolution layers allows detection of more complex patterns.

---

## Live Code

```rust
fn main() {
    println!("=== Filters as Pattern Detectors ===\n");

    // Create a 9x9 image with distinct regions:
    // Top: horizontal lines, Left: vertical lines, Center: diagonal
    let mut image = vec![vec![0.0_f64; 9]; 9];

    // Horizontal stripes in top rows
    for j in 0..9 {
        image[0][j] = 1.0;
        image[2][j] = 1.0;
    }
    // Vertical stripes in left columns
    for i in 0..9 {
        image[i][0] = 1.0;
        image[i][2] = 1.0;
    }
    // Diagonal in center
    for k in 3..8 {
        image[k][k] = 1.0;
    }

    println!("Input image (H lines top, V lines left, diagonal center):");
    print_binary_grid(&image);

    // Define three oriented edge filters
    let filters: Vec<(&str, Vec<Vec<f64>>)> = vec![
        (
            "Horizontal",
            vec![
                vec![-1.0, -1.0, -1.0],
                vec![ 2.0,  2.0,  2.0],
                vec![-1.0, -1.0, -1.0],
            ],
        ),
        (
            "Vertical",
            vec![
                vec![-1.0, 2.0, -1.0],
                vec![-1.0, 2.0, -1.0],
                vec![-1.0, 2.0, -1.0],
            ],
        ),
        (
            "Diagonal (\\)",
            vec![
                vec![ 2.0, -1.0, -1.0],
                vec![-1.0,  2.0, -1.0],
                vec![-1.0, -1.0,  2.0],
            ],
        ),
    ];

    let mut feature_maps = Vec::new();

    for (name, kernel) in &filters {
        let fmap = convolve2d(&image, kernel);
        let relu_map: Vec<Vec<f64>> = fmap
            .iter()
            .map(|row| row.iter().map(|v| v.max(0.0)).collect())
            .collect();

        // Compute total activation as a measure of pattern presence
        let total: f64 = relu_map.iter().flatten().sum();
        let max_val = relu_map
            .iter()
            .flatten()
            .cloned()
            .fold(0.0_f64, f64::max);

        println!("Filter: {} -> total activation: {:.1}, max: {:.1}", name, total, max_val);
        print_activation_grid(&relu_map);

        feature_maps.push(relu_map);
    }

    // Demonstrate how a second layer can combine first-layer outputs
    println!("=== Stacking Layers: Detecting Corners ===\n");
    println!("A corner is where a horizontal and vertical edge meet.");
    println!("We simulate this by combining feature maps from layer 1.\n");

    // Element-wise minimum of horizontal and vertical maps detects co-occurrence
    let h_map = &feature_maps[0];
    let v_map = &feature_maps[1];
    let rows = h_map.len();
    let cols = h_map[0].len();

    let corner_map: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| h_map[i][j].min(v_map[i][j]))
                .collect()
        })
        .collect();

    println!("Corner detection (min of horizontal and vertical responses):");
    print_activation_grid(&corner_map);

    let corner_total: f64 = corner_map.iter().flatten().sum();
    println!(
        "Corner activation is concentrated where H and V edges meet: {:.1}",
        corner_total
    );
    println!("This is how deeper layers compose simple features into complex ones.");
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

fn print_binary_grid(grid: &[Vec<f64>]) {
    for row in grid {
        let cells: Vec<&str> = row.iter().map(|v| if *v > 0.5 { "#" } else { "." }).collect();
        println!("  {}", cells.join(" "));
    }
    println!();
}

fn print_activation_grid(grid: &[Vec<f64>]) {
    for row in grid {
        let cells: Vec<String> = row.iter().map(|v| {
            if *v > 3.0 {
                " ## ".to_string()
            } else if *v > 0.5 {
                format!("{:4.1}", v)
            } else {
                "  . ".to_string()
            }
        }).collect();
        println!("  {}", cells.join(""));
    }
    println!();
}
```

---

## Key Takeaways

- Each convolutional filter acts as a learned pattern detector, producing a feature map that highlights where its target pattern occurs in the input.
- Multiple filters run in parallel to capture different features simultaneously, creating a multi-channel representation.
- Stacking convolutional layers allows the network to detect increasingly complex patterns by composing simpler ones from earlier layers.
- Learned filters outperform hand-crafted ones because gradient descent optimizes them specifically for the task at hand.
