# Feature Maps

> Phase 5 — CNN | Kata 5.4

---

## Concept & Intuition

### What problem are we solving?

A feature map is the output of applying a single convolutional filter to an input. When we stack multiple filters, we get a volume of feature maps, one per filter. Understanding feature maps is essential because they are the internal representation that the CNN builds and refines layer by layer. Each feature map is a spatial grid where each cell answers the question: "How strongly does my filter's pattern appear at this location?"

As data flows through a CNN, feature maps undergo a transformation at each layer. The first layer produces feature maps responding to simple patterns (edges, colors). The second layer's filters operate on these first-layer feature maps, producing feature maps that respond to combinations of simple patterns (corners, textures). By the time we reach deeper layers, feature maps encode highly abstract concepts (faces, wheels, letters). This progressive abstraction is the key mechanism that allows CNNs to bridge the gap from raw pixels to semantic understanding.

The shape of a feature map volume tells us important architectural information. If the input is H x W x C (height, width, channels), and we apply F filters of size K x K, the output is (H-K+1) x (W-K+1) x F. Each layer can change the spatial dimensions and the channel depth independently, allowing architects to design networks that gradually shrink spatially while growing in channel depth, compressing spatial detail while expanding representational richness.

### Why naive approaches fail

Without understanding feature maps, it is tempting to think of CNN layers as black boxes. This makes architecture design guesswork rather than engineering. For example, if your feature maps become 1x1 spatially too early, you have lost all spatial information and cannot recover it. If you have too few channels, the representation may lack the capacity to distinguish between classes.

Another pitfall is ignoring the relationship between feature map size and receptive field. Each layer's feature map cell "sees" a region of the input (its receptive field). If the receptive field is too small at the classification stage, the network has never seen the whole object, only fragments. Understanding how feature map dimensions and receptive fields evolve through the network is critical for effective architecture design.

### Mental models

- **Stack of transparent overlays**: Each feature map is a transparent sheet highlighting one type of pattern. Stack them all together and you get a rich, multi-faceted view of the image.
- **Progressive summarization**: Like summarizing a book into chapters, then into a synopsis, then into a one-line description. Each layer summarizes spatial details while preserving essential information.
- **Trading space for depth**: As feature maps shrink spatially, they grow in channel count, trading "where" information for "what" information.

### Visual explanations

```
  Feature map dimensions through a CNN:

  Input          Layer 1         Layer 2         Layer 3
  32x32x3   -->  30x30x16   -->  14x14x32   -->  6x6x64
  (3 colors)     (16 filters)    (32 filters)    (64 filters)
                 3x3 conv        3x3 conv+pool   3x3 conv+pool

  Spatial:  32   ->  30   ->  14   ->  6     (shrinking)
  Channels:  3   ->  16   ->  32   ->  64    (growing)

  Each cell's receptive field:
  Layer 1: 3x3 of input
  Layer 2: 7x7 of input (3x3 of 3x3 patches, roughly)
  Layer 3: ~15x15 of input (sees larger context)

  Feature map volume at Layer 2:

  Channel 0    Channel 1    ...   Channel 31
  +--------+   +--------+        +--------+
  |  edge  |   | corner |        | texture|
  |  map   |   |  map   |   ...  |  map   |
  |14 x 14 |   |14 x 14 |        |14 x 14 |
  +--------+   +--------+        +--------+
        \          |           /
         \         |          /
          Combined: 14 x 14 x 32 volume
```

---

## Hands-on Exploration

1. Build a two-layer convolutional network from scratch.
2. Observe how feature map dimensions change through the layers.
3. Track how the receptive field grows with depth.
4. Visualize the feature maps at each layer to see progressive abstraction.

---

## Live Code

```rust
fn main() {
    println!("=== Feature Maps ===\n");

    // Create a 10x10 input with two distinct patterns:
    // An L-shape (top-left) and a T-shape (bottom-right)
    let mut image = vec![vec![0.0_f64; 10]; 10];

    // L-shape: vertical bar + horizontal base
    for i in 1..6 {
        image[i][1] = 1.0; // vertical part
    }
    for j in 1..4 {
        image[5][j] = 1.0; // horizontal base
    }

    // T-shape: horizontal bar + vertical stem
    for j in 5..9 {
        image[5][j] = 1.0; // horizontal part
    }
    for i in 5..9 {
        image[i][7] = 1.0; // vertical stem
    }

    println!("Input (10x10) with L-shape (top-left) and T-shape (bottom-right):");
    print_binary_grid(&image);

    // Layer 1: Three 3x3 filters (horizontal, vertical, diagonal)
    let filters_l1: Vec<(&str, Vec<Vec<f64>>)> = vec![
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
            "Corner",
            vec![
                vec![ 0.0, -1.0, -1.0],
                vec![-1.0,  2.0, -1.0],
                vec![-1.0, -1.0,  0.0],
            ],
        ),
    ];

    println!("--- Layer 1: Applying {} filters (3x3) ---\n", filters_l1.len());
    println!(
        "Input: 10x10x1 -> Output: {}x{}x{}",
        10 - 2, 10 - 2, filters_l1.len()
    );

    let mut layer1_maps: Vec<Vec<Vec<f64>>> = Vec::new();

    for (name, kernel) in &filters_l1 {
        let fmap = convolve2d(&image, kernel);
        let relu_map: Vec<Vec<f64>> = fmap
            .iter()
            .map(|row| row.iter().map(|v| v.max(0.0)).collect())
            .collect();

        let total: f64 = relu_map.iter().flatten().sum();
        println!(
            "Feature map '{}': {}x{}, total activation={:.1}",
            name,
            relu_map.len(),
            relu_map[0].len(),
            total
        );
        print_heat_grid(&relu_map);

        layer1_maps.push(relu_map);
    }

    // Layer 2: Apply a 3x3 filter across ALL layer-1 feature maps
    println!("--- Layer 2: Combining Layer-1 feature maps ---\n");
    println!(
        "Input: {}x{}x{} -> Output: {}x{}x1",
        layer1_maps[0].len(),
        layer1_maps[0][0].len(),
        layer1_maps.len(),
        layer1_maps[0].len() - 2,
        layer1_maps[0][0].len() - 2
    );

    // A layer-2 filter has one 3x3 kernel per input channel
    // This filter combines edges to detect junction-like structures
    let filter_l2: Vec<Vec<Vec<f64>>> = vec![
        // kernel for horizontal channel
        vec![
            vec![0.0, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.5, 0.0],
        ],
        // kernel for vertical channel
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.0, 0.0],
        ],
        // kernel for corner channel
        vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
    ];

    let l2_map = convolve2d_multichannel(&layer1_maps, &filter_l2);
    let l2_relu: Vec<Vec<f64>> = l2_map
        .iter()
        .map(|row| row.iter().map(|v| v.max(0.0)).collect())
        .collect();

    println!(
        "Layer-2 feature map (junction detector): {}x{}",
        l2_relu.len(),
        l2_relu[0].len()
    );
    print_heat_grid(&l2_relu);

    // Receptive field analysis
    println!("=== Receptive Field Growth ===");
    println!("Layer 1 cell sees: 3x3 = 9 input pixels");
    println!("Layer 2 cell sees: 5x5 = 25 input pixels (3x3 window over 3x3 windows)");
    println!("With pooling, receptive field grows even faster.\n");

    // Dimension tracking
    println!("=== Dimension Tracking (typical CNN) ===");
    let configs = vec![
        ("Input", 32, 3),
        ("Conv 3x3, 16 filters", 30, 16),
        ("MaxPool 2x2", 15, 16),
        ("Conv 3x3, 32 filters", 13, 32),
        ("MaxPool 2x2", 6, 32),
        ("Conv 3x3, 64 filters", 4, 64),
        ("Global Avg Pool", 1, 64),
    ];

    println!("{:<30} {:>8} {:>8} {:>10}", "Layer", "Spatial", "Channels", "Values");
    println!("{}", "-".repeat(60));
    for (name, spatial, channels) in &configs {
        println!(
            "{:<30} {:>5}x{:<5} {:>5} {:>10}",
            name,
            spatial,
            spatial,
            channels,
            spatial * spatial * channels
        );
    }
}

fn convolve2d(input: &[Vec<f64>], kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let in_h = input.len();
    let in_w = input[0].len();
    let k_size = kernel.len();
    let out_h = in_h - k_size + 1;
    let out_w = in_w - k_size + 1;

    let mut output = vec![vec![0.0; out_w]; out_h];
    for i in 0..out_h {
        for j in 0..out_w {
            let mut sum = 0.0;
            for ki in 0..k_size {
                for kj in 0..k_size {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    output
}

fn convolve2d_multichannel(
    inputs: &[Vec<Vec<f64>>],
    kernels: &[Vec<Vec<f64>>],
) -> Vec<Vec<f64>> {
    let out_h = inputs[0].len() - kernels[0].len() + 1;
    let out_w = inputs[0][0].len() - kernels[0][0].len() + 1;
    let mut output = vec![vec![0.0; out_w]; out_h];

    for (ch, kernel) in kernels.iter().enumerate() {
        let contribution = convolve2d(&inputs[ch], kernel);
        for i in 0..out_h {
            for j in 0..out_w {
                output[i][j] += contribution[i][j];
            }
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

fn print_heat_grid(grid: &[Vec<f64>]) {
    let max_val = grid
        .iter()
        .flatten()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(0.001);
    for row in grid {
        let cells: Vec<&str> = row
            .iter()
            .map(|v| {
                let normalized = v / max_val;
                if normalized > 0.7 {
                    "##"
                } else if normalized > 0.3 {
                    "++"
                } else if normalized > 0.05 {
                    ".."
                } else {
                    "  "
                }
            })
            .collect();
        println!("  |{}|", cells.join(""));
    }
    println!();
}
```

---

## Key Takeaways

- A feature map is a spatial grid showing where a particular pattern was detected, one per filter per layer.
- Feature maps form volumes (height x width x channels) that grow deeper and shrink spatially as data flows through the network, trading spatial resolution for representational richness.
- Deeper layers have larger receptive fields, allowing their feature maps to capture increasingly global and abstract patterns.
- Tracking feature map dimensions through a network is essential for architecture design: spatial resolution, channel count, and total parameter count must all be balanced.
