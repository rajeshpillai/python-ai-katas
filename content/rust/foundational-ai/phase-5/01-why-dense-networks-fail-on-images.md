# Why Dense Networks Fail on Images

> Phase 5 — CNN | Kata 5.1

---

## Concept & Intuition

### What problem are we solving?

When we attempt to classify images using fully connected (dense) neural networks, we immediately run into fundamental problems. Consider a modest 28x28 grayscale image: it has 784 pixels. A dense layer connecting these to just 128 hidden neurons requires 100,352 weights. Scale to a 224x224 RGB image and the first layer alone demands over 150 million parameters. This explosion makes training slow, memory-hungry, and prone to overfitting.

But the parameter count is only half the story. Dense networks treat every pixel as an independent feature with no concept of spatial locality. A cat's ear in the top-left corner and the same ear shifted ten pixels to the right look like completely different patterns to a dense network. It must learn each shifted version independently, which is catastrophically wasteful.

The core issue is that dense networks lack two properties that vision naturally demands: translational equivariance (recognizing a pattern regardless of where it appears) and local connectivity (pixels near each other matter more than distant ones). Without these inductive biases, dense networks must brute-force learn spatial structure from data alone, requiring far more examples than we typically have.

### Why naive approaches fail

A dense network memorizes pixel positions rather than learning spatial patterns. If you train on centered digits and test on slightly shifted ones, accuracy drops dramatically. The network has no mechanism to generalize across spatial translations because every input position is an independent dimension.

Furthermore, the sheer number of parameters creates severe overfitting. With millions of weights and only thousands of training images, the network memorizes training examples rather than extracting generalizable features. Regularization helps but cannot overcome the fundamental architectural mismatch between dense connectivity and spatial data.

### Mental models

- **Reading through a straw**: A dense network is like trying to read a page by looking at each letter position through a tiny straw, memorizing that "position 47 is an 'a'" rather than recognizing the word "cat" as a pattern that could appear anywhere.
- **Jigsaw with no edges**: Feeding a flattened image into a dense network is like solving a jigsaw puzzle after removing all edge shapes. The spatial relationships that make the puzzle tractable are destroyed.
- **Overly specific memory**: Dense networks act like someone who can only recognize their friend when standing in the exact same spot where they first met.

### Visual explanations

```
  Dense network view of a 4x4 image:

  Image:              Flattened input:
  [1 0 1 0]           [1,0,1,0,0,1,1,0,0,0,1,1,0,0,0,1]
  [0 1 1 0]            |  |  |  |  |  |  |  |  |  |  ...
  [0 0 1 1]            v  v  v  v  v  v  v  v  v  v
  [0 0 0 1]           All 16 inputs connect to ALL hidden neurons
                      = 16 x H weights (no spatial structure!)

  Shifted image:       Flattened input:
  [0 1 0 1]           [0,1,0,1,1,0,0,1,0,0,1,1,0,0,0,1]
  [1 0 0 1]            Completely different pattern to the network!
  [0 0 1 1]            Same features, but network cannot tell.
  [0 0 0 1]

  Parameter explosion:
  Image size     Pixels    Dense(128)    Dense(256)
  ─────────────────────────────────────────────────
  28x28 (MNIST)    784     100,352       200,704
  32x32x3 (CIFAR)  3,072   393,216       786,432
  224x224x3        150,528  19,267,584    38,535,168
```

---

## Hands-on Exploration

1. Create a small synthetic 6x6 "image" containing a recognizable pattern (a diagonal line).
2. Flatten the image and pass it through a simulated dense layer, observing the weight count.
3. Shift the pattern by one pixel and show that the dense layer activations change drastically.
4. Compare the number of parameters with what a local (convolutional) approach would need.

---

## Live Code

```rust
fn main() {
    // Simulate a 6x6 binary image with a diagonal pattern
    let image: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];

    // Same diagonal shifted right by 1
    let shifted: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    // Flatten both images
    let flat_orig: Vec<f64> = image.iter().flatten().cloned().collect();
    let flat_shift: Vec<f64> = shifted.iter().flatten().cloned().collect();

    let input_size = flat_orig.len(); // 36
    let hidden_size = 8;

    println!("=== Why Dense Networks Fail on Images ===\n");
    println!("Image size: 6x6 = {} pixels", input_size);
    println!(
        "Dense layer to {} hidden neurons: {} weights",
        hidden_size,
        input_size * hidden_size
    );
    println!(
        "Equivalent conv layer (3x3 kernel, {} filters): {} weights\n",
        hidden_size,
        3 * 3 * hidden_size
    );

    // Simulate a dense layer with fixed pseudo-random weights
    let weights: Vec<Vec<f64>> = (0..hidden_size)
        .map(|h| {
            (0..input_size)
                .map(|i| {
                    let seed = (h * input_size + i) as f64;
                    (seed * 2.7183).sin() * 0.5
                })
                .collect()
        })
        .collect();

    let biases = vec![0.0_f64; hidden_size];

    // Forward pass: dense layer + ReLU
    let act_orig = dense_forward(&flat_orig, &weights, &biases);
    let act_shift = dense_forward(&flat_shift, &weights, &biases);

    println!("Activations for ORIGINAL diagonal:");
    print_vec(&act_orig);

    println!("Activations for SHIFTED diagonal (1px right):");
    print_vec(&act_shift);

    // Measure how different the activations are
    let diff: f64 = act_orig
        .iter()
        .zip(act_shift.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let mag: f64 = act_orig.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();

    println!(
        "L2 distance between activations: {:.4}",
        diff
    );
    println!(
        "Relative change: {:.1}%",
        if mag > 0.0 { diff / mag * 100.0 } else { 0.0 }
    );
    println!("\nThe same pattern shifted by 1 pixel produces");
    println!("drastically different activations in a dense layer.");
    println!("A convolutional layer would produce the SAME pattern,");
    println!("just shifted in the output feature map.");

    // Parameter comparison for realistic sizes
    println!("\n=== Parameter Comparison ===");
    let sizes = [(28, 28, 1, "MNIST"), (32, 32, 3, "CIFAR-10"), (224, 224, 3, "ImageNet")];
    let hidden = 256;
    for (h, w, c, name) in &sizes {
        let pixels = h * w * c;
        let dense_params = pixels * hidden;
        let conv_params = 3 * 3 * c * hidden; // 3x3 conv, same output filters
        println!(
            "{:>10}: dense={:>12} params | conv={:>8} params | ratio={:.0}x",
            name,
            dense_params,
            conv_params,
            dense_params as f64 / conv_params as f64
        );
    }
}

fn dense_forward(input: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(biases.iter())
        .map(|(w, b)| {
            let sum: f64 = w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum();
            (sum + b).max(0.0) // ReLU
        })
        .collect()
}

fn print_vec(v: &[f64]) {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.4}", x)).collect();
    println!("  [{}]\n", parts.join(", "));
}
```

---

## Key Takeaways

- Dense networks treat every pixel independently, destroying spatial structure when the image is flattened.
- Parameter counts explode with image size because every pixel connects to every hidden neuron, making training impractical for realistic images.
- A one-pixel shift produces entirely different activations in a dense layer, meaning the network must memorize every possible position of every pattern.
- Convolutional layers solve these problems by sharing weights across spatial positions, achieving translational equivariance with orders-of-magnitude fewer parameters.
