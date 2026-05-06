# Multi-Layer Perceptron

> Phase 3 — Artificial Neural Networks | Kata 3.2

---

## Concept & Intuition

### What problem are we solving?

A multi-layer perceptron (MLP) stacks multiple layers of neurons to learn complex, non-linear patterns that a single neuron cannot. Each layer transforms its input through a linear operation (matrix multiply + bias) followed by a non-linear activation function. By composing multiple such transformations, the network can represent arbitrarily complex functions — this is the Universal Approximation Theorem.

The key insight of MLPs is that hidden layers learn intermediate representations. The first layer might learn simple features (edges, thresholds), the second layer combines those into more complex features (shapes, patterns), and the final layer uses those features for the task (classification, regression). This hierarchical feature learning is what makes deep learning powerful.

Training an MLP uses the same principle as training a single neuron — gradient descent — but the gradients must flow backward through multiple layers. This is backpropagation: the chain rule applied systematically from the output layer back to the input layer. Each layer receives gradients from the layer above and passes gradients to the layer below.

### Why naive approaches fail

A single-layer network (no hidden layers) can only learn linear decision boundaries. The classic example is XOR: no single line can separate the inputs (0,0), (1,1) from (0,1), (1,0). An MLP with one hidden layer can solve XOR by first transforming the inputs into a space where they become linearly separable, then applying a linear classifier. This transformation is what the hidden layer learns.

### Mental models

- **Assembly line**: Each layer processes the data and passes a refined version to the next layer. Early layers do rough shaping; later layers do fine details.
- **Function composition**: An MLP computes f(x) = f3(f2(f1(x))), where each fi is a layer. Composing simple functions creates complex ones.
- **XOR as the litmus test**: If your architecture can solve XOR, it can (in principle) solve any classification problem. XOR requires at least one hidden layer.

### Visual explanations

```
  Multi-layer perceptron (2 → 4 → 1):

  Input      Hidden       Output
  layer      layer        layer

  x1 ──→ [h1]
      ╲  ╱    ╲
       ╲╱      ╲
      ╱╲[h2]───→ [out] → ŷ
     ╱  ╲      ╱
  x2 ──→ [h3]╱
      ╲      ╱
       ╲[h4]╱

  Each arrow has a weight. Each neuron has a bias.
  Total parameters: (2*4 + 4) + (4*1 + 1) = 17
```

---

## Hands-on Exploration

1. Build a multi-layer perceptron with configurable layer sizes.
2. Train it to solve XOR — the problem a single neuron cannot solve.
3. Observe how the hidden layer transforms the input space.

---

## Live Code

```rust
fn main() {
    // === Multi-Layer Perceptron ===
    // Stacking layers to learn non-linear patterns.

    // Pseudo-random number generator
    let mut seed: u64 = 123;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // === MLP components ===
    // Layer: y = activation(W @ x + b)

    struct Layer {
        weights: Vec<Vec<f64>>,  // [out_size][in_size]
        biases: Vec<f64>,        // [out_size]
        // Cached for backprop
        last_input: Vec<f64>,
        last_z: Vec<f64>,       // pre-activation
        last_output: Vec<f64>,  // post-activation
    }

    impl Layer {
        fn new(in_size: usize, out_size: usize, rand: &mut dyn FnMut() -> f64) -> Self {
            // Xavier initialization
            let scale = (2.0 / (in_size + out_size) as f64).sqrt();
            let weights: Vec<Vec<f64>> = (0..out_size)
                .map(|_| (0..in_size).map(|_| rand() * scale).collect())
                .collect();
            let biases = vec![0.0; out_size];
            Layer {
                weights, biases,
                last_input: vec![],
                last_z: vec![],
                last_output: vec![],
            }
        }

        fn forward(&mut self, input: &[f64], use_relu: bool) -> Vec<f64> {
            self.last_input = input.to_vec();
            self.last_z = Vec::new();
            self.last_output = Vec::new();

            for i in 0..self.weights.len() {
                let z: f64 = self.weights[i].iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>() + self.biases[i];
                self.last_z.push(z);

                let a = if use_relu {
                    if z > 0.0 { z } else { 0.01 * z } // leaky ReLU
                } else {
                    1.0 / (1.0 + (-z).exp()) // sigmoid
                };
                self.last_output.push(a);
            }
            self.last_output.clone()
        }
    }

    // === Build MLP: 2 → 8 → 4 → 1 ===
    let mut layer1 = Layer::new(2, 8, &mut rand_f64);
    let mut layer2 = Layer::new(8, 4, &mut rand_f64);
    let mut layer3 = Layer::new(4, 1, &mut rand_f64);

    println!("=== Multi-Layer Perceptron ===\n");
    println!("  Architecture: 2 → 8 (ReLU) → 4 (ReLU) → 1 (sigmoid)");
    println!("  Parameters: {} + {} + {} = {}",
        2*8 + 8, 8*4 + 4, 4*1 + 1,
        (2*8 + 8) + (8*4 + 4) + (4*1 + 1));

    // === XOR dataset ===
    let xor_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    println!("  Task: Learn XOR (non-linearly separable)\n");

    // Forward pass through entire network
    let mut forward_all = |x: &[f64],
                           l1: &mut Layer, l2: &mut Layer, l3: &mut Layer| -> f64 {
        let h1 = l1.forward(x, true);      // ReLU
        let h2 = l2.forward(&h1, true);     // ReLU
        let out = l3.forward(&h2, false);   // sigmoid
        out[0]
    };

    // === Training with manual backpropagation ===
    let lr = 0.1;
    let n_epochs = 2000;

    println!("  {:>6} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "epoch", "loss", "XOR(00)", "XOR(01)", "XOR(10)", "XOR(11)");
    println!("  {:->6} {:->10} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "", "");

    for epoch in 0..n_epochs {
        let mut total_loss = 0.0;

        for (x, target) in &xor_data {
            // Forward pass
            let h1 = layer1.forward(x, true);
            let h2 = layer2.forward(&h1, true);
            let output = layer3.forward(&h2, false);
            let pred = output[0];

            // BCE loss
            let p = pred.max(1e-15).min(1.0 - 1e-15);
            total_loss += -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());

            // === Backpropagation ===

            // Output layer gradient: d_loss/d_z3 = pred - target (for sigmoid + BCE)
            let d_out = vec![pred - target];

            // Layer 3 weight gradients
            let mut d_h2 = vec![0.0; layer2.weights.len()];
            for i in 0..layer3.weights.len() {
                for j in 0..layer3.weights[i].len() {
                    layer3.weights[i][j] -= lr * d_out[i] * layer3.last_input[j];
                    d_h2[j] += d_out[i] * layer3.weights[i][j];
                }
                layer3.biases[i] -= lr * d_out[i];
            }

            // Layer 2 gradient (through leaky ReLU)
            let d_z2: Vec<f64> = d_h2.iter().enumerate().map(|(i, &d)| {
                let relu_grad = if layer2.last_z[i] > 0.0 { 1.0 } else { 0.01 };
                d * relu_grad
            }).collect();

            let mut d_h1 = vec![0.0; layer1.weights.len()];
            for i in 0..layer2.weights.len() {
                for j in 0..layer2.weights[i].len() {
                    layer2.weights[i][j] -= lr * d_z2[i] * layer2.last_input[j];
                    d_h1[j] += d_z2[i] * layer2.weights[i][j];
                }
                layer2.biases[i] -= lr * d_z2[i];
            }

            // Layer 1 gradient (through leaky ReLU)
            let d_z1: Vec<f64> = d_h1.iter().enumerate().map(|(i, &d)| {
                let relu_grad = if layer1.last_z[i] > 0.0 { 1.0 } else { 0.01 };
                d * relu_grad
            }).collect();

            for i in 0..layer1.weights.len() {
                for j in 0..layer1.weights[i].len() {
                    layer1.weights[i][j] -= lr * d_z1[i] * layer1.last_input[j];
                }
                layer1.biases[i] -= lr * d_z1[i];
            }
        }

        total_loss /= xor_data.len() as f64;

        if epoch < 10 || epoch % 200 == 0 || epoch == n_epochs - 1 {
            let p00 = forward_all(&[0.0, 0.0], &mut layer1, &mut layer2, &mut layer3);
            let p01 = forward_all(&[0.0, 1.0], &mut layer1, &mut layer2, &mut layer3);
            let p10 = forward_all(&[1.0, 0.0], &mut layer1, &mut layer2, &mut layer3);
            let p11 = forward_all(&[1.0, 1.0], &mut layer1, &mut layer2, &mut layer3);

            println!("  {:>6} {:>10.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                epoch, total_loss, p00, p01, p10, p11);
        }
    }

    // === Final results ===
    println!("\n=== XOR Results ===\n");
    println!("  {:>10} {:>10} {:>10} {:>10}",
        "Input", "Target", "Output", "Correct?");
    println!("  {:->10} {:->10} {:->10} {:->10}", "", "", "", "");

    for (x, target) in &xor_data {
        let pred = forward_all(x, &mut layer1, &mut layer2, &mut layer3);
        let class = if pred >= 0.5 { 1.0 } else { 0.0 };
        let correct = if (class - target).abs() < 0.1 { "yes" } else { "NO" };
        println!("  [{:.0}, {:.0}] {:>10.0} {:>10.3} {:>10}",
            x[0], x[1], target, pred, correct);
    }

    // === Visualize hidden representations ===
    println!("\n=== Hidden Layer Representations ===\n");
    println!("  The hidden layer transforms inputs into a linearly separable space:\n");

    for (x, target) in &xor_data {
        let h1 = layer1.forward(x, true);
        let active: Vec<String> = h1.iter()
            .map(|&v| if v > 0.1 { format!("{:.2}", v) } else { " -- ".to_string() })
            .collect();
        println!("  [{:.0},{:.0}] (class {:.0}) → hidden: [{}]",
            x[0], x[1], target, active.join(", "));
    }

    println!();
    println!("  The hidden layer maps XOR inputs to a space where a");
    println!("  single linear boundary can separate the classes.");

    // === Decision boundary visualization ===
    println!("\n=== Learned Decision Boundary ===\n");

    let grid_size = 20;
    println!("  x2");
    for row in (0..grid_size).rev() {
        let x2 = row as f64 / (grid_size - 1) as f64;
        let mut line = String::new();
        for col in 0..grid_size {
            let x1 = col as f64 / (grid_size - 1) as f64;
            let pred = forward_all(&[x1, x2], &mut layer1, &mut layer2, &mut layer3);

            // Check if data point
            let mut is_data = false;
            for (dx, dt) in &xor_data {
                if ((dx[0] - x1).abs() < 0.08) && ((dx[1] - x2).abs() < 0.08) {
                    line.push(if *dt > 0.5 { '1' } else { '0' });
                    is_data = true;
                    break;
                }
            }
            if !is_data {
                if pred > 0.45 && pred < 0.55 {
                    line.push('/');
                } else if pred >= 0.5 {
                    line.push('+');
                } else {
                    line.push('.');
                }
            }
        }
        if row == grid_size - 1 || row == 0 {
            println!("  {:>3.1}|{}", x2, line);
        } else {
            println!("     |{}", line);
        }
    }
    println!("     +{}", "-".repeat(grid_size));
    println!("      0.0{}x1{}1.0",
        " ".repeat(grid_size / 2 - 4), " ".repeat(grid_size / 2 - 4));

    println!();
    println!("Key insight: Multiple layers enable non-linear decision boundaries.");
    println!("The hidden layer transforms inputs into a linearly separable space.");
}
```

---

## Key Takeaways

- A multi-layer perceptron stacks layers of neurons with non-linear activations to learn complex, non-linear patterns.
- Hidden layers learn intermediate representations that transform the input into a space where the problem becomes linearly separable.
- Backpropagation applies the chain rule layer by layer, propagating gradients from the output back to the input.
- An MLP with at least one hidden layer can approximate any continuous function (Universal Approximation Theorem) — but learning that function requires good training.
