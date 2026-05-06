# Vanishing Gradients

> Phase 3 — Artificial Neural Networks | Kata 3.4

---

## Concept & Intuition

### What problem are we solving?

The vanishing gradient problem is one of the most important challenges in training deep neural networks. During backpropagation, gradients are multiplied by the derivative of the activation function at each layer. If this derivative is consistently less than 1 (as it is for sigmoid, whose maximum derivative is 0.25), gradients shrink exponentially as they flow backward through the network. After just a few layers, the gradient becomes so small that the early layers barely update — they cannot learn.

This was the primary obstacle that prevented training deep networks for decades. A 10-layer network with sigmoid activations might have gradients in the first layer that are 0.25^10 = 0.0000001 times smaller than gradients in the last layer. The first layer effectively receives zero gradient signal — it is frozen while only the last few layers learn.

The discovery that ReLU activations largely solve this problem (because their derivative is 1 for positive inputs) was a breakthrough that enabled modern deep learning. Combined with better initialization methods (Xavier, He) and normalization techniques (batch norm), deep networks can now train hundreds of layers.

### Why naive approaches fail

Simply stacking more layers with sigmoid or tanh activations does not make a better model. It makes a worse one, because the additional layers cannot learn. The model has capacity it cannot use. This paradox — more layers leading to worse performance — confused researchers for years until the vanishing gradient problem was understood and addressed.

### Mental models

- **Telephone game**: The gradient signal is like a message passed through many people. Each sigmoid layer reduces the message volume by 75%. After 10 people, the message is inaudible.
- **Multiplicative shrinkage**: If you multiply 0.25 by itself repeatedly: 0.25, 0.0625, 0.0156, 0.004, 0.001, ... The gradient becomes negligible in just a few layers.
- **Gradient as water flow**: Sigmoid layers are narrow pipes that restrict flow. ReLU layers are wide-open pipes. Deep networks need wide pipes to let gradients flow to early layers.

### Visual explanations

```
  Gradient magnitude through layers (sigmoid):

  Layer:    10    9    8    7    6    5    4    3    2    1
  Grad:  |████|███|██ |█  |▌  |▎  |▏  |   |   |   |
          1.0  0.25 0.06 0.016 ...         → vanishingly small

  With ReLU:
  Layer:    10    9    8    7    6    5    4    3    2    1
  Grad:  |████|████|████|████|████|████|████|████|████|████|
          1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
```

---

## Hands-on Exploration

1. Build a deep network and measure gradient magnitudes at each layer.
2. Compare gradient flow with sigmoid vs. ReLU activations.
3. Observe how vanishing gradients prevent early layers from learning.

---

## Live Code

```rust
fn main() {
    // === Vanishing Gradients ===
    // Why deep networks with sigmoid activations cannot learn.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    println!("=== Vanishing Gradients Problem ===\n");

    // === Mathematical demonstration ===
    println!("--- Mathematical Foundation ---\n");
    println!("  Sigmoid derivative: max = 0.25 (at z=0)");
    println!("  In backpropagation, gradient is multiplied by this at each layer.\n");

    println!("  {:>6} {:>15} {:>15} {:>15}",
        "Layers", "Sigmoid grad", "Tanh grad", "ReLU grad");
    println!("  {:->6} {:->15} {:->15} {:->15}", "", "", "", "");

    for n_layers in [1, 2, 3, 5, 10, 20, 50] {
        let sig_grad = 0.25_f64.powi(n_layers as i32);
        let tanh_grad = 0.65_f64.powi(n_layers as i32); // typical tanh derivative
        let relu_grad = 1.0_f64.powi(n_layers as i32);

        println!("  {:>6} {:>15.2e} {:>15.2e} {:>15.2e}",
            n_layers, sig_grad, tanh_grad, relu_grad);
    }

    println!();
    println!("  After 10 sigmoid layers: gradient is ~0.0000001 of the output gradient!");
    println!("  Early layers receive essentially zero learning signal.\n");

    // === Simulate gradient flow through a deep network ===
    println!("--- Simulated Gradient Flow Through a Deep Network ---\n");

    let n_layers = 10;
    let layer_size = 8;

    // Initialize weights (Xavier initialization)
    let mut all_weights: Vec<Vec<Vec<f64>>> = Vec::new();
    for layer in 0..n_layers {
        let in_size = if layer == 0 { 4 } else { layer_size };
        let out_size = if layer == n_layers - 1 { 1 } else { layer_size };
        let scale = (2.0 / (in_size + out_size) as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..out_size)
            .map(|_| (0..in_size).map(|_| rand_f64() * scale).collect())
            .collect();
        all_weights.push(weights);
    }

    // Forward pass to get typical activations
    let input = vec![0.5, -0.3, 0.8, -0.1];

    let simulate_gradient_flow = |activation: &str| -> Vec<f64> {
        let mut activations: Vec<Vec<f64>> = vec![input.clone()];
        let mut pre_activations: Vec<Vec<f64>> = Vec::new();

        // Forward pass
        for layer in 0..n_layers {
            let prev = activations.last().unwrap();
            let weights = &all_weights[layer];
            let mut z: Vec<f64> = Vec::new();
            let mut a: Vec<f64> = Vec::new();

            for i in 0..weights.len() {
                let zi: f64 = weights[i].iter()
                    .zip(prev.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                z.push(zi);

                let ai = match activation {
                    "sigmoid" => 1.0 / (1.0 + (-zi).exp()),
                    "tanh" => zi.tanh(),
                    "relu" => if zi > 0.0 { zi } else { 0.01 * zi },
                    _ => zi,
                };
                a.push(ai);
            }
            pre_activations.push(z);
            activations.push(a);
        }

        // Backward pass: measure gradient magnitude at each layer
        let mut grad = vec![1.0]; // start with gradient = 1 at output
        let mut grad_magnitudes: Vec<f64> = Vec::new();

        for layer in (0..n_layers).rev() {
            let z = &pre_activations[layer];

            // Compute activation derivative
            let act_deriv: Vec<f64> = z.iter().map(|&zi| {
                match activation {
                    "sigmoid" => {
                        let s = 1.0 / (1.0 + (-zi).exp());
                        s * (1.0 - s)
                    },
                    "tanh" => 1.0 - zi.tanh() * zi.tanh(),
                    "relu" => if zi > 0.0 { 1.0 } else { 0.01 },
                    _ => 1.0,
                }
            }).collect();

            // Gradient magnitude = ||grad * act_deriv * weights||
            let weights = &all_weights[layer];
            let grad_norm: f64 = grad.iter()
                .enumerate()
                .map(|(i, g)| {
                    let d = if i < act_deriv.len() { act_deriv[i] } else { 1.0 };
                    (g * d).abs()
                })
                .sum::<f64>() / grad.len() as f64;

            grad_magnitudes.push(grad_norm);

            // Propagate gradient backward through weights
            let in_size = weights[0].len();
            let mut new_grad = vec![0.0; in_size];
            for i in 0..weights.len().min(grad.len()) {
                let d = if i < act_deriv.len() { act_deriv[i] } else { 1.0 };
                for j in 0..in_size {
                    new_grad[j] += grad[i] * d * weights[i][j];
                }
            }
            grad = new_grad;
        }

        grad_magnitudes.reverse();
        grad_magnitudes
    };

    // Compare activations
    let activations_to_test = ["sigmoid", "tanh", "relu"];

    for &act in &activations_to_test {
        let grad_mags = simulate_gradient_flow(act);

        println!("  {} activation — gradient magnitude per layer:", act);
        let max_grad = grad_mags.iter().cloned().fold(0.0_f64, f64::max).max(1e-20);

        for (i, &g) in grad_mags.iter().enumerate() {
            let bar_len = (g / max_grad * 40.0) as usize;
            let bar_len = bar_len.min(40).max(0);
            let bar = "█".repeat(bar_len);
            let layer_label = if i == 0 { "(input)" }
                else if i == n_layers - 1 { "(output)" }
                else { "" };
            println!("    layer {:>2}: {:.2e} |{:<40}| {}",
                i + 1, g, bar, layer_label);
        }

        let ratio = if grad_mags.last().unwrap_or(&1.0) > &0.0 {
            grad_mags.first().unwrap_or(&0.0) / grad_mags.last().unwrap_or(&1.0)
        } else {
            0.0
        };
        println!("    Ratio (first/last layer): {:.2e}\n", ratio);
    }

    // === Show how vanishing gradients affect learning ===
    println!("--- Effect on Learning Speed ---\n");
    println!("  Simulating weight updates across layers (sigmoid vs ReLU):\n");

    let base_grad = 1.0;
    println!("  {:>8} {:>15} {:>15} {:>15} {:>15}",
        "Layer", "Sig grad", "Sig update", "ReLU grad", "ReLU update");
    println!("  {:->8} {:->15} {:->15} {:->15} {:->15}", "", "", "", "", "");

    let lr = 0.01;
    for layer in (1..=10).rev() {
        let sig_grad = base_grad * 0.25_f64.powi((10 - layer) as i32);
        let relu_grad = base_grad * 1.0_f64.powi((10 - layer) as i32);
        let sig_update = lr * sig_grad;
        let relu_update = lr * relu_grad;

        println!("  {:>8} {:>15.2e} {:>15.2e} {:>15.2e} {:>15.2e}",
            layer, sig_grad, sig_update, relu_grad, relu_update);
    }

    // === Solutions ===
    println!("\n=== Solutions to Vanishing Gradients ===\n");
    println!("  1. ReLU activation:       derivative = 1 for z > 0 (no shrinkage)");
    println!("  2. Residual connections:   gradient can skip layers (ResNet)");
    println!("  3. Proper initialization:  Xavier/He init keeps variance stable");
    println!("  4. Batch normalization:    re-centers activations each layer");
    println!("  5. Gradient clipping:      prevents exploding (the opposite problem)");
    println!("  6. LSTM/GRU:              gated architectures for recurrent networks");

    println!();
    println!("Key insight: Sigmoid derivatives (max 0.25) multiply across layers,");
    println!("causing gradients to shrink exponentially. ReLU (derivative = 1)");
    println!("solves this, enabling training of deep networks.");
}
```

---

## Key Takeaways

- Vanishing gradients occur when activation derivatives are consistently less than 1, causing gradients to shrink exponentially with network depth.
- Sigmoid (max derivative 0.25) makes deep training nearly impossible: after 10 layers, gradients are ~10^-7 of their original magnitude.
- ReLU solves the vanishing gradient problem by having a derivative of exactly 1 for positive inputs, allowing gradient flow without shrinkage.
- Additional techniques like residual connections, proper initialization, and batch normalization further stabilize gradient flow in very deep networks.
