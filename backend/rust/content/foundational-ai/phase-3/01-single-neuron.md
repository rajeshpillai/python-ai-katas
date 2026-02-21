# Single Neuron

> Phase 3 — Artificial Neural Networks | Kata 3.1

---

## Concept & Intuition

### What problem are we solving?

A single artificial neuron is the fundamental building block of neural networks. It takes a vector of inputs, computes a weighted sum, adds a bias, and passes the result through an activation function. Mathematically: output = activation(w1*x1 + w2*x2 + ... + wn*xn + b). This is remarkably similar to linear regression, but the activation function introduces non-linearity, allowing the neuron to model non-linear relationships.

A single neuron with a sigmoid activation function is exactly logistic regression — one of the most widely used classification algorithms. With a step function activation, it is a perceptron — the original neural network from the 1950s. With a ReLU activation, it is a rectified linear unit. The activation function determines what kind of decision boundary the neuron can learn.

Understanding a single neuron deeply — its forward pass, its loss computation, its gradient calculation, and its weight update — is essential because a neural network is just many neurons connected together. If you understand one neuron, you understand the principle behind all of deep learning.

### Why naive approaches fail

A single neuron can only learn linear decision boundaries (or linearly separable patterns transformed by the activation function). It cannot solve XOR or any problem where the classes are not linearly separable. This limitation motivated the development of multi-layer networks. But the training algorithm for a single neuron (forward pass → loss → backward pass → update) is identical to training an entire network.

### Mental models

- **Neuron as a weighted vote**: Each input casts a vote weighted by its importance. The bias shifts the threshold for "firing." The activation function decides whether the total vote is strong enough.
- **Neuron as a linear classifier with a twist**: The weighted sum creates a hyperplane; the activation function shapes the output into a probability or binary decision.
- **Biological inspiration**: Like a biological neuron that fires when its inputs exceed a threshold, an artificial neuron activates when the weighted sum exceeds the bias threshold.

### Visual explanations

```
  Single neuron:

  x1 ──w1──╲
             ╲
  x2 ──w2────⊕──→ activation(Σ + b) ──→ output
             ╱
  x3 ──w3──╱
              ↑
              b (bias)

  Forward: z = w1*x1 + w2*x2 + w3*x3 + b
           output = activation(z)
```

---

## Hands-on Exploration

1. Implement a single neuron with configurable activation function.
2. Train it on a simple binary classification task using gradient descent.
3. Observe how the decision boundary evolves during training.

---

## Live Code

```rust
fn main() {
    // === Single Neuron ===
    // The fundamental building block of neural networks.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // Activation functions
    let sigmoid = |z: f64| -> f64 { 1.0 / (1.0 + (-z).exp()) };
    let sigmoid_deriv = |z: f64| -> f64 {
        let s = 1.0 / (1.0 + (-z).exp());
        s * (1.0 - s)
    };

    // === Single neuron structure ===
    // 2 inputs → 1 output (binary classification)
    let mut w = vec![rand_f64() * 0.5, rand_f64() * 0.5];
    let mut b = 0.0;

    println!("=== Single Neuron (Logistic Regression) ===\n");
    println!("  Architecture: 2 inputs → sigmoid → 1 output");
    println!("  Initial weights: w=[{:.3}, {:.3}], b={:.3}\n", w[0], w[1], b);

    // Dataset: simple linearly separable problem
    // Class 1: points in upper-right region
    // Class 0: points in lower-left region
    let data: Vec<(Vec<f64>, f64)> = vec![
        (vec![2.0, 3.0], 1.0),
        (vec![3.0, 2.5], 1.0),
        (vec![2.5, 4.0], 1.0),
        (vec![3.5, 3.0], 1.0),
        (vec![4.0, 3.5], 1.0),
        (vec![1.5, 3.5], 1.0),
        (vec![0.5, 1.0], 0.0),
        (vec![1.0, 0.5], 0.0),
        (vec![0.0, 1.5], 0.0),
        (vec![1.5, 0.0], 0.0),
        (vec![0.5, 0.5], 0.0),
        (vec![1.0, 1.5], 0.0),
    ];

    // Forward pass
    let forward = |x: &[f64], w: &[f64], b: f64| -> (f64, f64) {
        let z = x[0] * w[0] + x[1] * w[1] + b;
        let a = sigmoid(z);
        (z, a)
    };

    // Binary cross-entropy loss
    let bce = |pred: f64, target: f64| -> f64 {
        let p = pred.max(1e-15).min(1.0 - 1e-15);
        -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
    };

    // === Training loop ===
    let lr = 0.5;
    let n_epochs = 100;

    println!("=== Training ===\n");
    println!("  {:>5} {:>10} {:>8} {:>8} {:>8} {:>10}",
        "epoch", "loss", "w[0]", "w[1]", "b", "accuracy");
    println!("  {:->5} {:->10} {:->8} {:->8} {:->8} {:->10}", "", "", "", "", "", "");

    for epoch in 0..n_epochs {
        let mut total_loss = 0.0;
        let mut grad_w = vec![0.0, 0.0];
        let mut grad_b = 0.0;
        let mut correct = 0;

        for (x, target) in &data {
            // Forward
            let (z, pred) = forward(x, &w, b);

            // Loss
            total_loss += bce(pred, *target);

            // Accuracy
            let predicted_class = if pred >= 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.1 { correct += 1; }

            // Backward (gradient of BCE with sigmoid)
            // dL/dz = pred - target (elegant simplification!)
            let dz = pred - target;
            grad_w[0] += dz * x[0];
            grad_w[1] += dz * x[1];
            grad_b += dz;
        }

        let n = data.len() as f64;
        total_loss /= n;
        grad_w[0] /= n;
        grad_w[1] /= n;
        grad_b /= n;

        if epoch < 10 || epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  {:>5} {:>10.4} {:>8.3} {:>8.3} {:>8.3} {:>9.0}%",
                epoch, total_loss, w[0], w[1], b,
                correct as f64 / n * 100.0);
        }

        // Update
        w[0] -= lr * grad_w[0];
        w[1] -= lr * grad_w[1];
        b -= lr * grad_b;
    }

    println!();

    // === Show predictions ===
    println!("=== Predictions ===\n");
    println!("  {:>10} {:>6} {:>8} {:>10}",
        "input", "true", "pred", "correct?");
    println!("  {:->10} {:->6} {:->8} {:->10}", "", "", "", "");

    for (x, target) in &data {
        let (_, pred) = forward(x, &w, b);
        let predicted_class = if pred >= 0.5 { 1.0 } else { 0.0 };
        let correct = if (predicted_class - target).abs() < 0.1 { "yes" } else { "NO" };
        println!("  [{:.1},{:.1}] {:>6.0} {:>8.3} {:>10}",
            x[0], x[1], target, pred, correct);
    }

    // === Decision boundary ===
    println!("\n=== Decision Boundary ===\n");
    println!("  The neuron learned: {:.3}*x1 + {:.3}*x2 + {:.3} = 0", w[0], w[1], b);
    println!("  Points above this line → class 1, below → class 0\n");

    // ASCII visualization
    let grid_size = 20;
    println!("  x2");
    for row in (0..grid_size).rev() {
        let x2 = row as f64 * 5.0 / grid_size as f64;
        let mut line = String::new();
        for col in 0..grid_size {
            let x1 = col as f64 * 5.0 / grid_size as f64;
            let (_, pred) = forward(&[x1, x2], &w, b);

            // Check if a data point is near this position
            let mut is_data = false;
            for (dx, dt) in &data {
                if ((dx[0] - x1).abs() < 0.2) && ((dx[1] - x2).abs() < 0.2) {
                    line.push(if *dt > 0.5 { '●' } else { '○' });
                    is_data = true;
                    break;
                }
            }
            if !is_data {
                if pred > 0.48 && pred < 0.52 {
                    line.push('/');  // decision boundary
                } else if pred >= 0.5 {
                    line.push('+');
                } else {
                    line.push('.');
                }
            }
        }
        if row == grid_size - 1 || row == grid_size / 2 || row == 0 {
            println!("  {:>3.1}|{}", x2, line);
        } else {
            println!("     |{}", line);
        }
    }
    println!("     +{}", "-".repeat(grid_size));
    println!("      0.0{}x1{}5.0",
        " ".repeat(grid_size / 2 - 4), " ".repeat(grid_size / 2 - 4));
    println!("  Legend: ● = class 1, ○ = class 0, / = boundary, + = pred 1, . = pred 0");

    println!();
    println!("Key insight: A single neuron = linear boundary + activation function.");
    println!("Training adjusts weights to position the boundary between classes.");
}
```

---

## Key Takeaways

- A single neuron computes output = activation(w . x + b) — a weighted sum of inputs followed by a non-linear activation function.
- With sigmoid activation and cross-entropy loss, a single neuron is exactly logistic regression — a powerful binary classifier.
- The gradient of cross-entropy loss with sigmoid simplifies elegantly to (prediction - target), making the backward pass very clean.
- A single neuron can only learn linear decision boundaries — this limitation motivates multi-layer networks.
