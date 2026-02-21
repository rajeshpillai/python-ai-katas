# Manual Gradient Descent

> Phase 2 — Optimization | Kata 2.1

---

## Concept & Intuition

### What problem are we solving?

Gradient descent is the workhorse algorithm of machine learning. When the Normal Equation is too expensive (large datasets, many features) or the loss function is not amenable to a closed-form solution (neural networks), we need an iterative approach. Gradient descent starts at a random point on the error surface and repeatedly takes steps in the direction of steepest descent — the negative gradient.

The gradient of the loss function with respect to the parameters tells you two things: the direction of steepest increase and the steepness of the slope. By moving in the opposite direction (negative gradient) and scaling the step size by a learning rate, you iteratively approach the minimum. For convex loss functions (like MSE with linear regression), gradient descent is guaranteed to converge to the global minimum.

Computing gradients by hand is essential for understanding what gradient descent actually does. For MSE loss with a linear model y = wx + b, the gradients are: dL/dw = (-2/n) * sum(x_i * (y_i - pred_i)) and dL/db = (-2/n) * sum(y_i - pred_i). Each gradient measures how much the loss would change if you slightly increased that parameter.

### Why naive approaches fail

Without understanding gradients, you cannot debug optimization failures. If the learning rate is too large, gradient descent overshoots the minimum and diverges. If it is too small, convergence takes forever. If the features are not scaled, the gradient in one direction is much larger than another, causing zigzag paths. Manual gradient computation builds the intuition needed to diagnose these problems.

### Mental models

- **Rolling a ball downhill**: The gradient tells you which direction is downhill. The learning rate controls how big a step you take. Too big and you overshoot; too small and you barely move.
- **Gradient as compass**: At any point on the error surface, the gradient points toward steepest ascent. Negate it to point downhill. The magnitude tells you how steep the slope is.
- **Parameter update rule**: new_param = old_param - learning_rate * gradient. This single equation is the entire algorithm.

### Visual explanations

```
  Gradient descent on error surface:

  loss │
       │  ●                  ● = starting point
       │   ╲
       │    ╲                Each step follows
       │     ●               the negative gradient
       │      ╲
       │       ╲
       │        ●
       │         ╲
       │          ★          ★ = minimum
       └──────────────── parameter
```

---

## Hands-on Exploration

1. Compute gradients by hand for a simple linear model.
2. Implement the gradient descent update loop from scratch.
3. Track the loss at each iteration and observe convergence.

---

## Live Code

```rust
fn main() {
    // === Manual Gradient Descent ===
    // Iteratively move parameters toward the minimum loss.

    // Dataset: y = 2x + 3 + noise
    let data: Vec<(f64, f64)> = vec![
        (1.0, 5.2), (2.0, 6.8), (3.0, 9.1), (4.0, 10.9),
        (5.0, 13.2), (6.0, 15.0), (7.0, 16.8), (8.0, 19.1),
        (9.0, 21.3), (10.0, 23.0), (11.0, 24.9), (12.0, 27.2),
    ];

    let n = data.len() as f64;

    // Loss function: MSE
    let compute_loss = |w: f64, b: f64| -> f64 {
        data.iter()
            .map(|(x, y)| { let p = w * x + b; (y - p) * (y - p) })
            .sum::<f64>() / n
    };

    // Gradients (analytically derived)
    let compute_gradients = |w: f64, b: f64| -> (f64, f64) {
        let mut grad_w = 0.0;
        let mut grad_b = 0.0;
        for &(x, y) in &data {
            let pred = w * x + b;
            let error = pred - y;
            grad_w += 2.0 * error * x / n;
            grad_b += 2.0 * error / n;
        }
        (grad_w, grad_b)
    };

    // === Step-by-step gradient descent ===
    let mut w = 0.0; // start far from optimal
    let mut b = 0.0;
    let learning_rate = 0.005;
    let n_epochs = 200;

    println!("=== Manual Gradient Descent ===\n");
    println!("  Model: y = w*x + b");
    println!("  True parameters: w=2.0, b=3.0");
    println!("  Starting parameters: w={:.1}, b={:.1}", w, b);
    println!("  Learning rate: {}\n", learning_rate);

    println!("  {:>5} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "Epoch", "w", "b", "Loss", "∂L/∂w", "∂L/∂b");
    println!("  {:->5} {:->8} {:->8} {:->10} {:->10} {:->10}", "", "", "", "", "", "");

    let mut loss_history: Vec<f64> = Vec::new();

    for epoch in 0..=n_epochs {
        let loss = compute_loss(w, b);
        let (grad_w, grad_b) = compute_gradients(w, b);
        loss_history.push(loss);

        // Print selected epochs
        if epoch <= 10 || epoch % 20 == 0 || epoch == n_epochs {
            println!("  {:>5} {:>8.4} {:>8.4} {:>10.4} {:>+10.4} {:>+10.4}",
                epoch, w, b, loss, grad_w, grad_b);
        }

        // Update parameters
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;
    }

    println!();
    println!("  Final: w={:.4}, b={:.4} (true: w=2.0, b=3.0)\n", w, b);

    // === Visualize convergence ===
    println!("=== Loss Curve ===\n");

    let max_loss = loss_history[0];
    let checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200];
    for &epoch in &checkpoints {
        if epoch < loss_history.len() {
            let loss = loss_history[epoch];
            let bar_len = (loss / max_loss * 50.0) as usize;
            let bar_len = bar_len.min(50);
            println!("    epoch {:>4}: loss={:>8.4}  |{}|",
                epoch, loss, "█".repeat(bar_len));
        }
    }

    println!();

    // === Verify gradients numerically ===
    println!("=== Gradient Verification (Numerical vs Analytical) ===\n");

    let test_w = 1.5;
    let test_b = 2.0;
    let eps = 1e-5;

    let (analytical_gw, analytical_gb) = compute_gradients(test_w, test_b);
    let numerical_gw = (compute_loss(test_w + eps, test_b) - compute_loss(test_w - eps, test_b)) / (2.0 * eps);
    let numerical_gb = (compute_loss(test_w, test_b + eps) - compute_loss(test_w, test_b - eps)) / (2.0 * eps);

    println!("  At w={}, b={}:", test_w, test_b);
    println!("    ∂L/∂w: analytical={:+.6}, numerical={:+.6}, diff={:.2e}",
        analytical_gw, numerical_gw, (analytical_gw - numerical_gw).abs());
    println!("    ∂L/∂b: analytical={:+.6}, numerical={:+.6}, diff={:.2e}",
        analytical_gb, numerical_gb, (analytical_gb - numerical_gb).abs());
    println!("  The analytical and numerical gradients match!\n");

    // === Show the gradient descent path on error surface ===
    println!("=== Gradient Descent Path (w trajectory) ===\n");

    let mut w_path = 0.0;
    let mut b_path = 0.0;
    let path_points: Vec<(f64, f64)> = (0..=50).map(|i| {
        if i > 0 {
            let (gw, gb) = compute_gradients(w_path, b_path);
            w_path -= learning_rate * gw;
            b_path -= learning_rate * gb;
        }
        (w_path, b_path)
    }).collect();

    // ASCII plot of w over iterations
    let w_min = path_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let w_max = path_points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);

    for (i, (w_val, _)) in path_points.iter().enumerate() {
        if i % 5 == 0 || i <= 5 {
            let pos = ((w_val - w_min) / (w_max - w_min) * 40.0) as usize;
            let pos = pos.min(40);
            let line = format!("{}*", " ".repeat(pos));
            println!("    step {:>3}: w={:.3} |{:<41}|", i, w_val, line);
        }
    }

    println!();
    println!("Key insight: Gradient descent iteratively follows the slope downhill.");
    println!("new_param = old_param - learning_rate * gradient");
    println!("This simple rule powers all of deep learning.");
}
```

---

## Key Takeaways

- Gradient descent iteratively updates parameters by moving in the direction of steepest descent (negative gradient), scaled by a learning rate.
- The gradient measures how much the loss would change if you slightly increased each parameter — it is both a direction and a magnitude.
- Analytical gradients can (and should) be verified against numerical gradients computed via finite differences.
- The learning rate controls the step size: too large causes divergence, too small causes slow convergence.
