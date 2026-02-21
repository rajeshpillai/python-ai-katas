# Convergence vs Divergence

> Phase 2 — Optimization | Kata 2.3

---

## Concept & Intuition

### What problem are we solving?

Convergence means the optimization process is approaching a solution — the loss is decreasing and the parameters are stabilizing. Divergence means the opposite — the loss is increasing and the parameters are growing without bound. Understanding the boundary between convergence and divergence is critical for training models successfully.

For gradient descent on a quadratic loss surface, there is a mathematical condition for convergence: the learning rate must be less than 2/L, where L is the largest eigenvalue of the Hessian (the matrix of second derivatives). Intuitively, L measures the maximum curvature of the loss surface. Steep curvature means the gradient changes rapidly, so large steps overshoot. Flat curvature means small gradients, so convergence is slow but stable.

Divergence is not always dramatic. Sometimes the loss oscillates around a value without decreasing — this is marginal divergence. Sometimes it slowly creeps upward. Sometimes it explodes to infinity in a few steps. Monitoring the loss curve is the primary diagnostic tool: healthy training shows smooth, monotonic decrease (possibly with small fluctuations in stochastic gradient descent).

### Why naive approaches fail

Without monitoring convergence, you might train for hours only to discover the model diverged early and produced garbage. Or you might stop training too early, before convergence, and get a suboptimal model. The loss curve tells you everything: is the model converging? How fast? Has it plateaued? Is it oscillating?

### Mental models

- **Convergence as a damped oscillation**: Good learning rates cause the parameters to oscillate with decreasing amplitude around the optimum, like a ball settling at the bottom of a bowl.
- **Divergence as resonance**: Too-large learning rates cause the parameters to oscillate with increasing amplitude, like pushing a swing at its natural frequency — each push adds energy.
- **Critical learning rate**: There is a sharp threshold. Below it, convergence. Above it, divergence. The threshold depends on the curvature of the loss surface.

### Visual explanations

```
  Convergence:              Divergence:
  loss │                    loss │           ╱
       │╲                        │         ╱
       │ ╲                       │   ╱╲  ╱
       │  ╲                      │  ╱  ╲╱
       │   ╲╲╲                   │ ╱
       │      ╲╲╲____           │╱
       └──────────── epoch      └──────────── epoch

  Parameters settle down       Parameters blow up
```

---

## Hands-on Exploration

1. Find the critical learning rate threshold for a simple problem.
2. Visualize parameter trajectories for converging vs. diverging runs.
3. Implement an early stopping criterion based on loss monitoring.

---

## Live Code

```rust
fn main() {
    // === Convergence vs Divergence ===
    // Understanding when gradient descent works and when it fails.

    // Dataset: y = 2x + 1
    let data: Vec<(f64, f64)> = vec![
        (1.0, 3.1), (2.0, 5.0), (3.0, 7.2), (4.0, 8.9),
        (5.0, 11.0), (6.0, 13.1), (7.0, 15.0), (8.0, 16.9),
    ];
    let n = data.len() as f64;

    let compute_loss = |w: f64, b: f64| -> f64 {
        data.iter()
            .map(|(x, y)| { let p = w * x + b; (y - p) * (y - p) })
            .sum::<f64>() / n
    };

    let compute_gradients = |w: f64, b: f64| -> (f64, f64) {
        let mut gw = 0.0;
        let mut gb = 0.0;
        for &(x, y) in &data {
            let err = w * x + b - y;
            gw += 2.0 * err * x / n;
            gb += 2.0 * err / n;
        }
        (gw, gb)
    };

    // === Find the critical learning rate ===
    println!("=== Finding the Critical Learning Rate ===\n");

    let test_lrs = vec![0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05];
    let n_epochs = 200;

    println!("  {:>8} {:>12} {:>12} {:>10}",
        "lr", "final_loss", "loss_trend", "status");
    println!("  {:->8} {:->12} {:->12} {:->10}", "", "", "", "");

    let mut critical_lr = 0.0;

    for &lr in &test_lrs {
        let mut w = 0.0;
        let mut b = 0.0;
        let mut losses: Vec<f64> = Vec::new();
        let mut diverged = false;

        for _epoch in 0..n_epochs {
            let loss = compute_loss(w, b);
            if loss.is_nan() || loss > 1e10 {
                diverged = true;
                break;
            }
            losses.push(loss);
            let (gw, gb) = compute_gradients(w, b);
            w -= lr * gw;
            b -= lr * gb;
        }

        let status;
        let trend;
        if diverged || losses.is_empty() {
            status = "DIVERGED";
            trend = "↑↑↑";
        } else {
            let last = losses.last().unwrap();
            let mid = &losses[losses.len() / 2];
            if *last < 0.1 {
                status = "converged";
                trend = "↓↓↓";
            } else if last < mid {
                status = "converging";
                trend = "↓↓";
            } else if (last - mid).abs() / mid < 0.1 {
                status = "oscillating";
                trend = "↔";
                if critical_lr == 0.0 { critical_lr = lr; }
            } else {
                status = "DIVERGING";
                trend = "↑↑";
                if critical_lr == 0.0 { critical_lr = lr; }
            }
        };

        let final_loss = if !losses.is_empty() {
            format!("{:.4}", losses.last().unwrap())
        } else {
            "INF".to_string()
        };

        println!("  {:>8.4} {:>12} {:>12} {:>10}", lr, final_loss, trend, status);
    }

    println!("\n  Critical learning rate ≈ {:.3}\n", critical_lr);

    // === Parameter trajectories ===
    println!("=== Parameter Trajectories ===\n");

    let scenarios = vec![
        ("Converging (lr=0.005)", 0.005),
        ("Oscillating (lr=0.02)", 0.02),
        ("Diverging (lr=0.05)", 0.05),
    ];

    for (name, lr) in &scenarios {
        println!("  {}:", name);
        let mut w = 0.0;
        let mut b = 0.0;

        for epoch in 0..30 {
            let loss = compute_loss(w, b);
            if loss > 1e6 || loss.is_nan() {
                println!("    epoch {:>3}: w={:>10}, b={:>10}, loss=DIVERGED", epoch, "INF", "INF");
                break;
            }

            if epoch < 10 || epoch % 5 == 0 {
                // ASCII position indicator for w
                let w_display = ((w + 1.0) / 4.0 * 30.0) as i32;
                let w_display = w_display.max(0).min(30) as usize;
                let pos_str = format!("{}●", " ".repeat(w_display));
                println!("    epoch {:>3}: w={:>7.3}, b={:>7.3}, loss={:>9.3}  |{:<31}|",
                    epoch, w, b, loss, pos_str);
            }

            let (gw, gb) = compute_gradients(w, b);
            w -= lr * gw;
            b -= lr * gb;
        }
        println!();
    }

    // === Early stopping implementation ===
    println!("=== Early Stopping ===\n");
    println!("  Stop training when the loss stops improving.\n");

    let lr = 0.008;
    let patience = 10; // stop if no improvement for this many epochs
    let min_delta = 0.0001; // minimum improvement to count

    let mut w = 0.0;
    let mut b = 0.0;
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0;
    let mut wait = 0;
    let mut stopped_epoch = 0;

    for epoch in 0..500 {
        let loss = compute_loss(w, b);

        if loss < best_loss - min_delta {
            best_loss = loss;
            best_epoch = epoch;
            wait = 0;
        } else {
            wait += 1;
        }

        if wait >= patience {
            stopped_epoch = epoch;
            break;
        }

        let (gw, gb) = compute_gradients(w, b);
        w -= lr * gw;
        b -= lr * gb;
    }

    println!("  Training stopped at epoch {} (patience={})", stopped_epoch, patience);
    println!("  Best loss {:.6} at epoch {}", best_loss, best_epoch);
    println!("  Final parameters: w={:.4}, b={:.4}\n", w, b);

    // === Convergence criteria ===
    println!("=== Convergence Criteria ===\n");
    println!("  1. Loss below threshold:     loss < epsilon");
    println!("  2. Loss not improving:       |loss_new - loss_old| < delta");
    println!("  3. Gradient near zero:       ||gradient|| < epsilon");
    println!("  4. Parameters stabilized:    ||params_new - params_old|| < delta");
    println!();

    // Demonstrate gradient norm criterion
    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.005;
    let grad_threshold = 0.01;

    println!("  Gradient norm convergence criterion (threshold={}):\n", grad_threshold);
    for epoch in 0..300 {
        let (gw, gb) = compute_gradients(w, b);
        let grad_norm = (gw * gw + gb * gb).sqrt();

        if epoch < 10 || epoch % 30 == 0 || grad_norm < grad_threshold {
            println!("    epoch {:>4}: ||∇L|| = {:.6}", epoch, grad_norm);
        }

        if grad_norm < grad_threshold {
            println!("    Converged! Gradient norm below threshold at epoch {}.", epoch);
            break;
        }

        w -= lr * gw;
        b -= lr * gb;
    }

    println!();
    println!("Key insight: Convergence means the loss is decreasing toward a minimum.");
    println!("Monitor loss curves, use early stopping, and check gradient norms.");
}
```

---

## Key Takeaways

- Convergence means the optimization is approaching a minimum (loss decreasing, parameters stabilizing); divergence means the loss is growing unboundedly.
- There is a critical learning rate threshold: below it, convergence; above it, divergence. This threshold depends on the curvature of the loss surface.
- Early stopping halts training when the loss stops improving, preventing wasted computation and potential numerical instability.
- Multiple convergence criteria exist: loss threshold, loss plateau, gradient norm, and parameter stability — use them together for robust training.
