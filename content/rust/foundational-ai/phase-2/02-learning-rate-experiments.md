# Learning Rate Experiments

> Phase 2 — Optimization | Kata 2.2

---

## Concept & Intuition

### What problem are we solving?

The learning rate is the most important hyperparameter in gradient descent. It controls how big a step the optimizer takes at each iteration. Too small, and training takes forever — the model inches toward the minimum. Too large, and the model overshoots the minimum, bouncing back and forth or diverging entirely. The right learning rate depends on the loss landscape and changes throughout training.

Finding a good learning rate is often the first thing you tune when training a model. A common strategy is to try several orders of magnitude (0.1, 0.01, 0.001, 0.0001) and see which converges fastest without diverging. More sophisticated approaches include learning rate schedules (reducing the rate over time) and adaptive methods (like Adam, which adjusts the rate per parameter).

Understanding learning rate behavior builds intuition for optimization. When you see a loss curve that oscillates wildly, you know the learning rate is too high. When you see a loss curve that barely moves, the rate is too low. When you see steady, smooth decrease that levels off, you have found a good rate.

### Why naive approaches fail

A fixed learning rate is almost never optimal throughout training. At the start, when parameters are far from optimal, you want large steps to make progress. Near the end, when you are close to the minimum, you want small steps to fine-tune. A single rate that is good for the beginning is too large for the end, and vice versa. This is why learning rate schedules are essential for practical training.

### Mental models

- **Walking in fog**: You are on a hillside in thick fog and want to reach the valley. Small steps are safe but slow. Large steps cover ground but might walk off a cliff. The learning rate is your stride length.
- **Goldilocks zone**: There is a range of learning rates that converge. Below that range, progress is too slow. Above it, divergence. The best rate is the largest one that still converges stably.
- **Learning rate as trust**: A small learning rate says "I don't trust the gradient much, so I'll be cautious." A large rate says "I fully trust this direction, let's go far."

### Visual explanations

```
  Learning rate effects:

  Too small:           Just right:          Too large:
  loss │               loss │               loss │
       │╲                   │╲                   │╲   ╱╲
       │ ╲                  │ ╲                  │ ╲ ╱  ╲ ╱
       │  ╲                 │  ╲                 │  ╲    ╲
       │   ╲╲               │   ╲                │   diverges!
       │     ╲╲╲            │    ╲___            │
       │        ╲╲╲╲        │                    │
       └──────── epoch      └──────── epoch      └──────── epoch
       Very slow converge   Fast convergence     Unstable / diverges
```

---

## Hands-on Exploration

1. Run gradient descent with multiple learning rates and compare convergence.
2. Observe divergence when the learning rate is too large.
3. Implement a simple learning rate schedule and see how it improves convergence.

---

## Live Code

```rust
fn main() {
    // === Learning Rate Experiments ===
    // The learning rate controls how big each optimization step is.

    // Dataset: y = 3x + 1
    let data: Vec<(f64, f64)> = vec![
        (1.0, 4.1), (2.0, 7.0), (3.0, 9.9), (4.0, 13.2),
        (5.0, 16.0), (6.0, 19.1), (7.0, 21.8), (8.0, 25.1),
        (9.0, 28.0), (10.0, 31.1),
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

    // === Experiment with different learning rates ===
    println!("=== Learning Rate Comparison ===\n");
    println!("  Target: w=3.0, b=1.0  Starting: w=0.0, b=0.0\n");

    let learning_rates = vec![0.0001, 0.001, 0.005, 0.01, 0.02, 0.05];
    let n_epochs = 300;

    let mut all_histories: Vec<(f64, Vec<f64>, f64, f64)> = Vec::new();

    for &lr in &learning_rates {
        let mut w = 0.0;
        let mut b = 0.0;
        let mut history: Vec<f64> = Vec::new();
        let mut diverged = false;

        for _epoch in 0..n_epochs {
            let loss = compute_loss(w, b);
            history.push(loss);

            if loss.is_nan() || loss > 1e10 {
                diverged = true;
                break;
            }

            let (gw, gb) = compute_gradients(w, b);
            w -= lr * gw;
            b -= lr * gb;
        }

        let final_loss = if diverged {
            f64::INFINITY
        } else {
            compute_loss(w, b)
        };

        all_histories.push((lr, history, w, b));

        let status = if diverged {
            "DIVERGED".to_string()
        } else if final_loss > 1.0 {
            format!("slow (loss={:.2})", final_loss)
        } else {
            format!("w={:.3}, b={:.3}, loss={:.4}", w, b, final_loss)
        };

        println!("  lr={:.4}: {}", lr, status);
    }

    println!();

    // === Loss curves comparison ===
    println!("=== Loss Curves (log scale) ===\n");
    let checkpoints = [0, 1, 5, 10, 25, 50, 100, 200, 299];

    // Header
    print!("  {:>6}", "epoch");
    for &lr in &learning_rates {
        print!(" {:>10}", format!("lr={}", lr));
    }
    println!();
    print!("  {:->6}", "");
    for _ in &learning_rates {
        print!(" {:->10}", "");
    }
    println!();

    for &epoch in &checkpoints {
        print!("  {:>6}", epoch);
        for (_, history, _, _) in &all_histories {
            if epoch < history.len() {
                let loss = history[epoch];
                if loss > 1e6 {
                    print!(" {:>10}", "INF");
                } else {
                    print!(" {:>10.2}", loss);
                }
            } else {
                print!(" {:>10}", "DIV");
            }
        }
        println!();
    }

    println!();

    // === Visual loss curves ===
    println!("=== Loss Over Time (selected learning rates) ===\n");

    let display_lrs = vec![0.001, 0.005, 0.01];
    for &target_lr in &display_lrs {
        if let Some((lr, history, _, _)) = all_histories.iter().find(|(lr, _, _, _)| (*lr - target_lr).abs() < 1e-6) {
            println!("  lr={}:", lr);
            let max_loss = history[0].max(1.0);
            for &epoch in &[0, 5, 10, 20, 50, 100, 200, 299] {
                if epoch < history.len() {
                    let loss = history[epoch];
                    let bar_len = ((loss / max_loss).sqrt() * 40.0) as usize;
                    let bar_len = bar_len.min(50);
                    println!("    epoch {:>4}: {:>8.3} |{}|",
                        epoch, loss, "█".repeat(bar_len));
                }
            }
            println!();
        }
    }

    // === Learning rate schedule ===
    println!("=== Learning Rate Schedule ===\n");
    println!("  Comparing fixed lr=0.01 vs step decay schedule:\n");

    // Fixed learning rate
    let mut w_fixed = 0.0;
    let mut b_fixed = 0.0;
    let mut history_fixed: Vec<f64> = Vec::new();

    // Step decay: start at 0.02, halve every 50 epochs
    let mut w_decay = 0.0;
    let mut b_decay = 0.0;
    let mut history_decay: Vec<f64> = Vec::new();

    for epoch in 0..n_epochs {
        // Fixed
        history_fixed.push(compute_loss(w_fixed, b_fixed));
        let (gw, gb) = compute_gradients(w_fixed, b_fixed);
        w_fixed -= 0.01 * gw;
        b_fixed -= 0.01 * gb;

        // Decay
        let lr_decay = 0.02 * (0.5_f64).powi((epoch / 50) as i32);
        history_decay.push(compute_loss(w_decay, b_decay));
        let (gw, gb) = compute_gradients(w_decay, b_decay);
        w_decay -= lr_decay * gw;
        b_decay -= lr_decay * gb;
    }

    println!("  {:>6} {:>12} {:>12} {:>12}",
        "epoch", "fixed(0.01)", "decay", "decay_lr");
    println!("  {:->6} {:->12} {:->12} {:->12}", "", "", "", "");

    for &epoch in &[0, 10, 50, 100, 150, 200, 250, 299] {
        let lr_at_epoch = 0.02 * (0.5_f64).powi((epoch / 50) as i32);
        println!("  {:>6} {:>12.4} {:>12.4} {:>12.5}",
            epoch, history_fixed[epoch], history_decay[epoch], lr_at_epoch);
    }

    println!();
    println!("  Fixed final: w={:.4}, b={:.4}, loss={:.6}",
        w_fixed, b_fixed, compute_loss(w_fixed, b_fixed));
    println!("  Decay final: w={:.4}, b={:.4}, loss={:.6}",
        w_decay, b_decay, compute_loss(w_decay, b_decay));
    println!();
    println!("  The decay schedule starts fast and fine-tunes near the end,");
    println!("  often achieving lower final loss than a fixed rate.\n");

    println!("Key insight: The learning rate is the most important hyperparameter.");
    println!("Too small → slow. Too large → divergence. Schedules → best of both worlds.");
}
```

---

## Key Takeaways

- The learning rate is the single most important hyperparameter: it determines whether gradient descent converges, how fast, and how precisely.
- Too small a learning rate means painfully slow convergence; too large means oscillation or divergence.
- The optimal learning rate depends on the loss landscape geometry — try several orders of magnitude to find the right range.
- Learning rate schedules (starting large and decaying over time) combine fast initial progress with precise final convergence.
