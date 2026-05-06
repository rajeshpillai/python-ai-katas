# Visualizing Loss Curves

> Phase 2 — Optimization | Kata 2.4

---

## Concept & Intuition

### What problem are we solving?

A loss curve is a plot of the training loss over time (epochs or iterations). It is the single most important diagnostic tool in machine learning. By reading loss curves, you can diagnose nearly every training problem: learning rate too high (oscillation), learning rate too low (slow decrease), overfitting (training loss drops while validation loss rises), underfitting (both losses plateau high), and more.

Loss curves tell a story about the training process. A smooth, rapid decrease that levels off means healthy convergence. A jagged, oscillating curve means the learning rate is too high or the batch size is too small. A curve that decreases then increases means overfitting. A curve that barely changes means the model is not learning — perhaps the features are uninformative or the model capacity is too low.

Comparing training and validation loss curves is especially powerful. The gap between them measures generalization: a small gap means the model generalizes well; a large gap means overfitting. The point where validation loss starts increasing while training loss continues to decrease is the ideal stopping point.

### Why naive approaches fail

Without loss curves, training is a black box. You set hyperparameters, wait, and hope for the best. Loss curves transform training from guesswork into science — they give you real-time feedback about what is working and what needs adjustment. Skipping this visualization means flying blind.

### Mental models

- **Loss curve as a heartbeat monitor**: Just as doctors read EKG traces to diagnose heart conditions, ML practitioners read loss curves to diagnose training problems.
- **Train vs. validation gap**: The gap between training and validation loss is the "overfitting thermometer." It should stay small and not grow over time.
- **Shape tells the story**: Exponential decay = healthy. Plateau = stuck. Oscillation = unstable. Increase = diverging or overfitting.

### Visual explanations

```
  Common loss curve patterns:

  Healthy:          Overfitting:       Too high lr:      Too low lr:
  loss │             loss │             loss │            loss │
       │╲ train          │╲ train          │╲╱╲╱╲          │╲
       │ ╲               │ ╲               │  ╲╱ ╲         │ ╲
       │  ╲─ valid       │  ╲──            │      ╲        │  ╲╲
       │   ───           │   ╱ valid       │       ╲       │    ╲╲
       │                 │  ╱              │        ╲      │      ╲╲╲╲
       └──── epoch       └──── epoch       └──── epoch     └───── epoch
```

---

## Hands-on Exploration

1. Train models with different hyperparameters and record loss curves.
2. Build ASCII loss curve visualizations.
3. Diagnose training problems by reading the shapes of loss curves.

---

## Live Code

```rust
fn main() {
    // === Visualizing Loss Curves ===
    // The loss curve is your primary diagnostic tool.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // Generate training and validation data
    // True function: y = 0.5x^2 - x + 2
    let true_fn = |x: f64| -> f64 { 0.5 * x * x - x + 2.0 };

    let n_train = 20;
    let n_val = 15;

    let x_train: Vec<f64> = (0..n_train).map(|i| -3.0 + 8.0 * i as f64 / (n_train - 1) as f64).collect();
    let y_train: Vec<f64> = x_train.iter().map(|&x| true_fn(x) + rand_f64() * 1.5).collect();

    let x_val: Vec<f64> = (0..n_val).map(|i| -3.0 + 8.0 * i as f64 / (n_val - 1) as f64).collect();
    let y_val: Vec<f64> = x_val.iter().map(|&x| true_fn(x) + rand_f64() * 1.5).collect();

    // Linear model: y = w*x + b
    let compute_loss = |xs: &[f64], ys: &[f64], w: f64, b: f64| -> f64 {
        let n = xs.len() as f64;
        xs.iter().zip(ys.iter())
            .map(|(&x, &y)| { let p = w * x + b; (y - p) * (y - p) })
            .sum::<f64>() / n
    };

    let compute_grads = |xs: &[f64], ys: &[f64], w: f64, b: f64| -> (f64, f64) {
        let n = xs.len() as f64;
        let mut gw = 0.0;
        let mut gb = 0.0;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let err = w * x + b - y;
            gw += 2.0 * err * x / n;
            gb += 2.0 * err / n;
        }
        (gw, gb)
    };

    // === ASCII loss curve plotter ===
    let plot_loss_curves = |train_losses: &[f64], val_losses: &[f64], title: &str, height: usize, width: usize| {
        println!("  {}\n", title);

        let all_losses: Vec<f64> = train_losses.iter().chain(val_losses.iter()).cloned().collect();
        let min_loss = all_losses.iter().cloned().fold(f64::INFINITY, f64::min).max(0.0);
        let max_loss = all_losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_loss - min_loss).max(0.01);

        let n_epochs = train_losses.len();

        // Downsample to fit width
        let sample_every = (n_epochs as f64 / width as f64).ceil() as usize;
        let sample_every = sample_every.max(1);

        let mut grid = vec![vec![' '; width]; height];

        // Plot training loss
        for col in 0..width {
            let epoch = (col * n_epochs / width).min(n_epochs - 1);
            if epoch < train_losses.len() {
                let loss = train_losses[epoch];
                let row = ((max_loss - loss) / range * (height - 1) as f64) as usize;
                let row = row.min(height - 1);
                grid[row][col] = 'T';
            }
        }

        // Plot validation loss
        for col in 0..width {
            let epoch = (col * n_epochs / width).min(n_epochs - 1);
            if epoch < val_losses.len() {
                let loss = val_losses[epoch];
                let row = ((max_loss - loss) / range * (height - 1) as f64) as usize;
                let row = row.min(height - 1);
                if grid[row][col] == 'T' {
                    grid[row][col] = 'X'; // overlap
                } else {
                    grid[row][col] = 'V';
                }
            }
        }

        // Print grid
        for (i, row) in grid.iter().enumerate() {
            let loss_at_row = max_loss - range * i as f64 / (height - 1) as f64;
            let line: String = row.iter().collect();
            if i == 0 || i == height / 2 || i == height - 1 {
                println!("    {:>7.2} |{}", loss_at_row, line);
            } else {
                println!("            |{}", line);
            }
        }
        println!("            +{}", "-".repeat(width));
        println!("             0{}epochs{}{}",
            " ".repeat(width / 3), " ".repeat(width / 3), n_epochs);
        println!("    T=train, V=validation, X=overlap\n");
    };

    // === Scenario 1: Healthy convergence ===
    println!("=== Loss Curve Scenarios ===\n");

    let mut w = 0.0;
    let mut b = 0.0;
    let lr = 0.008;
    let n_epochs = 200;

    let mut train_losses_1: Vec<f64> = Vec::new();
    let mut val_losses_1: Vec<f64> = Vec::new();

    for _epoch in 0..n_epochs {
        train_losses_1.push(compute_loss(&x_train, &y_train, w, b));
        val_losses_1.push(compute_loss(&x_val, &y_val, w, b));
        let (gw, gb) = compute_grads(&x_train, &y_train, w, b);
        w -= lr * gw;
        b -= lr * gb;
    }

    plot_loss_curves(&train_losses_1, &val_losses_1,
        "Scenario 1: Healthy Convergence (lr=0.008)", 12, 50);

    // === Scenario 2: Learning rate too high ===
    let mut w = 0.0;
    let mut b = 0.0;
    let lr_high = 0.03;

    let mut train_losses_2: Vec<f64> = Vec::new();
    let mut val_losses_2: Vec<f64> = Vec::new();

    for _epoch in 0..100 {
        let tl = compute_loss(&x_train, &y_train, w, b);
        let vl = compute_loss(&x_val, &y_val, w, b);
        if tl.is_nan() || tl > 1e6 { break; }
        train_losses_2.push(tl);
        val_losses_2.push(vl);
        let (gw, gb) = compute_grads(&x_train, &y_train, w, b);
        w -= lr_high * gw;
        b -= lr_high * gb;
    }

    plot_loss_curves(&train_losses_2, &val_losses_2,
        "Scenario 2: Learning Rate Too High (lr=0.03)", 12, 50);

    // === Scenario 3: Learning rate too low ===
    let mut w = 0.0;
    let mut b = 0.0;
    let lr_low = 0.0005;

    let mut train_losses_3: Vec<f64> = Vec::new();
    let mut val_losses_3: Vec<f64> = Vec::new();

    for _epoch in 0..n_epochs {
        train_losses_3.push(compute_loss(&x_train, &y_train, w, b));
        val_losses_3.push(compute_loss(&x_val, &y_val, w, b));
        let (gw, gb) = compute_grads(&x_train, &y_train, w, b);
        w -= lr_low * gw;
        b -= lr_low * gb;
    }

    plot_loss_curves(&train_losses_3, &val_losses_3,
        "Scenario 3: Learning Rate Too Low (lr=0.0005)", 12, 50);

    // === Diagnostic summary ===
    println!("=== Loss Curve Diagnostic Guide ===\n");
    println!("  Pattern               │ Diagnosis           │ Fix");
    println!("  ──────────────────────┼─────────────────────┼──────────────────────");
    println!("  Smooth rapid decrease  │ Healthy convergence │ Keep going");
    println!("  Wild oscillation       │ lr too high         │ Reduce learning rate");
    println!("  Very slow decrease     │ lr too low          │ Increase learning rate");
    println!("  Train↓ Val↑            │ Overfitting         │ More data, regularize");
    println!("  Both plateau high      │ Underfitting        │ More capacity, features");
    println!("  Sudden spike           │ Bad data/gradient   │ Gradient clipping");
    println!("  Staircase pattern      │ lr schedule active  │ Normal behavior");

    println!();
    println!("Key insight: Loss curves are the #1 diagnostic tool in ML.");
    println!("Learn to read them and you can diagnose almost any training problem.");
}
```

---

## Key Takeaways

- Loss curves plot training loss over time and are the most important diagnostic tool in machine learning.
- Comparing training vs. validation loss reveals overfitting: if validation loss increases while training loss decreases, the model is memorizing noise.
- The shape of the loss curve diagnoses specific problems: oscillation (lr too high), slow descent (lr too low), plateau (insufficient capacity).
- Always plot loss curves when training models — they transform optimization from guesswork into data-driven debugging.
