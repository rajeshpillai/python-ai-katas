# What Is Signal?

> Phase 0 — Foundations | Kata 0.4

---

## Concept & Intuition

### What problem are we solving?

Signal is the true, systematic pattern in data — the part that carries meaningful information. If noise is the fog, signal is the mountain range underneath. The entire goal of machine learning is to extract signal from data: to find the consistent, reproducible relationships that let us make predictions about unseen examples.

Signal can take many forms: a linear relationship between house size and price, a seasonal pattern in sales data, or a complex non-linear boundary separating two classes of images. What makes it "signal" is that it reflects a real relationship in the world — it is not an artifact of the particular sample you happened to collect.

The signal-to-noise ratio (SNR) determines how easy a learning problem is. High SNR means the pattern is clear and even simple models can find it. Low SNR means the pattern is buried under randomness, requiring more data, better features, or more sophisticated models to uncover. Knowing the SNR of your problem sets realistic expectations for what any model can achieve.

### Why naive approaches fail

Without understanding signal, you cannot tell whether a model is learning something real or just memorizing noise. A model with 100% training accuracy on noisy data is almost certainly overfitting. The signal is the part of the data that is consistent across different samples — if you retrain on new data and get very different results, you were fitting noise, not signal.

### Mental models

- **Signal is repeatable**: If you collect new data from the same process, the signal will be there again. Noise will be different. This is the basis for train/test splits.
- **Signal-to-noise ratio (SNR)**: Think of SNR like trying to hear a conversation at a party. High SNR = quiet room, easy to hear. Low SNR = loud party, you need to listen very carefully (more data, better features).
- **Signal has structure**: Noise is random and patternless. Signal has structure — correlations, trends, clusters. Learning algorithms are pattern-detectors.

### Visual explanations

```
  High SNR:                     Low SNR:
  (easy to learn)               (hard to learn)

  y │    . .  * * *             y │  .   *  .  *
    │   * *                       │ *  .  *  .  *
    │  * .                        │  . *  .  *
    │ * .                         │ * .  *  .  *
    │* .                          │ . *  .  *  .
    └───────── x                  └───────── x

  Signal clearly visible         Signal buried in noise
```

---

## Hands-on Exploration

1. Generate data with a known signal and varying amounts of noise.
2. Measure the signal-to-noise ratio and observe how it affects learnability.
3. Demonstrate that signal is consistent across samples while noise changes.

---

## Live Code

```rust
fn main() {
    // === What is signal? ===
    // Signal is the true pattern in data — the part that generalizes.

    // Pseudo-random number generator
    let mut seed: u64 = 12345;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // True signal: y = 0.5*x^2 + 2*x + 1 (a quadratic)
    let signal_fn = |x: f64| -> f64 { 0.5 * x * x + 2.0 * x + 1.0 };

    println!("=== Signal-to-Noise Ratio (SNR) ===\n");
    println!("  True signal: y = 0.5x² + 2x + 1\n");

    let n = 50;
    let x_values: Vec<f64> = (0..n).map(|i| -5.0 + 10.0 * i as f64 / (n - 1) as f64).collect();

    // Generate data at different noise levels and compute SNR
    for &noise_std in &[0.5, 2.0, 5.0, 10.0] {
        let signal_values: Vec<f64> = x_values.iter().map(|&x| signal_fn(x)).collect();
        let noisy_values: Vec<f64> = signal_values.iter()
            .map(|&s| s + rand_f64() * noise_std)
            .collect();

        // Signal power = variance of signal
        let sig_mean: f64 = signal_values.iter().sum::<f64>() / n as f64;
        let signal_power: f64 = signal_values.iter()
            .map(|s| (s - sig_mean) * (s - sig_mean))
            .sum::<f64>() / n as f64;

        // Noise power = variance of noise
        let noise: Vec<f64> = noisy_values.iter().zip(signal_values.iter())
            .map(|(obs, sig)| obs - sig)
            .collect();
        let noise_mean: f64 = noise.iter().sum::<f64>() / n as f64;
        let noise_power: f64 = noise.iter()
            .map(|n| (n - noise_mean) * (n - noise_mean))
            .sum::<f64>() / n as f64;

        let snr = signal_power / noise_power;
        let snr_db = 10.0 * snr.log10();

        let difficulty = if snr_db > 20.0 { "easy" }
            else if snr_db > 10.0 { "moderate" }
            else if snr_db > 3.0 { "hard" }
            else { "very hard" };

        println!("  noise_std={:>4.1}: SNR={:>6.1} ({:>5.1} dB) — {} to learn",
            noise_std, snr, snr_db, difficulty);
    }

    // === Demonstrate that signal is consistent across samples ===
    println!("\n=== Signal Is Consistent Across Samples ===\n");
    println!("  Fitting a line (y = ax + b) to two independent samples:\n");

    let noise_std = 3.0;

    for sample_id in 1..=3 {
        // Generate a sample
        let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter()
            .map(|&x| signal_fn(x) + rand_f64() * noise_std)
            .collect();

        // Fit a line using least squares: y = ax + b
        // a = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        // b = (Σy - a*Σx) / n
        let n_pts = xs.len() as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

        let a = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x);
        let b = (sum_y - a * sum_x) / n_pts;

        // Compute R² (how much variance is explained)
        let y_mean = sum_y / n_pts;
        let ss_tot: f64 = ys.iter().map(|y| (y - y_mean) * (y - y_mean)).sum();
        let ss_res: f64 = xs.iter().zip(ys.iter())
            .map(|(x, y)| { let pred = a * x + b; (y - pred) * (y - pred) })
            .sum();
        let r_squared = 1.0 - ss_res / ss_tot;

        println!("  Sample {}: y = {:.2}x + {:.2}  (R² = {:.3})",
            sample_id, a, b, r_squared);
    }

    println!("\n  The slope (a) stays roughly consistent — that's the signal.");
    println!("  The intercept (b) and R² vary a bit — that's the noise.\n");

    // === Extracting signal by averaging ===
    println!("=== Extracting Signal by Ensemble Averaging ===\n");

    let test_x = 3.0;
    let true_y = signal_fn(test_x);
    println!("  At x={:.1}, true y = {:.2}", test_x, true_y);

    let mut running_sum = 0.0;
    let trials = 200;
    let checkpoints = [1, 5, 10, 20, 50, 100, 200];
    let mut check_idx = 0;

    for t in 1..=trials {
        let noisy_y = true_y + rand_f64() * 5.0;
        running_sum += noisy_y;

        if check_idx < checkpoints.len() && t == checkpoints[check_idx] {
            let avg = running_sum / t as f64;
            let error = (avg - true_y).abs();
            let bar_len = (error * 5.0).min(30.0) as usize;
            let bar = "#".repeat(bar_len);
            println!("    After {:>4} observations: avg = {:>7.2}, error = {:>5.2}  |{}|",
                t, avg, error, bar);
            check_idx += 1;
        }
    }

    println!("\n  Signal emerges from noise when you have enough data.\n");

    // === Signal types ===
    println!("=== Types of Signal ===\n");

    let examples = [
        ("Linear",    "y = 3x + 2",         |x: f64| 3.0 * x + 2.0),
        ("Quadratic", "y = x² - 4x + 3",    |x: f64| x * x - 4.0 * x + 3.0),
        ("Periodic",  "y = sin(x)",          |x: f64| x.sin()),
        ("Step",      "y = sign(x)",         |x: f64| if x >= 0.0 { 1.0 } else { -1.0 }),
    ];

    for (name, formula, func) in &examples {
        let vals: Vec<f64> = (-5..=5).map(|i| func(i as f64)).collect();
        let vals_str: Vec<String> = vals.iter().map(|v| format!("{:>6.1}", v)).collect();
        println!("  {:>10} ({}): [{}]", name, formula, vals_str.join(", "));
    }

    println!();
    println!("Key insight: Signal is the true pattern that persists across samples.");
    println!("Your model's job: find the signal, ignore the noise.");
}
```

---

## Key Takeaways

- Signal is the true, systematic pattern in data — the part that reflects a real relationship and generalizes to new observations.
- The signal-to-noise ratio (SNR) determines how difficult a learning problem is: high SNR problems are easy, low SNR problems require more data or better features.
- Signal is consistent across independent samples; noise changes. This is why train/test splits work as a validation strategy.
- Averaging multiple noisy observations is one of the simplest ways to extract signal — more data literally reveals the truth.
