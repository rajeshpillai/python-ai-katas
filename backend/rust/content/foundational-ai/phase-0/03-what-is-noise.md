# What Is Noise?

> Phase 0 — Foundations | Kata 0.3

---

## Concept & Intuition

### What problem are we solving?

Noise is the random, unpredictable component in data that does not carry useful information about the underlying pattern. Every real-world measurement contains some noise: sensor imprecision, human error, environmental fluctuations, or inherent randomness in the process being measured. A thermometer might read 72.3 degrees when the true temperature is 72.0 — that 0.3 is noise.

The fundamental challenge of machine learning is to learn the signal (the true pattern) while ignoring the noise. If a model memorizes the noise, it will perform well on training data but poorly on new data — this is overfitting. If it ignores too much, it misses the signal — this is underfitting. Understanding noise is the first step toward building models that generalize.

Noise comes in many forms: measurement noise (instrument imprecision), sampling noise (your sample does not perfectly represent the population), and irreducible noise (genuine randomness in the process). The key insight is that noise is not an error to be fixed — it is a fundamental property of real data that must be understood and managed.

### Why naive approaches fail

If you treat every fluctuation in your data as meaningful, you will build a model that chases random variations. Fitting a high-degree polynomial to noisy data produces wild oscillations between data points. The model "explains" every bump in the training data but makes absurd predictions on new inputs. Recognizing noise means accepting that not every data point needs to be fit perfectly.

### Mental models

- **Signal + Noise**: Every observation = true_value + random_noise. Your job is to estimate the true_value while acknowledging that the noise exists.
- **Noise as fog**: Imagine looking at a mountain range through fog. The mountains (signal) are always there, but the fog (noise) obscures your view. More data is like waiting for the fog to thin — the mountains become clearer.
- **Noise floor**: There is a minimum level of noise you cannot eliminate. Trying to push accuracy below the noise floor leads to overfitting, not better predictions.

### Visual explanations

```
  True signal:          y = 2x + 1
                        │
  Observed data:        y = 2x + 1 + noise
                        │
    y │        . *          * = true signal
      │      *   .          . = noisy observation
      │    . *
      │  *   .
      │ .  *
      │*  .
      └──────────── x

  The dots scatter around the line.
  A good model finds the line, not the dots.
```

---

## Hands-on Exploration

1. Generate a clean signal (a known function) and add random noise to it.
2. Observe how increasing noise levels make the underlying pattern harder to detect.
3. See how averaging multiple noisy observations reveals the true signal.

---

## Live Code

```rust
fn main() {
    // === What is noise? ===
    // Noise is the random component that obscures the true signal.

    // Simple pseudo-random number generator (linear congruential)
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-1, 1]
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // True signal: y = 2x + 1
    let true_fn = |x: f64| -> f64 { 2.0 * x + 1.0 };

    let n = 20;
    let x_values: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();

    println!("=== Noise at Different Levels ===\n");
    println!("  True signal: y = 2x + 1\n");

    let noise_levels = [0.0, 0.5, 2.0, 5.0];

    for &noise_level in &noise_levels {
        let mut total_error = 0.0;
        let mut noisy_values: Vec<f64> = Vec::new();

        for &x in &x_values {
            let true_y = true_fn(x);
            let noise = rand_f64() * noise_level;
            let noisy_y = true_y + noise;
            noisy_values.push(noisy_y);
            total_error += (noisy_y - true_y).abs();
        }

        let mean_abs_error = total_error / n as f64;
        println!("  Noise level σ={:.1}: mean absolute error = {:.3}", noise_level, mean_abs_error);

        // Show a few sample points
        if noise_level == 2.0 {
            println!("    Sample points (noise_level=2.0):");
            for i in 0..5 {
                let true_y = true_fn(x_values[i]);
                println!("      x={:.1}: true={:.1}, observed={:.2}, error={:+.2}",
                    x_values[i], true_y, noisy_values[i], noisy_values[i] - true_y);
            }
        }
    }

    println!();

    // === Averaging reduces noise ===
    println!("=== Averaging Multiple Noisy Observations ===\n");
    println!("  Taking N measurements at x=5.0 (true y = {:.1}):\n", true_fn(5.0));

    let x_fixed = 5.0;
    let true_y = true_fn(x_fixed);
    let noise_std = 3.0;

    for &num_samples in &[1, 5, 10, 50, 100, 500] {
        let mut sum = 0.0;
        for _ in 0..num_samples {
            sum += true_y + rand_f64() * noise_std;
        }
        let average = sum / num_samples as f64;
        let error = (average - true_y).abs();

        let bar_len = (error * 5.0).min(40.0) as usize;
        let bar: String = "#".repeat(bar_len);

        println!("    N={:>4}: avg={:>7.2}, error={:>5.2}  |{}|",
            num_samples, average, error, bar);
    }

    println!();
    println!("  As N increases, the average converges to the true value.");
    println!("  This is the Law of Large Numbers in action.\n");

    // === ASCII visualization of signal vs noise ===
    println!("=== Visualizing Signal vs Noise ===\n");

    let plot_width = 50;
    let plot_height = 15;
    let x_min = 0.0_f64;
    let x_max = 5.0_f64;
    let y_min = 0.0_f64;
    let y_max = 15.0_f64;

    // Generate noisy points
    let mut points: Vec<(f64, f64)> = Vec::new();
    for i in 0..30 {
        let x = x_min + (x_max - x_min) * (i as f64 / 29.0);
        let y = true_fn(x) + rand_f64() * 2.0;
        points.push((x, y));
    }

    // Create grid
    let mut grid = vec![vec![' '; plot_width]; plot_height];

    // Plot true signal
    for col in 0..plot_width {
        let x = x_min + (x_max - x_min) * (col as f64 / (plot_width - 1) as f64);
        let y = true_fn(x);
        let row = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as i32;
        let row = (plot_height as i32 - 1 - row) as usize;
        if row < plot_height {
            grid[row][col] = '-';
        }
    }

    // Plot noisy points
    for (x, y) in &points {
        let col = ((x - x_min) / (x_max - x_min) * (plot_width - 1) as f64) as usize;
        let row = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as i32;
        let row = (plot_height as i32 - 1 - row) as usize;
        if row < plot_height && col < plot_width {
            grid[row][col] = '*';
        }
    }

    println!("    y");
    for row in &grid {
        let line: String = row.iter().collect();
        println!("    |{}", line);
    }
    println!("    +{} x", "-".repeat(plot_width));
    println!("    Legend: '-' = true signal, '*' = noisy observations");

    println!();
    println!("Key insight: Noise is unavoidable. The goal is to learn the signal, not memorize the noise.");
}
```

---

## Key Takeaways

- Noise is the random, unpredictable component in data that does not reflect the true underlying pattern.
- Every real-world dataset contains noise — from measurement error, sampling variability, or inherent randomness.
- Averaging multiple noisy observations reduces noise (Law of Large Numbers), revealing the true signal underneath.
- A model that fits noise instead of signal will fail on new data — this is the core motivation for regularization and validation.
