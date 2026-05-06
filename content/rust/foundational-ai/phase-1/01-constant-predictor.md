# Constant Predictor

> Phase 1 — What Does It Mean to Learn? | Kata 1.1

---

## Concept & Intuition

### What problem are we solving?

Before building sophisticated models, we need a baseline — the simplest possible prediction strategy. A constant predictor always outputs the same value regardless of the input. It completely ignores the features and just guesses a fixed number. This sounds useless, but it is profoundly important: it defines the floor that any useful model must beat.

The optimal constant prediction depends on the loss function. If you are minimizing squared error, the best constant is the mean of the target values. If you are minimizing absolute error, the best constant is the median. If you are maximizing accuracy for classification, the best constant is the most common class. Understanding why these are optimal requires understanding the mathematics of loss minimization — which is the foundation of all machine learning.

Every model you ever build should be compared against a constant predictor. If your fancy neural network cannot beat "just predict the average," something is wrong with your data, features, or implementation. The constant predictor is your sanity check.

### Why naive approaches fail

Some practitioners skip baselines and go straight to complex models. Without a baseline, you cannot tell if your model is actually learning patterns or just getting lucky. A model that achieves 90% accuracy sounds great — until you realize that 90% of the data belongs to one class, and a constant predictor achieves the same accuracy for free.

### Mental models

- **The "always guess the average" strategy**: If someone asks you to predict house prices and you know nothing about the house, your best guess is the average price. This is the constant predictor.
- **Baseline as a bar to clear**: Your model's value is measured by how much it improves over the baseline, not by its raw performance.
- **Constant predictor = zero learning**: It uses zero information from the input. Any model that uses features should do at least a little better.

### Visual explanations

```
  Constant predictor (mean):

  price │
        │  .        .
        │     .  .     .
        │──────────────────── predicted = mean
        │  .     .
        │     .     .
        └───────────────── sq_feet

  Every prediction is the same horizontal line.
  The errors are the vertical distances from each dot to the line.
```

---

## Hands-on Exploration

1. Compute the mean and median of a target variable.
2. Use each as a constant prediction and measure the resulting error.
3. Prove empirically that the mean minimizes squared error and the median minimizes absolute error.

---

## Live Code

```rust
fn main() {
    // === Constant Predictor ===
    // The simplest baseline: always predict the same value.

    // Dataset: house prices (our target)
    let prices = vec![
        250000.0, 320000.0, 195000.0, 410000.0, 275000.0,
        150000.0, 380000.0, 220000.0, 310000.0, 185000.0,
        425000.0, 290000.0, 175000.0, 345000.0, 260000.0,
    ];

    let n = prices.len() as f64;

    // Compute mean
    let mean: f64 = prices.iter().sum::<f64>() / n;

    // Compute median
    let mut sorted = prices.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    println!("=== Constant Predictor Baseline ===\n");
    println!("  Target: house prices ({} observations)", prices.len());
    println!("  Mean:   ${:.0}", mean);
    println!("  Median: ${:.0}\n", median);

    // Compare different constant predictions
    println!("=== Comparing Constant Predictions ===\n");
    println!("  {:>12} {:>14} {:>14} {:>14}",
        "Prediction", "MSE", "MAE", "Max Error");
    println!("  {:->12} {:->14} {:->14} {:->14}", "", "", "", "");

    let candidates = vec![
        ("Min", sorted[0]),
        ("Q1", sorted[sorted.len() / 4]),
        ("Median", median),
        ("Mean", mean),
        ("Q3", sorted[3 * sorted.len() / 4]),
        ("Max", sorted[sorted.len() - 1]),
    ];

    let mut best_mse = ("", f64::INFINITY);
    let mut best_mae = ("", f64::INFINITY);

    for (name, pred) in &candidates {
        let mse: f64 = prices.iter()
            .map(|p| (p - pred) * (p - pred))
            .sum::<f64>() / n;
        let mae: f64 = prices.iter()
            .map(|p| (p - pred).abs())
            .sum::<f64>() / n;
        let max_err: f64 = prices.iter()
            .map(|p| (p - pred).abs())
            .fold(0.0_f64, f64::max);

        if mse < best_mse.1 { best_mse = (name, mse); }
        if mae < best_mae.1 { best_mae = (name, mae); }

        println!("  {:>12} {:>14.0} {:>14.0} {:>14.0}", name, mse, mae, max_err);
    }

    println!();
    println!("  Best MSE:  {} (as expected, the MEAN minimizes squared error)", best_mse.0);
    println!("  Best MAE:  {} (as expected, the MEDIAN minimizes absolute error)", best_mae.0);

    // === Prove it: sweep over all possible constants ===
    println!("\n=== Proof: Mean Minimizes MSE ===\n");
    println!("  Sweeping prediction from ${:.0} to ${:.0}:\n", sorted[0], sorted[sorted.len()-1]);

    let steps = 20;
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    let mut min_mse = f64::INFINITY;
    let mut min_mse_pred = 0.0;

    for i in 0..=steps {
        let pred = lo + (hi - lo) * i as f64 / steps as f64;
        let mse: f64 = prices.iter()
            .map(|p| (p - pred) * (p - pred))
            .sum::<f64>() / n;

        if mse < min_mse {
            min_mse = mse;
            min_mse_pred = pred;
        }

        let bar_len = (mse / 1e9 * 40.0) as usize;
        let bar_len = bar_len.min(60);
        let marker = if (pred - mean).abs() < (hi - lo) / steps as f64 / 2.0 { " ← mean" } else { "" };
        println!("    ${:>7.0}: MSE={:>12.0}  |{}|{}",
            pred, mse, "█".repeat(bar_len), marker);
    }

    println!();
    println!("  Minimum MSE at prediction = ${:.0} (actual mean = ${:.0})", min_mse_pred, mean);

    // === Classification baseline ===
    println!("\n=== Classification Baseline: Most Frequent Class ===\n");

    let labels = vec!["cat", "cat", "cat", "dog", "cat", "dog", "cat", "cat", "dog", "cat"];
    let n_labels = labels.len();
    let cat_count = labels.iter().filter(|&&l| l == "cat").count();
    let dog_count = n_labels - cat_count;

    println!("  Labels: {} cats, {} dogs", cat_count, dog_count);
    println!("  Most frequent class: cat ({:.0}%)", cat_count as f64 / n_labels as f64 * 100.0);
    println!("  Constant predictor accuracy: {:.0}%",
        cat_count.max(dog_count) as f64 / n_labels as f64 * 100.0);
    println!("  Any useful model must beat this baseline.\n");

    println!("Key insight: The constant predictor is your minimum bar.");
    println!("If your model can't beat 'predict the average,' it learned nothing.");
}
```

---

## Key Takeaways

- A constant predictor always outputs the same value — it is the simplest possible baseline and uses zero information from the input features.
- The mean minimizes mean squared error (MSE); the median minimizes mean absolute error (MAE). These are mathematically provable facts.
- Every model should be compared against the constant predictor baseline — improvement over this baseline is the true measure of learning.
- For classification, the baseline is predicting the most frequent class — if your model cannot beat this, it has not learned to discriminate between classes.
