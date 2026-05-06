# What Is a Feature?

> Phase 0 — Foundations | Kata 0.2

---

## Concept & Intuition

### What problem are we solving?

A feature is a measurable property of the data that we use as input to a model. If data is a table, features are the columns we feed to the algorithm. Choosing the right features is often more important than choosing the right algorithm — the quality of your inputs determines the ceiling for your outputs.

Feature engineering is the process of transforming raw data into features that better represent the underlying patterns. A raw timestamp is hard for a model to use, but extracting "hour of day" or "day of week" gives the model something meaningful. A street address is opaque, but the zip code or distance to city center is informative. Good features make the signal in your data accessible to the learning algorithm.

In Rust, features are typically represented as `f64` values in vectors. Since we work without external crates, we transform and compute features manually, which gives us a deep understanding of exactly what each feature represents and how it was derived.

### Why naive approaches fail

Using raw data without feature engineering forces the model to discover transformations on its own — if it can at all. Many models (especially linear ones) cannot learn non-linear transformations. Feeding a model "date of birth" when it needs "age" means the model must implicitly learn subtraction, which wastes capacity and may not converge. Thoughtful feature engineering does the heavy lifting upfront.

### Mental models

- **Features as coordinates**: Each observation is a point in a high-dimensional space. Each feature is one axis. Good features spread the points out so that similar observations cluster together.
- **Features as questions**: Each feature answers one question about an observation. "How big is the house?" (sq_feet). "Where is it?" (zip_code). Better questions lead to better predictions.
- **Derived features amplify signal**: Combining raw features (e.g., price_per_sqft = price / sq_feet) can create features that are more directly related to the target.

### Visual explanations

```
  Raw data:              Engineered features:
  ┌──────────────┐       ┌──────────────────────────────────┐
  │ timestamp    │  ──→  │ hour_of_day, day_of_week, month  │
  │ address      │  ──→  │ zip_code, dist_to_center         │
  │ sq_feet      │  ──→  │ sq_feet, log_sq_feet             │
  │ price        │  ──→  │ price, price_per_sqft            │
  └──────────────┘       └──────────────────────────────────┘
  Hard for model          Easy for model to use
```

---

## Hands-on Exploration

1. Start with raw observations and identify which columns are useful features.
2. Create derived features by combining existing ones (ratios, differences, logs).
3. Compare how well raw vs. engineered features correlate with the target variable.

---

## Live Code

```rust
fn main() {
    // === What is a feature? ===
    // Features are the measurable properties we feed to a model.

    // Raw dataset: [sq_feet, bedrooms, lot_size_acres, year_built, price]
    let raw_data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 0.25, 1985.0, 250000.0],
        vec![1800.0, 4.0, 0.50, 2001.0, 320000.0],
        vec![1100.0, 2.0, 0.15, 1972.0, 195000.0],
        vec![2200.0, 5.0, 1.00, 2015.0, 410000.0],
        vec![1600.0, 3.0, 0.30, 1990.0, 275000.0],
        vec![900.0,  1.0, 0.10, 1965.0, 150000.0],
        vec![2000.0, 4.0, 0.75, 2010.0, 380000.0],
        vec![1300.0, 2.0, 0.20, 1978.0, 220000.0],
    ];

    let current_year = 2026.0;

    println!("=== Raw Features vs Engineered Features ===\n");

    // Engineer new features
    let mut engineered: Vec<Vec<f64>> = Vec::new();
    let eng_names = [
        "sq_feet", "bedrooms", "price_per_sqft", "age_years",
        "sqft_per_bed", "log_sqft", "price"
    ];

    println!("  {:>12} {:>10} {:>14} {:>10} {:>12} {:>10} {:>10}",
        eng_names[0], eng_names[1], eng_names[2], eng_names[3],
        eng_names[4], eng_names[5], eng_names[6]);
    println!("  {:->12} {:->10} {:->14} {:->10} {:->12} {:->10} {:->10}",
        "", "", "", "", "", "", "");

    for row in &raw_data {
        let sq_feet = row[0];
        let bedrooms = row[1];
        let price = row[4];
        let year_built = row[3];

        // Derived features
        let price_per_sqft = price / sq_feet;
        let age_years = current_year - year_built;
        let sqft_per_bedroom = sq_feet / bedrooms;
        let log_sqft = sq_feet.ln();

        println!("  {:>12.0} {:>10.0} {:>14.2} {:>10.0} {:>12.1} {:>10.2} {:>10.0}",
            sq_feet, bedrooms, price_per_sqft, age_years, sqft_per_bedroom, log_sqft, price);

        engineered.push(vec![
            sq_feet, bedrooms, price_per_sqft, age_years,
            sqft_per_bedroom, log_sqft, price
        ]);
    }

    println!();

    // Compute correlation of each feature with price
    // Pearson correlation: r = Σ((x-μx)(y-μy)) / (n·σx·σy)
    let n = engineered.len() as f64;
    let price_col: Vec<f64> = engineered.iter().map(|r| r[6]).collect();
    let price_mean: f64 = price_col.iter().sum::<f64>() / n;
    let price_std: f64 = (price_col.iter()
        .map(|p| (p - price_mean) * (p - price_mean))
        .sum::<f64>() / n).sqrt();

    println!("=== Feature Correlations with Price ===\n");

    for feat_idx in 0..6 {
        let feat_col: Vec<f64> = engineered.iter().map(|r| r[feat_idx]).collect();
        let feat_mean: f64 = feat_col.iter().sum::<f64>() / n;
        let feat_std: f64 = (feat_col.iter()
            .map(|x| (x - feat_mean) * (x - feat_mean))
            .sum::<f64>() / n).sqrt();

        let covariance: f64 = feat_col.iter().zip(price_col.iter())
            .map(|(x, y)| (x - feat_mean) * (y - price_mean))
            .sum::<f64>() / n;

        let correlation = if feat_std * price_std > 0.0 {
            covariance / (feat_std * price_std)
        } else {
            0.0
        };

        let bar_len = (correlation.abs() * 30.0) as usize;
        let bar: String = if correlation >= 0.0 {
            "+".repeat(bar_len)
        } else {
            "-".repeat(bar_len)
        };

        println!("  {:>14}: r = {:>+6.3}  [{}]", eng_names[feat_idx], correlation, bar);
    }

    println!();

    // Feature scaling demonstration
    println!("=== Feature Scaling (Min-Max Normalization) ===\n");
    println!("  Without scaling, features with large ranges dominate distance calculations.\n");

    for feat_idx in [0, 1, 3] {
        let col: Vec<f64> = engineered.iter().map(|r| r[feat_idx]).collect();
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = col.iter().map(|x| (x - min) / (max - min)).collect();

        println!("  {:>14}: raw range [{:.0}, {:.0}] → scaled range [{:.2}, {:.2}]",
            eng_names[feat_idx], min, max,
            scaled.iter().cloned().fold(f64::INFINITY, f64::min),
            scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }

    println!();
    println!("Key insight: Good features make patterns accessible to the model.");
    println!("Derived features (ratios, logs, ages) often correlate better with the target.");
}
```

---

## Key Takeaways

- A feature is any measurable property used as input to a model — choosing good features is often more important than choosing a good algorithm.
- Feature engineering (creating derived features like ratios, logarithms, or ages) can dramatically improve model performance by making patterns explicit.
- Correlation analysis helps identify which features carry the most predictive signal for the target variable.
- Feature scaling ensures that no single feature dominates simply because of its numeric range.
