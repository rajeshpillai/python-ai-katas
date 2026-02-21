# Feature Scaling

> Phase 1 — Data Wrangling | Kata 1.05

---

## Concept & Intuition

### What problem are we solving?

Features in a dataset often have vastly different scales. House area might range from 500 to 5000 square feet while the number of bedrooms ranges from 1 to 6. If you feed these directly into a distance-based algorithm like KNN, the area feature will dominate simply because its values are much larger, not because it is more important. A 100 sqft difference and a 1 bedroom difference are treated very differently even though they might carry similar predictive power.

Feature scaling transforms all features to comparable ranges so that no single feature dominates due to its scale. The two most common approaches are min-max normalization (scaling to [0, 1]) and standardization (scaling to zero mean and unit variance). The choice depends on your algorithm and data distribution.

In this kata, we implement both scaling methods from scratch, apply them to a multi-feature dataset, and observe how scaling affects distance calculations — the fundamental operation in many ML algorithms.

### Why naive approaches fail

Without scaling, gradient-based algorithms (like linear regression with gradient descent) converge slowly because the loss surface becomes elongated in the direction of large-scale features. Distance-based algorithms (KNN, SVM, K-means) give disproportionate weight to high-magnitude features. Even PCA will be dominated by the feature with the largest variance, regardless of whether that variance is meaningful.

### Mental models

- **Min-max normalization**: Squeeze all values into [0, 1]. The smallest value maps to 0, the largest to 1. Good when you want bounded outputs and your data has no extreme outliers.
- **Standardization (Z-score)**: Center each feature at 0 with standard deviation 1. Good when your data is approximately normal and you want to compare relative deviations.
- **Robustness to outliers**: Min-max is sensitive to outliers (one extreme value compresses all others). Robust scaling uses the median and IQR instead.

### Visual explanations

```
  Before Scaling:                After Min-Max [0,1]:       After Standardization:
  sqft: [500, 5000]             sqft: [0.0, 1.0]           sqft: [-1.5, 1.5]
  beds: [1, 6]                  beds: [0.0, 1.0]           beds: [-1.5, 1.5]

  Distance(A,B) dominated       All features contribute     All features contribute
  by sqft                       equally by range             equally by variance

  Point A: (1000, 3)            (0.11, 0.40)                (-0.8, 0.0)
  Point B: (1500, 4)            (0.22, 0.60)                (-0.2, 0.5)
  dist = ~500 (sqft wins)       dist = 0.22                 dist = 0.78
```

---

## Hands-on Exploration

1. Create a multi-feature dataset with intentionally different scales.
2. Implement min-max normalization and standardization.
3. Compute distances between data points before and after scaling.
4. Observe how the relative importance of features changes with scaling.

---

## Live Code

```rust
fn main() {
    println!("=== Feature Scaling ===\n");

    // Dataset: features at very different scales
    let columns = vec!["sqft", "bedrooms", "age_years", "price"];
    let data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 10.0, 250000.0],
        vec![1600.0, 3.0, 5.0,  310000.0],
        vec![1700.0, 4.0, 8.0,  340000.0],
        vec![1100.0, 2.0, 25.0, 180000.0],
        vec![2100.0, 4.0, 2.0,  420000.0],
        vec![1500.0, 3.0, 15.0, 275000.0],
        vec![1800.0, 4.0, 3.0,  365000.0],
        vec![950.0,  2.0, 30.0, 150000.0],
    ];

    println!("--- Original Data ---");
    print_table(&columns, &data);

    // Show feature ranges
    println!("\n--- Feature Ranges (unscaled) ---");
    let n_cols = columns.len();
    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        println!("  {:<12} range: [{:.0}, {:.0}] (span: {:.0})", columns[j], min, max, range);
    }

    // Min-Max Normalization
    println!("\n--- Min-Max Normalization [0, 1] ---");
    let (minmax_data, minmax_params) = min_max_normalize(&data);
    print_table(&columns, &minmax_data);

    // Standardization (Z-score)
    println!("\n--- Standardization (Z-score) ---");
    let (std_data, std_params) = standardize(&data);
    print_table(&columns, &std_data);

    // Robust Scaling (median + IQR)
    println!("\n--- Robust Scaling (median + IQR) ---");
    let robust_data = robust_scale(&data);
    print_table(&columns, &robust_data);

    // Distance comparison
    println!("\n--- Distance Comparison (Point 0 vs Point 4) ---");
    let i = 0;
    let j = 4;

    let dist_raw = euclidean_distance(&data[i], &data[j]);
    let dist_minmax = euclidean_distance(&minmax_data[i], &minmax_data[j]);
    let dist_std = euclidean_distance(&std_data[i], &std_data[j]);
    let dist_robust = euclidean_distance(&robust_data[i], &robust_data[j]);

    println!("  Raw distance:     {:.2}", dist_raw);
    println!("  Min-Max distance: {:.4}", dist_minmax);
    println!("  Std distance:     {:.4}", dist_std);
    println!("  Robust distance:  {:.4}", dist_robust);

    // Feature contribution analysis
    println!("\n--- Feature Contribution to Distance ---");
    println!("  Raw data: which feature dominates?");
    for f in 0..n_cols {
        let diff = (data[i][f] - data[j][f]).abs();
        let pct = (diff * diff) / (dist_raw * dist_raw) * 100.0;
        println!("    {:<12}: diff={:.0}, contribution={:.1}%", columns[f], diff, pct);
    }

    println!("  After min-max: feature contributions");
    for f in 0..n_cols {
        let diff = (minmax_data[i][f] - minmax_data[j][f]).abs();
        let pct = (diff * diff) / (dist_minmax * dist_minmax) * 100.0;
        println!("    {:<12}: diff={:.3}, contribution={:.1}%", columns[f], diff, pct);
    }

    // Inverse transform verification
    println!("\n--- Inverse Transform Verification ---");
    let restored = min_max_inverse(&minmax_data[0], &minmax_params);
    println!("  Original:  {:?}", data[0]);
    println!("  Restored:  {:?}", restored.iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>());

    kata_metric("raw_distance", dist_raw);
    kata_metric("minmax_distance", dist_minmax);
    kata_metric("std_distance", dist_std);
    kata_metric("robust_distance", dist_robust);
}

fn min_max_normalize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<(f64, f64)>) {
    let n_cols = data[0].len();
    let mut params: Vec<(f64, f64)> = Vec::new();

    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        params.push((min, max));
    }

    let normalized = data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            let (min, max) = params[j];
            if (max - min).abs() < 1e-10 { 0.0 } else { (v - min) / (max - min) }
        }).collect()
    }).collect();

    (normalized, params)
}

fn min_max_inverse(scaled: &[f64], params: &[(f64, f64)]) -> Vec<f64> {
    scaled.iter().enumerate().map(|(j, &v)| {
        let (min, max) = params[j];
        v * (max - min) + min
    }).collect()
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<(f64, f64)>) {
    let n_cols = data[0].len();
    let n = data.len() as f64;
    let mut params: Vec<(f64, f64)> = Vec::new();

    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let mean = col.iter().sum::<f64>() / n;
        let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        params.push((mean, std));
    }

    let standardized = data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            let (mean, std) = params[j];
            if std.abs() < 1e-10 { 0.0 } else { (v - mean) / std }
        }).collect()
    }).collect();

    (standardized, params)
}

fn robust_scale(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_cols = data[0].len();
    let mut medians = Vec::new();
    let mut iqrs = Vec::new();

    for j in 0..n_cols {
        let mut col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = median_sorted(&col);
        let q1 = percentile_sorted(&col, 25.0);
        let q3 = percentile_sorted(&col, 75.0);
        medians.push(med);
        iqrs.push(q3 - q1);
    }

    data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            if iqrs[j].abs() < 1e-10 { 0.0 } else { (v - medians[j]) / iqrs[j] }
        }).collect()
    }).collect()
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn print_table(columns: &[&str], data: &[Vec<f64>]) {
    for col in columns {
        print!("{:>12}", col);
    }
    println!();
    println!("{}", "-".repeat(12 * columns.len()));
    for row in data {
        for val in row {
            print!("{:>12.4}", val);
        }
        println!();
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Features at different scales cause distance-based and gradient-based algorithms to give disproportionate weight to high-magnitude features.
- Min-max normalization maps values to [0, 1] — simple and bounded, but sensitive to outliers.
- Standardization (Z-score) centers features at mean 0 with unit variance — works well for normally distributed data.
- Robust scaling uses the median and IQR, making it resistant to outliers. Choose your scaling method based on your data distribution and algorithm requirements.
