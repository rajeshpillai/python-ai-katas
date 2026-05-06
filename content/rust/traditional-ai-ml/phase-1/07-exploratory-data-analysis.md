# Exploratory Data Analysis

> Phase 1 — Data Wrangling | Kata 1.07

---

## Concept & Intuition

### What problem are we solving?

Exploratory Data Analysis (EDA) is the process of investigating your dataset to discover patterns, spot anomalies, test hypotheses, and check assumptions before building any model. It is the bridge between "I have data" and "I understand my data." EDA combines statistical summaries with visualizations (here rendered as ASCII) to build intuition about feature distributions, relationships between variables, and potential problems.

Good EDA answers critical questions: Are features normally distributed or skewed? Are there strong correlations that suggest multicollinearity? Do certain features clearly separate the classes? Are there clusters or patterns that suggest natural groupings? Skipping EDA is like navigating without a map — you might eventually get somewhere, but you will waste time and miss important landmarks.

In this kata, we implement a comprehensive EDA toolkit: distribution analysis with histograms, correlation matrices, feature-target relationship analysis, and summary statistics. All visualizations are ASCII-based to work in any terminal.

### Why naive approaches fail

Jumping straight to model building without EDA often leads to poor results and wasted time. You might train a complex model only to discover that a simple correlation between two features explains most of the variance. Or you might not notice that a feature has a bimodal distribution that needs special handling. EDA is not optional overhead — it is a critical investment that pays dividends in model quality and development speed.

### Mental models

- **EDA as detective work**: You are looking for clues in the data. Each statistic and plot is evidence that helps you understand the data-generating process.
- **Distributions tell stories**: A normal distribution suggests random variation around a central value. A bimodal distribution suggests two distinct populations. Heavy skew suggests outliers or bounded quantities.
- **Correlation is not causation, but it is a useful clue**: Strong correlation between features suggests redundancy. Strong correlation between a feature and the target suggests predictive power.

### Visual explanations

```
  EDA Pipeline:

  Raw Data -> Summary Stats -> Distributions -> Correlations -> Feature-Target
      |            |                |                |              |
  "What do     "What are        "What shape      "Which         "Which features
   I have?"    the ranges?"     is the data?"    features       predict the
                                                  overlap?"      target?"
```

---

## Hands-on Exploration

1. Compute comprehensive summary statistics for all features.
2. Build ASCII histograms to visualize feature distributions.
3. Compute and display a full correlation matrix.
4. Analyze feature-target relationships to identify the most predictive features.

---

## Live Code

```rust
fn main() {
    println!("=== Exploratory Data Analysis ===\n");

    // Dataset: student performance
    let columns = vec![
        "hours_study", "hours_sleep", "attendance_pct", "prev_grade", "final_grade",
    ];

    let data: Vec<Vec<f64>> = vec![
        vec![1.0, 4.0, 60.0, 45.0, 38.0],
        vec![2.0, 5.0, 65.0, 50.0, 45.0],
        vec![2.5, 7.0, 70.0, 55.0, 52.0],
        vec![3.0, 6.0, 72.0, 58.0, 55.0],
        vec![3.5, 7.5, 78.0, 62.0, 60.0],
        vec![4.0, 8.0, 80.0, 65.0, 65.0],
        vec![4.5, 7.0, 82.0, 68.0, 68.0],
        vec![5.0, 7.5, 85.0, 70.0, 72.0],
        vec![5.5, 6.5, 88.0, 72.0, 75.0],
        vec![6.0, 8.0, 90.0, 75.0, 78.0],
        vec![6.5, 7.0, 88.0, 78.0, 80.0],
        vec![7.0, 7.5, 92.0, 80.0, 82.0],
        vec![7.5, 8.0, 90.0, 82.0, 85.0],
        vec![8.0, 6.5, 95.0, 85.0, 88.0],
        vec![8.5, 7.0, 93.0, 88.0, 90.0],
        vec![9.0, 8.5, 95.0, 90.0, 92.0],
        vec![2.0, 4.5, 55.0, 40.0, 35.0],
        vec![3.0, 5.0, 60.0, 48.0, 42.0],
        vec![6.0, 5.5, 75.0, 60.0, 62.0],
        vec![9.5, 7.5, 98.0, 92.0, 95.0],
    ];

    // Step 1: Summary Statistics
    println!("--- Summary Statistics ---");
    println!(
        "{:<16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Feature", "Count", "Mean", "Std", "Min", "Median", "Max"
    );
    println!("{}", "-".repeat(72));

    let n_cols = columns.len();
    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let n = col.len();
        let mean = col.iter().sum::<f64>() / n as f64;
        let std = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let med = median(&col);

        println!(
            "{:<16} {:>8} {:>8.2} {:>8.2} {:>8.1} {:>8.1} {:>8.1}",
            columns[j], n, mean, std, min, med, max
        );
    }

    // Step 2: ASCII Histograms
    println!("\n--- Feature Distributions (ASCII Histograms) ---");
    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        println!("\n{}", columns[j]);
        ascii_histogram(&col, 10);
    }

    // Step 3: Correlation Matrix
    println!("\n--- Correlation Matrix ---");
    let mut corr_matrix = vec![vec![0.0; n_cols]; n_cols];

    // Header
    print!("{:<16}", "");
    for col in &columns {
        let short: String = col.chars().take(10).collect();
        print!("{:>11}", short);
    }
    println!();
    println!("{}", "-".repeat(16 + 11 * n_cols));

    for i in 0..n_cols {
        let col_i: Vec<f64> = data.iter().map(|r| r[i]).collect();
        print!("{:<16}", columns[i]);
        for j in 0..n_cols {
            let col_j: Vec<f64> = data.iter().map(|r| r[j]).collect();
            let corr = correlation(&col_i, &col_j);
            corr_matrix[i][j] = corr;
            print!("{:>11.3}", corr);
        }
        println!();
    }

    // Step 4: Feature-Target Analysis
    let target_col = n_cols - 1;
    println!("\n--- Feature-Target Correlations (target: {}) ---", columns[target_col]);
    let target: Vec<f64> = data.iter().map(|r| r[target_col]).collect();

    let mut feature_corrs: Vec<(String, f64)> = Vec::new();
    for j in 0..n_cols - 1 {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let corr = correlation(&col, &target);
        feature_corrs.push((columns[j].to_string(), corr));
    }
    feature_corrs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    for (name, corr) in &feature_corrs {
        let bar_len = (corr.abs() * 30.0) as usize;
        let bar: String = "#".repeat(bar_len);
        let sign = if *corr >= 0.0 { "+" } else { "-" };
        println!("  {:<16} {:>7.3} {} {}", name, corr, sign, bar);
    }

    // Step 5: Skewness analysis
    println!("\n--- Skewness Analysis ---");
    for j in 0..n_cols {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let skew = skewness(&col);
        let interpretation = if skew.abs() < 0.5 {
            "approximately symmetric"
        } else if skew > 0.0 {
            "right-skewed (long right tail)"
        } else {
            "left-skewed (long left tail)"
        };
        println!("  {:<16} skewness={:>6.3}  {}", columns[j], skew, interpretation);
    }

    // Step 6: Detect potential multicollinearity
    println!("\n--- Multicollinearity Check (|corr| > 0.8) ---");
    let mut collinear_pairs = 0;
    for i in 0..n_cols {
        for j in (i + 1)..n_cols {
            if corr_matrix[i][j].abs() > 0.8 {
                println!(
                    "  WARNING: {} and {} have correlation {:.3}",
                    columns[i], columns[j], corr_matrix[i][j]
                );
                collinear_pairs += 1;
            }
        }
    }
    if collinear_pairs == 0 {
        println!("  No highly collinear pairs found.");
    }

    // Metrics
    let best_feature = &feature_corrs[0];
    kata_metric("n_samples", data.len() as f64);
    kata_metric("n_features", (n_cols - 1) as f64);
    kata_metric("best_feature_correlation", best_feature.1);
    kata_metric("collinear_pairs", collinear_pairs as f64);
}

fn median(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn skewness(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    let m3 = v.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
    m3
}

fn ascii_histogram(data: &[f64], n_bins: usize) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        println!("  All values are {:.2}", min);
        return;
    }

    let bin_width = range / n_bins as f64;
    let mut bins = vec![0usize; n_bins];

    for &val in data {
        let bin = ((val - min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        bins[bin] += 1;
    }

    let max_count = *bins.iter().max().unwrap();
    let bar_scale = if max_count > 0 { 30.0 / max_count as f64 } else { 1.0 };

    for (i, &count) in bins.iter().enumerate() {
        let lo = min + i as f64 * bin_width;
        let hi = lo + bin_width;
        let bar_len = (count as f64 * bar_scale) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  [{:>7.1}, {:>7.1}) {:>3} |{}", lo, hi, count, bar);
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- EDA is not optional — it is the foundation for informed modeling decisions. Always explore before you model.
- Summary statistics, distributions, and correlations each reveal different aspects of the data that guide feature selection and algorithm choice.
- Multicollinearity (highly correlated features) can destabilize linear models and should be detected early.
- The strongest feature-target correlations indicate where the predictive signal lives, helping prioritize feature engineering efforts.
