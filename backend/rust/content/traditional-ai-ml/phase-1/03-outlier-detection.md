# Outlier Detection

> Phase 1 — Data Wrangling | Kata 1.03

---

## Concept & Intuition

### What problem are we solving?

Outliers are data points that deviate significantly from the rest of the dataset. A house listed at $1 when similar houses sell for $300,000 is almost certainly a data entry error. A transaction of $50,000 on an account that typically sees $50 transactions might be fraud. Outliers can arise from measurement errors, data entry mistakes, natural variation, or genuinely rare events. Identifying them is crucial because many ML algorithms are sensitive to extreme values — a single outlier can dramatically shift a regression line.

The challenge is distinguishing between errors (which should be removed) and legitimate rare observations (which might be the most interesting data points). In fraud detection, the outliers are exactly what you want to find. In regression, they might be what you want to remove. Context determines the appropriate action.

In this kata, we implement three outlier detection methods: the Z-score method (assumes normal distribution), the IQR method (robust to non-normal distributions), and a simple isolation-based approach. We compare their behavior on different data distributions.

### Why naive approaches fail

Simply removing the top and bottom 5% of values (percentile trimming) is arbitrary and treats all distributions the same. A normally distributed feature and a heavily skewed feature have very different tail behaviors. Using only the mean and standard deviation (Z-score) works well for normal distributions but is itself influenced by outliers — a single extreme value inflates the standard deviation, making it harder to detect other outliers. Robust methods that use the median and interquartile range avoid this circular dependency.

### Mental models

- **Z-score as standardized distance**: A Z-score tells you how many standard deviations a point is from the mean. Beyond 3 standard deviations is a common outlier threshold.
- **IQR as a robust fence**: The IQR (Q3 - Q1) captures the middle 50% of data. Points beyond 1.5 * IQR from the quartiles are outliers. This method resists the influence of extreme values.
- **Outlier vs. anomaly**: "Outlier" implies the point does not belong. "Anomaly" implies the point is unusual but might be genuinely interesting. The same detection method serves both purposes; the interpretation differs.

### Visual explanations

```
  Normal Distribution with Outliers:

  Count
    |         ***
    |       **   **
    |     **       **
    |   **           **
    | **               **      x     x    <- outliers
    +--+---+---+---+---+---+---+---+---
         Q1   Med  Q3       1.5*IQR

  IQR = Q3 - Q1
  Lower fence = Q1 - 1.5 * IQR
  Upper fence = Q3 + 1.5 * IQR
```

---

## Hands-on Exploration

1. Generate a dataset with known outliers injected at specific positions.
2. Implement Z-score based outlier detection.
3. Implement IQR based outlier detection.
4. Compare which methods catch which outliers and measure false positive/negative rates.

---

## Live Code

```rust
fn main() {
    println!("=== Outlier Detection ===\n");

    // Generate normal-ish data with known outliers
    let mut data = vec![
        // Normal range: roughly 50-150
        52.0, 58.0, 61.0, 65.0, 70.0, 72.0, 75.0, 78.0, 80.0, 82.0,
        85.0, 87.0, 90.0, 92.0, 95.0, 98.0, 100.0, 102.0, 105.0, 108.0,
        110.0, 112.0, 115.0, 118.0, 120.0, 122.0, 125.0, 128.0, 132.0, 138.0,
    ];

    // Known outliers (indices 30-34)
    let outlier_indices: Vec<usize> = vec![30, 31, 32, 33, 34];
    data.push(5.0);    // extreme low
    data.push(250.0);  // extreme high
    data.push(300.0);  // extreme high
    data.push(-20.0);  // extreme low
    data.push(500.0);  // extreme high

    println!("Dataset: {} values ({} known outliers)\n", data.len(), outlier_indices.len());

    // Method 1: Z-Score
    println!("--- Method 1: Z-Score (threshold = 2.0) ---");
    let z_outliers = z_score_outliers(&data, 2.0);
    print_outlier_results(&data, &z_outliers, &outlier_indices);

    println!("\n--- Method 1b: Z-Score (threshold = 3.0) ---");
    let z_outliers_3 = z_score_outliers(&data, 3.0);
    print_outlier_results(&data, &z_outliers_3, &outlier_indices);

    // Method 2: IQR
    println!("\n--- Method 2: IQR (multiplier = 1.5) ---");
    let iqr_outliers = iqr_outliers(&data, 1.5);
    print_outlier_results(&data, &iqr_outliers, &outlier_indices);

    println!("\n--- Method 2b: IQR (multiplier = 3.0, extreme) ---");
    let iqr_outliers_3 = iqr_outliers(&data, 3.0);
    print_outlier_results(&data, &iqr_outliers_3, &outlier_indices);

    // Method 3: Modified Z-Score (using median, more robust)
    println!("\n--- Method 3: Modified Z-Score (MAD-based, threshold = 3.5) ---");
    let mad_outliers = mad_outliers(&data, 3.5);
    print_outlier_results(&data, &mad_outliers, &outlier_indices);

    // Test on skewed data
    println!("\n\n=== Skewed Distribution Test ===");
    let skewed: Vec<f64> = vec![
        1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0, 200.0,
    ];
    println!("Skewed data: {:?}\n", skewed);

    let z_skewed = z_score_outliers(&skewed, 2.0);
    let iqr_skewed = iqr_outliers(&skewed, 1.5);
    let mad_skewed = mad_outliers(&skewed, 3.5);

    println!("Z-Score outliers: {:?}", indices_to_values(&skewed, &z_skewed));
    println!("IQR outliers:     {:?}", indices_to_values(&skewed, &iqr_skewed));
    println!("MAD outliers:     {:?}", indices_to_values(&skewed, &mad_skewed));

    // Summary metrics
    kata_metric("z_score_detected", z_outliers.len() as f64);
    kata_metric("iqr_detected", iqr_outliers.len() as f64);
    kata_metric("mad_detected", mad_outliers.len() as f64);
}

fn z_score_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let mut outliers = Vec::new();
    for (i, &val) in data.iter().enumerate() {
        let z = (val - mean).abs() / std_dev;
        if z > threshold {
            outliers.push(i);
        }
    }
    outliers
}

fn iqr_outliers(data: &[f64], multiplier: f64) -> Vec<usize> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();

    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;

    let lower = q1 - multiplier * iqr;
    let upper = q3 + multiplier * iqr;

    println!("  Q1={:.1}, Q3={:.1}, IQR={:.1}, Lower={:.1}, Upper={:.1}",
        q1, q3, iqr, lower, upper);

    let mut outliers = Vec::new();
    for (i, &val) in data.iter().enumerate() {
        if val < lower || val > upper {
            outliers.push(i);
        }
    }
    outliers
}

fn mad_outliers(data: &[f64], threshold: f64) -> Vec<usize> {
    let med = median(data);

    // MAD = median of |x_i - median|
    let mut abs_devs: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = median(&abs_devs);

    // Modified Z-score: 0.6745 is the 0.75th quantile of the standard normal
    let factor = 0.6745;

    let mut outliers = Vec::new();
    for (i, &val) in data.iter().enumerate() {
        let modified_z = if mad > 0.0 {
            factor * (val - med).abs() / mad
        } else {
            0.0
        };
        if modified_z > threshold {
            outliers.push(i);
        }
    }
    outliers
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = p / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn print_outlier_results(data: &[f64], detected: &[usize], known: &[usize]) {
    let true_pos = detected.iter().filter(|i| known.contains(i)).count();
    let false_pos = detected.iter().filter(|i| !known.contains(i)).count();
    let false_neg = known.iter().filter(|i| !detected.contains(i)).count();

    let precision = if true_pos + false_pos > 0 {
        true_pos as f64 / (true_pos + false_pos) as f64
    } else {
        0.0
    };
    let recall = if true_pos + false_neg > 0 {
        true_pos as f64 / (true_pos + false_neg) as f64
    } else {
        0.0
    };

    println!("  Detected: {:?}", indices_to_values(data, detected));
    println!("  True positives: {}, False positives: {}, False negatives: {}",
        true_pos, false_pos, false_neg);
    println!("  Precision: {:.2}, Recall: {:.2}", precision, recall);
}

fn indices_to_values(data: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| data[i]).collect()
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Outliers can be errors to remove or valuable signals to investigate — context determines the appropriate response.
- Z-score detection works well for normally distributed data but is itself influenced by outliers (the mean and std dev shift).
- IQR-based detection is robust because the median and quartiles resist the influence of extreme values.
- The Modified Z-Score (MAD-based) combines the interpretability of Z-scores with the robustness of median-based statistics.
