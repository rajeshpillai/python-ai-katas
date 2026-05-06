# Missing Values

> Phase 1 — Data Wrangling | Kata 1.02

---

## Concept & Intuition

### What problem are we solving?

Real-world data is messy. Sensors malfunction, survey respondents skip questions, databases have null entries, and data merges create gaps. Missing values are ubiquitous, and how you handle them can make or break your model. Simply ignoring rows with missing data wastes potentially valuable information, while feeding missing values directly into an algorithm will either cause errors or produce garbage results.

There are three types of missingness. Missing Completely At Random (MCAR) means the probability of being missing is unrelated to any data. Missing At Random (MAR) means missingness depends on observed values (e.g., older patients are less likely to report weight). Missing Not At Random (MNAR) means missingness depends on the missing value itself (e.g., high-income people refuse to report income). Each type requires different handling strategies.

In this kata, we implement common strategies for detecting and handling missing values: deletion, mean/median imputation, and a more sophisticated approach using nearest neighbors. We use `f64::NAN` to represent missing values in our Rust implementation.

### Why naive approaches fail

Dropping every row with any missing value (listwise deletion) can dramatically reduce your dataset. If you have 20 features and each has 5% missing independently, you lose 64% of your data. Mean imputation is better but reduces variance and distorts correlations — every imputed value is the same, creating an artificial spike in the distribution. The right strategy depends on why data is missing and how much is missing.

### Mental models

- **Missing data as information loss**: Each missing value reduces the information content of your dataset. The goal of imputation is to recover as much information as possible without introducing bias.
- **Imputation as prediction**: Filling a missing value is fundamentally a prediction task. Mean imputation uses the simplest predictor; KNN imputation uses nearby data points; model-based imputation uses the full feature set.
- **The missing indicator trick**: Sometimes the fact that a value is missing is itself informative. Adding a binary column "feature_X_was_missing" preserves this signal.

### Visual explanations

```
  Original Data:          After Mean Imputation:    After KNN Imputation:
  +----+----+----+        +----+----+----+          +----+----+----+
  | 10 |  5 | 20 |        | 10 |  5 | 20 |          | 10 |  5 | 20 |
  | 12 | NA | 24 |  --->  | 12 | 4.3| 24 |    --->  | 12 | 5.5| 24 |
  |  8 |  3 | NA |        |  8 |  3 |22.7|          |  8 |  3 | 22 |
  | 11 |  5 | 24 |        | 11 |  5 | 24 |          | 11 |  5 | 24 |
  +----+----+----+        +----+----+----+          +----+----+----+

  Mean: uses column avg     KNN: uses similar rows' values
  (ignores row context)     (preserves local structure)
```

---

## Hands-on Exploration

1. Create a dataset with intentionally introduced missing values (NaN).
2. Detect missing values: which columns, how many, what percentage.
3. Implement mean and median imputation strategies.
4. Implement KNN imputation — fill missing values using the K nearest complete rows.
5. Compare the imputed datasets against the original complete data.

---

## Live Code

```rust
fn main() {
    println!("=== Missing Values: Detection and Imputation ===\n");

    // Original complete dataset (ground truth)
    let complete_data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 250000.0],
        vec![1600.0, 3.0, 310000.0],
        vec![1700.0, 4.0, 340000.0],
        vec![1100.0, 2.0, 180000.0],
        vec![2100.0, 4.0, 420000.0],
        vec![1500.0, 3.0, 275000.0],
        vec![1800.0, 4.0, 365000.0],
        vec![950.0,  2.0, 150000.0],
        vec![2400.0, 5.0, 500000.0],
        vec![1300.0, 3.0, 230000.0],
    ];

    let columns = vec!["sqft", "bedrooms", "price"];

    // Introduce missing values (NaN)
    let mut data_with_missing = complete_data.clone();
    data_with_missing[1][1] = f64::NAN; // bedrooms missing
    data_with_missing[2][2] = f64::NAN; // price missing
    data_with_missing[4][0] = f64::NAN; // sqft missing
    data_with_missing[7][1] = f64::NAN; // bedrooms missing
    data_with_missing[7][2] = f64::NAN; // price missing
    data_with_missing[9][0] = f64::NAN; // sqft missing

    // Step 1: Detect missing values
    println!("--- Missing Value Detection ---");
    print_data(&columns, &data_with_missing);

    println!("\nMissing value summary:");
    println!("{:<12} {:>8} {:>10}", "Column", "Missing", "Pct");
    println!("{}", "-".repeat(30));
    for (j, col) in columns.iter().enumerate() {
        let missing = data_with_missing.iter().filter(|row| row[j].is_nan()).count();
        let pct = missing as f64 / data_with_missing.len() as f64 * 100.0;
        println!("{:<12} {:>8} {:>9.1}%", col, missing, pct);
    }

    // Strategy 1: Listwise deletion
    println!("\n--- Strategy 1: Listwise Deletion ---");
    let deleted: Vec<Vec<f64>> = data_with_missing
        .iter()
        .filter(|row| !row.iter().any(|v| v.is_nan()))
        .cloned()
        .collect();
    println!("Rows remaining: {} / {} ({:.0}% data loss)",
        deleted.len(),
        data_with_missing.len(),
        (1.0 - deleted.len() as f64 / data_with_missing.len() as f64) * 100.0
    );
    print_data(&columns, &deleted);

    // Strategy 2: Mean imputation
    println!("\n--- Strategy 2: Mean Imputation ---");
    let mean_imputed = mean_impute(&data_with_missing);
    print_data(&columns, &mean_imputed);

    // Strategy 3: Median imputation
    println!("\n--- Strategy 3: Median Imputation ---");
    let median_imputed = median_impute(&data_with_missing);
    print_data(&columns, &median_imputed);

    // Strategy 4: KNN imputation (k=3)
    println!("\n--- Strategy 4: KNN Imputation (k=3) ---");
    let knn_imputed = knn_impute(&data_with_missing, 3);
    print_data(&columns, &knn_imputed);

    // Compare accuracy of imputation vs. ground truth
    println!("\n--- Imputation Accuracy (RMSE vs. ground truth) ---");
    let rmse_mean = imputation_rmse(&complete_data, &mean_imputed, &data_with_missing);
    let rmse_median = imputation_rmse(&complete_data, &median_imputed, &data_with_missing);
    let rmse_knn = imputation_rmse(&complete_data, &knn_imputed, &data_with_missing);

    println!("Mean imputation RMSE:   {:.2}", rmse_mean);
    println!("Median imputation RMSE: {:.2}", rmse_median);
    println!("KNN imputation RMSE:    {:.2}", rmse_knn);

    kata_metric("data_loss_deletion_pct",
        (1.0 - deleted.len() as f64 / data_with_missing.len() as f64) * 100.0);
    kata_metric("rmse_mean_imputation", rmse_mean);
    kata_metric("rmse_median_imputation", rmse_median);
    kata_metric("rmse_knn_imputation", rmse_knn);
}

fn mean_impute(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_cols = data[0].len();
    let mut means = vec![0.0; n_cols];

    for j in 0..n_cols {
        let valid: Vec<f64> = data.iter().map(|r| r[j]).filter(|v| !v.is_nan()).collect();
        means[j] = valid.iter().sum::<f64>() / valid.len() as f64;
    }

    data.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &v)| if v.is_nan() { means[j] } else { v })
                .collect()
        })
        .collect()
}

fn median_impute(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_cols = data[0].len();
    let mut medians = vec![0.0; n_cols];

    for j in 0..n_cols {
        let mut valid: Vec<f64> = data.iter().map(|r| r[j]).filter(|v| !v.is_nan()).collect();
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = valid.len();
        medians[j] = if n % 2 == 0 {
            (valid[n / 2 - 1] + valid[n / 2]) / 2.0
        } else {
            valid[n / 2]
        };
    }

    data.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &v)| if v.is_nan() { medians[j] } else { v })
                .collect()
        })
        .collect()
}

fn knn_impute(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n_rows = data.len();
    let n_cols = data[0].len();
    let mut result = data.to_vec();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if data[i][j].is_nan() {
                // Find k nearest neighbors using non-missing shared features
                let mut distances: Vec<(usize, f64)> = Vec::new();

                for other in 0..n_rows {
                    if other == i || data[other][j].is_nan() {
                        continue;
                    }
                    // Compute distance using shared non-missing features
                    let mut dist = 0.0;
                    let mut count = 0;
                    for f in 0..n_cols {
                        if f != j && !data[i][f].is_nan() && !data[other][f].is_nan() {
                            let diff = data[i][f] - data[other][f];
                            dist += diff * diff;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        distances.push((other, (dist / count as f64).sqrt()));
                    }
                }

                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // Average the values from k nearest neighbors
                let neighbors: Vec<f64> = distances
                    .iter()
                    .take(k)
                    .map(|(idx, _)| data[*idx][j])
                    .collect();

                if !neighbors.is_empty() {
                    result[i][j] = neighbors.iter().sum::<f64>() / neighbors.len() as f64;
                }
            }
        }
    }
    result
}

fn imputation_rmse(
    original: &[Vec<f64>],
    imputed: &[Vec<f64>],
    with_missing: &[Vec<f64>],
) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0;
    for i in 0..original.len() {
        for j in 0..original[0].len() {
            if with_missing[i][j].is_nan() {
                let diff = original[i][j] - imputed[i][j];
                sum_sq += diff * diff;
                count += 1;
            }
        }
    }
    if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    }
}

fn print_data(columns: &[&str], data: &[Vec<f64>]) {
    for col in columns {
        print!("{:>12}", col);
    }
    println!();
    println!("{}", "-".repeat(12 * columns.len()));
    for row in data {
        for val in row {
            if val.is_nan() {
                print!("{:>12}", "NaN");
            } else if *val == val.floor() && val.abs() < 1e6 {
                print!("{:>12.0}", val);
            } else {
                print!("{:>12.1}", val);
            }
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

- Missing values are inevitable in real data. Ignoring them silently leads to biased or broken models.
- Listwise deletion is simple but wasteful, especially when missingness is spread across many columns.
- Mean/median imputation is easy to implement but distorts the data distribution and underestimates variance.
- KNN imputation preserves local data structure by filling missing values based on similar observations, generally producing better results.
