# Anomaly Detection

> Phase 5 — Unsupervised Learning | Kata 5.06

---

## Concept & Intuition

### What problem are we solving?

Anomaly detection identifies data points that deviate significantly from the expected pattern. Unlike classification, you typically have abundant examples of "normal" behavior but few or no examples of anomalies. This makes it an unsupervised or semi-supervised problem. Applications include fraud detection, network intrusion detection, equipment failure prediction, and quality control.

The core idea is to model what "normal" looks like, then flag anything that falls far outside this model as anomalous. Different approaches define "normal" differently: statistical methods use probability distributions, distance-based methods use neighborhood density, and isolation-based methods measure how easily a point can be separated from the rest.

In this kata, we implement three anomaly detection approaches: statistical (Gaussian model), distance-based (Local Outlier Factor concept), and Isolation Forest (a tree-based method). We compare their ability to detect known anomalies injected into a dataset.

### Why naive approaches fail

Simple threshold rules (e.g., "flag anything above 3 standard deviations") work only for single-feature, normally distributed data. Real anomalies are often multivariate — a transaction might have a normal amount and a normal time individually, but the combination (large amount at unusual time) is anomalous. Multivariate anomaly detection requires modeling the joint distribution or using distance/isolation-based methods that naturally handle multiple features.

### Mental models

- **Anomaly as low probability**: Under a Gaussian model, anomalies fall in the tails of the distribution — regions where the probability density is very low.
- **Anomaly as isolation**: Anomalies are few and different. In a random tree, they are separated from the majority with fewer splits (shorter path length). Normal points require more splits.
- **Local vs. global anomaly**: A point might be globally normal (within the overall range) but locally anomalous (far from its nearest neighbors). The LOF-style approach captures this distinction.

### Visual explanations

```
  Normal data:                    With anomalies:

    . . . .                        . . . .
   . . . . .                      . . . . .
  . . . . . .                    . . . . . .    x  <- anomaly
   . . . . .                      . . . . .
    . . . .                        . . . .
                              x                  <- anomaly

  Gaussian: low P(x)          Isolation: short path
  Distance: far from others    All methods flag these
```

---

## Hands-on Exploration

1. Generate normal data and inject known anomalies.
2. Implement Gaussian anomaly detection using Mahalanobis distance.
3. Implement a simplified Isolation Forest.
4. Compare detection rates and false positive rates.

---

## Live Code

```rust
fn main() {
    println!("=== Anomaly Detection ===\n");

    // Generate normal data (2D Gaussian-like)
    let mut rng = 42u64;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut is_anomaly: Vec<bool> = Vec::new();

    // Normal points: centered at (5, 5)
    for _ in 0..80 {
        let x = 5.0 + randf(&mut rng) * 1.5;
        let y = 5.0 + randf(&mut rng) * 1.5;
        data.push(vec![x, y]);
        is_anomaly.push(false);
    }

    // Anomalies: scattered far from center
    let anomaly_points = vec![
        vec![0.5, 0.5], vec![9.5, 9.5], vec![0.5, 9.0],
        vec![9.0, 1.0], vec![5.0, 0.0], vec![5.0, 10.0],
        vec![0.0, 5.0], vec![10.0, 5.0],
    ];
    for p in &anomaly_points {
        data.push(p.clone());
        is_anomaly.push(true);
    }

    let n = data.len();
    let n_anomalies = is_anomaly.iter().filter(|&&a| a).count();
    println!("Dataset: {} points ({} normal, {} anomalies)\n", n, n - n_anomalies, n_anomalies);

    // Method 1: Gaussian (Mahalanobis distance)
    println!("--- Method 1: Gaussian Model ---");
    let gaussian_scores = gaussian_anomaly_scores(&data);
    let gaussian_threshold = find_threshold(&gaussian_scores, &is_anomaly, n_anomalies);
    let gaussian_preds: Vec<bool> = gaussian_scores.iter().map(|&s| s > gaussian_threshold).collect();
    print_detection_results("Gaussian", &is_anomaly, &gaussian_preds);

    // Method 2: K-distance (simplified LOF idea)
    println!("\n--- Method 2: K-Distance Anomaly Score ---");
    let k = 5;
    let kdist_scores = k_distance_scores(&data, k);
    let kdist_threshold = find_threshold(&kdist_scores, &is_anomaly, n_anomalies);
    let kdist_preds: Vec<bool> = kdist_scores.iter().map(|&s| s > kdist_threshold).collect();
    print_detection_results("K-Distance", &is_anomaly, &kdist_preds);

    // Method 3: Isolation Forest
    println!("\n--- Method 3: Isolation Forest ---");
    let n_trees = 50;
    let iso_scores = isolation_forest_scores(&data, n_trees, &mut rng);
    let iso_threshold = find_threshold(&iso_scores, &is_anomaly, n_anomalies);
    let iso_preds: Vec<bool> = iso_scores.iter().map(|&s| s > iso_threshold).collect();
    print_detection_results("Isolation Forest", &is_anomaly, &iso_preds);

    // Detailed anomaly scores
    println!("\n--- Anomaly Scores for Known Anomalies ---");
    println!("{:<8} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "Idx", "x", "y", "Gaussian", "K-Dist", "IsoForest");
    println!("{}", "-".repeat(56));
    for i in 0..n {
        if is_anomaly[i] {
            println!("{:<8} {:>8.2} {:>8.2} {:>10.3} {:>10.3} {:>10.3}",
                i, data[i][0], data[i][1],
                gaussian_scores[i], kdist_scores[i], iso_scores[i]);
        }
    }

    // Score distribution
    println!("\n--- Score Distribution ---");
    println!("  Gaussian scores:");
    println!("    Normal mean:   {:.3}", score_stats(&gaussian_scores, &is_anomaly, false).0);
    println!("    Anomaly mean:  {:.3}", score_stats(&gaussian_scores, &is_anomaly, true).0);
    println!("  K-Distance scores:");
    println!("    Normal mean:   {:.3}", score_stats(&kdist_scores, &is_anomaly, false).0);
    println!("    Anomaly mean:  {:.3}", score_stats(&kdist_scores, &is_anomaly, true).0);
    println!("  Isolation scores:");
    println!("    Normal mean:   {:.3}", score_stats(&iso_scores, &is_anomaly, false).0);
    println!("    Anomaly mean:  {:.3}", score_stats(&iso_scores, &is_anomaly, true).0);

    // Comparison
    println!("\n--- Method Comparison ---");
    println!("{:<20} {:>10} {:>10} {:>10}", "Method", "Precision", "Recall", "F1");
    println!("{}", "-".repeat(52));
    for (name, preds) in &[
        ("Gaussian", &gaussian_preds),
        ("K-Distance", &kdist_preds),
        ("Isolation Forest", &iso_preds),
    ] {
        let (prec, rec, f1) = detection_metrics(&is_anomaly, preds);
        println!("{:<20} {:>10.3} {:>10.3} {:>10.3}", name, prec, rec, f1);
    }

    let (_, _, f1_gauss) = detection_metrics(&is_anomaly, &gaussian_preds);
    let (_, _, f1_iso) = detection_metrics(&is_anomaly, &iso_preds);
    kata_metric("gaussian_f1", f1_gauss);
    kata_metric("isolation_forest_f1", f1_iso);
    kata_metric("n_anomalies", n_anomalies as f64);
    kata_metric("n_total", n as f64);
}

fn gaussian_anomaly_scores(data: &[Vec<f64>]) -> Vec<f64> {
    let n = data.len() as f64;
    let p = data[0].len();

    // Compute mean and covariance
    let mut mean = vec![0.0; p];
    for row in data { for j in 0..p { mean[j] += row[j]; } }
    for j in 0..p { mean[j] /= n; }

    let mut cov = vec![vec![0.0; p]; p];
    for row in data {
        for i in 0..p { for j in 0..p {
            cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
        }}
    }
    for i in 0..p { for j in 0..p { cov[i][j] /= n; } }

    // Invert covariance
    let cov_inv = invert_2x2(&cov);

    // Mahalanobis distance
    data.iter().map(|x| {
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();
        let mut md = 0.0;
        for i in 0..p { for j in 0..p { md += diff[i] * cov_inv[i][j] * diff[j]; } }
        md.sqrt()
    }).collect()
}

fn invert_2x2(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-10 { return vec![vec![1.0, 0.0], vec![0.0, 1.0]]; }
    vec![
        vec![m[1][1] / det, -m[0][1] / det],
        vec![-m[1][0] / det, m[0][0] / det],
    ]
}

fn k_distance_scores(data: &[Vec<f64>], k: usize) -> Vec<f64> {
    data.iter().map(|p| {
        let mut dists: Vec<f64> = data.iter()
            .map(|q| euclidean(p, q))
            .filter(|&d| d > 0.0)
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if dists.len() >= k { dists[k - 1] } else { *dists.last().unwrap_or(&0.0) }
    }).collect()
}

fn isolation_forest_scores(data: &[Vec<f64>], n_trees: usize, rng: &mut u64) -> Vec<f64> {
    let n = data.len();
    let max_depth = (n as f64).log2().ceil() as usize + 2;
    let mut all_depths = vec![0.0f64; n];

    for _ in 0..n_trees {
        // Build isolation tree
        let indices: Vec<usize> = (0..n).collect();
        let depths = isolation_tree_depths(data, &indices, 0, max_depth, rng);
        for (i, &d) in depths.iter().enumerate() {
            all_depths[i] += d as f64;
        }
    }

    // Average path length and convert to anomaly score
    let c_n = if n > 2 {
        2.0 * ((n as f64 - 1.0).ln() + 0.5772) - 2.0 * (n as f64 - 1.0) / n as f64
    } else { 1.0 };

    all_depths.iter().map(|&avg_depth| {
        let avg = avg_depth / n_trees as f64;
        2.0_f64.powf(-avg / c_n)
    }).collect()
}

fn isolation_tree_depths(
    data: &[Vec<f64>], indices: &[usize], depth: usize, max_depth: usize, rng: &mut u64,
) -> Vec<(usize, usize)> {
    // Returns (index, depth) pairs
    if indices.len() <= 1 || depth >= max_depth {
        return indices.iter().map(|&i| (i, depth)).collect();
    }

    let p = data[0].len();
    *rng = lcg(*rng);
    let feature = (*rng as usize) % p;

    let vals: Vec<f64> = indices.iter().map(|&i| data[i][feature]).collect();
    let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return indices.iter().map(|&i| (i, depth)).collect();
    }

    *rng = lcg(*rng);
    let split = min_val + (*rng as f64 / u64::MAX as f64) * (max_val - min_val);

    let left: Vec<usize> = indices.iter().filter(|&&i| data[i][feature] < split).cloned().collect();
    let right: Vec<usize> = indices.iter().filter(|&&i| data[i][feature] >= split).cloned().collect();

    let mut result = isolation_tree_depths(data, &left, depth + 1, max_depth, rng);
    result.extend(isolation_tree_depths(data, &right, depth + 1, max_depth, rng));

    // Convert to indexed format
    let mut depths = vec![depth; data.len()];
    for (idx, d) in &result { depths[*idx] = *d; }

    indices.iter().map(|&i| (i, depths[i])).collect()
}

fn find_threshold(scores: &[f64], is_anomaly: &[bool], n_anomalies: usize) -> f64 {
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if n_anomalies < sorted.len() { sorted[n_anomalies] } else { sorted[sorted.len() - 1] }
}

fn detection_metrics(actual: &[bool], predicted: &[bool]) -> (f64, f64, f64) {
    let mut tp=0; let mut fp=0; let mut fn_=0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if *a && *p { tp += 1; } else if !*a && *p { fp += 1; } else if *a && !*p { fn_ += 1; }
    }
    let prec = if tp+fp>0 { tp as f64/(tp+fp) as f64 } else { 0.0 };
    let rec = if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 };
    let f1 = if prec+rec>0.0 { 2.0*prec*rec/(prec+rec) } else { 0.0 };
    (prec, rec, f1)
}

fn print_detection_results(name: &str, actual: &[bool], predicted: &[bool]) {
    let (prec, rec, f1) = detection_metrics(actual, predicted);
    let tp = actual.iter().zip(predicted.iter()).filter(|(&a, &p)| a && p).count();
    let fp = actual.iter().zip(predicted.iter()).filter(|(&a, &p)| !a && p).count();
    let fn_ = actual.iter().zip(predicted.iter()).filter(|(&a, &p)| a && !p).count();
    println!("  TP={}, FP={}, FN={}", tp, fp, fn_);
    println!("  Precision={:.3}, Recall={:.3}, F1={:.3}", prec, rec, f1);
}

fn score_stats(scores: &[f64], is_anomaly: &[bool], anomaly: bool) -> (f64, f64) {
    let filtered: Vec<f64> = scores.iter().zip(is_anomaly.iter())
        .filter(|(_, &a)| a == anomaly).map(|(&s, _)| s).collect();
    let mean = filtered.iter().sum::<f64>() / filtered.len().max(1) as f64;
    let std = (filtered.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / filtered.len().max(1) as f64).sqrt();
    (mean, std)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)|(x-y).powi(2)).sum::<f64>().sqrt() }

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Anomaly detection models "normal" behavior and flags deviations. It is inherently unsupervised since anomaly examples are scarce or absent.
- Gaussian models using Mahalanobis distance work well for normally distributed data and capture multivariate relationships.
- Isolation Forest is effective for high-dimensional data — anomalies have shorter average path lengths in random trees because they are easier to isolate.
- The right method depends on the data: statistical methods for well-behaved distributions, distance methods for local anomalies, isolation for general-purpose detection.
