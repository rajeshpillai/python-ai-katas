# Sequence Classification

> Phase 8 — Time Series & Sequential Data | Kata 8.6

---

## Concept & Intuition

### What problem are we solving?

Not all time series problems are about forecasting. **Sequence classification** assigns a label to an entire sequence: is this ECG reading normal or abnormal? Is this sensor trace from a faulty machine? Is this user's activity pattern fraudulent? The input is a variable-length time series; the output is a class label.

The challenge is transforming a sequence of values into a fixed-length feature vector that captures the essential characteristics. Simple approaches extract **statistical features** (mean, variance, skewness, number of peaks, slope). More sophisticated approaches use **shapelet-based features** (characteristic subsequences) or **distance-based methods** (Dynamic Time Warping to compare entire sequences).

Feature extraction is the key to sequence classification. A raw sequence of 1000 data points is too high-dimensional and misaligned for direct use. By computing summary statistics, frequency-domain features (dominant frequencies), and shape-based features, you compress the sequence into a manageable feature vector that standard classifiers (KNN, SVM, random forests) can handle.

### Why naive approaches fail

Treating each time step as an independent feature fails because: (1) sequences may have different lengths, (2) the same pattern at different time offsets should be recognized as identical, and (3) the feature space is enormous (one dimension per time step). Feature extraction solves all three problems by mapping variable-length sequences to fixed-length vectors that capture patterns regardless of timing.

### Mental models

- **Sequence as a fingerprint**: the statistical features (mean, variance, peaks, slope) form a "fingerprint" that characterizes each sequence type.
- **DTW as elastic matching**: Dynamic Time Warping stretches and compresses time to align two sequences, measuring similarity despite differences in speed.
- **Feature extraction as compression**: we compress a 1000-point sequence into 20 informative features, discarding noise and preserving signal.

### Visual explanations

```
Sequence Classification Pipeline:

  Raw sequences:     Feature extraction:     Classification:
  ~~~~/\~~~~         mean=5.2                \
  ~~~~\/~~~~   -->   std=1.8           -->    Classifier --> Label
  __/\__/\__         n_peaks=3               /
                     trend_slope=0.1
                     max_freq=0.25

Feature Types:
  Statistical:  mean, std, min, max, skewness, kurtosis
  Temporal:     autocorrelation, trend slope, n_crossings
  Frequency:    dominant frequency, spectral energy
  Shape:        n_peaks, peak_height, rise_time
```

---

## Hands-on Exploration

1. Generate three classes of sequences (sinusoidal, linear trend, random walk) with noise. Extract statistical features.
2. Implement a distance-based classifier using Dynamic Time Warping distance. Compare to Euclidean distance.
3. Build a feature-based classifier: extract mean, std, trend, peaks, autocorrelation. Train a simple KNN on the features.
4. Evaluate using proper train/test split. Compare feature-based vs distance-based approaches.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_normal = |s: &mut u64| -> f64 {
        let u1 = rand_f64(s).max(1e-10);
        let u2 = rand_f64(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };

    // --- Generate 3 classes of sequences ---
    let seq_len = 50;
    let n_per_class = 40;
    let n_total = n_per_class * 3;

    let mut sequences: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Class 0: Sinusoidal (oscillating)
    for _ in 0..n_per_class {
        let freq = 0.1 + rand_f64(&mut rng) * 0.15;
        let phase = rand_f64(&mut rng) * 2.0 * pi;
        let amp = 3.0 + rand_f64(&mut rng) * 2.0;
        let seq: Vec<f64> = (0..seq_len).map(|t| {
            amp * (2.0 * pi * freq * t as f64 + phase).sin()
            + rand_normal(&mut rng) * 0.5
        }).collect();
        sequences.push(seq);
        labels.push(0);
    }

    // Class 1: Trending (upward or downward)
    for _ in 0..n_per_class {
        let slope = (rand_f64(&mut rng) - 0.3) * 0.3;
        let start = rand_f64(&mut rng) * 5.0 - 2.5;
        let seq: Vec<f64> = (0..seq_len).map(|t| {
            start + slope * t as f64 + rand_normal(&mut rng) * 0.5
        }).collect();
        sequences.push(seq);
        labels.push(1);
    }

    // Class 2: Step function (abrupt change)
    for _ in 0..n_per_class {
        let change_point = 15 + (rand_f64(&mut rng) * 20.0) as usize;
        let level1 = rand_f64(&mut rng) * 4.0 - 2.0;
        let level2 = level1 + (rand_f64(&mut rng) * 6.0 - 3.0);
        let seq: Vec<f64> = (0..seq_len).map(|t| {
            let base = if t < change_point { level1 } else { level2 };
            base + rand_normal(&mut rng) * 0.5
        }).collect();
        sequences.push(seq);
        labels.push(2);
    }

    let class_names = ["Sinusoidal", "Trending", "Step Function"];
    println!("=== Sequence Classification ===");
    println!("Classes: {:?}", class_names);
    println!("{} sequences per class, length={}\n", n_per_class, seq_len);

    // --- Feature Extraction ---
    fn extract_features(seq: &[f64]) -> Vec<f64> {
        let n = seq.len() as f64;
        let mean: f64 = seq.iter().sum::<f64>() / n;
        let var: f64 = seq.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let min = seq.iter().cloned().fold(f64::MAX, f64::min);
        let max = seq.iter().cloned().fold(f64::MIN, f64::max);

        // Skewness
        let skew = if std > 1e-10 {
            seq.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>() / n
        } else { 0.0 };

        // Trend (linear regression slope)
        let t_mean = (n - 1.0) / 2.0;
        let cov: f64 = seq.iter().enumerate()
            .map(|(t, &v)| (t as f64 - t_mean) * (v - mean)).sum::<f64>();
        let t_var: f64 = (0..seq.len()).map(|t| (t as f64 - t_mean).powi(2)).sum::<f64>();
        let slope = if t_var > 1e-10 { cov / t_var } else { 0.0 };

        // Number of zero crossings (crossings of the mean)
        let crossings = seq.windows(2).filter(|w| {
            (w[0] - mean).signum() != (w[1] - mean).signum()
        }).count() as f64;

        // Number of peaks
        let peaks = (1..seq.len() - 1).filter(|&t| {
            seq[t] > seq[t - 1] && seq[t] > seq[t + 1]
        }).count() as f64;

        // Autocorrelation at lag 1
        let ac1 = if var > 1e-10 {
            (1..seq.len()).map(|t| {
                (seq[t] - mean) * (seq[t - 1] - mean)
            }).sum::<f64>() / (n * var)
        } else { 0.0 };

        // Max absolute difference (captures step changes)
        let max_diff = seq.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);

        vec![mean, std, min, max, skew, slope, crossings, peaks, ac1, max_diff]
    }

    let feature_names = [
        "mean", "std", "min", "max", "skewness",
        "slope", "crossings", "peaks", "autocorr_1", "max_diff",
    ];

    // Extract features for all sequences
    let features: Vec<Vec<f64>> = sequences.iter().map(|s| extract_features(s)).collect();

    // Show feature statistics per class
    println!("=== Feature Statistics per Class ===\n");
    println!("{:<12}", "Feature");
    for name in &class_names {
        print!("{:>15}", name);
    }
    println!();
    println!("{}", "-".repeat(57));

    for (f, fname) in feature_names.iter().enumerate() {
        print!("{:<12}", fname);
        for c in 0..3 {
            let vals: Vec<f64> = (0..n_total)
                .filter(|&i| labels[i] == c)
                .map(|i| features[i][f])
                .collect();
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            print!("{:>15.3}", mean);
        }
        println!();
    }

    // --- Train/Test Split ---
    let split = 90; // 30 per class for training
    // Interleave: first 30 of each class for train
    let mut train_idx: Vec<usize> = Vec::new();
    let mut test_idx: Vec<usize> = Vec::new();
    for c in 0..3 {
        let class_start = c * n_per_class;
        for i in 0..n_per_class {
            if i < 30 {
                train_idx.push(class_start + i);
            } else {
                test_idx.push(class_start + i);
            }
        }
    }

    // --- KNN Classifier on extracted features ---
    fn knn_predict(
        train_features: &[Vec<f64>], train_labels: &[usize],
        test_point: &[f64], k: usize,
    ) -> usize {
        let mut distances: Vec<(f64, usize)> = train_features.iter()
            .zip(train_labels)
            .map(|(feat, &label)| {
                let dist: f64 = feat.iter().zip(test_point)
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (dist, label)
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut votes = vec![0; 3];
        for &(_, label) in distances.iter().take(k) {
            votes[label] += 1;
        }
        votes.iter().enumerate().max_by_key(|&(_, v)| v).unwrap().0
    }

    // Standardize features
    let n_feat = feature_names.len();
    let mut means = vec![0.0; n_feat];
    let mut stds = vec![1.0; n_feat];
    for f in 0..n_feat {
        means[f] = train_idx.iter().map(|&i| features[i][f]).sum::<f64>() / train_idx.len() as f64;
        let var: f64 = train_idx.iter().map(|&i| (features[i][f] - means[f]).powi(2)).sum::<f64>()
            / train_idx.len() as f64;
        stds[f] = var.sqrt().max(1e-8);
    }

    let standardize = |feat: &[f64]| -> Vec<f64> {
        feat.iter().enumerate().map(|(f, &v)| (v - means[f]) / stds[f]).collect()
    };

    let train_feat_std: Vec<Vec<f64>> = train_idx.iter().map(|&i| standardize(&features[i])).collect();
    let train_labels_vec: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();

    // Evaluate KNN with different k
    println!("\n=== KNN on Extracted Features ===\n");
    println!("{:>5} {:>10}", "k", "Accuracy");
    println!("{}", "-".repeat(17));

    for &k in &[1, 3, 5, 7] {
        let mut correct = 0;
        for &i in &test_idx {
            let test_std = standardize(&features[i]);
            let pred = knn_predict(&train_feat_std, &train_labels_vec, &test_std, k);
            if pred == labels[i] { correct += 1; }
        }
        let acc = correct as f64 / test_idx.len() as f64;
        println!("{:>5} {:>10.3}", k, acc);
    }

    // --- DTW Distance ---
    fn dtw_distance(s1: &[f64], s2: &[f64]) -> f64 {
        let n = s1.len();
        let m = s2.len();
        let mut dp = vec![vec![f64::MAX; m + 1]; n + 1];
        dp[0][0] = 0.0;

        for i in 1..=n {
            for j in 1..=m {
                let cost = (s1[i - 1] - s2[j - 1]).powi(2);
                dp[i][j] = cost + dp[i-1][j].min(dp[i][j-1]).min(dp[i-1][j-1]);
            }
        }
        dp[n][m].sqrt()
    }

    // DTW-based 1NN classifier
    println!("\n=== DTW-based 1-NN Classifier ===\n");

    let mut dtw_correct = 0;
    for &i in &test_idx {
        let mut best_dist = f64::MAX;
        let mut best_label = 0;
        for &j in &train_idx {
            let dist = dtw_distance(&sequences[i], &sequences[j]);
            if dist < best_dist {
                best_dist = dist;
                best_label = labels[j];
            }
        }
        if best_label == labels[i] { dtw_correct += 1; }
    }
    let dtw_acc = dtw_correct as f64 / test_idx.len() as f64;
    println!("DTW 1-NN accuracy: {:.3}", dtw_acc);

    // --- Confusion matrix ---
    println!("\n=== Confusion Matrix (KNN k=5, features) ===");
    let mut confusion = vec![vec![0; 3]; 3];
    for &i in &test_idx {
        let test_std = standardize(&features[i]);
        let pred = knn_predict(&train_feat_std, &train_labels_vec, &test_std, 5);
        confusion[labels[i]][pred] += 1;
    }

    println!("            Predicted");
    println!("            {:>8} {:>8} {:>8}", class_names[0], class_names[1], class_names[2]);
    for (c, name) in class_names.iter().enumerate() {
        println!("Actual {:<10} {:>5} {:>8} {:>8}",
            name, confusion[c][0], confusion[c][1], confusion[c][2]);
    }

    // Per-class accuracy
    println!("\nPer-class accuracy:");
    for c in 0..3 {
        let total: usize = confusion[c].iter().sum();
        let acc = if total > 0 { confusion[c][c] as f64 / total as f64 } else { 0.0 };
        println!("  {:<15}: {:.1}% ({}/{})", class_names[c], acc * 100.0, confusion[c][c], total);
    }

    println!();
    let knn5_acc = {
        let mut c = 0;
        for &i in &test_idx {
            let test_std = standardize(&features[i]);
            let pred = knn_predict(&train_feat_std, &train_labels_vec, &test_std, 5);
            if pred == labels[i] { c += 1; }
        }
        c as f64 / test_idx.len() as f64
    };
    println!("kata_metric(\"knn_feature_accuracy\", {:.3})", knn5_acc);
    println!("kata_metric(\"dtw_1nn_accuracy\", {:.3})", dtw_acc);
    println!("kata_metric(\"n_features\", {})", n_feat);
}
```

---

## Key Takeaways

- **Sequence classification assigns labels to entire time series,** not individual points. This requires transforming variable-length sequences into fixed-length representations.
- **Feature extraction is the key step.** Statistical features (mean, std, slope, peaks, autocorrelation) compress sequences into informative, classifier-friendly vectors.
- **DTW provides an elastic distance measure** that handles timing differences, but is computationally expensive (O(n*m) per pair).
- **Feature-based approaches are often faster and competitive** with distance-based methods, especially when domain knowledge guides feature selection.
