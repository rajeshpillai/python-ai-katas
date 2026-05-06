# Data Drift Detection

> Phase 11 — Productionizing ML | Kata 11.2

---

## Concept & Intuition

### What problem are we solving?

A model trained on historical data makes an implicit assumption: **future data will look like past data**. When this assumption breaks -- when the statistical properties of incoming data shift -- the model's performance degrades silently. This is **data drift**, and it is one of the most common reasons ML systems fail in production.

There are several types of drift. **Covariate drift** occurs when the input distribution P(X) changes (e.g., your customer demographics shift). **Concept drift** occurs when the relationship P(Y|X) changes (e.g., what makes a customer churn changes over time). **Prior probability drift** occurs when P(Y) changes (e.g., fraud becomes more common). All require different detection and mitigation strategies.

Detecting drift early is critical. Without monitoring, a model can silently decay for months before someone notices the business metrics dropping. A good drift detection system raises alerts when input distributions shift significantly, triggering investigation, retraining, or model rollback.

### Why naive approaches fail

Simply monitoring prediction accuracy is not enough -- in many production systems, ground truth labels arrive with a delay (sometimes weeks or months). You need to detect drift from the **inputs alone**, before you know whether predictions are wrong. Statistical tests comparing reference distributions (from training data) to production distributions catch shifts early, even without labels.

### Mental models

- **Drift as concept aging**: a model is like a map. The map was accurate when drawn, but the terrain changes over time. Drift detection tells you when the map is outdated.
- **Two-sample testing**: "Are these two batches of data from the same distribution?" This is the core statistical question. If the answer is "no," drift has occurred.
- **Training distribution as the reference**: everything the model learned came from the training data. Any significant deviation from that distribution is a potential problem.

### Visual explanations

```
Types of drift:

  Covariate drift (P(X) changes):
    Training:   age ~ N(35, 10)     income ~ N(50k, 15k)
    Production: age ~ N(45, 12)     income ~ N(70k, 20k)
    --> Model inputs look different, predictions may be unreliable

  Concept drift (P(Y|X) changes):
    Training:   high spending --> loyal customer
    Production: high spending --> about to churn (economy changed)
    --> Same inputs, different correct outputs

  Label drift (P(Y) changes):
    Training:   5% fraud rate
    Production: 15% fraud rate (new attack vector)
    --> Class balance shifted

Detection methods:
  Statistical tests:
    - KS test: compares CDFs, works per-feature
    - PSI: binned distribution comparison
    - Summary statistics: mean, std, percentile shifts

  Monitoring pipeline:
    Training data --> compute reference statistics
                         |
    Production data --> compute window statistics --> compare --> ALERT?
                         |
                     [sliding window or batch]
```

---

## Hands-on Exploration

1. Generate two datasets with the same distribution. Run a KS test -- it should not detect drift.
2. Shift the mean of one feature by 1 standard deviation. Run the KS test again -- does it detect the shift?
3. Compute PSI for a gradually drifting feature. At what point does PSI exceed the alert threshold (0.2)?
4. Simulate concept drift: keep X the same but change the decision boundary. Can you detect this from X alone?

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64, mean: f64, std: f64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };

    // --- Generate reference (training) and production data ---
    let n_ref = 1000;
    let n_prod = 500;
    let feature_names = ["age", "income", "tenure", "usage", "support_calls"];
    let n_features = feature_names.len();

    let ref_means = [35.0, 50000.0, 24.0, 150.0, 2.0];
    let ref_stds = [10.0, 15000.0, 12.0, 50.0, 1.5];

    let mut x_ref: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_ref {
        let row: Vec<f64> = (0..n_features)
            .map(|j| rand_normal(&mut rng, ref_means[j], ref_stds[j]))
            .collect();
        x_ref.push(row);
    }

    // Production distribution (with drift in age and income)
    let drift_magnitude = 0.5; // std devs to shift
    let drift_features = [0_usize, 1]; // age and income drift
    let mut prod_means = ref_means;
    for &f in &drift_features {
        prod_means[f] += drift_magnitude * ref_stds[f];
    }

    let mut x_prod: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_prod {
        let row: Vec<f64> = (0..n_features)
            .map(|j| rand_normal(&mut rng, prod_means[j], ref_stds[j]))
            .collect();
        x_prod.push(row);
    }

    println!("=== Data Drift Detection ===\n");
    println!("Reference samples: {}, Production samples: {}", n_ref, n_prod);
    println!("Drift magnitude: {} std devs in: {:?}\n",
        drift_magnitude,
        drift_features.iter().map(|&i| feature_names[i]).collect::<Vec<_>>());

    // --- Method 1: Kolmogorov-Smirnov Test ---
    // KS statistic: max difference between empirical CDFs
    fn ks_test(ref_data: &[f64], prod_data: &[f64]) -> (f64, f64) {
        let mut all: Vec<(f64, bool)> = Vec::new();
        for &v in ref_data { all.push((v, true)); }
        for &v in prod_data { all.push((v, false)); }
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let n_ref = ref_data.len() as f64;
        let n_prod = prod_data.len() as f64;
        let mut cdf_ref = 0.0;
        let mut cdf_prod = 0.0;
        let mut max_diff = 0.0_f64;

        for (_, is_ref) in &all {
            if *is_ref { cdf_ref += 1.0 / n_ref; }
            else { cdf_prod += 1.0 / n_prod; }
            max_diff = max_diff.max((cdf_ref - cdf_prod).abs());
        }

        // Approximate p-value using the asymptotic distribution
        let n_eff = (n_ref * n_prod) / (n_ref + n_prod);
        let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * max_diff;
        let p_value = (-2.0 * lambda * lambda).exp() * 2.0;
        let p_value = p_value.max(0.0).min(1.0);

        (max_diff, p_value)
    }

    println!("=== KS Test (per feature) ===\n");
    println!("{:>15} {:>8} {:>10} {:>8} {:>10} {:>10}",
        "Feature", "KS Stat", "p-value", "Drift?", "Ref Mean", "Prod Mean");
    println!("{}", "-".repeat(65));

    let mut n_drifted = 0;
    let mut drifted_features: Vec<&str> = Vec::new();

    for j in 0..n_features {
        let ref_col: Vec<f64> = x_ref.iter().map(|r| r[j]).collect();
        let prod_col: Vec<f64> = x_prod.iter().map(|r| r[j]).collect();

        let ref_mean: f64 = ref_col.iter().sum::<f64>() / n_ref as f64;
        let prod_mean: f64 = prod_col.iter().sum::<f64>() / n_prod as f64;

        let (ks_stat, p_value) = ks_test(&ref_col, &prod_col);
        let drift_detected = p_value < 0.05;

        if drift_detected {
            n_drifted += 1;
            drifted_features.push(feature_names[j]);
        }

        let marker = if drift_detected { "YES" } else { "no" };
        println!("{:>15} {:>8.4} {:>10.6} {:>8} {:>10.1} {:>10.1}",
            feature_names[j], ks_stat, p_value, marker, ref_mean, prod_mean);
    }

    // --- Method 2: Population Stability Index (PSI) ---
    fn compute_psi(reference: &[f64], production: &[f64], n_bins: usize) -> f64 {
        let mut sorted_ref = reference.to_vec();
        sorted_ref.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Create bins from reference percentiles
        let mut breakpoints: Vec<f64> = Vec::new();
        breakpoints.push(f64::NEG_INFINITY);
        for i in 1..n_bins {
            let idx = (i as f64 / n_bins as f64 * sorted_ref.len() as f64) as usize;
            breakpoints.push(sorted_ref[idx.min(sorted_ref.len() - 1)]);
        }
        breakpoints.push(f64::INFINITY);

        // Count per bin
        let bin_of = |x: f64| -> usize {
            for i in 1..breakpoints.len() {
                if x < breakpoints[i] { return i - 1; }
            }
            breakpoints.len() - 2
        };

        let mut ref_counts = vec![0_usize; n_bins];
        let mut prod_counts = vec![0_usize; n_bins];
        for &v in reference { ref_counts[bin_of(v)] += 1; }
        for &v in production { prod_counts[bin_of(v)] += 1; }

        let mut psi = 0.0;
        for i in 0..n_bins {
            let ref_pct = (ref_counts[i] as f64 + 1.0) / (reference.len() as f64 + n_bins as f64);
            let prod_pct = (prod_counts[i] as f64 + 1.0) / (production.len() as f64 + n_bins as f64);
            psi += (prod_pct - ref_pct) * (prod_pct / ref_pct).ln();
        }
        psi
    }

    println!("\n=== Population Stability Index (PSI) ===");
    println!("Thresholds: PSI < 0.1 = stable, 0.1-0.2 = moderate, > 0.2 = significant\n");
    println!("{:>15} {:>8} {:>18}", "Feature", "PSI", "Status");
    println!("{}", "-".repeat(44));

    for j in 0..n_features {
        let ref_col: Vec<f64> = x_ref.iter().map(|r| r[j]).collect();
        let prod_col: Vec<f64> = x_prod.iter().map(|r| r[j]).collect();
        let psi = compute_psi(&ref_col, &prod_col, 10);

        let status = if psi < 0.1 { "Stable" }
            else if psi < 0.2 { "Moderate shift" }
            else { "SIGNIFICANT DRIFT" };

        println!("{:>15} {:>8.4} {:>18}", feature_names[j], psi, status);
    }

    // --- Method 3: Summary Statistics Monitoring ---
    println!("\n=== Summary Statistics Comparison ===\n");
    println!("{:>15} {:>10} {:>12} {:>12} {:>10}",
        "Feature", "Metric", "Reference", "Production", "% Change");
    println!("{}", "-".repeat(62));

    for j in 0..n_features {
        let ref_col: Vec<f64> = x_ref.iter().map(|r| r[j]).collect();
        let prod_col: Vec<f64> = x_prod.iter().map(|r| r[j]).collect();

        let ref_mean = ref_col.iter().sum::<f64>() / n_ref as f64;
        let prod_mean = prod_col.iter().sum::<f64>() / n_prod as f64;
        let ref_std = (ref_col.iter().map(|v| (v - ref_mean).powi(2)).sum::<f64>() / n_ref as f64).sqrt();
        let prod_std = (prod_col.iter().map(|v| (v - prod_mean).powi(2)).sum::<f64>() / n_prod as f64).sqrt();

        for &(metric, rv, pv) in &[("Mean", ref_mean, prod_mean), ("Std", ref_std, prod_std)] {
            let pct_change = 100.0 * (pv - rv) / (rv.abs() + 1e-10);
            let flag = if pct_change.abs() > 10.0 { " !" } else { "" };
            println!("{:>15} {:>10} {:>12.2} {:>12.2} {:>9.1}%{}",
                feature_names[j], metric, rv, pv, pct_change, flag);
        }
    }

    // --- Gradual drift simulation ---
    println!("\n=== Gradual Drift Simulation ===");
    println!("Monitoring 'age' feature over 10 time windows\n");

    let n_windows = 10;
    let window_size = 200;
    println!("{:>8} {:>8} {:>10} {:>8} {:>8}",
        "Window", "Mean", "KS p-val", "PSI", "Alert");
    println!("{}", "-".repeat(46));

    let ref_age: Vec<f64> = x_ref.iter().map(|r| r[0]).collect();

    for w in 0..n_windows {
        let window_drift = drift_magnitude * (w as f64 / (n_windows - 1) as f64);
        let window_mean = ref_means[0] + window_drift * ref_stds[0];

        let window_data: Vec<f64> = (0..window_size)
            .map(|_| rand_normal(&mut rng, window_mean, ref_stds[0]))
            .collect();

        let actual_mean = window_data.iter().sum::<f64>() / window_size as f64;
        let (_, ks_p) = ks_test(&ref_age, &window_data);
        let psi = compute_psi(&ref_age, &window_data, 10);

        let alert = if ks_p < 0.05 || psi > 0.2 { "ALERT" } else { "" };
        println!("{:>8} {:>8.1} {:>10.6} {:>8.4} {:>8}",
            w + 1, actual_mean, ks_p, psi, alert);
    }

    // --- Correlation shift detection ---
    println!("\n=== Multivariate Drift (Correlation Changes) ===\n");

    let compute_corr = |data: &[Vec<f64>], i: usize, j: usize| -> f64 {
        let n = data.len() as f64;
        let mean_i = data.iter().map(|r| r[i]).sum::<f64>() / n;
        let mean_j = data.iter().map(|r| r[j]).sum::<f64>() / n;
        let cov: f64 = data.iter().map(|r| (r[i] - mean_i) * (r[j] - mean_j)).sum::<f64>() / n;
        let std_i = (data.iter().map(|r| (r[i] - mean_i).powi(2)).sum::<f64>() / n).sqrt();
        let std_j = (data.iter().map(|r| (r[j] - mean_j).powi(2)).sum::<f64>() / n).sqrt();
        if std_i * std_j < 1e-10 { 0.0 } else { cov / (std_i * std_j) }
    };

    println!("Correlation difference (|ref - prod|):");
    print!("{:>15}", "");
    for name in &feature_names {
        print!("  {:>8}", &name[..name.len().min(8)]);
    }
    println!();
    for i in 0..n_features {
        print!("{:>15}", feature_names[i]);
        for j in 0..n_features {
            let ref_corr = compute_corr(&x_ref, i, j);
            let prod_corr = compute_corr(&x_prod, i, j);
            let diff = (ref_corr - prod_corr).abs();
            let marker = if diff > 0.1 { "*" } else { " " };
            print!("  {:>7.3}{}", diff, marker);
        }
        println!();
    }
    println!("(* = correlation shift > 0.1)");

    // --- Summary ---
    println!("\n=== Drift Detection Summary ===\n");
    println!("Features with detected drift (KS test): {}/{}", n_drifted, n_features);
    if n_drifted > 0 {
        println!("  Drifted features: {:?}", drifted_features);
        println!("\nRecommended actions:");
        println!("  1. Investigate root cause of drift");
        println!("  2. Evaluate model performance on recent labeled data");
        println!("  3. Consider retraining model with recent data");
        println!("  4. Set up automated alerts for PSI > 0.2");
    } else {
        println!("No significant drift detected. Model inputs appear stable.");
    }

    println!();
    println!("kata_metric(\"features_drifted\", {})", n_drifted);
    println!("kata_metric(\"total_features\", {})", n_features);
    let age_ref: Vec<f64> = x_ref.iter().map(|r| r[0]).collect();
    let age_prod: Vec<f64> = x_prod.iter().map(|r| r[0]).collect();
    let (age_ks, _) = ks_test(&age_ref, &age_prod);
    println!("kata_metric(\"age_ks_statistic\", {:.4})", age_ks);
}
```

---

## Key Takeaways

- **Data drift is the silent killer of ML models.** Performance can degrade for months before anyone notices, because drift happens in the inputs, not in a visible error message.
- **The KS test and PSI are complementary detection methods.** KS is a formal statistical test with p-values; PSI gives an interpretable stability score. Use both.
- **Monitor individual features AND multivariate relationships.** Two features can individually look fine while their correlation changes dramatically.
- **You often must detect drift without labels.** Ground truth may be delayed by weeks or months. Input-based drift detection gives early warning before you can measure accuracy.
- **Drift detection triggers action, not panic.** Not all drift degrades model performance. The workflow is: detect, investigate, evaluate impact, then retrain if needed.
