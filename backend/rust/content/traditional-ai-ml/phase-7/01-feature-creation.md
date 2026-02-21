# Feature Creation

> Phase 7 — Feature Engineering & Pipelines | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

Raw data rarely comes in a form that machine learning algorithms can use directly. Feature creation (also called feature engineering) is the art of transforming raw data into representations that make patterns more obvious to the model. A model might struggle to learn that "price per square foot" matters when given only raw price and area as separate columns. Creating that ratio explicitly gives the model a shortcut to an important pattern.

Feature creation encompasses many techniques: mathematical transformations (log, square root, polynomial terms), interaction features (products and ratios of existing features), binning continuous variables, and encoding domain knowledge as computed features. The best features encode relationships that the model would otherwise have to discover on its own -- which may be difficult or impossible with limited data.

Good feature engineering often matters more than algorithm choice. A linear model with well-crafted features frequently outperforms a complex model with raw features. This is because features can make nonlinear relationships linear, remove skewness, and expose hidden structure.

### Why naive approaches fail

Throwing raw features into a model and hoping it figures out the relationships works only if the model is powerful enough and has enough data to learn those relationships. Linear models cannot learn interactions unless you create them. Even tree-based models, which can capture interactions, benefit from explicit feature creation because it reduces the depth of trees needed to represent a pattern. Creating too many features without validation can lead to overfitting, so feature creation must be paired with proper evaluation.

### Mental models

- **Translation for the model**: feature engineering is like translating data from human-understandable form into model-understandable form.
- **Shortcut creation**: every engineered feature is a shortcut that saves the model from having to discover a pattern on its own.
- **Domain knowledge as code**: the best features encode expert knowledge about the problem domain.

### Visual explanations

```
Raw features:            Engineered features:
  price    area            price_per_sqft = price / area
  $200K    1000            log_price = ln(price)
  $500K    2500            area_bins = [small, medium, large]
  $150K    800             price_x_area = price * area (interaction)

Before feature engineering:
  Linear model sees: y ~ a*price + b*area  (cannot capture ratio)

After feature engineering:
  Linear model sees: y ~ a*price + b*area + c*price_per_sqft
  (now the ratio relationship is directly accessible)
```

---

## Hands-on Exploration

1. Generate a dataset where the true relationship involves a ratio of two features. Show that a linear model fails on raw features but succeeds with the ratio feature.
2. Create polynomial features (x^2, x1*x2) and show how they enable linear models to fit nonlinear patterns.
3. Apply log transformation to a skewed feature and show the improvement in model performance.
4. Bin a continuous variable and compare the information preserved versus lost.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    let n = 200;
    let split = 150;

    // --- Dataset: housing price depends on price_per_sqft and age ---
    let mut area: Vec<f64> = Vec::new();
    let mut bedrooms: Vec<f64> = Vec::new();
    let mut age: Vec<f64> = Vec::new();
    let mut price: Vec<f64> = Vec::new();

    for _ in 0..n {
        let a = 500.0 + rand_f64(&mut rng) * 3000.0;
        let b = 1.0 + (rand_f64(&mut rng) * 5.0).floor();
        let ag = rand_f64(&mut rng) * 50.0;
        // True relationship: price depends on area/bedrooms ratio and age
        let p = 100.0 * (a / b) - 500.0 * ag + 50000.0
            + (rand_f64(&mut rng) - 0.5) * 20000.0;
        area.push(a);
        bedrooms.push(b);
        age.push(ag);
        price.push(p.max(10000.0));
    }

    // --- Linear regression helper ---
    fn fit_linear_multi(features: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        // Simple gradient descent for multiple linear regression
        let n = y.len();
        let n_feat = features.len();
        let mut weights = vec![0.0; n_feat + 1]; // +1 for bias
        let lr = 0.0000001;
        let n_iter = 2000;

        // Normalize features for stable gradient descent
        let mut means = vec![0.0; n_feat];
        let mut stds = vec![1.0; n_feat];
        for f in 0..n_feat {
            means[f] = features[f].iter().sum::<f64>() / n as f64;
            let var: f64 = features[f].iter()
                .map(|&x| (x - means[f]).powi(2)).sum::<f64>() / n as f64;
            stds[f] = var.sqrt().max(1e-8);
        }

        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_std: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>()
            / n as f64;
        let y_std = y_std.sqrt().max(1e-8);

        for _ in 0..n_iter {
            let mut grad = vec![0.0; n_feat + 1];
            for i in 0..n {
                let mut pred = weights[n_feat]; // bias
                for f in 0..n_feat {
                    pred += weights[f] * (features[f][i] - means[f]) / stds[f];
                }
                let err = pred - (y[i] - y_mean) / y_std;
                for f in 0..n_feat {
                    grad[f] += err * (features[f][i] - means[f]) / stds[f];
                }
                grad[n_feat] += err;
            }
            for j in 0..=n_feat {
                weights[j] -= lr * 2.0 * grad[j] / n as f64;
            }
        }

        // Return denormalized predictions aren't needed -- return weights for prediction
        // Instead, just return predictions directly
        let mut preds = Vec::new();
        // We need to return the model -- let's just compute and return predictions
        weights.push(y_mean);
        weights.push(y_std);
        for f in 0..n_feat {
            weights.push(means[f]);
            weights.push(stds[f]);
        }
        weights
    }

    fn predict_multi(features: &[Vec<f64>], model: &[f64], n_feat: usize) -> Vec<f64> {
        let n = features[0].len();
        let y_mean = model[n_feat + 1];
        let y_std = model[n_feat + 2];
        let means_stds = &model[n_feat + 3..];

        (0..n).map(|i| {
            let mut pred = model[n_feat]; // bias
            for f in 0..n_feat {
                let mean = means_stds[f * 2];
                let std = means_stds[f * 2 + 1];
                pred += model[f] * (features[f][i] - mean) / std;
            }
            pred * y_std + y_mean
        }).collect()
    }

    let mse = |pred: &[f64], truth: &[f64]| -> f64 {
        pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
    };

    // --- Model 1: Raw features only ---
    let raw_features = vec![
        area[..split].to_vec(),
        bedrooms[..split].to_vec(),
        age[..split].to_vec(),
    ];
    let raw_model = fit_linear_multi(&raw_features, &price[..split]);

    let test_raw = vec![
        area[split..].to_vec(),
        bedrooms[split..].to_vec(),
        age[split..].to_vec(),
    ];
    let pred_raw = predict_multi(&test_raw, &raw_model, 3);
    let mse_raw = mse(&pred_raw, &price[split..]);

    // --- Model 2: With engineered features ---
    let area_per_bed: Vec<f64> = area.iter().zip(&bedrooms)
        .map(|(&a, &b)| a / b).collect();
    let log_area: Vec<f64> = area.iter().map(|&a| (a + 1.0).ln()).collect();
    let age_squared: Vec<f64> = age.iter().map(|&a| a * a).collect();

    let eng_features = vec![
        area[..split].to_vec(),
        bedrooms[..split].to_vec(),
        age[..split].to_vec(),
        area_per_bed[..split].to_vec(),
        log_area[..split].to_vec(),
        age_squared[..split].to_vec(),
    ];
    let eng_model = fit_linear_multi(&eng_features, &price[..split]);

    let test_eng = vec![
        area[split..].to_vec(),
        bedrooms[split..].to_vec(),
        age[split..].to_vec(),
        area_per_bed[split..].to_vec(),
        log_area[split..].to_vec(),
        age_squared[split..].to_vec(),
    ];
    let pred_eng = predict_multi(&test_eng, &eng_model, 6);
    let mse_eng = mse(&pred_eng, &price[split..]);

    println!("=== Feature Creation: Impact on Linear Regression ===\n");
    println!("{:<35} {:>12}", "Features", "Test MSE");
    println!("{}", "-".repeat(49));
    println!("{:<35} {:>12.0}", "Raw (area, beds, age)", mse_raw);
    println!("{:<35} {:>12.0}", "+ ratio, log, polynomial", mse_eng);
    println!("\nImprovement: {:.1}%", (1.0 - mse_eng / mse_raw) * 100.0);

    // --- Demonstrate log transform on skewed data ---
    println!("\n=== Log Transform on Skewed Feature ===");
    let mut skewed: Vec<f64> = (0..n).map(|_| {
        let u = rand_f64(&mut rng);
        (u * 10.0).exp() // exponentially distributed
    }).collect();

    let skew_mean: f64 = skewed.iter().sum::<f64>() / n as f64;
    let skew_var: f64 = skewed.iter().map(|x| (x - skew_mean).powi(2)).sum::<f64>() / n as f64;
    let skew_std = skew_var.sqrt();
    let skew_median = {
        let mut s = skewed.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[n / 2]
    };

    let log_skewed: Vec<f64> = skewed.iter().map(|&x| (x + 1.0).ln()).collect();
    let log_mean: f64 = log_skewed.iter().sum::<f64>() / n as f64;
    let log_var: f64 = log_skewed.iter().map(|x| (x - log_mean).powi(2)).sum::<f64>() / n as f64;
    let log_std = log_var.sqrt();
    let log_median = {
        let mut s = log_skewed.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[n / 2]
    };

    println!("  Raw:    mean={:.1}, std={:.1}, median={:.1}, mean/median={:.2}",
        skew_mean, skew_std, skew_median, skew_mean / skew_median);
    println!("  Log:    mean={:.2}, std={:.2}, median={:.2}, mean/median={:.2}",
        log_mean, log_std, log_median, log_mean / log_median);
    println!("  (closer mean/median ratio = less skewed)");

    // --- Demonstrate binning ---
    println!("\n=== Binning Continuous Variables ===");
    let bin_edges = [0.0, 10.0, 25.0, 50.0];
    let bin_labels = ["young (0-10)", "mid (10-25)", "old (25-50)"];
    let mut bin_counts = vec![0; bin_labels.len()];
    let mut bin_price_sums = vec![0.0; bin_labels.len()];
    let mut bin_price_counts = vec![0; bin_labels.len()];

    for i in 0..split {
        for b in 0..bin_labels.len() {
            if age[i] >= bin_edges[b] && age[i] < bin_edges[b + 1] {
                bin_counts[b] += 1;
                bin_price_sums[b] += price[i];
                bin_price_counts[b] += 1;
                break;
            }
        }
    }

    println!("  Age bins and mean price:");
    for b in 0..bin_labels.len() {
        let avg = if bin_price_counts[b] > 0 {
            bin_price_sums[b] / bin_price_counts[b] as f64
        } else { 0.0 };
        println!("    {:<15}: n={:>3}, mean_price=${:.0}", bin_labels[b], bin_counts[b], avg);
    }

    // --- Polynomial feature expansion ---
    println!("\n=== Polynomial Feature Expansion ===");
    println!("  Original: [x1, x2]");
    println!("  Degree 2: [x1, x2, x1^2, x1*x2, x2^2]");
    println!("  Degree 3: [x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3]");

    // Show with example
    let x1 = 2.0;
    let x2 = 3.0;
    println!("\n  Example: x1={}, x2={}", x1, x2);
    println!("  Degree 2 features: [{}, {}, {}, {}, {}]",
        x1, x2, x1*x1, x1*x2, x2*x2);

    println!();
    println!("kata_metric(\"mse_raw_features\", {:.0})", mse_raw);
    println!("kata_metric(\"mse_engineered_features\", {:.0})", mse_eng);
    println!("kata_metric(\"improvement_pct\", {:.1})", (1.0 - mse_eng / mse_raw) * 100.0);
}
```

---

## Key Takeaways

- **Feature creation transforms raw data into representations that expose hidden patterns.** Ratios, logarithms, polynomials, and interactions make relationships explicit.
- **Good features can compensate for model simplicity.** A linear model with engineered features often beats a complex model with raw features.
- **Log transforms tame skewed distributions.** Many real-world quantities (income, prices, counts) are right-skewed; log transform makes them approximately normal.
- **Domain knowledge is your most powerful tool.** The best features come from understanding the problem, not from automated feature generation.
