# Interpretability

> Phase 11 — Productionizing ML | Kata 11.5

---

## Concept & Intuition

### What problem are we solving?

A model predicts that a loan application should be denied. The applicant asks: "Why?" If you cannot answer, you have a problem -- legally, ethically, and practically. **Model interpretability** is the ability to explain how a model makes its predictions. It ranges from intrinsically interpretable models (linear regression, decision trees) to post-hoc explanation methods for black-box models (feature importance, SHAP values, partial dependence plots).

There are two levels of interpretability. **Global interpretability** explains how the model works overall: "Which features matter most? What is the general relationship between income and approval probability?" **Local interpretability** explains individual predictions: "Why was *this particular application* denied? Which features pushed the decision toward denial?"

Key methods include: **Feature importance** (how much does each feature contribute to predictions overall?), **Partial Dependence Plots** (how does the prediction change as one feature varies, holding others constant?), **SHAP values** (Shapley values that fairly attribute each feature's contribution to a specific prediction), and **LIME** (local linear approximations around individual predictions).

### Why naive approaches fail

Looking at model coefficients only works for linear models. For complex models (random forests, gradient boosting, neural networks), the relationship between inputs and outputs is non-linear and involves interactions. Simply looking at which features the model "uses" does not tell you how it uses them. SHAP values provide a theoretically grounded attribution that accounts for feature interactions and is consistent (a feature that contributes more always gets a higher value).

### Mental models

- **Feature importance as a budget**: if the model has 100 units of predictive power to distribute, how many go to each feature?
- **Partial dependence as a controlled experiment**: "What happens to the prediction if I change only this one feature?" This isolates the marginal effect.
- **SHAP as fair credit assignment**: like splitting a group project grade among team members based on each person's actual contribution. SHAP values are the unique fair way to do this.

### Visual explanations

```
Feature Importance (Global):
  income        ████████████████  0.35
  credit_score  ██████████████    0.30
  debt_ratio    ██████████        0.20
  education     █████             0.10
  experience    ██                0.05

Partial Dependence (credit_score):
  Prediction
  0.8 |            ...........
      |         ...
  0.6 |       ..
      |     ..
  0.4 |   ..
      | ..
  0.2 |.
      +----------------------->
       500  600  700  800  credit_score

SHAP (Local - why this application was denied):
  Base prediction:  0.55 (average approval rate)
  + income = 80k:   +0.12
  + education = 16: +0.05
  - credit = 580:   -0.25  <-- biggest reason for denial
  - debt = 0.6:     -0.15
  + experience = 8: +0.02
  = Final:          0.34 (denied, threshold = 0.5)
```

---

## Hands-on Exploration

1. Train a model and compute permutation feature importance. Which features matter most?
2. Compute partial dependence for the top feature. How does the prediction change as that feature varies?
3. Implement a simple SHAP-like attribution for a single prediction. Show how each feature pushes the prediction up or down.
4. Compare feature importance rankings from different methods. Do they agree?

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_int = |s: &mut u64, max: usize| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as usize) % max
    };

    println!("=== Model Interpretability ===\n");

    // --- Generate loan approval dataset ---
    let feature_names = ["income", "credit_score", "debt_ratio", "education", "experience"];
    let n_feat = feature_names.len();
    let n = 300;

    let mut x_data: Vec<Vec<f64>> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    // True relationship (non-linear)
    for _ in 0..n {
        let income = 30.0 + rand_normal(&mut rng) * 20.0;
        let credit = 650.0 + rand_normal(&mut rng) * 80.0;
        let debt = 0.35 + rand_normal(&mut rng) * 0.15;
        let education = 14.0 + rand_normal(&mut rng) * 3.0;
        let experience = 10.0 + rand_normal(&mut rng) * 5.0;

        let logit = 0.04 * (income - 40.0) + 0.015 * (credit - 650.0)
            - 3.0 * (debt - 0.3) + 0.1 * (education - 12.0)
            + 0.02 * (experience - 8.0);
        let prob = 1.0 / (1.0 + (-logit).exp());
        let label = if rand_f64(&mut rng) < prob { 1.0 } else { 0.0 };

        x_data.push(vec![income, credit, debt, education, experience]);
        y_data.push(label);
    }

    // Split into train/test
    let n_train = 200;
    let x_train = &x_data[..n_train];
    let y_train = &y_data[..n_train];
    let x_test = &x_data[n_train..];
    let y_test = &y_data[n_train..];

    // --- Train a gradient boosted ensemble (simplified) ---
    // Use multiple decision stumps
    let n_stumps = 50;
    let lr = 0.3;

    struct Stump {
        feature: usize,
        threshold: f64,
        left_val: f64,
        right_val: f64,
    }

    impl Stump {
        fn predict_one(&self, x: &[f64]) -> f64 {
            if x[self.feature] <= self.threshold { self.left_val } else { self.right_val }
        }
    }

    let mut stumps: Vec<Stump> = Vec::new();
    let mut base_pred = y_train.iter().sum::<f64>() / n_train as f64;

    // Current predictions
    let mut preds: Vec<f64> = vec![base_pred; n_train];

    for _ in 0..n_stumps {
        // Compute residuals (pseudo-residuals for logistic loss)
        let residuals: Vec<f64> = preds.iter().zip(y_train)
            .map(|(&p, &y)| y - 1.0 / (1.0 + (-p).exp()))
            .collect();

        // Find best stump
        let mut best_stump = Stump { feature: 0, threshold: 0.0, left_val: 0.0, right_val: 0.0 };
        let mut best_loss = f64::MAX;

        for feat in 0..n_feat {
            let mut vals: Vec<f64> = x_train.iter().map(|r| r[feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();

            for &thresh in vals.iter().step_by(vals.len().max(1) / 10 + 1) {
                let mut left_sum = 0.0;
                let mut left_count = 0.0;
                let mut right_sum = 0.0;
                let mut right_count = 0.0;

                for (i, row) in x_train.iter().enumerate() {
                    if row[feat] <= thresh {
                        left_sum += residuals[i];
                        left_count += 1.0;
                    } else {
                        right_sum += residuals[i];
                        right_count += 1.0;
                    }
                }

                let left_val = if left_count > 0.0 { left_sum / left_count } else { 0.0 };
                let right_val = if right_count > 0.0 { right_sum / right_count } else { 0.0 };

                let loss: f64 = x_train.iter().enumerate().map(|(i, row)| {
                    let v = if row[feat] <= thresh { left_val } else { right_val };
                    (residuals[i] - v).powi(2)
                }).sum();

                if loss < best_loss {
                    best_loss = loss;
                    best_stump = Stump { feature: feat, threshold: thresh, left_val, right_val };
                }
            }
        }

        // Update predictions
        for (i, row) in x_train.iter().enumerate() {
            preds[i] += lr * best_stump.predict_one(row);
        }
        stumps.push(best_stump);
    }

    // Model prediction function
    let model_predict = |x: &[f64]| -> f64 {
        let raw = base_pred + stumps.iter().map(|s| lr * s.predict_one(x)).sum::<f64>();
        1.0 / (1.0 + (-raw).exp())
    };

    // Test accuracy
    let test_acc: f64 = x_test.iter().zip(y_test).map(|(x, &y)| {
        let pred = if model_predict(x) > 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.01 { 1.0 } else { 0.0 }
    }).sum::<f64>() / x_test.len() as f64;

    println!("Model: Gradient Boosted Stumps ({} stumps)", n_stumps);
    println!("Test accuracy: {:.3}\n", test_acc);

    // --- Method 1: Permutation Feature Importance ---
    println!("=== Permutation Feature Importance (Global) ===\n");

    let baseline_acc = test_acc;
    let mut importances: Vec<(usize, f64)> = Vec::new();

    for feat in 0..n_feat {
        let mut shuffled_acc_sum = 0.0;
        let n_shuffles = 10;

        for _ in 0..n_shuffles {
            // Create shuffled copy
            let mut x_shuffled: Vec<Vec<f64>> = x_test.to_vec();
            // Fisher-Yates shuffle for this feature
            for i in (1..x_shuffled.len()).rev() {
                let j = rand_int(&mut rng, i + 1);
                let tmp = x_shuffled[i][feat];
                x_shuffled[i][feat] = x_shuffled[j][feat];
                x_shuffled[j][feat] = tmp;
            }

            let shuf_acc: f64 = x_shuffled.iter().zip(y_test).map(|(x, &y)| {
                let pred = if model_predict(x) > 0.5 { 1.0 } else { 0.0 };
                if (pred - y).abs() < 0.01 { 1.0 } else { 0.0 }
            }).sum::<f64>() / x_test.len() as f64;

            shuffled_acc_sum += shuf_acc;
        }

        let mean_shuffled_acc = shuffled_acc_sum / n_shuffles as f64;
        let importance = baseline_acc - mean_shuffled_acc;
        importances.push((feat, importance));
    }

    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:>15} {:>12} {:>20}", "Feature", "Importance", "Visual");
    println!("{}", "-".repeat(50));

    let max_imp = importances[0].1.max(0.001);
    for &(feat, imp) in &importances {
        let bar_len = ((imp / max_imp) * 20.0).max(0.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("{:>15} {:>12.4} {:>20}", feature_names[feat], imp, bar);
    }

    // --- Method 2: Partial Dependence ---
    println!("\n=== Partial Dependence (Top Feature) ===\n");

    let top_feature = importances[0].0;
    let top_name = feature_names[top_feature];

    println!("Feature: {} (index {})\n", top_name, top_feature);

    // Compute PDP: average prediction as feature varies
    let mut feat_vals: Vec<f64> = x_test.iter().map(|r| r[top_feature]).collect();
    feat_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_val = feat_vals[0];
    let max_val = feat_vals[feat_vals.len() - 1];

    println!("{:>10} {:>12} {:>20}", top_name, "Avg Pred", "Visual");
    println!("{}", "-".repeat(45));

    let n_grid = 10;
    for i in 0..=n_grid {
        let val = min_val + (max_val - min_val) * i as f64 / n_grid as f64;

        // Average prediction with this feature value, other features from data
        let avg_pred: f64 = x_test.iter().map(|row| {
            let mut modified = row.clone();
            modified[top_feature] = val;
            model_predict(&modified)
        }).sum::<f64>() / x_test.len() as f64;

        let bar_len = (avg_pred * 30.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("{:>10.1} {:>12.3} {}", val, avg_pred, bar);
    }

    // --- Method 3: Local Explanation (SHAP-like) ---
    println!("\n=== Local Explanation (SHAP-like attribution) ===\n");

    // Pick a specific test case
    let test_idx = 0;
    let test_point = &x_test[test_idx];
    let test_pred = model_predict(test_point);
    let test_label = y_test[test_idx];

    println!("Test instance #{}: prediction={:.3}, actual={}",
        test_idx, test_pred, test_label);
    println!("Features: {:?}\n",
        test_point.iter().enumerate()
            .map(|(j, &v)| format!("{}={:.1}", feature_names[j], v))
            .collect::<Vec<_>>());

    // Approximate SHAP using marginal contributions
    // For each feature, compare prediction with and without it
    // (replace with average value when "without")
    let feature_means: Vec<f64> = (0..n_feat).map(|j| {
        x_train.iter().map(|r| r[j]).sum::<f64>() / n_train as f64
    }).collect();

    let base_prediction: f64 = {
        // Prediction with all features at their mean
        model_predict(&feature_means)
    };

    let mut shap_values: Vec<(usize, f64)> = Vec::new();

    // Simple marginal contribution (not full Shapley, but illustrative)
    for feat in 0..n_feat {
        // Prediction with this feature present, others at mean
        let mut with_feature = feature_means.clone();
        with_feature[feat] = test_point[feat];

        // Average over multiple background samples
        let mut contribution = 0.0;
        let n_bg = 50;
        for bg_idx in 0..n_bg {
            let bg = &x_train[bg_idx % n_train];

            // With feature
            let mut x_with = bg.to_vec();
            x_with[feat] = test_point[feat];
            let pred_with = model_predict(&x_with);

            // Without feature (use background)
            let pred_without = model_predict(bg);

            contribution += pred_with - pred_without;
        }
        contribution /= n_bg as f64;

        shap_values.push((feat, contribution));
    }

    // Normalize so they sum to (prediction - base)
    let shap_sum: f64 = shap_values.iter().map(|(_, v)| v).sum();
    let target_sum = test_pred - base_prediction;
    if shap_sum.abs() > 1e-10 {
        let scale = target_sum / shap_sum;
        for sv in &mut shap_values {
            sv.1 *= scale;
        }
    }

    shap_values.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Base prediction (average): {:.3}", base_prediction);
    println!();
    println!("{:>15} {:>10} {:>10} {:>15}", "Feature", "Value", "SHAP", "Direction");
    println!("{}", "-".repeat(53));

    for &(feat, shap) in &shap_values {
        let direction = if shap > 0.01 { "+" }
            else if shap < -0.01 { "-" }
            else { "~" };
        let bar_len = (shap.abs() * 50.0).min(20.0) as usize;
        let bar: String = if shap >= 0.0 {
            std::iter::repeat('+').take(bar_len).collect()
        } else {
            std::iter::repeat('-').take(bar_len).collect()
        };
        println!("{:>15} {:>10.1} {:>+10.3} {:>15}", feature_names[feat], test_point[feat], shap, bar);
    }

    println!("\n= Final prediction: {:.3}", test_pred);
    println!("= Decision: {}", if test_pred > 0.5 { "APPROVED" } else { "DENIED" });

    // --- Method 4: Stump-based feature usage ---
    println!("\n=== Model Structure: Feature Usage in Stumps ===\n");

    let mut stump_counts = vec![0_usize; n_feat];
    for s in &stumps {
        stump_counts[s.feature] += 1;
    }

    println!("{:>15} {:>10} {:>10}", "Feature", "# Stumps", "Usage %");
    println!("{}", "-".repeat(38));
    for feat in 0..n_feat {
        println!("{:>15} {:>10} {:>9.1}%",
            feature_names[feat], stump_counts[feat],
            stump_counts[feat] as f64 / n_stumps as f64 * 100.0);
    }

    // --- Comparison of methods ---
    println!("\n=== Feature Ranking Comparison ===\n");
    let mut perm_rank: Vec<(usize, usize)> = importances.iter().enumerate()
        .map(|(rank, &(feat, _))| (feat, rank + 1)).collect();
    perm_rank.sort_by_key(|&(feat, _)| feat);

    let mut usage_order: Vec<(usize, usize)> = (0..n_feat).collect::<Vec<_>>().iter()
        .map(|&f| (f, stump_counts[f])).collect();
    usage_order.sort_by(|a, b| b.1.cmp(&a.1));
    let mut usage_rank: Vec<(usize, usize)> = usage_order.iter().enumerate()
        .map(|(rank, &(feat, _))| (feat, rank + 1)).collect();
    usage_rank.sort_by_key(|&(feat, _)| feat);

    println!("{:>15} {:>15} {:>15}", "Feature", "Perm Rank", "Usage Rank");
    println!("{}", "-".repeat(48));
    for feat in 0..n_feat {
        println!("{:>15} {:>15} {:>15}",
            feature_names[feat], perm_rank[feat].1, usage_rank[feat].1);
    }

    let top_feature_name = feature_names[importances[0].0];
    println!();
    println!("kata_metric(\"test_accuracy\", {:.3})", test_acc);
    println!("kata_metric(\"top_feature\", \"{}\")", top_feature_name);
    println!("kata_metric(\"top_importance\", {:.4})", importances[0].1);
}
```

---

## Key Takeaways

- **Model interpretability is essential for trust, debugging, and compliance.** Stakeholders need to understand why a model makes specific predictions, especially in high-stakes domains.
- **Global methods (feature importance, PDP) explain the model overall**, while local methods (SHAP) explain individual predictions. Both are needed.
- **Permutation importance measures how much accuracy drops when a feature is shuffled.** It works for any model and accounts for feature interactions.
- **Partial dependence plots show the marginal effect of a feature on predictions**, isolating one variable while averaging over others.
- **SHAP values provide theoretically grounded, fair attributions for individual predictions**, making them the gold standard for local explanation.
