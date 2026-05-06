# Responsible AI

> Phase 11 — Productionizing ML | Kata 11.6

---

## Concept & Intuition

### What problem are we solving?

Machine learning models increasingly make decisions that affect people's lives: who gets a loan, who gets hired, who gets paroled, what news you see, what medical treatment you receive. These decisions carry enormous ethical weight. A model trained on historically biased data will perpetuate and even amplify those biases. A model that performs well on average may systematically fail for minority groups. **Responsible AI** is the discipline of detecting, measuring, and mitigating these harms.

**Fairness** in ML means that a model's predictions do not systematically disadvantage people based on protected attributes like race, gender, age, or disability. But "fair" can mean many things mathematically -- and different fairness definitions are often mutually incompatible. **Demographic parity** requires equal positive prediction rates across groups. **Equalized odds** requires equal true positive and false positive rates. **Calibration** requires that a predicted probability of 70% means the same thing for all groups. You cannot satisfy all three simultaneously (except in trivial cases), so choosing a fairness metric is itself an ethical decision.

**Bias detection** involves auditing a model for disparities across groups. **Bias mitigation** involves fixing those disparities, through preprocessing (rebalancing data), in-processing (adding fairness constraints to the loss function), or post-processing (adjusting thresholds per group). Each approach has trade-offs between fairness and accuracy.

### Why naive approaches fail

Removing protected attributes (like race or gender) from the input does not make a model fair. Other features -- zip code, name, purchasing habits -- can be correlated with protected attributes and serve as proxies. This is called "fairness through unawareness," and it is widely recognized as insufficient. You must measure outcomes across groups to detect bias, not just check whether the model has access to protected attributes.

### Mental models

- **Bias in, bias out**: a model trained on historical hiring data where women were underrepresented in leadership will learn to penalize women. The model is faithfully learning patterns in the data -- but those patterns reflect historical injustice, not ground truth.
- **Fairness as a multi-stakeholder problem**: different stakeholders (applicants, lenders, regulators) may prefer different fairness definitions. There is no universal "correct" definition -- it depends on the context and values.
- **The impossibility theorem**: you cannot simultaneously achieve demographic parity, equalized odds, and calibration (except when base rates are equal across groups). Choosing a fairness metric is a value judgment, not a technical decision.

### Visual explanations

```
Fairness Metrics:

  Demographic Parity:
    P(Y_hat=1 | group=A) = P(Y_hat=1 | group=B)
    "Equal acceptance rates regardless of group"

  Equalized Odds:
    P(Y_hat=1 | Y=1, group=A) = P(Y_hat=1 | Y=1, group=B)  (equal TPR)
    P(Y_hat=1 | Y=0, group=A) = P(Y_hat=1 | Y=0, group=B)  (equal FPR)
    "Equal accuracy regardless of group"

  Calibration:
    P(Y=1 | Y_hat=p, group=A) = P(Y=1 | Y_hat=p, group=B) = p
    "Predicted probabilities mean the same thing for all groups"

Sources of bias in ML:
  Data collection    --> Who is in the dataset? Who is missing?
  Labeling           --> Are labels equally accurate for all groups?
  Feature selection  --> Do features serve as proxies for protected attributes?
  Model training     --> Does the model amplify existing disparities?
  Deployment         --> Are all groups equally served by the system?
```

---

## Hands-on Exploration

1. Generate a synthetic dataset with a protected attribute (group A and group B). Train a classifier. Compute accuracy separately for each group -- are they the same?
2. Compute demographic parity, equalized odds, and calibration. Does the model satisfy all three?
3. Apply a simple mitigation: adjust the classification threshold separately for each group to achieve demographic parity. What happens to overall accuracy?
4. Discuss: if the base rates are genuinely different between groups, is it "fair" to force equal prediction rates?

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

    println!("=== Responsible AI: Fairness Analysis ===\n");

    // --- Generate synthetic data with protected attribute ---
    let n = 2000;
    let bias_strength = 1.0;

    let mut group: Vec<usize> = Vec::new(); // 0 = Group A, 1 = Group B
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    let feature_names = ["income", "credit_score", "education", "experience", "debt_ratio"];
    let n_feat = feature_names.len();

    for _ in 0..n {
        let g = if rand_f64(&mut rng) < 0.4 { 1 } else { 0 }; // 40% Group B
        group.push(g);

        let income = 50.0 + rand_normal(&mut rng) * 15.0 + g as f64 * 5.0;
        let credit = 650.0 + rand_normal(&mut rng) * 80.0 - g as f64 * bias_strength * 20.0;
        let education = 14.0 + rand_normal(&mut rng) * 3.0 + g as f64 * 1.0;
        let experience = 10.0 + rand_normal(&mut rng) * 5.0;
        let debt = 0.35 + rand_normal(&mut rng) * 0.15;

        features.push(vec![income, credit, education, experience, debt]);

        // Label influenced by features AND group (systemic bias)
        let logit = 0.02 * (credit - 650.0) + 0.01 * (income - 50.0)
            + 0.05 * (education - 14.0) - 0.5 * debt
            - bias_strength * 0.3 * g as f64;
        let prob = 1.0 / (1.0 + (-logit).exp());
        let label = if rand_f64(&mut rng) < prob { 1 } else { 0 };
        labels.push(label);
    }

    let n_a = group.iter().filter(|&&g| g == 0).count();
    let n_b = group.iter().filter(|&&g| g == 1).count();
    let rate_a = labels.iter().zip(&group).filter(|(&l, &&g)| g == 0 && l == 1).count() as f64 / n_a as f64;
    let rate_b = labels.iter().zip(&group).filter(|(&l, &&g)| g == 1 && l == 1).count() as f64 / n_b as f64;

    println!("Samples: {}, Bias strength: {}", n, bias_strength);
    println!("Group A: {}, Group B: {}", n_a, n_b);
    println!("Base rate Group A: {:.3}", rate_a);
    println!("Base rate Group B: {:.3}\n", rate_b);

    // --- Split data ---
    let n_train = 1400;

    // --- Train logistic regression (without group as feature) ---
    let mut weights = vec![0.0; n_feat];
    let mut bias = 0.0;
    let lr = 0.001;

    for _ in 0..1000 {
        let mut grad_w = vec![0.0; n_feat];
        let mut grad_b = 0.0;
        for i in 0..n_train {
            let logit: f64 = bias + weights.iter().zip(&features[i]).map(|(w, x)| w * x).sum::<f64>();
            let pred = 1.0 / (1.0 + (-logit).exp());
            let err = pred - labels[i] as f64;
            for j in 0..n_feat {
                grad_w[j] += err * features[i][j];
            }
            grad_b += err;
        }
        for j in 0..n_feat {
            weights[j] -= lr * grad_w[j] / n_train as f64;
        }
        bias -= lr * grad_b / n_train as f64;
    }

    // Predict on test set
    let test_start = n_train;
    let n_test = n - n_train;

    let predict_prob = |x: &[f64]| -> f64 {
        let logit = bias + weights.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>();
        1.0 / (1.0 + (-logit).exp())
    };

    let probs: Vec<f64> = (test_start..n).map(|i| predict_prob(&features[i])).collect();
    let preds: Vec<usize> = probs.iter().map(|&p| if p > 0.5 { 1 } else { 0 }).collect();
    let test_labels = &labels[test_start..];
    let test_group = &group[test_start..];

    let overall_acc = preds.iter().zip(test_labels)
        .filter(|(&p, &y)| p == y).count() as f64 / n_test as f64;

    println!("Model: Logistic Regression (no group feature)");
    println!("Overall accuracy: {:.4}\n", overall_acc);

    // --- Fairness Metrics ---
    struct GroupMetrics {
        n: usize,
        base_rate: f64,
        positive_rate: f64,
        accuracy: f64,
        tpr: f64,
        fpr: f64,
        avg_prob: f64,
    }

    let compute_group_metrics = |preds: &[usize], probs: &[f64], labels: &[usize], groups: &[usize], target_group: usize| -> GroupMetrics {
        let mask: Vec<bool> = groups.iter().map(|&g| g == target_group).collect();
        let n = mask.iter().filter(|&&m| m).count();
        let mut tp = 0; let mut fp = 0; let mut tn = 0; let mut fn_ = 0;
        let mut prob_sum = 0.0;
        let mut base_pos = 0;
        let mut pred_pos = 0;

        for i in 0..preds.len() {
            if !mask[i] { continue; }
            prob_sum += probs[i];
            if labels[i] == 1 { base_pos += 1; }
            if preds[i] == 1 { pred_pos += 1; }

            match (preds[i], labels[i]) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_ += 1,
                _ => {}
            }
        }

        GroupMetrics {
            n,
            base_rate: base_pos as f64 / n as f64,
            positive_rate: pred_pos as f64 / n as f64,
            accuracy: (tp + tn) as f64 / n as f64,
            tpr: if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 },
            fpr: if fp + tn > 0 { fp as f64 / (fp + tn) as f64 } else { 0.0 },
            avg_prob: prob_sum / n as f64,
        }
    };

    let metrics_a = compute_group_metrics(&preds, &probs, test_labels, test_group, 0);
    let metrics_b = compute_group_metrics(&preds, &probs, test_labels, test_group, 1);

    println!("=== Per-Group Performance ===\n");
    println!("{:>20} {:>10} {:>10} {:>10}", "Metric", "Group A", "Group B", "Disparity");
    println!("{}", "-".repeat(55));

    let metrics_table: Vec<(&str, f64, f64)> = vec![
        ("Sample Size", metrics_a.n as f64, metrics_b.n as f64),
        ("Base Rate", metrics_a.base_rate, metrics_b.base_rate),
        ("Positive Rate", metrics_a.positive_rate, metrics_b.positive_rate),
        ("Accuracy", metrics_a.accuracy, metrics_b.accuracy),
        ("True Pos Rate", metrics_a.tpr, metrics_b.tpr),
        ("False Pos Rate", metrics_a.fpr, metrics_b.fpr),
        ("Avg Pred Prob", metrics_a.avg_prob, metrics_b.avg_prob),
    ];

    for (label, a, b) in &metrics_table {
        let disp = (a - b).abs();
        let flag = if disp > 0.05 && *label != "Sample Size" && *label != "Base Rate" { " !" } else { "" };
        println!("{:>20} {:>10.4} {:>10.4} {:>10.4}{}", label, a, b, disp, flag);
    }

    // --- Formal Fairness Definitions ---
    println!("\n=== Fairness Definitions ===\n");

    // Demographic Parity
    let dp_diff = (metrics_a.positive_rate - metrics_b.positive_rate).abs();
    let dp_max = metrics_a.positive_rate.max(metrics_b.positive_rate);
    let dp_ratio = if dp_max > 0.0 {
        metrics_a.positive_rate.min(metrics_b.positive_rate) / dp_max
    } else { 1.0 };

    println!("1. Demographic Parity:");
    println!("   P(Y_hat=1|A) = {:.4}", metrics_a.positive_rate);
    println!("   P(Y_hat=1|B) = {:.4}", metrics_b.positive_rate);
    println!("   Difference:   {:.4} (ideal: 0)", dp_diff);
    println!("   Ratio:        {:.4} (4/5 rule threshold: 0.80)", dp_ratio);
    println!("   Status:       {}", if dp_ratio >= 0.80 { "FAIR" } else { "UNFAIR (violates 4/5 rule)" });

    // Equalized Odds
    let tpr_diff = (metrics_a.tpr - metrics_b.tpr).abs();
    let fpr_diff = (metrics_a.fpr - metrics_b.fpr).abs();

    println!("\n2. Equalized Odds:");
    println!("   TPR Group A: {:.4}, TPR Group B: {:.4} (diff: {:.4})",
        metrics_a.tpr, metrics_b.tpr, tpr_diff);
    println!("   FPR Group A: {:.4}, FPR Group B: {:.4} (diff: {:.4})",
        metrics_a.fpr, metrics_b.fpr, fpr_diff);
    println!("   Status:       {}", if tpr_diff < 0.05 && fpr_diff < 0.05 { "FAIR" } else { "UNFAIR" });

    // --- Bias Mitigation: Threshold Adjustment ---
    println!("\n=== Bias Mitigation: Threshold Adjustment ===\n");

    // Find threshold for Group B that matches Group A's positive rate
    let target_rate = metrics_a.positive_rate;

    // Binary search for threshold
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let rate: f64 = probs.iter().zip(test_group)
            .filter(|(_, &&g)| g == 1)
            .map(|(&p, _)| if p >= mid { 1.0 } else { 0.0 })
            .sum::<f64>() / test_group.iter().filter(|&&g| g == 1).count() as f64;
        if rate > target_rate { lo = mid; } else { hi = mid; }
    }
    let threshold_b = (lo + hi) / 2.0;

    // Apply adjusted thresholds
    let fair_preds: Vec<usize> = probs.iter().zip(test_group).map(|(&p, &g)| {
        let thresh = if g == 0 { 0.5 } else { threshold_b };
        if p >= thresh { 1 } else { 0 }
    }).collect();

    let fair_acc = fair_preds.iter().zip(test_labels)
        .filter(|(&p, &y)| p == y).count() as f64 / n_test as f64;

    let fair_metrics_a = compute_group_metrics(&fair_preds, &probs, test_labels, test_group, 0);
    let fair_metrics_b = compute_group_metrics(&fair_preds, &probs, test_labels, test_group, 1);

    println!("Original threshold: 0.5 for both groups");
    println!("Adjusted thresholds: Group A = 0.500, Group B = {:.3}\n", threshold_b);

    println!("{:>20} {:>10} {:>10}", "Metric", "Before", "After");
    println!("{}", "-".repeat(45));
    println!("{:>20} {:>10.4} {:>10.4}", "Accuracy (overall)", overall_acc, fair_acc);
    println!("{:>20} {:>10.4} {:>10.4}", "Pos Rate A", metrics_a.positive_rate, fair_metrics_a.positive_rate);
    println!("{:>20} {:>10.4} {:>10.4}", "Pos Rate B", metrics_b.positive_rate, fair_metrics_b.positive_rate);
    let new_dp_diff = (fair_metrics_a.positive_rate - fair_metrics_b.positive_rate).abs();
    println!("{:>20} {:>10.4} {:>10.4}", "DP Difference", dp_diff, new_dp_diff);

    if fair_acc < overall_acc {
        println!("\nAccuracy decreased by {:.2}% to achieve fairness.",
            (overall_acc - fair_acc) * 100.0);
        println!("This is the fairness-accuracy trade-off.");
    }

    // --- Proxy Detection ---
    println!("\n=== Proxy Detection ===\n");
    println!("Correlation between features and protected attribute (group):\n");
    println!("{:>15} {:>12} {:>18}", "Feature", "Correlation", "Potential Proxy?");
    println!("{}", "-".repeat(48));

    for j in 0..n_feat {
        let feat_vals: Vec<f64> = (test_start..n).map(|i| features[i][j]).collect();
        let group_vals: Vec<f64> = test_group.iter().map(|&g| g as f64).collect();

        let mean_f = feat_vals.iter().sum::<f64>() / n_test as f64;
        let mean_g = group_vals.iter().sum::<f64>() / n_test as f64;
        let cov: f64 = feat_vals.iter().zip(&group_vals)
            .map(|(f, g)| (f - mean_f) * (g - mean_g)).sum::<f64>() / n_test as f64;
        let std_f = (feat_vals.iter().map(|f| (f - mean_f).powi(2)).sum::<f64>() / n_test as f64).sqrt();
        let std_g = (group_vals.iter().map(|g| (g - mean_g).powi(2)).sum::<f64>() / n_test as f64).sqrt();
        let corr = if std_f * std_g > 1e-10 { cov / (std_f * std_g) } else { 0.0 };

        let is_proxy = if corr.abs() > 0.1 { "YES" } else { "no" };
        println!("{:>15} {:>+12.4} {:>18}", feature_names[j], corr, is_proxy);
    }

    println!("\nNote: Even without 'group' as a feature, the model can learn");
    println!("to discriminate through proxy features correlated with group membership.");

    // --- Responsible AI Checklist ---
    println!("\n=== Responsible AI Checklist ===\n");
    let checklist = [
        ("Data representativeness", "Are all groups adequately represented in training data?"),
        ("Label quality", "Are labels equally accurate/available across groups?"),
        ("Feature audit", "Do any features serve as proxies for protected attributes?"),
        ("Fairness metrics", "Which fairness definition is appropriate for this use case?"),
        ("Disparate impact", "Does the model's error rate differ across groups?"),
        ("Mitigation strategy", "What interventions are appropriate if bias is found?"),
        ("Transparency", "Can affected individuals understand why a decision was made?"),
        ("Recourse", "Can affected individuals contest or appeal a decision?"),
        ("Monitoring", "Is fairness monitored continuously in production?"),
        ("Stakeholder input", "Have affected communities been consulted?"),
    ];

    for (i, (item, question)) in checklist.iter().enumerate() {
        println!("  {:>2}. [ ] {}", i + 1, item);
        println!("       {}", question);
    }

    println!();
    println!("kata_metric(\"demographic_parity_ratio\", {:.4})", dp_ratio);
    println!("kata_metric(\"tpr_difference\", {:.4})", tpr_diff);
    println!("kata_metric(\"accuracy_before\", {:.4})", overall_acc);
    println!("kata_metric(\"accuracy_after_mitigation\", {:.4})", fair_acc);
}
```

---

## Key Takeaways

- **Removing protected attributes does not make a model fair.** Other features can serve as proxies for race, gender, or other protected characteristics. You must measure outcomes across groups.
- **Fairness has multiple, often conflicting mathematical definitions.** Demographic parity, equalized odds, and calibration cannot all be satisfied simultaneously (when base rates differ). Choosing a fairness metric is a value judgment.
- **Bias mitigation involves trade-offs.** Adjusting thresholds to achieve demographic parity may reduce overall accuracy. Pre-processing, in-processing, and post-processing methods each have different trade-off profiles.
- **The four-fifths rule is a practical heuristic.** If the positive prediction rate for any group is less than 80% of the rate for the most favored group, disparate impact may be present.
- **Responsible AI is not just a technical problem.** It requires stakeholder engagement, transparency, recourse mechanisms, and ongoing monitoring. Technical tools like fairness metrics are necessary but not sufficient.
