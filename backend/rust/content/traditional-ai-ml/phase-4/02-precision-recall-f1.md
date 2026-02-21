# Precision, Recall, and F1-Score

> Phase 4 — Model Evaluation & Selection | Kata 4.02

---

## Concept & Intuition

### What problem are we solving?

When classes are imbalanced or error costs are asymmetric, accuracy alone is insufficient. Precision and recall decompose model performance into two complementary questions. Precision asks: "Of all items the model flagged as positive, what fraction are actually positive?" Recall asks: "Of all actually positive items, what fraction did the model find?" The F1-score is their harmonic mean, providing a single balanced metric.

These metrics are essential in domains where one type of error matters more. In spam filtering, low precision means legitimate emails go to spam (annoying). In cancer screening, low recall means sick patients go undiagnosed (dangerous). Understanding precision-recall tradeoffs lets you tune your model to the specific costs of your application.

In this kata, we implement precision, recall, F1-score, and the precision-recall curve from scratch. We demonstrate how threshold tuning trades precision for recall and vice versa.

### Why naive approaches fail

Optimizing for accuracy on imbalanced data often produces models that ignore the minority class entirely. Even among models that detect the minority class, there is always a tradeoff: being more aggressive (lower threshold) catches more true positives but also more false positives. Without precision-recall analysis, you cannot make this tradeoff explicit. The F1-score provides a useful default balance, but the right tradeoff point depends on the application.

### Mental models

- **Precision as purity**: Of everything the model calls positive, how pure is that set? High precision means few false alarms.
- **Recall as completeness**: Of everything that truly is positive, how complete is the model's detection? High recall means few missed cases.
- **F1 as harmonic mean**: The harmonic mean penalizes extreme imbalances. Precision=0.9 and recall=0.1 gives F1=0.18, not the arithmetic mean 0.5. Both must be high for F1 to be high.

### Visual explanations

```
  All items:    [+ + + - - - - - - - - -]  (3 positive, 9 negative)

  Model says:   [Y Y N N N Y N N N N N N]  (Y = predicted positive)

  TP=2  FP=1  FN=1  TN=8

  Precision = TP/(TP+FP) = 2/3 = 0.67   ("Of 3 flagged, 2 are real")
  Recall    = TP/(TP+FN) = 2/3 = 0.67   ("Of 3 real, found 2")
  F1        = 2*P*R/(P+R) = 0.67

  Lower threshold -> more Y's -> higher recall, lower precision
  Higher threshold -> fewer Y's -> lower recall, higher precision
```

---

## Hands-on Exploration

1. Compute precision, recall, and F1 from a confusion matrix.
2. Build a precision-recall curve by varying the classification threshold.
3. Identify the threshold that maximizes F1.
4. Compare two models using precision-recall analysis.

---

## Live Code

```rust
fn main() {
    println!("=== Precision, Recall, and F1-Score ===\n");

    // Simulate two models' probability outputs
    let actual = vec![1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
    // 8 positive, 22 negative

    // Model A: decent, separates well
    let probs_a = vec![
        0.95, 0.88, 0.82, 0.75, 0.70, 0.60, 0.55, 0.45,  // positives
        0.40, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18,  // negatives
        0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03,
        0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
    ];

    // Model B: less calibrated, more overlap
    let probs_b = vec![
        0.90, 0.80, 0.65, 0.50, 0.45, 0.40, 0.35, 0.30,  // positives
        0.55, 0.48, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15,  // negatives
        0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02,
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01,
    ];

    println!("Dataset: {} samples ({} positive, {} negative)\n",
        actual.len(),
        actual.iter().filter(|&&a| a == 1).count(),
        actual.iter().filter(|&&a| a == 0).count());

    // Default threshold (0.5)
    println!("--- At Default Threshold (0.5) ---");
    let preds_a: Vec<usize> = probs_a.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect();
    let preds_b: Vec<usize> = probs_b.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect();

    println!("  Model A:");
    print_full_metrics(&actual, &preds_a);
    println!("  Model B:");
    print_full_metrics(&actual, &preds_b);

    // Precision-Recall curve for Model A
    println!("\n--- Precision-Recall Curve: Model A ---");
    let pr_a = precision_recall_curve(&actual, &probs_a);
    println!("{:<12} {:>10} {:>10} {:>10}", "Threshold", "Precision", "Recall", "F1");
    println!("{}", "-".repeat(44));

    let mut best_f1_a = 0.0;
    let mut best_thresh_a = 0.5;

    for &(thresh, prec, rec, f1) in &pr_a {
        if f1 > best_f1_a { best_f1_a = f1; best_thresh_a = thresh; }
        println!("{:<12.2} {:>10.3} {:>10.3} {:>10.3}", thresh, prec, rec, f1);
    }
    println!("\n  Best F1: {:.3} at threshold {:.2}", best_f1_a, best_thresh_a);

    // Precision-Recall curve for Model B
    println!("\n--- Precision-Recall Curve: Model B ---");
    let pr_b = precision_recall_curve(&actual, &probs_b);
    let mut best_f1_b = 0.0;
    let mut best_thresh_b = 0.5;

    for &(thresh, prec, rec, f1) in &pr_b {
        if f1 > best_f1_b { best_f1_b = f1; best_thresh_b = thresh; }
    }
    println!("  Best F1: {:.3} at threshold {:.2}", best_f1_b, best_thresh_b);

    // ASCII PR curve
    println!("\n--- ASCII Precision-Recall Curve ---");
    print_pr_curve(&pr_a, &pr_b);

    // F-beta scores for different betas
    println!("\n--- F-beta Scores (at optimal F1 threshold) ---");
    let preds_opt_a: Vec<usize> = probs_a.iter().map(|&p| if p >= best_thresh_a { 1 } else { 0 }).collect();
    let (_, prec_a, rec_a, _) = compute_metrics(&actual, &preds_opt_a);

    println!("{:<12} {:>10} {:>10} {}", "Beta", "F-beta", "Emphasis", "");
    println!("{}", "-".repeat(40));
    for &beta in &[0.5, 1.0, 2.0] {
        let fb = f_beta(prec_a, rec_a, beta);
        let emphasis = if beta < 1.0 { "precision" } else if beta > 1.0 { "recall" } else { "balanced" };
        println!("{:<12.1} {:>10.3} {:>10}", beta, fb, emphasis);
    }

    println!("\n  F0.5 emphasizes precision (fewer false positives)");
    println!("  F1   balances precision and recall equally");
    println!("  F2   emphasizes recall (fewer missed positives)");

    // Micro vs macro averaging
    println!("\n--- Micro vs Macro Averaging ---");
    let actual_mc = vec![0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2];
    let pred_mc   = vec![0,0,0,0,0,0,0,0,0,0, 1,1,1,0,0, 2,2,2,0,0];

    let (micro_p, micro_r, micro_f1) = micro_average(&actual_mc, &pred_mc, 3);
    let (macro_p, macro_r, macro_f1) = macro_average(&actual_mc, &pred_mc, 3);

    println!("  Micro: P={:.3}, R={:.3}, F1={:.3} (weights by class size)", micro_p, micro_r, micro_f1);
    println!("  Macro: P={:.3}, R={:.3}, F1={:.3} (treats classes equally)", macro_p, macro_r, macro_f1);

    kata_metric("model_a_best_f1", best_f1_a);
    kata_metric("model_b_best_f1", best_f1_b);
    kata_metric("model_a_best_threshold", best_thresh_a);
    kata_metric("micro_f1", micro_f1);
    kata_metric("macro_f1", macro_f1);
}

fn compute_metrics(actual: &[usize], predicted: &[usize]) -> (f64, f64, f64, f64) {
    let mut tp = 0; let mut fp = 0; let mut fn_ = 0; let mut tn = 0;
    for (&a, &p) in actual.iter().zip(predicted.iter()) {
        match (a, p) { (1,1)=>tp+=1, (0,1)=>fp+=1, (1,0)=>fn_+=1, (0,0)=>tn+=1, _=>{} }
    }
    let acc = (tp + tn) as f64 / actual.len() as f64;
    let prec = if tp+fp>0 { tp as f64/(tp+fp) as f64 } else { 0.0 };
    let rec = if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 };
    let f1 = if prec+rec>0.0 { 2.0*prec*rec/(prec+rec) } else { 0.0 };
    (acc, prec, rec, f1)
}

fn print_full_metrics(actual: &[usize], predicted: &[usize]) {
    let (acc, prec, rec, f1) = compute_metrics(actual, predicted);
    println!("    Accuracy={:.1}%, Precision={:.3}, Recall={:.3}, F1={:.3}",
        acc*100.0, prec, rec, f1);
}

fn precision_recall_curve(actual: &[usize], probs: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut curve = Vec::new();
    for &t in &thresholds {
        let preds: Vec<usize> = probs.iter().map(|&p| if p >= t { 1 } else { 0 }).collect();
        let (_, prec, rec, f1) = compute_metrics(actual, &preds);
        curve.push((t, prec, rec, f1));
    }
    curve
}

fn f_beta(precision: f64, recall: f64, beta: f64) -> f64 {
    let b2 = beta * beta;
    if precision + recall > 0.0 {
        (1.0 + b2) * precision * recall / (b2 * precision + recall)
    } else {
        0.0
    }
}

fn micro_average(actual: &[usize], predicted: &[usize], n_classes: usize) -> (f64, f64, f64) {
    let mut tp = 0; let mut fp = 0; let mut fn_ = 0;
    for c in 0..n_classes {
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            if a == c && p == c { tp += 1; }
            else if a != c && p == c { fp += 1; }
            else if a == c && p != c { fn_ += 1; }
        }
    }
    let p = if tp+fp>0 { tp as f64/(tp+fp) as f64 } else { 0.0 };
    let r = if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 };
    let f1 = if p+r>0.0 { 2.0*p*r/(p+r) } else { 0.0 };
    (p, r, f1)
}

fn macro_average(actual: &[usize], predicted: &[usize], n_classes: usize) -> (f64, f64, f64) {
    let mut precs = Vec::new(); let mut recs = Vec::new();
    for c in 0..n_classes {
        let mut tp=0; let mut fp=0; let mut fn_=0;
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            if a==c && p==c { tp+=1; } else if a!=c && p==c { fp+=1; } else if a==c && p!=c { fn_+=1; }
        }
        precs.push(if tp+fp>0 { tp as f64/(tp+fp) as f64 } else { 0.0 });
        recs.push(if tp+fn_>0 { tp as f64/(tp+fn_) as f64 } else { 0.0 });
    }
    let p = precs.iter().sum::<f64>()/n_classes as f64;
    let r = recs.iter().sum::<f64>()/n_classes as f64;
    let f1 = if p+r>0.0 { 2.0*p*r/(p+r) } else { 0.0 };
    (p, r, f1)
}

fn print_pr_curve(curve_a: &[(f64,f64,f64,f64)], curve_b: &[(f64,f64,f64,f64)]) {
    let height = 10;
    let width = 30;
    let mut grid = vec![vec![' '; width]; height];

    for &(_, prec, rec, _) in curve_a {
        let col = (rec * (width-1) as f64).round() as usize;
        let row = ((1.0-prec) * (height-1) as f64).round() as usize;
        if col < width && row < height { grid[row][col] = 'A'; }
    }
    for &(_, prec, rec, _) in curve_b {
        let col = (rec * (width-1) as f64).round() as usize;
        let row = ((1.0-prec) * (height-1) as f64).round() as usize;
        if col < width && row < height { grid[row][col] = 'B'; }
    }

    println!("  Precision");
    for (i, row) in grid.iter().enumerate() {
        let p = 1.0 - i as f64 / (height-1) as f64;
        let line: String = row.iter().collect();
        println!("  {:.1} |{}", p, line);
    }
    println!("      +{}", "-".repeat(width));
    println!("       0.0          Recall          1.0");
    println!("       A = Model A, B = Model B");
}

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Precision measures the purity of positive predictions; recall measures the completeness of positive detection. Both are needed for a full picture.
- The F1-score (harmonic mean of precision and recall) provides a single balanced metric. F-beta scores allow emphasizing one over the other.
- The precision-recall curve reveals how the tradeoff changes with threshold, helping you pick the right operating point for your application.
- Micro averaging weights by class size (favors majority); macro averaging treats classes equally (highlights minority class performance).
