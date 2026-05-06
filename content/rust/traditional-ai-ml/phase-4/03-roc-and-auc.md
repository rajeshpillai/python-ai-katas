# ROC and AUC

> Phase 4 — Model Evaluation & Selection | Kata 4.03

---

## Concept & Intuition

### What problem are we solving?

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (recall) against the False Positive Rate at every possible classification threshold. The Area Under the ROC Curve (AUC) summarizes a model's ability to discriminate between classes into a single number between 0 and 1. An AUC of 0.5 means the model is no better than random; an AUC of 1.0 means perfect separation.

Unlike precision-recall, the ROC curve is invariant to class balance, making it a standard way to compare classifiers across different datasets. However, this invariance can be a disadvantage: on highly imbalanced data, the ROC curve can look overly optimistic because the FPR denominator (total negatives) is large, making even many false positives appear as a small rate.

In this kata, we compute the ROC curve, calculate AUC using the trapezoidal rule, and compare ROC analysis with precision-recall analysis to understand when each is most appropriate.

### Why naive approaches fail

Evaluating a classifier at a single threshold ignores most of the model's behavior. The ROC curve shows performance across all thresholds, revealing whether the model fundamentally understands the data or just happens to work at one specific threshold. Two models might have identical accuracy at threshold 0.5 but very different ROC curves — one might be robust across thresholds while the other degrades rapidly.

### Mental models

- **ROC as a threshold sweep**: As you lower the threshold from 1 to 0, you predict more items as positive. The ROC curve traces the tradeoff between catching more true positives (TPR rises) and generating more false positives (FPR rises).
- **AUC as ranking quality**: AUC equals the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example. An AUC of 0.9 means 90% of the time, a positive is scored higher than a negative.
- **Random classifier = diagonal**: A model that assigns random scores produces a 45-degree line from (0,0) to (1,1) with AUC=0.5.

### Visual explanations

```
  ROC Curve:
  TPR (Recall)
  1.0 |         ____-----
      |      __/        Perfect (AUC=1.0)
      |    _/
      |   / Good model (AUC~0.85)
  0.5 |  /
      | /  .  .  .  .  Random (AUC=0.5)
      |/
  0.0 +----------------- FPR
      0.0              1.0

  AUC = area under the curve
  Higher = better discrimination
```

---

## Hands-on Exploration

1. Compute TPR and FPR at multiple thresholds to build the ROC curve.
2. Calculate AUC using the trapezoidal rule.
3. Compare two models using their ROC curves and AUC values.
4. Demonstrate when ROC is preferable to precision-recall and vice versa.

---

## Live Code

```rust
fn main() {
    println!("=== ROC Curve and AUC ===\n");

    // True labels (8 positive, 22 negative)
    let actual = vec![1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

    // Model A: good separation
    let scores_a = vec![
        0.95, 0.90, 0.85, 0.78, 0.72, 0.65, 0.55, 0.48,
        0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20,
        0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04,
        0.03, 0.03, 0.02, 0.01, 0.01, 0.01,
    ];

    // Model B: weaker separation
    let scores_b = vec![
        0.88, 0.75, 0.60, 0.52, 0.45, 0.40, 0.35, 0.30,
        0.58, 0.50, 0.44, 0.38, 0.33, 0.28, 0.22, 0.18,
        0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
        0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
    ];

    // Model C: random
    let mut rng = 42u64;
    let scores_c: Vec<f64> = (0..30).map(|_| { rng = lcg(rng); rng as f64 / u64::MAX as f64 }).collect();

    let n_pos = actual.iter().filter(|&&a| a == 1).count();
    let n_neg = actual.iter().filter(|&&a| a == 0).count();
    println!("Dataset: {} positive, {} negative\n", n_pos, n_neg);

    // Compute ROC curves
    let roc_a = compute_roc(&actual, &scores_a);
    let roc_b = compute_roc(&actual, &scores_b);
    let roc_c = compute_roc(&actual, &scores_c);

    let auc_a = compute_auc(&roc_a);
    let auc_b = compute_auc(&roc_b);
    let auc_c = compute_auc(&roc_c);

    println!("--- AUC Scores ---");
    println!("  Model A (good):   AUC = {:.4}", auc_a);
    println!("  Model B (decent): AUC = {:.4}", auc_b);
    println!("  Model C (random): AUC = {:.4}", auc_c);

    // Detailed ROC table for Model A
    println!("\n--- ROC Table: Model A ---");
    println!("{:<12} {:>8} {:>8} {:>8} {:>8}", "Threshold", "TPR", "FPR", "TP", "FP");
    println!("{}", "-".repeat(48));
    let thresholds = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    for &t in &thresholds {
        let preds: Vec<usize> = scores_a.iter().map(|&s| if s >= t { 1 } else { 0 }).collect();
        let (tp, fp, _, _) = confusion_counts(&actual, &preds);
        let tpr = tp as f64 / n_pos as f64;
        let fpr = fp as f64 / n_neg as f64;
        println!("{:<12.1} {:>8.3} {:>8.3} {:>8} {:>8}", t, tpr, fpr, tp, fp);
    }

    // ASCII ROC curve
    println!("\n--- ASCII ROC Curves ---");
    print_roc_curves(&roc_a, &roc_b, &roc_c);

    // Youden's J statistic (optimal threshold)
    println!("\n--- Optimal Threshold (Youden's J) ---");
    let (best_thresh, best_j) = youdens_j(&actual, &scores_a);
    println!("  Model A: best threshold={:.2}, J={:.4}", best_thresh, best_j);
    let (best_thresh_b, best_j_b) = youdens_j(&actual, &scores_b);
    println!("  Model B: best threshold={:.2}, J={:.4}", best_thresh_b, best_j_b);

    // AUC interpretation
    println!("\n--- AUC Interpretation Guide ---");
    println!("  0.90 - 1.00  Excellent");
    println!("  0.80 - 0.90  Good");
    println!("  0.70 - 0.80  Fair");
    println!("  0.60 - 0.70  Poor");
    println!("  0.50 - 0.60  No better than random");

    // ROC vs PR comparison
    println!("\n--- When to Use ROC vs Precision-Recall ---");
    println!("  ROC: Balanced classes, care about both TPR and FPR");
    println!("  PR:  Imbalanced classes, care mainly about positive class");
    println!("  ROC can look optimistic on imbalanced data because");
    println!("  FPR = FP/N is small even with many false positives when N is large.");

    kata_metric("auc_model_a", auc_a);
    kata_metric("auc_model_b", auc_b);
    kata_metric("auc_random", auc_c);
    kata_metric("optimal_threshold_a", best_thresh);
}

fn compute_roc(actual: &[usize], scores: &[f64]) -> Vec<(f64, f64)> {
    let n_pos = actual.iter().filter(|&&a| a == 1).count() as f64;
    let n_neg = actual.iter().filter(|&&a| a == 0).count() as f64;

    let mut indexed: Vec<(f64, usize)> = scores.iter().zip(actual.iter())
        .map(|(&s, &a)| (s, a)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut roc = vec![(0.0_f64, 0.0_f64)];
    let mut tp = 0.0;
    let mut fp = 0.0;

    for (_, label) in &indexed {
        if *label == 1 { tp += 1.0; } else { fp += 1.0; }
        roc.push((fp / n_neg, tp / n_pos));
    }

    roc
}

fn compute_auc(roc: &[(f64, f64)]) -> f64 {
    let mut auc = 0.0;
    for i in 1..roc.len() {
        let dx = roc[i].0 - roc[i-1].0;
        let avg_y = (roc[i].1 + roc[i-1].1) / 2.0;
        auc += dx * avg_y;
    }
    auc
}

fn confusion_counts(actual: &[usize], predicted: &[usize]) -> (usize, usize, usize, usize) {
    let mut tp=0; let mut fp=0; let mut fn_=0; let mut tn=0;
    for (&a, &p) in actual.iter().zip(predicted.iter()) {
        match (a, p) { (1,1)=>tp+=1, (0,1)=>fp+=1, (1,0)=>fn_+=1, (0,0)=>tn+=1, _=>{} }
    }
    (tp, fp, fn_, tn)
}

fn youdens_j(actual: &[usize], scores: &[f64]) -> (f64, f64) {
    let n_pos = actual.iter().filter(|&&a| a == 1).count() as f64;
    let n_neg = actual.iter().filter(|&&a| a == 0).count() as f64;
    let mut best_j = -1.0;
    let mut best_t = 0.5;

    for t_i in 0..100 {
        let t = t_i as f64 / 100.0;
        let preds: Vec<usize> = scores.iter().map(|&s| if s >= t { 1 } else { 0 }).collect();
        let (tp, fp, _, _) = confusion_counts(actual, &preds);
        let tpr = tp as f64 / n_pos;
        let fpr = fp as f64 / n_neg;
        let j = tpr - fpr;
        if j > best_j { best_j = j; best_t = t; }
    }
    (best_t, best_j)
}

fn print_roc_curves(roc_a: &[(f64,f64)], roc_b: &[(f64,f64)], roc_c: &[(f64,f64)]) {
    let h = 12; let w = 30;
    let mut grid = vec![vec![' '; w]; h];

    // Diagonal (random)
    for i in 0..w {
        let r = ((1.0 - i as f64 / (w-1) as f64) * (h-1) as f64).round() as usize;
        if r < h { grid[r][i] = '.'; }
    }

    for &(fpr, tpr) in roc_a {
        let c = (fpr * (w-1) as f64).round() as usize;
        let r = ((1.0-tpr) * (h-1) as f64).round() as usize;
        if c < w && r < h { grid[r][c] = 'A'; }
    }
    for &(fpr, tpr) in roc_b {
        let c = (fpr * (w-1) as f64).round() as usize;
        let r = ((1.0-tpr) * (h-1) as f64).round() as usize;
        if c < w && r < h { grid[r][c] = 'B'; }
    }

    println!("  TPR");
    for (i, row) in grid.iter().enumerate() {
        let tpr = 1.0 - i as f64 / (h-1) as f64;
        let line: String = row.iter().collect();
        println!("  {:.1}|{}", tpr, line);
    }
    println!("     +{}", "-".repeat(w));
    println!("     0.0        FPR        1.0");
    println!("     A=Good, B=Decent, .=Random");
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- The ROC curve plots TPR vs FPR across all thresholds, providing a comprehensive view of classifier performance.
- AUC summarizes the ROC curve into a single number: the probability that a random positive is scored higher than a random negative.
- Youden's J statistic (TPR - FPR) identifies the optimal threshold where the model maximally separates the classes.
- Use ROC/AUC for balanced datasets and when both types of errors matter equally. Use precision-recall for imbalanced datasets where positive class detection is paramount.
