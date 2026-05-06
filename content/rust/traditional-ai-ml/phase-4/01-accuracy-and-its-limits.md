# Accuracy and Its Limits

> Phase 4 — Model Evaluation & Selection | Kata 4.01

---

## Concept & Intuition

### What problem are we solving?

Accuracy — the fraction of correct predictions — is the most intuitive evaluation metric. But it can be deeply misleading. In a medical test where 99% of patients are healthy, a model that always says "healthy" achieves 99% accuracy while missing every sick patient. In spam detection, a model that never flags spam achieves high accuracy on a mostly-legitimate inbox while letting all spam through.

Understanding when accuracy works and when it fails is essential for evaluating models honestly. We need a richer vocabulary of metrics that capture different aspects of model performance. The confusion matrix is the foundation: it breaks down predictions into true positives, true negatives, false positives, and false negatives, from which all other metrics are derived.

In this kata, we build a comprehensive evaluation framework, starting with the confusion matrix and deriving accuracy, error rate, and per-class metrics. We demonstrate scenarios where accuracy is misleading and where it works well.

### Why naive approaches fail

Reporting a single number (accuracy) hides critical information. Two models can have identical accuracy but vastly different behavior: one might correctly classify most of class A but miss class B, while the other is balanced. Without the confusion matrix, you cannot diagnose what the model gets wrong or why. In multi-class settings, accuracy obscures which classes are well-served and which are not.

### Mental models

- **Confusion matrix as the ground truth**: Every evaluation metric is a different view of the same confusion matrix. Understanding the matrix means understanding all metrics.
- **Accuracy works when**: Classes are balanced, errors in both directions are equally costly, and you care about overall performance rather than per-class performance.
- **Accuracy fails when**: Classes are imbalanced, one type of error is much costlier than another, or minority class performance matters.

### Visual explanations

```
  Confusion Matrix:

                  Predicted
                  Pos    Neg
  Actual  Pos  [  TP  |  FN  ]    TP = True Positive
          Neg  [  FP  |  TN  ]    FP = False Positive
                                   FN = False Negative
  Accuracy = (TP + TN) / Total     TN = True Negative
  Error Rate = (FP + FN) / Total
```

---

## Hands-on Exploration

1. Create a confusion matrix from predictions and actual labels.
2. Derive accuracy and demonstrate when it is meaningful.
3. Create scenarios where high accuracy masks poor model quality.
4. Introduce per-class accuracy to reveal hidden problems.

---

## Live Code

```rust
fn main() {
    println!("=== Accuracy and Its Limits ===\n");

    // Scenario 1: Balanced classes (accuracy works well)
    println!("--- Scenario 1: Balanced Classes ---");
    let actual_1 = vec![0,0,0,0,0,1,1,1,1,1, 0,0,0,0,0,1,1,1,1,1];
    let pred_1   = vec![0,0,0,1,0,1,1,0,1,1, 0,0,1,0,0,1,1,1,0,1];
    evaluate("Balanced (50/50)", &actual_1, &pred_1);

    // Scenario 2: Imbalanced (accuracy is misleading)
    println!("\n--- Scenario 2: Imbalanced Classes (95/5) ---");
    let mut actual_2 = vec![0; 95];
    actual_2.extend(vec![1; 5]);
    // Model A: always predicts majority
    let pred_2a = vec![0; 100];
    // Model B: catches some minority at cost of some majority errors
    let mut pred_2b = vec![0; 100];
    pred_2b[93] = 1; pred_2b[94] = 1; // 2 FP
    pred_2b[95] = 1; pred_2b[96] = 1; pred_2b[97] = 1; // 3 TP
    evaluate("Always majority", &actual_2, &pred_2a);
    println!();
    evaluate("Catches minority", &actual_2, &pred_2b);

    println!("\n  INSIGHT: Model A has higher accuracy (95%) but catches ZERO");
    println!("  minority samples. Model B has lower accuracy (95%) but");
    println!("  finds 3 out of 5 minority samples (60% recall).");

    // Scenario 3: Multi-class
    println!("\n--- Scenario 3: Multi-class (3 classes) ---");
    let actual_3 = vec![0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2];
    let pred_3   = vec![0,0,0,0,0,0,0,0,0,0, 0,0,1,1,1, 2,2,2,2,0];
    evaluate_multiclass("3-class", &actual_3, &pred_3, 3);

    // Scenario 4: Same accuracy, different confusion matrices
    println!("\n--- Scenario 4: Same Accuracy, Different Errors ---");
    let actual_4 = vec![0,0,0,0,0,1,1,1,1,1];
    let pred_4a  = vec![0,0,0,0,1,1,1,1,1,0]; // 1 FP + 1 FN
    let pred_4b  = vec![0,0,0,0,0,1,1,1,0,0]; // 0 FP + 2 FN
    println!("Model A:");
    evaluate("Balanced errors", &actual_4, &pred_4a);
    println!("\nModel B:");
    evaluate("Misses positives", &actual_4, &pred_4b);
    println!("\n  Both have 80% accuracy, but Model A has balanced errors");
    println!("  while Model B misses more positives (lower recall).");

    // Metrics comparison table
    println!("\n\n=== Why Accuracy Alone Is Not Enough ===");
    println!("{:<25} {:>10} {:>10} {:>10}", "Scenario", "Accuracy", "Balanced", "Useful?");
    println!("{}", "-".repeat(55));
    println!("{:<25} {:>9.1}% {:>9.1}% {:>10}", "Balanced classes",
        accuracy(&actual_1, &pred_1) * 100.0,
        balanced_accuracy(&actual_1, &pred_1) * 100.0, "Yes");
    println!("{:<25} {:>9.1}% {:>9.1}% {:>10}", "Always majority (95/5)",
        accuracy(&actual_2, &pred_2a) * 100.0,
        balanced_accuracy(&actual_2, &pred_2a) * 100.0, "No!");
    println!("{:<25} {:>9.1}% {:>9.1}% {:>10}", "Catches minority",
        accuracy(&actual_2, &pred_2b) * 100.0,
        balanced_accuracy(&actual_2, &pred_2b) * 100.0, "Better");

    kata_metric("balanced_accuracy_useful",
        accuracy(&actual_1, &pred_1));
    kata_metric("balanced_accuracy_misleading",
        accuracy(&actual_2, &pred_2a));
    kata_metric("balanced_accuracy_better",
        balanced_accuracy(&actual_2, &pred_2b));
}

fn confusion_matrix(actual: &[usize], predicted: &[usize], n_classes: usize) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0; n_classes]; n_classes];
    for (a, p) in actual.iter().zip(predicted.iter()) {
        cm[*a][*p] += 1;
    }
    cm
}

fn evaluate(name: &str, actual: &[usize], predicted: &[usize]) {
    let cm = confusion_matrix(actual, predicted, 2);
    let tp = cm[1][1]; let fp = cm[0][1]; let fn_ = cm[1][0]; let tn = cm[0][0];
    let total = actual.len();

    println!("  Model: {}", name);
    println!("  Confusion Matrix:");
    println!("              Pred 0   Pred 1");
    println!("  Actual 0    {:>6}   {:>6}", tn, fp);
    println!("  Actual 1    {:>6}   {:>6}", fn_, tp);
    println!("  Accuracy: {:.1}% ({}/{})", (tp + tn) as f64 / total as f64 * 100.0, tp + tn, total);
    println!("  Error rate: {:.1}%", (fp + fn_) as f64 / total as f64 * 100.0);
    println!("  Per-class accuracy: class0={:.1}%, class1={:.1}%",
        if tn + fp > 0 { tn as f64 / (tn + fp) as f64 * 100.0 } else { 0.0 },
        if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 * 100.0 } else { 0.0 });
}

fn evaluate_multiclass(name: &str, actual: &[usize], predicted: &[usize], n: usize) {
    let cm = confusion_matrix(actual, predicted, n);
    println!("  Model: {}", name);
    println!("  Confusion Matrix:");
    print!("              ");
    for c in 0..n { print!("Pred {}  ", c); }
    println!();
    for r in 0..n {
        print!("  Actual {}    ", r);
        for c in 0..n { print!("{:>6}  ", cm[r][c]); }
        println!();
    }

    let total_correct: usize = (0..n).map(|i| cm[i][i]).sum();
    let total = actual.len();
    println!("  Overall accuracy: {:.1}%", total_correct as f64 / total as f64 * 100.0);

    for c in 0..n {
        let class_total: usize = cm[c].iter().sum();
        let class_correct = cm[c][c];
        println!("  Class {} accuracy: {:.1}% ({}/{})",
            c, class_correct as f64 / class_total as f64 * 100.0, class_correct, class_total);
    }
}

fn accuracy(actual: &[usize], predicted: &[usize]) -> f64 {
    actual.iter().zip(predicted.iter()).filter(|(a, p)| a == p).count() as f64 / actual.len() as f64
}

fn balanced_accuracy(actual: &[usize], predicted: &[usize]) -> f64 {
    let classes: Vec<usize> = {
        let mut c: Vec<usize> = actual.to_vec();
        c.sort(); c.dedup(); c
    };
    let per_class: Vec<f64> = classes.iter().map(|&c| {
        let total = actual.iter().filter(|&&a| a == c).count();
        let correct = actual.iter().zip(predicted.iter()).filter(|(&a, &p)| a == c && p == c).count();
        if total > 0 { correct as f64 / total as f64 } else { 0.0 }
    }).collect();
    per_class.iter().sum::<f64>() / per_class.len() as f64
}

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Accuracy is meaningful only when classes are balanced and error costs are symmetric.
- The confusion matrix is the foundation of all classification metrics. Always examine it before trusting any single number.
- Balanced accuracy (average of per-class accuracies) is a better metric for imbalanced datasets than overall accuracy.
- Two models with identical accuracy can have very different confusion matrices and very different practical utility.
