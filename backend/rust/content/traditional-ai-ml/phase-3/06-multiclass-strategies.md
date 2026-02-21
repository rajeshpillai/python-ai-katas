# Multiclass Strategies

> Phase 3 — Supervised Learning: Classification | Kata 3.06

---

## Concept & Intuition

### What problem are we solving?

Many classification algorithms (logistic regression, SVM) are inherently binary: they separate two classes. But real-world problems often involve multiple classes — recognizing handwritten digits (10 classes), classifying news articles (dozens of categories), or diagnosing diseases (many possible conditions). Multiclass strategies extend binary classifiers to handle multiple classes through decomposition.

The two main approaches are One-vs-Rest (OvR) and One-vs-One (OvO). OvR trains one classifier per class, each distinguishing that class from all others. OvO trains one classifier for every pair of classes. Both approaches let you use your favorite binary classifier on multiclass problems, but they have different tradeoffs in terms of training time, prediction time, and decision boundary quality.

In this kata, we implement both OvR and OvO strategies using logistic regression as the base binary classifier, then compare their performance on a multiclass dataset.

### Why naive approaches fail

You might try assigning class labels as numbers (0, 1, 2, 3) and using regression to predict them. But this imposes a false ordering: class 2 is not "between" class 1 and class 3, and class 3 is not "more" than class 1. Multi-output regression also fails because the relationship between classes is not captured by numerical proximity. Proper multiclass strategies treat each class distinction as a separate binary classification problem.

### Mental models

- **OvR as "Is it this class or not?"**: For K classes, train K binary classifiers. Each one asks "Is this class C or not?" At prediction time, pick the class whose classifier is most confident.
- **OvO as "Which of these two?"**: For K classes, train K*(K-1)/2 binary classifiers. Each one asks "Is it class A or class B?" At prediction time, each classifier votes and the class with the most votes wins.
- **Tradeoff**: OvR trains K models (fast) but each sees imbalanced data. OvO trains K*(K-1)/2 models (more models) but each sees balanced pairwise data.

### Visual explanations

```
  One-vs-Rest (K=3):                One-vs-One (K=3):

  Classifier 1: A vs {B,C}          Classifier 1: A vs B
  Classifier 2: B vs {A,C}          Classifier 2: A vs C
  Classifier 3: C vs {A,B}          Classifier 3: B vs C

  Prediction: max confidence         Prediction: majority vote
  among 3 classifiers                among 3 classifiers

  3 models total                     3 models total (K*(K-1)/2)
  (For K=10: 10 models)              (For K=10: 45 models)
```

---

## Hands-on Exploration

1. Create a 3-class dataset that is not trivially separable.
2. Implement OvR using logistic regression as the base classifier.
3. Implement OvO using logistic regression as the base classifier.
4. Compare accuracy, training time (number of model trains), and predictions.

---

## Live Code

```rust
fn main() {
    println!("=== Multiclass Strategies ===\n");

    // Generate 3-class dataset
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Class 0: centered at (1, 5)
    for _ in 0..20 {
        let x1 = 1.0 + randf(&mut rng) * 3.0;
        let x2 = 5.0 + randf(&mut rng) * 3.0;
        features.push(vec![x1, x2]);
        labels.push(0);
    }

    // Class 1: centered at (5, 1)
    for _ in 0..20 {
        let x1 = 5.0 + randf(&mut rng) * 3.0;
        let x2 = 1.0 + randf(&mut rng) * 3.0;
        features.push(vec![x1, x2]);
        labels.push(1);
    }

    // Class 2: centered at (6, 7)
    for _ in 0..20 {
        let x1 = 6.0 + randf(&mut rng) * 3.0;
        let x2 = 7.0 + randf(&mut rng) * 3.0;
        features.push(vec![x1, x2]);
        labels.push(2);
    }

    let n = features.len();
    let n_classes = 3;
    let class_names = vec!["ClassA", "ClassB", "ClassC"];

    // Standardize
    let (scaled, _, _) = standardize(&features);

    // Split
    let train_n = 45;
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 77);
    let x_train: Vec<Vec<f64>> = indices[..train_n].iter().map(|&i| scaled[i].clone()).collect();
    let y_train: Vec<usize> = indices[..train_n].iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = indices[train_n..].iter().map(|&i| scaled[i].clone()).collect();
    let y_test: Vec<usize> = indices[train_n..].iter().map(|&i| labels[i]).collect();

    println!("Dataset: {} samples, {} classes", n, n_classes);
    println!("Train: {}, Test: {}\n", train_n, n - train_n);

    // Strategy 1: One-vs-Rest
    println!("--- Strategy 1: One-vs-Rest (OvR) ---");
    let ovr_models = train_ovr(&x_train, &y_train, n_classes);
    println!("  Trained {} binary classifiers", ovr_models.len());

    for (c, model) in ovr_models.iter().enumerate() {
        println!("  Classifier {}: {} vs Rest (w=[{:.3}, {:.3}], b={:.3})",
            c, class_names[c], model.0[0], model.0[1], model.1);
    }

    let ovr_train_preds = predict_ovr(&x_train, &ovr_models);
    let ovr_test_preds = predict_ovr(&x_test, &ovr_models);
    let ovr_train_acc = acc(&y_train, &ovr_train_preds);
    let ovr_test_acc = acc(&y_test, &ovr_test_preds);
    println!("  Train accuracy: {:.1}%", ovr_train_acc * 100.0);
    println!("  Test accuracy:  {:.1}%", ovr_test_acc * 100.0);

    // Strategy 2: One-vs-One
    println!("\n--- Strategy 2: One-vs-One (OvO) ---");
    let ovo_models = train_ovo(&x_train, &y_train, n_classes);
    println!("  Trained {} binary classifiers", ovo_models.len());

    for &(c1, c2, ref w, b) in &ovo_models {
        println!("  Classifier: {} vs {} (w=[{:.3}, {:.3}], b={:.3})",
            class_names[c1], class_names[c2], w[0], w[1], b);
    }

    let ovo_train_preds = predict_ovo(&x_train, &ovo_models, n_classes);
    let ovo_test_preds = predict_ovo(&x_test, &ovo_models, n_classes);
    let ovo_train_acc = acc(&y_train, &ovo_train_preds);
    let ovo_test_acc = acc(&y_test, &ovo_test_preds);
    println!("  Train accuracy: {:.1}%", ovo_train_acc * 100.0);
    println!("  Test accuracy:  {:.1}%", ovo_test_acc * 100.0);

    // Comparison
    println!("\n--- Comparison ---");
    println!("{:<20} {:>12} {:>12} {:>10}", "Strategy", "Train Acc", "Test Acc", "# Models");
    println!("{}", "-".repeat(56));
    println!("{:<20} {:>11.1}% {:>11.1}% {:>10}", "One-vs-Rest",
        ovr_train_acc * 100.0, ovr_test_acc * 100.0, n_classes);
    println!("{:<20} {:>11.1}% {:>11.1}% {:>10}", "One-vs-One",
        ovo_train_acc * 100.0, ovo_test_acc * 100.0, n_classes * (n_classes - 1) / 2);

    // Scaling analysis
    println!("\n--- Scaling with Number of Classes ---");
    println!("{:<10} {:>10} {:>10}", "Classes", "OvR models", "OvO models");
    println!("{}", "-".repeat(30));
    for k in &[3, 5, 10, 20, 50, 100] {
        println!("{:<10} {:>10} {:>10}", k, k, k * (k - 1) / 2);
    }

    // Detailed test predictions
    println!("\n--- Test Predictions Detail ---");
    println!("{:<4} {:>8} {:>8} {:>8} {:>8}", "Idx", "Actual", "OvR", "OvO", "Agree?");
    println!("{}", "-".repeat(38));
    for i in 0..x_test.len() {
        let agree = if ovr_test_preds[i] == ovo_test_preds[i] { "yes" } else { "no" };
        println!("{:<4} {:>8} {:>8} {:>8} {:>8}",
            i, class_names[y_test[i]], class_names[ovr_test_preds[i]],
            class_names[ovo_test_preds[i]], agree);
    }

    kata_metric("ovr_test_accuracy", ovr_test_acc);
    kata_metric("ovo_test_accuracy", ovo_test_acc);
    kata_metric("ovr_models", n_classes as f64);
    kata_metric("ovo_models", (n_classes * (n_classes - 1) / 2) as f64);
}

fn train_binary_logistic(x: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) -> (Vec<f64>, f64) {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut w = vec![0.0; p];
    let mut b = 0.0;

    for _ in 0..epochs {
        let mut gw = vec![0.0; p];
        let mut gb = 0.0;
        for i in 0..x.len() {
            let z: f64 = x[i].iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
            let pred = 1.0 / (1.0 + (-z).exp());
            let err = pred - y[i];
            for j in 0..p { gw[j] += err * x[i][j]; }
            gb += err;
        }
        for j in 0..p { w[j] -= lr * gw[j] / n; }
        b -= lr * gb / n;
    }
    (w, b)
}

fn train_ovr(x: &[Vec<f64>], y: &[usize], n_classes: usize) -> Vec<(Vec<f64>, f64)> {
    let mut models = Vec::new();
    for c in 0..n_classes {
        let binary_y: Vec<f64> = y.iter().map(|&yi| if yi == c { 1.0 } else { 0.0 }).collect();
        let (w, b) = train_binary_logistic(x, &binary_y, 0.1, 500);
        models.push((w, b));
    }
    models
}

fn predict_ovr(x: &[Vec<f64>], models: &[(Vec<f64>, f64)]) -> Vec<usize> {
    x.iter().map(|xi| {
        let scores: Vec<f64> = models.iter().map(|(w, b)| {
            let z: f64 = xi.iter().zip(w.iter()).map(|(x, w)| x * w).sum::<f64>() + b;
            1.0 / (1.0 + (-z).exp())
        }).collect();
        scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }).collect()
}

fn train_ovo(
    x: &[Vec<f64>], y: &[usize], n_classes: usize,
) -> Vec<(usize, usize, Vec<f64>, f64)> {
    let mut models = Vec::new();
    for c1 in 0..n_classes {
        for c2 in (c1 + 1)..n_classes {
            let mut sub_x = Vec::new();
            let mut sub_y = Vec::new();
            for i in 0..x.len() {
                if y[i] == c1 {
                    sub_x.push(x[i].clone());
                    sub_y.push(0.0);
                } else if y[i] == c2 {
                    sub_x.push(x[i].clone());
                    sub_y.push(1.0);
                }
            }
            let (w, b) = train_binary_logistic(&sub_x, &sub_y, 0.1, 500);
            models.push((c1, c2, w, b));
        }
    }
    models
}

fn predict_ovo(
    x: &[Vec<f64>], models: &[(usize, usize, Vec<f64>, f64)], n_classes: usize,
) -> Vec<usize> {
    x.iter().map(|xi| {
        let mut votes = vec![0usize; n_classes];
        for &(c1, c2, ref w, b) in models {
            let z: f64 = xi.iter().zip(w.iter()).map(|(x, w)| x * w).sum::<f64>() + b;
            let prob = 1.0 / (1.0 + (-z).exp());
            if prob < 0.5 { votes[c1] += 1; } else { votes[c2] += 1; }
        }
        votes.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0
    }).collect()
}

fn acc(a: &[usize], p: &[usize]) -> f64 {
    a.iter().zip(p.iter()).filter(|(ai, pi)| ai == pi).count() as f64 / a.len() as f64
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    let p = data[0].len();
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        means[j] = col.iter().sum::<f64>() / n;
        stds[j] = (col.iter().map(|x| (x - means[j]).powi(2)).sum::<f64>() / n).sqrt();
    }
    let scaled = data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            if stds[j].abs() < 1e-10 { 0.0 } else { (v - means[j]) / stds[j] }
        }).collect()
    }).collect();
    (scaled, means, stds)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() { rng = lcg(rng); arr.swap(i, (rng % (i as u64 + 1)) as usize); }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- One-vs-Rest trains K classifiers (one per class) and picks the most confident. It is simple and efficient but each classifier sees imbalanced data.
- One-vs-One trains K*(K-1)/2 classifiers (one per pair) and uses voting. Each classifier sees balanced data but the number of models grows quadratically.
- For small K, the strategies perform similarly. For large K, OvR is more practical due to linear scaling.
- Some algorithms (like decision trees, Naive Bayes, and neural networks) handle multiclass natively, making decomposition unnecessary.
