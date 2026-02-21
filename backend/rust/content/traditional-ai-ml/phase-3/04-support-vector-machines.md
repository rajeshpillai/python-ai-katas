# Support Vector Machines

> Phase 3 — Supervised Learning: Classification | Kata 3.04

---

## Concept & Intuition

### What problem are we solving?

Support Vector Machines (SVMs) find the hyperplane that separates two classes with the maximum margin — the widest possible gap between the nearest points of each class. These nearest points are the "support vectors," and the margin is the perpendicular distance from the hyperplane to the nearest support vector on each side. Maximum margin classification is motivated by statistical learning theory: wider margins tend to generalize better.

While logistic regression finds a separating hyperplane that minimizes classification error, many such hyperplanes may exist. SVM is more principled: among all hyperplanes that separate the data, it finds the one with the largest margin. This geometric optimization objective has deep theoretical foundations and often produces models that generalize well, especially in high-dimensional spaces.

In this kata, we implement a simplified SVM using gradient descent on the hinge loss with L2 regularization. The hinge loss is zero when a point is correctly classified with margin greater than 1, and increases linearly as the point moves toward or past the wrong side of the boundary.

### Why naive approaches fail

Simply finding any separating hyperplane (like the perceptron algorithm) gives no guarantee about generalization. The chosen hyperplane might be very close to some training points, meaning small perturbations could cause misclassification. By maximizing the margin, SVM creates a buffer zone that makes the classifier robust to noise and new data.

### Mental models

- **Maximum margin as robustness**: The wider the margin, the more noise the classifier can tolerate before making errors. SVM maximizes this safety zone.
- **Hinge loss as relaxed constraint**: The hinge loss, max(0, 1 - y*(w*x + b)), is zero for correctly classified points far from the boundary and increases linearly for points near or on the wrong side.
- **Support vectors are critical**: Only the points closest to the boundary (support vectors) determine the position of the hyperplane. Removing any non-support-vector point has zero effect on the model.

### Visual explanations

```
  Logistic Regression:           SVM (Maximum Margin):

  + + +  |  - - -               + + +  |     |  - - -
  + + + |  - - - -              + + +  |     |  - - -
  + +|+ - - - - -               + + + *|     |* - - -
  + +| - - - - -                + + +  | <m> |  - - -
  + + |  - - -                  + + +  |     |  - - -

  Any separating line           Maximum margin line
  (not unique)                  * = support vectors
                                m = margin (maximized)
```

---

## Hands-on Exploration

1. Implement the hinge loss function.
2. Train a linear SVM using gradient descent on hinge loss + L2 regularization.
3. Identify the support vectors (points closest to the decision boundary).
4. Compare SVM with logistic regression on the same dataset.

---

## Live Code

```rust
fn main() {
    println!("=== Support Vector Machine ===\n");

    // Generate linearly separable data with some overlap
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new(); // -1 or +1

    // Class -1
    for _ in 0..25 {
        rng = lcg(rng);
        let x1 = (rng as f64 / u64::MAX as f64) * 6.0;
        rng = lcg(rng);
        let x2 = (rng as f64 / u64::MAX as f64) * 6.0;
        features.push(vec![x1, x2]);
        labels.push(-1.0);
    }

    // Class +1
    for _ in 0..25 {
        rng = lcg(rng);
        let x1 = 3.0 + (rng as f64 / u64::MAX as f64) * 6.0;
        rng = lcg(rng);
        let x2 = 3.0 + (rng as f64 / u64::MAX as f64) * 6.0;
        features.push(vec![x1, x2]);
        labels.push(1.0);
    }

    // Standardize
    let (scaled, _, _) = standardize(&features);
    let n = scaled.len();

    // Split
    let train_n = 38;
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 55);
    let x_train: Vec<Vec<f64>> = indices[..train_n].iter().map(|&i| scaled[i].clone()).collect();
    let y_train: Vec<f64> = indices[..train_n].iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = indices[train_n..].iter().map(|&i| scaled[i].clone()).collect();
    let y_test: Vec<f64> = indices[train_n..].iter().map(|&i| labels[i]).collect();

    println!("Train: {}, Test: {}\n", train_n, n - train_n);

    // Train SVM with different C values (inverse of regularization)
    println!("--- C Parameter Sweep ---");
    println!("{:<10} {:>10} {:>10} {:>12} {:>12}", "C", "Train Acc", "Test Acc", "Margin", "# SV");
    println!("{}", "-".repeat(56));

    let c_values = vec![0.01, 0.1, 1.0, 10.0, 100.0];
    let mut best_c = 1.0;
    let mut best_test_acc = 0.0;

    for &c in &c_values {
        let (w, b, _) = train_svm(&x_train, &y_train, c, 0.001, 2000);
        let train_preds: Vec<f64> = x_train.iter().map(|x| svm_predict(x, &w, b)).collect();
        let test_preds: Vec<f64> = x_test.iter().map(|x| svm_predict(x, &w, b)).collect();

        let train_acc = svm_accuracy(&y_train, &train_preds);
        let test_acc = svm_accuracy(&y_test, &test_preds);
        let margin = 2.0 / w.iter().map(|wi| wi * wi).sum::<f64>().sqrt();
        let n_sv = count_support_vectors(&x_train, &y_train, &w, b);

        println!("{:<10.2} {:>9.1}% {:>9.1}% {:>12.4} {:>12}",
            c, train_acc * 100.0, test_acc * 100.0, margin, n_sv);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_c = c;
        }
    }

    // Train best model
    let (w, b, loss_history) = train_svm(&x_train, &y_train, best_c, 0.001, 2000);
    println!("\nBest C: {} (test acc: {:.1}%)", best_c, best_test_acc * 100.0);
    println!("Weights: [{:.4}, {:.4}], Bias: {:.4}", w[0], w[1], b);

    // Loss convergence
    println!("\n--- Training Loss ---");
    for &ep in &[0, 10, 50, 100, 500, 1999] {
        if ep < loss_history.len() {
            println!("  Epoch {:>5}: loss = {:.4}", ep, loss_history[ep]);
        }
    }

    // Support vectors
    println!("\n--- Support Vectors ---");
    let margin = 2.0 / w.iter().map(|wi| wi * wi).sum::<f64>().sqrt();
    println!("  Margin width: {:.4}", margin);

    let mut sv_count = 0;
    for (i, x) in x_train.iter().enumerate() {
        let functional_margin = y_train[i] * (dot(x, &w) + b);
        if functional_margin < 1.1 {
            if sv_count < 10 {
                println!("  SV: x=[{:.3}, {:.3}], y={:+.0}, margin={:.3}",
                    x[0], x[1], y_train[i], functional_margin);
            }
            sv_count += 1;
        }
    }
    println!("  Total support vectors: {} / {} ({:.0}%)",
        sv_count, train_n, sv_count as f64 / train_n as f64 * 100.0);

    // Hinge loss demonstration
    println!("\n--- Hinge Loss Examples ---");
    println!("  y*f(x) = 2.0 (correct, far):    loss = {:.2}", hinge_loss(2.0));
    println!("  y*f(x) = 1.0 (correct, margin):  loss = {:.2}", hinge_loss(1.0));
    println!("  y*f(x) = 0.5 (correct, inside):  loss = {:.2}", hinge_loss(0.5));
    println!("  y*f(x) = 0.0 (on boundary):      loss = {:.2}", hinge_loss(0.0));
    println!("  y*f(x) = -1.0 (wrong side):       loss = {:.2}", hinge_loss(-1.0));

    kata_metric("best_c", best_c);
    kata_metric("best_test_accuracy", best_test_acc);
    kata_metric("margin_width", margin);
    kata_metric("n_support_vectors", sv_count as f64);
}

fn hinge_loss(margin: f64) -> f64 {
    (1.0 - margin).max(0.0)
}

fn train_svm(
    x: &[Vec<f64>],
    y: &[f64],
    c: f64,
    lr: f64,
    epochs: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let n = x.len();
    let p = x[0].len();
    let mut w = vec![0.0; p];
    let mut b = 0.0;
    let mut loss_history = Vec::new();

    for epoch in 0..epochs {
        let current_lr = lr / (1.0 + 0.001 * epoch as f64);
        let mut total_loss = 0.0;

        for i in 0..n {
            let margin = y[i] * (dot(&x[i], &w) + b);

            if margin < 1.0 {
                // Misclassified or within margin
                for j in 0..p {
                    w[j] -= current_lr * (w[j] / (c * n as f64) - y[i] * x[i][j]);
                }
                b -= current_lr * (-y[i]);
                total_loss += 1.0 - margin;
            } else {
                // Correctly classified outside margin
                for j in 0..p {
                    w[j] -= current_lr * w[j] / (c * n as f64);
                }
            }

            // Regularization term
            total_loss += 0.5 * w.iter().map(|wi| wi * wi).sum::<f64>() / (c * n as f64);
        }

        loss_history.push(total_loss / n as f64);
    }

    (w, b, loss_history)
}

fn svm_predict(x: &[f64], w: &[f64], b: f64) -> f64 {
    if dot(x, w) + b >= 0.0 { 1.0 } else { -1.0 }
}

fn count_support_vectors(x: &[Vec<f64>], y: &[f64], w: &[f64], b: f64) -> usize {
    x.iter().zip(y.iter())
        .filter(|(xi, &yi)| yi * (dot(xi, w) + b) < 1.1)
        .count()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn svm_accuracy(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter())
        .filter(|(&a, &p)| a * p > 0.0)
        .count() as f64 / actual.len() as f64
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

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() { rng = lcg(rng); arr.swap(i, (rng % (i as u64 + 1)) as usize); }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- SVMs find the maximum-margin hyperplane, providing a geometrically motivated and theoretically grounded classifier.
- The hinge loss penalizes points that are misclassified or within the margin, encouraging a wide separation between classes.
- Support vectors are the critical training points closest to the decision boundary. Only they determine the model; all other points can be removed without effect.
- The C parameter controls the tradeoff between margin width and training error: large C means narrow margin with fewer misclassifications, small C means wide margin allowing some errors.
