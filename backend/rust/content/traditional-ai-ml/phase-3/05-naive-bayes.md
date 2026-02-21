# Naive Bayes

> Phase 3 — Supervised Learning: Classification | Kata 3.05

---

## Concept & Intuition

### What problem are we solving?

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class. Despite this strong (and usually wrong) assumption, Naive Bayes is surprisingly effective in practice, especially for text classification, spam filtering, and medical diagnosis. It is fast, scales well, and works with very little training data.

Bayes' theorem says: P(class|features) = P(features|class) * P(class) / P(features). The classifier picks the class with the highest posterior probability. The "naive" part is assuming P(features|class) = P(f1|class) * P(f2|class) * ... * P(fn|class), which breaks the joint probability into a product of individual feature probabilities.

In this kata, we implement Gaussian Naive Bayes (for continuous features) from scratch. For each class and each feature, we estimate a Gaussian distribution (mean and variance) from the training data, then use these distributions to compute the likelihood of new observations.

### Why naive approaches fail

Without the independence assumption, estimating P(features|class) requires a joint probability table that grows exponentially with the number of features. With 10 binary features, you need 1024 entries per class. With continuous features, you need multivariate density estimation. The naive assumption makes the problem tractable by reducing it to estimating individual feature distributions. The amazing empirical finding is that this simplification works remarkably well even when features are correlated.

### Mental models

- **Bayes as updating beliefs**: Start with a prior belief (class frequencies). Each feature provides evidence that updates this belief. The final posterior is the class with the strongest combined evidence.
- **Gaussian bell curves per class**: For each feature and each class, imagine a bell curve. A new point's value falls somewhere on each bell curve. The class whose bell curves collectively assign the highest probability wins.
- **Naive does not mean stupid**: The independence assumption is "naive" but the classifier is not. It often outperforms more complex models, especially with limited data, because the bias introduced by the independence assumption reduces variance.

### Visual explanations

```
  Feature X1 distributions by class:

  Class A        Class B
  P(x1)          P(x1)
    |  __           |      __
    | /  \          |     /  \
    |/    \         |    /    \
    +------x1       +------x1
    mean=3          mean=7

  New point x1=5:
  P(x1=5|A) = 0.10  (far from A's center)
  P(x1=5|B) = 0.25  (closer to B's center)
  -> Evidence favors class B for this feature
```

---

## Hands-on Exploration

1. Implement Gaussian probability density function.
2. Train Gaussian Naive Bayes: compute mean and variance per class per feature.
3. Predict by computing posterior probabilities using Bayes' theorem.
4. Compare with KNN on the same dataset.

---

## Live Code

```rust
fn main() {
    println!("=== Gaussian Naive Bayes ===\n");

    // Dataset: classify iris-like flowers (3 classes, 4 features)
    let feature_names = vec!["sepal_len", "sepal_wid", "petal_len", "petal_wid"];
    let class_names = vec!["Setosa", "Versicolor", "Virginica"];

    // Simplified iris-like data
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    let mut rng = 42u64;

    // Setosa: small petals
    for _ in 0..20 {
        rng = lcg(rng); let sl = 5.0 + r(&mut rng) * 0.8;
        rng = lcg(rng); let sw = 3.4 + r(&mut rng) * 0.6;
        rng = lcg(rng); let pl = 1.4 + r(&mut rng) * 0.4;
        rng = lcg(rng); let pw = 0.2 + r(&mut rng) * 0.2;
        features.push(vec![sl, sw, pl, pw]);
        labels.push(0);
    }

    // Versicolor: medium petals
    for _ in 0..20 {
        rng = lcg(rng); let sl = 5.9 + r(&mut rng) * 0.8;
        rng = lcg(rng); let sw = 2.8 + r(&mut rng) * 0.4;
        rng = lcg(rng); let pl = 4.2 + r(&mut rng) * 0.8;
        rng = lcg(rng); let pw = 1.3 + r(&mut rng) * 0.4;
        features.push(vec![sl, sw, pl, pw]);
        labels.push(1);
    }

    // Virginica: large petals
    for _ in 0..20 {
        rng = lcg(rng); let sl = 6.6 + r(&mut rng) * 1.0;
        rng = lcg(rng); let sw = 3.0 + r(&mut rng) * 0.6;
        rng = lcg(rng); let pl = 5.5 + r(&mut rng) * 0.8;
        rng = lcg(rng); let pw = 2.0 + r(&mut rng) * 0.4;
        features.push(vec![sl, sw, pl, pw]);
        labels.push(2);
    }

    let n = features.len();
    let n_classes = 3;
    let n_features = 4;

    // Split
    let train_n = 45;
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 33);
    let x_train: Vec<Vec<f64>> = indices[..train_n].iter().map(|&i| features[i].clone()).collect();
    let y_train: Vec<usize> = indices[..train_n].iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = indices[train_n..].iter().map(|&i| features[i].clone()).collect();
    let y_test: Vec<usize> = indices[train_n..].iter().map(|&i| labels[i]).collect();

    println!("Dataset: {} samples, {} classes, {} features", n, n_classes, n_features);
    println!("Train: {}, Test: {}\n", train_n, n - train_n);

    // Train: compute mean and variance per class per feature
    let model = train_gaussian_nb(&x_train, &y_train, n_classes, n_features);

    // Print learned parameters
    println!("--- Learned Gaussian Parameters ---");
    for c in 0..n_classes {
        println!("\n  Class {} ({}):", c, class_names[c]);
        println!("    Prior: {:.3}", model.priors[c]);
        println!("    {:<12} {:>8} {:>8}", "Feature", "Mean", "Std");
        for f in 0..n_features {
            println!("    {:<12} {:>8.3} {:>8.3}",
                feature_names[f], model.means[c][f], model.variances[c][f].sqrt());
        }
    }

    // Predict
    let test_preds: Vec<usize> = x_test.iter()
        .map(|x| predict_nb(&model, x, n_classes))
        .collect();
    let train_preds: Vec<usize> = x_train.iter()
        .map(|x| predict_nb(&model, x, n_classes))
        .collect();

    let train_acc = acc(&y_train, &train_preds);
    let test_acc = acc(&y_test, &test_preds);

    println!("\n--- Results ---");
    println!("  Train accuracy: {:.1}%", train_acc * 100.0);
    println!("  Test accuracy:  {:.1}%", test_acc * 100.0);

    // Detailed predictions with probabilities
    println!("\n--- Test Predictions with Probabilities ---");
    println!("{:<4} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "Idx", "P(Set)", "P(Ver)", "P(Vir)", "Actual", "Pred", "OK?");
    println!("{}", "-".repeat(64));

    for (i, x) in x_test.iter().enumerate() {
        let probs = predict_proba_nb(&model, x, n_classes);
        let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let ok = if pred == y_test[i] { "yes" } else { "NO" };
        println!("{:<4} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>8} {:>8}",
            i, probs[0], probs[1], probs[2], class_names[y_test[i]], class_names[pred], ok);
    }

    // Confusion matrix
    println!("\n--- Confusion Matrix ---");
    let mut cm = vec![vec![0usize; n_classes]; n_classes];
    for (i, &pred) in test_preds.iter().enumerate() {
        cm[y_test[i]][pred] += 1;
    }
    print!("{:>12}", "");
    for name in &class_names { print!("{:>10}", name); }
    println!();
    for (i, row) in cm.iter().enumerate() {
        print!("{:>12}", class_names[i]);
        for &val in row { print!("{:>10}", val); }
        println!();
    }

    // Feature discriminative power
    println!("\n--- Feature Discriminative Power ---");
    for f in 0..n_features {
        let mean_range = model.means.iter().map(|m| m[f])
            .fold(f64::INFINITY, f64::min);
        let mean_max = model.means.iter().map(|m| m[f])
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_std: f64 = model.variances.iter().map(|v| v[f].sqrt()).sum::<f64>() / n_classes as f64;
        let separation = (mean_max - mean_range) / avg_std;
        let bar = "#".repeat((separation * 5.0).min(30.0) as usize);
        println!("  {:<12} separation={:.2}  |{}", feature_names[f], separation, bar);
    }

    kata_metric("train_accuracy", train_acc);
    kata_metric("test_accuracy", test_acc);
    kata_metric("n_classes", n_classes as f64);
}

struct GaussianNB {
    means: Vec<Vec<f64>>,
    variances: Vec<Vec<f64>>,
    priors: Vec<f64>,
}

fn train_gaussian_nb(
    x: &[Vec<f64>], y: &[usize], n_classes: usize, n_features: usize,
) -> GaussianNB {
    let n = x.len() as f64;
    let mut means = vec![vec![0.0; n_features]; n_classes];
    let mut variances = vec![vec![0.0; n_features]; n_classes];
    let mut counts = vec![0usize; n_classes];

    // Count and sum
    for (i, row) in x.iter().enumerate() {
        let c = y[i];
        counts[c] += 1;
        for f in 0..n_features {
            means[c][f] += row[f];
        }
    }

    // Compute means
    for c in 0..n_classes {
        for f in 0..n_features {
            means[c][f] /= counts[c] as f64;
        }
    }

    // Compute variances
    for (i, row) in x.iter().enumerate() {
        let c = y[i];
        for f in 0..n_features {
            variances[c][f] += (row[f] - means[c][f]).powi(2);
        }
    }

    for c in 0..n_classes {
        for f in 0..n_features {
            variances[c][f] = variances[c][f] / counts[c] as f64 + 1e-9; // smoothing
        }
    }

    let priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n).collect();

    GaussianNB { means, variances, priors }
}

fn predict_nb(model: &GaussianNB, x: &[f64], n_classes: usize) -> usize {
    let probs = predict_proba_nb(model, x, n_classes);
    probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

fn predict_proba_nb(model: &GaussianNB, x: &[f64], n_classes: usize) -> Vec<f64> {
    let mut log_probs = vec![0.0; n_classes];

    for c in 0..n_classes {
        log_probs[c] = model.priors[c].ln();
        for f in 0..x.len() {
            log_probs[c] += gaussian_log_pdf(x[f], model.means[c][f], model.variances[c][f]);
        }
    }

    // Convert to probabilities via softmax
    let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_log).exp()).collect();
    let sum: f64 = exp_probs.iter().sum();
    exp_probs.iter().map(|&e| e / sum).collect()
}

fn gaussian_log_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    -0.5 * ((x - mean).powi(2) / variance + variance.ln() + (2.0 * std::f64::consts::PI).ln())
}

fn acc(a: &[usize], p: &[usize]) -> f64 {
    a.iter().zip(p.iter()).filter(|(ai, pi)| ai == pi).count() as f64 / a.len() as f64
}

fn r(rng: &mut u64) -> f64 {
    *rng = lcg(*rng);
    (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0
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

- Naive Bayes applies Bayes' theorem with the assumption that features are conditionally independent given the class, making computation tractable.
- Gaussian Naive Bayes models each feature with a Gaussian distribution per class, requiring only mean and variance estimates from training data.
- Despite the naive independence assumption, the algorithm often performs surprisingly well, especially with limited data and high-dimensional feature spaces.
- The posterior probabilities provide well-calibrated confidence estimates, making Naive Bayes useful not just for classification but for ranking and scoring.
