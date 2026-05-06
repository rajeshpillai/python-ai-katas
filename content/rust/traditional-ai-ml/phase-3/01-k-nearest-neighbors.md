# K-Nearest Neighbors

> Phase 3 — Supervised Learning: Classification | Kata 3.01

---

## Concept & Intuition

### What problem are we solving?

K-Nearest Neighbors (KNN) is the simplest classification algorithm: to classify a new point, find the K closest training points and let them vote. The majority class among the K neighbors becomes the prediction. KNN embodies the principle that similar inputs should produce similar outputs — points close together in feature space tend to share a class.

KNN is a "lazy learner" — it does no work during training (it just stores the data) and does all the work at prediction time. This makes training instant but prediction slow for large datasets, since every prediction requires computing distances to all training points. Despite its simplicity, KNN can capture complex, nonlinear decision boundaries because the boundary emerges from the local structure of the data rather than a global function.

In this kata, we implement KNN from scratch, including the distance computation, neighbor selection, and voting mechanism. We explore how the choice of K affects the decision boundary and model performance.

### Why naive approaches fail

Using K=1 (the single nearest neighbor) is tempting but dangerous: it creates highly irregular decision boundaries that follow every noise point in the training data. This is extreme overfitting. Using a very large K smooths the boundary too much, eventually just predicting the majority class everywhere (extreme underfitting). The choice of K is critical, and it must be validated on held-out data.

### Mental models

- **Voting democracy**: Each neighbor gets one vote. K=1 is a dictatorship (one neighbor decides). K=N is pure majority rule (always predicts the most common class). The sweet spot is in between.
- **Decision boundary from data**: Unlike parametric models (linear regression, logistic regression), KNN's decision boundary is implicit — defined by the data itself. Add a new training point and the boundary shifts.
- **Curse of dimensionality**: In high dimensions, all points become approximately equidistant. KNN's distance-based reasoning breaks down. Feature selection and dimensionality reduction are essential companions.

### Visual explanations

```
  K=1 (overfitting):          K=5 (good):              K=N (underfitting):

  A A|B B B                   A A A|B B                 A A A A A
  A A|B B B                   A A A|B B                 A A A A A
  A|B B B B                   A A|B B B                 A A A A A
  A|B B B B                   A A|B B B                 A A A A A

  Jagged boundary             Smooth boundary           No boundary
  (follows noise)             (captures pattern)        (always majority)
```

---

## Hands-on Exploration

1. Create a 2D classification dataset with two classes.
2. Implement KNN with Euclidean distance.
3. Vary K and observe how the decision boundary changes.
4. Evaluate accuracy on a test set for different K values.

---

## Live Code

```rust
fn main() {
    println!("=== K-Nearest Neighbors Classification ===\n");

    // Generate 2D dataset: two clusters
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Class 0: centered at (2, 2)
    for _ in 0..25 {
        rng = lcg(rng);
        let x = 2.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
        rng = lcg(rng);
        let y = 2.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
        features.push(vec![x, y]);
        labels.push(0);
    }

    // Class 1: centered at (6, 6)
    for _ in 0..25 {
        rng = lcg(rng);
        let x = 6.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
        rng = lcg(rng);
        let y = 6.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
        features.push(vec![x, y]);
        labels.push(1);
    }

    // Shuffle and split
    let n = features.len();
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 123);

    let train_n = 35;
    let train_idx: Vec<usize> = indices[..train_n].to_vec();
    let test_idx: Vec<usize> = indices[train_n..].to_vec();

    let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| features[i].clone()).collect();
    let y_train: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| features[i].clone()).collect();
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    println!("Train: {} samples, Test: {} samples\n", train_n, n - train_n);

    // Test different K values
    println!("--- K Selection ---");
    println!("{:<6} {:>12} {:>12}", "K", "Train Acc", "Test Acc");
    println!("{}", "-".repeat(30));

    let k_values = vec![1, 3, 5, 7, 9, 11, 15, 25];
    let mut best_k = 1;
    let mut best_test_acc = 0.0;

    for &k in &k_values {
        let train_preds: Vec<usize> = x_train.iter()
            .map(|x| knn_predict(&x_train, &y_train, x, k, 2))
            .collect();
        let test_preds: Vec<usize> = x_test.iter()
            .map(|x| knn_predict(&x_train, &y_train, x, k, 2))
            .collect();

        let train_acc = accuracy(&y_train, &train_preds);
        let test_acc = accuracy(&y_test, &test_preds);

        let indicator = if k == 1 { " (overfit risk)" }
            else if k >= 25 { " (underfit risk)" }
            else { "" };

        println!("{:<6} {:>11.1}% {:>11.1}%{}",
            k, train_acc * 100.0, test_acc * 100.0, indicator);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_k = k;
        }
    }

    println!("\nBest K: {} (test accuracy: {:.1}%)", best_k, best_test_acc * 100.0);

    // Detailed prediction with K=best_k
    println!("\n--- Test Predictions (K={}) ---", best_k);
    println!("{:<6} {:>8} {:>8} {:>8} {:>12}", "Idx", "x1", "x2", "Actual", "Predicted");
    println!("{}", "-".repeat(44));

    for (i, x) in x_test.iter().enumerate() {
        let pred = knn_predict(&x_train, &y_train, x, best_k, 2);
        let correct = if pred == y_test[i] { "" } else { " WRONG" };
        println!("{:<6} {:>8.2} {:>8.2} {:>8} {:>8}{}",
            i, x[0], x[1], y_test[i], pred, correct);
    }

    // Distance weighting comparison
    println!("\n--- Weighted KNN (K=5) ---");
    let weighted_preds: Vec<usize> = x_test.iter()
        .map(|x| knn_predict_weighted(&x_train, &y_train, x, 5, 2))
        .collect();
    let uniform_preds: Vec<usize> = x_test.iter()
        .map(|x| knn_predict(&x_train, &y_train, x, 5, 2))
        .collect();
    let weighted_acc = accuracy(&y_test, &weighted_preds);
    let uniform_acc = accuracy(&y_test, &uniform_preds);
    println!("  Uniform voting accuracy:   {:.1}%", uniform_acc * 100.0);
    println!("  Distance-weighted accuracy: {:.1}%", weighted_acc * 100.0);

    // ASCII decision boundary
    println!("\n--- Decision Boundary (K={}) ---", best_k);
    print_decision_boundary(&x_train, &y_train, best_k);

    kata_metric("best_k", best_k as f64);
    kata_metric("best_test_accuracy", best_test_acc);
    kata_metric("weighted_accuracy", weighted_acc);
    kata_metric("train_samples", train_n as f64);
}

fn knn_predict(
    x_train: &[Vec<f64>],
    y_train: &[usize],
    query: &[f64],
    k: usize,
    n_classes: usize,
) -> usize {
    let mut distances: Vec<(usize, f64)> = x_train.iter().enumerate().map(|(i, x)| {
        let dist = euclidean_distance(x, query);
        (i, dist)
    }).collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut votes = vec![0usize; n_classes];
    for i in 0..k.min(distances.len()) {
        let label = y_train[distances[i].0];
        votes[label] += 1;
    }

    votes.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0
}

fn knn_predict_weighted(
    x_train: &[Vec<f64>],
    y_train: &[usize],
    query: &[f64],
    k: usize,
    n_classes: usize,
) -> usize {
    let mut distances: Vec<(usize, f64)> = x_train.iter().enumerate().map(|(i, x)| {
        (i, euclidean_distance(x, query))
    }).collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut weighted_votes = vec![0.0f64; n_classes];
    for i in 0..k.min(distances.len()) {
        let label = y_train[distances[i].0];
        let weight = 1.0 / (distances[i].1 + 1e-8);
        weighted_votes[label] += weight;
    }

    weighted_votes.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn accuracy(actual: &[usize], predicted: &[usize]) -> f64 {
    let correct = actual.iter().zip(predicted.iter()).filter(|(a, p)| a == p).count();
    correct as f64 / actual.len() as f64
}

fn print_decision_boundary(x_train: &[Vec<f64>], y_train: &[usize], k: usize) {
    let x_min = 0.0;
    let x_max = 8.0;
    let y_min = 0.0;
    let y_max = 8.0;
    let rows = 16;
    let cols = 32;

    for r in 0..rows {
        let y = y_max - (y_max - y_min) * r as f64 / (rows - 1) as f64;
        for c in 0..cols {
            let x = x_min + (x_max - x_min) * c as f64 / (cols - 1) as f64;
            let pred = knn_predict(x_train, y_train, &vec![x, y], k, 2);
            // Check if any training point is nearby
            let near_train = x_train.iter().zip(y_train.iter()).any(|(pt, _)| {
                (pt[0] - x).abs() < 0.3 && (pt[1] - y).abs() < 0.3
            });
            if near_train {
                let label = x_train.iter().zip(y_train.iter())
                    .min_by_key(|(pt, _)| ((pt[0] - x).powi(2) + (pt[1] - y).powi(2)) as i64)
                    .unwrap().1;
                print!("{}", if *label == 0 { 'A' } else { 'B' });
            } else {
                print!("{}", if pred == 0 { '.' } else { '+' });
            }
        }
        println!();
    }
    println!("A/. = Class 0, B/+ = Class 1");
}

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() {
        rng = lcg(rng);
        let j = (rng % (i as u64 + 1)) as usize;
        arr.swap(i, j);
    }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- KNN classifies by majority vote among the K nearest training points. It is simple, non-parametric, and makes no assumptions about data distribution.
- The choice of K controls the bias-variance tradeoff: small K overfits, large K underfits. Validate K on held-out data.
- Distance-weighted voting gives closer neighbors more influence, often improving performance near decision boundaries.
- KNN struggles with high-dimensional data (curse of dimensionality) and large datasets (slow prediction). Feature scaling is mandatory.
