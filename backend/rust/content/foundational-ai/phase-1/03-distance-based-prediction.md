# Distance-Based Prediction

> Phase 1 — What Does It Mean to Learn? | Kata 1.3

---

## Concept & Intuition

### What problem are we solving?

Distance-based prediction is built on a simple intuition: similar inputs should produce similar outputs. If you want to predict the price of a house, look at houses that are most similar to it and use their prices. This is the k-nearest neighbors (k-NN) algorithm — one of the simplest and most intuitive machine learning methods.

The core question is: how do you measure "similarity"? The most common approach is Euclidean distance — the straight-line distance between two points in feature space. If house A has 1500 sqft and 3 bedrooms, and house B has 1600 sqft and 3 bedrooms, they are "close" in feature space and likely have similar prices. The k-NN algorithm finds the k closest training examples to a query point and averages their targets (for regression) or takes a majority vote (for classification).

Distance-based methods are powerful because they make no assumptions about the underlying function shape. They can capture any pattern, given enough data. But they also have weaknesses: they are sensitive to feature scaling (a feature measured in thousands will dominate one measured in units), they struggle in high dimensions (the "curse of dimensionality"), and they require storing all training data.

### Why naive approaches fail

Using raw features without scaling makes the distance metric meaningless. If sq_feet ranges from 800 to 3000 while bedrooms ranges from 1 to 5, the distance will be dominated by sq_feet. A 200 sqft difference will overshadow a 4-bedroom difference, even though both might be equally important. Always normalize features before computing distances.

### Mental models

- **Neighborhood voting**: To predict a value, survey your nearest neighbors and average their answers. More neighbors (larger k) gives a smoother, more stable prediction.
- **Feature space as a map**: Each observation is a point on a map. Close points are similar. Distance-based methods draw circles around query points and look at what is inside.
- **k controls smoothness**: k=1 means copy the nearest neighbor exactly (noisy). Large k means average many neighbors (smooth but potentially biased). This is another manifestation of the bias-variance tradeoff.

### Visual explanations

```
  k-NN prediction for query point (?):

  feature2 │
           │  A(100)    B(120)
           │       ╲   ╱
           │        ? ?
           │       ╱
           │  C(110)      D(200)
           └────────────────── feature1

  k=3: nearest are A, B, C
  prediction = (100 + 120 + 110) / 3 = 110

  k=1: nearest is C
  prediction = 110
```

---

## Hands-on Exploration

1. Implement Euclidean distance between two feature vectors.
2. Build a k-NN predictor from scratch: find the k nearest neighbors and average their targets.
3. Experiment with different values of k and observe the effect on prediction smoothness.

---

## Live Code

```rust
fn main() {
    // === Distance-Based Prediction (k-NN) ===
    // Similar inputs → similar outputs.

    // Euclidean distance
    let euclidean_distance = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter())
            .map(|(ai, bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    };

    // Dataset: [sq_feet, bedrooms] → price
    let features: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0], vec![1800.0, 4.0], vec![1100.0, 2.0],
        vec![2200.0, 5.0], vec![1600.0, 3.0], vec![900.0, 1.0],
        vec![2000.0, 4.0], vec![1300.0, 2.0], vec![1700.0, 3.0],
        vec![1500.0, 3.0], vec![2100.0, 4.0], vec![1000.0, 2.0],
    ];
    let targets: Vec<f64> = vec![
        250000.0, 320000.0, 195000.0, 410000.0, 275000.0, 150000.0,
        380000.0, 220000.0, 310000.0, 265000.0, 400000.0, 180000.0,
    ];

    // === Feature scaling (min-max normalization) ===
    println!("=== Feature Scaling ===\n");
    let n_features = features[0].len();
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];
    for row in &features {
        for j in 0..n_features {
            if row[j] < mins[j] { mins[j] = row[j]; }
            if row[j] > maxs[j] { maxs[j] = row[j]; }
        }
    }

    let normalize = |row: &[f64]| -> Vec<f64> {
        row.iter().enumerate()
            .map(|(j, &v)| (v - mins[j]) / (maxs[j] - mins[j]))
            .collect()
    };

    let scaled_features: Vec<Vec<f64>> = features.iter().map(|r| normalize(r)).collect();

    println!("  Before scaling: distances dominated by sq_feet");
    let d_raw = euclidean_distance(&features[0], &features[1]);
    println!("    dist([1400,3], [1800,4]) = {:.1}", d_raw);
    println!("  After scaling: features contribute equally");
    let d_scaled = euclidean_distance(&scaled_features[0], &scaled_features[1]);
    println!("    dist(scaled) = {:.3}\n", d_scaled);

    // === k-NN predictor ===
    let knn_predict = |query: &[f64], k: usize| -> (f64, Vec<(usize, f64, f64)>) {
        let query_scaled = normalize(query);
        let mut distances: Vec<(usize, f64)> = scaled_features.iter()
            .enumerate()
            .map(|(i, f)| (i, euclidean_distance(&query_scaled, f)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let neighbors: Vec<(usize, f64, f64)> = distances[..k].iter()
            .map(|&(i, d)| (i, d, targets[i]))
            .collect();
        let prediction = neighbors.iter().map(|n| n.2).sum::<f64>() / k as f64;
        (prediction, neighbors)
    };

    // Query point
    let query = vec![1550.0, 3.0];
    println!("=== k-NN Predictions for query [{:.0}, {:.0}] ===\n", query[0], query[1]);

    for k in [1, 3, 5, 7] {
        let (prediction, neighbors) = knn_predict(&query, k);
        println!("  k={}: predicted price = ${:.0}", k, prediction);
        for (i, dist, target) in &neighbors {
            println!("    neighbor #{}: [{:.0}, {:.0}] dist={:.3} price=${:.0}",
                i, features[*i][0], features[*i][1], dist, target);
        }
        println!();
    }

    // === Effect of k on smoothness ===
    println!("=== Effect of k on Prediction Smoothness ===\n");
    println!("  Predictions across different square footages (bedrooms=3):\n");
    println!("  {:>8} {:>10} {:>10} {:>10} {:>10}",
        "sq_feet", "k=1", "k=3", "k=5", "k=7");
    println!("  {:->8} {:->10} {:->10} {:->10} {:->10}", "", "", "", "", "");

    for sqft in (900..=2200).step_by(100) {
        let q = vec![sqft as f64, 3.0];
        let p1 = knn_predict(&q, 1).0;
        let p3 = knn_predict(&q, 3).0;
        let p5 = knn_predict(&q, 5).0;
        let p7 = knn_predict(&q, 7).0;
        println!("  {:>8} {:>10.0} {:>10.0} {:>10.0} {:>10.0}",
            sqft, p1, p3, p5, p7);
    }

    println!();
    println!("  Notice: k=1 is jumpy (high variance), k=7 is smooth (high bias).");

    // === Distance metrics comparison ===
    println!("\n=== Distance Metrics ===\n");

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    // Euclidean (L2)
    let l2 = euclidean_distance(&a, &b);

    // Manhattan (L1)
    let l1: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).sum();

    // Chebyshev (L∞)
    let linf: f64 = a.iter().zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .fold(0.0_f64, f64::max);

    println!("  Comparing a={:?} and b={:?}", a, b);
    println!("    Euclidean (L2):  {:.3}", l2);
    println!("    Manhattan (L1):  {:.3}", l1);
    println!("    Chebyshev (L∞):  {:.3}", linf);

    println!();
    println!("Key insight: Similar inputs produce similar outputs.");
    println!("k-NN is simple, powerful, and requires no training — but needs good features and scaling.");
}
```

---

## Key Takeaways

- Distance-based prediction (k-NN) uses the principle that similar inputs should produce similar outputs — it finds the nearest training examples and averages their targets.
- Feature scaling is critical: without it, features with larger numeric ranges dominate the distance calculation, making other features irrelevant.
- The choice of k controls the bias-variance tradeoff: small k gives noisy, high-variance predictions; large k gives smooth, high-bias predictions.
- k-NN is a non-parametric method that makes no assumptions about the data distribution — it can model any pattern given enough data.
