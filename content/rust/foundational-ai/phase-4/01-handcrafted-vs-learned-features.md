# Handcrafted vs Learned Features

> Phase 4 — Representation Learning | Kata 4.1

---

## Concept & Intuition

### What problem are we solving?

Traditional machine learning relies on humans to design features: a data scientist examines the raw data, applies domain knowledge, and creates meaningful inputs for the model (feature engineering). Deep learning automates this process: the hidden layers learn to extract useful features directly from raw data. This is representation learning — the model discovers its own features.

Handcrafted features require domain expertise and are limited by human imagination. An expert might create "price per square foot" for real estate or "word frequency" for text classification. These features can be powerful, but they require extensive experimentation and domain knowledge. Worse, for complex data like images or audio, humans often cannot articulate what features matter.

Learned features are discovered automatically by the network. The first hidden layer might learn edge detectors, the second layer combines edges into shapes, and the third layer combines shapes into objects. This hierarchical feature learning is why deep learning excels on complex data — it finds features that humans would never think to create.

### Why naive approaches fail

Relying entirely on handcrafted features hits a ceiling: human intuition cannot capture all the subtle patterns in high-dimensional data. Conversely, relying entirely on learned features requires massive datasets and deep architectures. The best approaches often combine both: use domain knowledge to preprocess data and let the network learn the rest.

### Mental models

- **Handcrafted = human knowledge as code**: You translate your understanding of the problem into explicit mathematical transformations.
- **Learned = let the data teach**: Instead of telling the model what features to use, you give it raw data and enough capacity, and it discovers useful representations on its own.
- **Features as language**: Handcrafted features are like a human writing a description. Learned features are like a neural network developing its own internal language.

### Visual explanations

```
  Traditional ML pipeline:
  Raw data → [Human engineers features] → Features → [Simple model] → Prediction

  Deep learning pipeline:
  Raw data → [Layer 1: learns simple features]
           → [Layer 2: learns complex features]
           → [Layer 3: learns task-specific features]
           → Prediction

  The network IS the feature engineering.
```

---

## Hands-on Exploration

1. Solve a classification task with handcrafted features and a simple model.
2. Solve the same task with raw features and a multi-layer network.
3. Compare what each approach learns and which performs better.

---

## Live Code

```rust
fn main() {
    // === Handcrafted vs Learned Features ===
    // Comparing human-designed features with network-discovered features.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // === Dataset: Classify points inside/outside a ring ===
    // Points at distance 1.5-2.5 from origin → class 1 (ring)
    // Points at distance 0-1.0 or 3.0+ from origin → class 0

    let n_samples = 100;
    let mut raw_data: Vec<(f64, f64, f64)> = Vec::new(); // (x1, x2, label)

    for _ in 0..n_samples {
        let x1 = rand_f64() * 4.0 - 2.0; // [-2, 2]
        let x2 = rand_f64() * 4.0 - 2.0;
        let dist = (x1 * x1 + x2 * x2).sqrt();
        let label = if dist >= 1.2 && dist <= 2.2 { 1.0 } else { 0.0 };
        raw_data.push((x1, x2, label));
    }

    let n_class1: usize = raw_data.iter().filter(|d| d.2 > 0.5).count();
    println!("=== Handcrafted vs Learned Features ===\n");
    println!("  Task: classify points inside a ring (annulus)");
    println!("  Data: {} samples ({} in ring, {} outside)\n",
        n_samples, n_class1, n_samples - n_class1);

    // === Approach 1: Handcrafted features ===
    println!("--- Approach 1: Handcrafted Features + Simple Model ---\n");

    // Engineer a feature: distance from origin
    // This single feature perfectly captures the ring pattern!
    let handcrafted: Vec<(Vec<f64>, f64)> = raw_data.iter()
        .map(|(x1, x2, label)| {
            let dist = (x1 * x1 + x2 * x2).sqrt();
            let dist_sq = x1 * x1 + x2 * x2;
            (vec![dist, dist_sq], *label)
        })
        .collect();

    println!("  Handcrafted features: [distance, distance²]");
    println!("  These capture the ring pattern directly.\n");

    // Train a simple logistic regression on handcrafted features
    let mut w_hc = vec![0.0, 0.0];
    let mut b_hc = 0.0;
    let lr = 0.1;

    for _epoch in 0..200 {
        let mut gw = vec![0.0, 0.0];
        let mut gb = 0.0;
        let n = handcrafted.len() as f64;

        for (x, target) in &handcrafted {
            let z: f64 = w_hc.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b_hc;
            let pred = 1.0 / (1.0 + (-z).exp());
            let err = pred - target;
            gw[0] += err * x[0] / n;
            gw[1] += err * x[1] / n;
            gb += err / n;
        }

        w_hc[0] -= lr * gw[0];
        w_hc[1] -= lr * gw[1];
        b_hc -= lr * gb;
    }

    // Evaluate
    let mut correct_hc = 0;
    for (x, target) in &handcrafted {
        let z: f64 = w_hc.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b_hc;
        let pred = 1.0 / (1.0 + (-z).exp());
        let class = if pred >= 0.5 { 1.0 } else { 0.0 };
        if (class - target).abs() < 0.1 { correct_hc += 1; }
    }
    let acc_hc = correct_hc as f64 / n_samples as f64 * 100.0;
    println!("  Accuracy with handcrafted features: {:.1}% ({}/{})\n",
        acc_hc, correct_hc, n_samples);

    // === Approach 2: Raw features with linear model ===
    println!("--- Approach 2: Raw Features + Linear Model ---\n");
    println!("  Raw features: [x1, x2] (no engineering)\n");

    let raw: Vec<(Vec<f64>, f64)> = raw_data.iter()
        .map(|(x1, x2, label)| (vec![*x1, *x2], *label))
        .collect();

    let mut w_raw = vec![0.0, 0.0];
    let mut b_raw = 0.0;

    for _epoch in 0..200 {
        let mut gw = vec![0.0, 0.0];
        let mut gb = 0.0;
        let n = raw.len() as f64;

        for (x, target) in &raw {
            let z: f64 = w_raw.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b_raw;
            let pred = 1.0 / (1.0 + (-z).exp());
            let err = pred - target;
            gw[0] += err * x[0] / n;
            gw[1] += err * x[1] / n;
            gb += err / n;
        }

        w_raw[0] -= lr * gw[0];
        w_raw[1] -= lr * gw[1];
        b_raw -= lr * gb;
    }

    let mut correct_raw = 0;
    for (x, target) in &raw {
        let z: f64 = w_raw.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b_raw;
        let pred = 1.0 / (1.0 + (-z).exp());
        let class = if pred >= 0.5 { 1.0 } else { 0.0 };
        if (class - target).abs() < 0.1 { correct_raw += 1; }
    }
    let acc_raw = correct_raw as f64 / n_samples as f64 * 100.0;
    println!("  Accuracy with raw features (linear): {:.1}% ({}/{})",
        acc_raw, correct_raw, n_samples);
    println!("  Linear model CANNOT learn a ring-shaped boundary!\n");

    // === Approach 3: Raw features with MLP (learned features) ===
    println!("--- Approach 3: Raw Features + MLP (Learned Features) ---\n");
    println!("  Architecture: 2 → 16 → 8 → 1 (ReLU → ReLU → sigmoid)\n");

    // Initialize MLP
    seed = 99;
    let mut w1: Vec<Vec<f64>> = (0..16).map(|_|
        (0..2).map(|_| rand_f64() * (2.0 / 18.0_f64).sqrt()).collect()
    ).collect();
    let mut b1 = vec![0.0; 16];
    let mut w2: Vec<Vec<f64>> = (0..8).map(|_|
        (0..16).map(|_| rand_f64() * (2.0 / 24.0_f64).sqrt()).collect()
    ).collect();
    let mut b2 = vec![0.0; 8];
    let mut w3: Vec<f64> = (0..8).map(|_| rand_f64() * (2.0 / 9.0_f64).sqrt()).collect();
    let mut b3 = 0.0;

    let lr_mlp = 0.05;

    for epoch in 0..500 {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (x, target) in &raw {
            // Forward
            let z1: Vec<f64> = (0..16).map(|i|
                w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b1[i]
            ).collect();
            let h1: Vec<f64> = z1.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();

            let z2: Vec<f64> = (0..8).map(|i|
                w2[i].iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2[i]
            ).collect();
            let h2: Vec<f64> = z2.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();

            let z3: f64 = w3.iter().zip(h2.iter()).map(|(w, h)| w * h).sum::<f64>() + b3;
            let pred = 1.0 / (1.0 + (-z3).exp());

            let p = pred.max(1e-15).min(1.0 - 1e-15);
            total_loss += -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());
            if (if pred >= 0.5 { 1.0 } else { 0.0 } - target).abs() < 0.1 { correct += 1; }

            // Backward
            let d_out = pred - target;

            let d_h2: Vec<f64> = (0..8).map(|i| d_out * w3[i]).collect();
            for i in 0..8 {
                w3[i] -= lr_mlp * d_out * h2[i];
            }
            b3 -= lr_mlp * d_out;

            let d_z2: Vec<f64> = d_h2.iter().enumerate()
                .map(|(i, &d)| if z2[i] > 0.0 { d } else { 0.01 * d }).collect();

            let d_h1: Vec<f64> = (0..16).map(|j|
                (0..8).map(|i| d_z2[i] * w2[i][j]).sum::<f64>()
            ).collect();
            for i in 0..8 {
                for j in 0..16 {
                    w2[i][j] -= lr_mlp * d_z2[i] * h1[j];
                }
                b2[i] -= lr_mlp * d_z2[i];
            }

            let d_z1: Vec<f64> = d_h1.iter().enumerate()
                .map(|(i, &d)| if z1[i] > 0.0 { d } else { 0.01 * d }).collect();
            for i in 0..16 {
                for j in 0..2 {
                    w1[i][j] -= lr_mlp * d_z1[i] * x[j];
                }
                b1[i] -= lr_mlp * d_z1[i];
            }
        }

        if epoch < 10 || epoch % 50 == 0 || epoch == 499 {
            let acc = correct as f64 / n_samples as f64 * 100.0;
            println!("    epoch {:>4}: loss={:.4}, accuracy={:.1}%",
                epoch, total_loss / n_samples as f64, acc);
        }
    }

    // Final MLP accuracy
    let mut correct_mlp = 0;
    for (x, target) in &raw {
        let z1: Vec<f64> = (0..16).map(|i|
            w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b1[i]
        ).collect();
        let h1: Vec<f64> = z1.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();
        let z2: Vec<f64> = (0..8).map(|i|
            w2[i].iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2[i]
        ).collect();
        let h2: Vec<f64> = z2.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();
        let z3: f64 = w3.iter().zip(h2.iter()).map(|(w, h)| w * h).sum::<f64>() + b3;
        let pred = 1.0 / (1.0 + (-z3).exp());
        if (if pred >= 0.5 { 1.0 } else { 0.0 } - target).abs() < 0.1 { correct_mlp += 1; }
    }
    let acc_mlp = correct_mlp as f64 / n_samples as f64 * 100.0;

    // === Summary ===
    println!("\n=== Summary ===\n");
    println!("  {:>35} {:>12}", "Approach", "Accuracy");
    println!("  {:->35} {:->12}", "", "");
    println!("  {:>35} {:>11.1}%", "Handcrafted features + linear", acc_hc);
    println!("  {:>35} {:>11.1}%", "Raw features + linear", acc_raw);
    println!("  {:>35} {:>11.1}%", "Raw features + MLP (learned)", acc_mlp);
    println!();
    println!("  The handcrafted approach works because a human recognized the ring pattern.");
    println!("  The linear model with raw features fails because the pattern is non-linear.");
    println!("  The MLP discovers its own features that capture the ring pattern.");

    println!();
    println!("Key insight: Deep learning = automatic feature engineering.");
    println!("Hidden layers learn representations that make the task easier for the output layer.");
}
```

---

## Key Takeaways

- Handcrafted features require domain expertise but can be highly effective when the right features are known.
- Linear models fail on problems with non-linear structure (like ring-shaped decision boundaries) unless features are engineered to expose the structure.
- Deep networks learn their own features automatically — hidden layers discover representations that make the final classification easier.
- The power of deep learning is representation learning: the network finds features that humans might never think to engineer.
