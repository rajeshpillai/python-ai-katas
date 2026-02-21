# Visualizing Hidden Layers

> Phase 4 — Representation Learning | Kata 4.4

---

## Concept & Intuition

### What problem are we solving?

Hidden layers in a neural network learn intermediate representations of the data. But what do these representations look like? By examining the activations of hidden layers for different inputs, we can understand what the network has learned. This is not just intellectual curiosity — it is a powerful debugging and interpretability tool.

Each neuron in a hidden layer is a feature detector: it responds strongly to certain patterns in its input and weakly to others. In the first hidden layer, neurons might detect simple features (presence of a certain input range). In deeper layers, neurons combine those simple features into complex ones. By visualizing which neurons activate for which inputs, we can see the hierarchy of features the network has learned.

Understanding hidden layer representations helps answer critical questions: Is the network learning the right features? Are different classes well-separated in the hidden space? Are there neurons that never activate (dead neurons)? Are certain features redundant? This introspection is essential for debugging, improving, and trusting neural network models.

### Why naive approaches fail

Treating a neural network as a black box means you cannot diagnose failures. If the model misclassifies certain inputs, looking at the hidden layer activations for those inputs reveals whether the problem is in the learned representation (the features are not discriminative) or in the output layer (the features are good but the decision boundary is wrong). Without this visibility, debugging is blind guesswork.

### Mental models

- **Hidden layers as a camera**: Each hidden layer takes a "photo" of the data from a different perspective. Examining these photos reveals what the network "sees."
- **Activation patterns as fingerprints**: Each input creates a unique pattern of activations in the hidden layer. Similar inputs create similar patterns — this is how the network recognizes similarity.
- **Hidden space as a transformed world**: The hidden layer maps the original feature space into a new space where the task is easier. Visualizing this new space reveals the network's learned understanding.

### Visual explanations

```
  Original space:             Hidden layer space:
  (non-linearly separable)    (linearly separable!)

    ● ○ ○ ● ●                    ●●●
    ○ ● ● ○ ○                   ●●●●●
    ○ ● ● ○ ○                  ──────────
    ● ○ ○ ● ●                   ○○○○○
    ● ○ ● ● ○                    ○○○

  The hidden layer transforms the data so that
  a simple linear boundary can separate the classes.
```

---

## Hands-on Exploration

1. Train a network and extract hidden layer activations for each input.
2. Visualize activation patterns as heatmaps and histograms.
3. Show how the hidden space separates classes that overlap in the original space.

---

## Live Code

```rust
fn main() {
    // === Visualizing Hidden Layers ===
    // Peering inside the network to see what it learns.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    println!("=== Visualizing Hidden Layers ===\n");

    // === Dataset: concentric circles (non-linearly separable) ===
    let n_per_class = 30;
    let mut data: Vec<(Vec<f64>, f64)> = Vec::new();

    // Inner circle (class 0): radius ~0.5
    for _ in 0..n_per_class {
        let angle = rand_f64() * std::f64::consts::PI;
        let r = 0.3 + rand_f64() * 0.3;
        let x = r * angle.cos();
        let y = r * angle.sin();
        data.push((vec![x, y], 0.0));
    }

    // Outer ring (class 1): radius ~1.5
    for _ in 0..n_per_class {
        let angle = rand_f64() * std::f64::consts::PI;
        let r = 1.2 + rand_f64() * 0.4;
        let x = r * angle.cos();
        let y = r * angle.sin();
        data.push((vec![x, y], 1.0));
    }

    println!("  Task: separate inner circle from outer ring");
    println!("  Architecture: 2 → 8 → 4 → 1\n");

    // === Build and train network ===
    // Layer 1: 2 → 8
    let mut w1: Vec<Vec<f64>> = (0..8).map(|_|
        (0..2).map(|_| rand_f64() * 0.8).collect()
    ).collect();
    let mut b1 = vec![0.0; 8];

    // Layer 2: 8 → 4
    let mut w2: Vec<Vec<f64>> = (0..4).map(|_|
        (0..8).map(|_| rand_f64() * 0.5).collect()
    ).collect();
    let mut b2 = vec![0.0; 4];

    // Layer 3: 4 → 1
    let mut w3: Vec<f64> = (0..4).map(|_| rand_f64() * 0.5).collect();
    let mut b3 = 0.0;

    let lr = 0.1;

    // Train
    for _epoch in 0..500 {
        for (x, target) in &data {
            // Forward
            let z1: Vec<f64> = (0..8).map(|i|
                w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b1[i]
            ).collect();
            let h1: Vec<f64> = z1.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();

            let z2: Vec<f64> = (0..4).map(|i|
                w2[i].iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2[i]
            ).collect();
            let h2: Vec<f64> = z2.iter().map(|&z| if z > 0.0 { z } else { 0.01 * z }).collect();

            let z3: f64 = w3.iter().zip(h2.iter()).map(|(w, h)| w * h).sum::<f64>() + b3;
            let pred = 1.0 / (1.0 + (-z3).exp());

            // Backward
            let d_out = pred - target;
            let d_h2: Vec<f64> = (0..4).map(|i| d_out * w3[i]).collect();
            for i in 0..4 { w3[i] -= lr * d_out * h2[i]; }
            b3 -= lr * d_out;

            let d_z2: Vec<f64> = d_h2.iter().enumerate()
                .map(|(i, &d)| if z2[i] > 0.0 { d } else { 0.01 * d }).collect();
            let d_h1: Vec<f64> = (0..8).map(|j|
                (0..4).map(|i| d_z2[i] * w2[i][j]).sum::<f64>()
            ).collect();
            for i in 0..4 {
                for j in 0..8 { w2[i][j] -= lr * d_z2[i] * h1[j]; }
                b2[i] -= lr * d_z2[i];
            }

            let d_z1: Vec<f64> = d_h1.iter().enumerate()
                .map(|(i, &d)| if z1[i] > 0.0 { d } else { 0.01 * d }).collect();
            for i in 0..8 {
                for j in 0..2 { w1[i][j] -= lr * d_z1[i] * x[j]; }
                b1[i] -= lr * d_z1[i];
            }
        }
    }

    // === Extract hidden layer activations ===
    println!("=== Hidden Layer 1 Activations (8 neurons) ===\n");

    let forward_l1 = |x: &[f64]| -> Vec<f64> {
        (0..8).map(|i| {
            let z = w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b1[i];
            if z > 0.0 { z } else { 0.01 * z }
        }).collect()
    };

    let forward_l2 = |h1: &[f64]| -> Vec<f64> {
        (0..4).map(|i| {
            let z = w2[i].iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2[i];
            if z > 0.0 { z } else { 0.01 * z }
        }).collect()
    };

    let forward_all = |x: &[f64]| -> f64 {
        let h1 = forward_l1(x);
        let h2 = forward_l2(&h1);
        let z3: f64 = w3.iter().zip(h2.iter()).map(|(w, h)| w * h).sum::<f64>() + b3;
        1.0 / (1.0 + (-z3).exp())
    };

    // Show activations for sample inputs
    println!("  Activation heatmap (high=█, medium=▓, low=░, zero=·):\n");
    println!("  {:>14} {:>6} {:>24} {:>6}",
        "Input", "Class", "Layer 1 activations", "Pred");
    println!("  {:->14} {:->6} {:->24} {:->6}", "", "", "", "");

    let activation_char = |val: f64, max_val: f64| -> char {
        let normalized = val / max_val.max(0.01);
        if normalized > 0.7 { '█' }
        else if normalized > 0.3 { '▓' }
        else if normalized > 0.05 { '░' }
        else { '·' }
    };

    // Find max activation for scaling
    let mut max_act = 0.0_f64;
    for (x, _) in &data {
        let h1 = forward_l1(x);
        for &a in &h1 {
            if a > max_act { max_act = a; }
        }
    }

    // Show a subset
    for i in (0..data.len()).step_by(5) {
        let (x, target) = &data[i];
        let h1 = forward_l1(x);
        let pred = forward_all(x);

        let activation_str: String = h1.iter()
            .map(|&a| activation_char(a, max_act))
            .collect();

        println!("  [{:>5.2},{:>5.2}] {:>6.0} [{}] {:>6.2}",
            x[0], x[1], target, activation_str, pred);
    }

    // === Neuron activation statistics ===
    println!("\n=== Per-Neuron Statistics (Layer 1) ===\n");
    println!("  {:>8} {:>10} {:>10} {:>10} {:>12}",
        "Neuron", "Mean act", "Max act", "Dead %", "Role");
    println!("  {:->8} {:->10} {:->10} {:->10} {:->12}", "", "", "", "", "");

    for neuron in 0..8 {
        let acts: Vec<f64> = data.iter()
            .map(|(x, _)| forward_l1(x)[neuron])
            .collect();

        let mean_act: f64 = acts.iter().sum::<f64>() / acts.len() as f64;
        let max_act_n: f64 = acts.iter().cloned().fold(0.0_f64, f64::max);
        let dead_pct = acts.iter().filter(|&&a| a <= 0.0).count() as f64
            / acts.len() as f64 * 100.0;

        // Determine role by looking at class-specific activation
        let class0_mean: f64 = data.iter()
            .filter(|(_, t)| *t < 0.5)
            .map(|(x, _)| forward_l1(x)[neuron])
            .sum::<f64>() / n_per_class as f64;
        let class1_mean: f64 = data.iter()
            .filter(|(_, t)| *t > 0.5)
            .map(|(x, _)| forward_l1(x)[neuron])
            .sum::<f64>() / n_per_class as f64;

        let role = if dead_pct > 80.0 { "mostly dead" }
            else if (class1_mean - class0_mean).abs() > 0.3 {
                if class1_mean > class0_mean { "outer-detector" } else { "inner-detector" }
            } else { "shared" };

        println!("  {:>8} {:>10.3} {:>10.3} {:>9.0}% {:>12}",
            neuron, mean_act, max_act_n, dead_pct, role);
    }

    // === Hidden Layer 2 activations ===
    println!("\n=== Hidden Layer 2 Activations (4 neurons) ===\n");
    println!("  Layer 2 combines Layer 1 features into higher-level patterns.\n");

    let mut class0_h2: Vec<Vec<f64>> = Vec::new();
    let mut class1_h2: Vec<Vec<f64>> = Vec::new();

    for (x, target) in &data {
        let h1 = forward_l1(x);
        let h2 = forward_l2(&h1);
        if *target < 0.5 {
            class0_h2.push(h2);
        } else {
            class1_h2.push(h2);
        }
    }

    println!("  Mean activations by class:");
    println!("  {:>8} {:>12} {:>12} {:>12}",
        "Neuron", "Class 0", "Class 1", "Separation");
    println!("  {:->8} {:->12} {:->12} {:->12}", "", "", "", "");

    for d in 0..4 {
        let c0_mean: f64 = class0_h2.iter().map(|h| h[d]).sum::<f64>() / class0_h2.len() as f64;
        let c1_mean: f64 = class1_h2.iter().map(|h| h[d]).sum::<f64>() / class1_h2.len() as f64;
        let sep = (c1_mean - c0_mean).abs();
        let bar_len = (sep * 10.0) as usize;
        println!("  {:>8} {:>12.3} {:>12.3} {:>12.3} |{}|",
            d, c0_mean, c1_mean, sep, "█".repeat(bar_len.min(30)));
    }

    // === Accuracy check ===
    println!("\n=== Model Performance ===\n");

    let mut correct = 0;
    for (x, target) in &data {
        let pred = forward_all(x);
        let class = if pred >= 0.5 { 1.0 } else { 0.0 };
        if (class - target).abs() < 0.1 { correct += 1; }
    }
    println!("  Accuracy: {}/{} ({:.1}%)\n", correct, data.len(),
        correct as f64 / data.len() as f64 * 100.0);

    // === Decision boundary visualization ===
    println!("=== Decision Boundary ===\n");

    let grid_size = 25;
    println!("  y");
    for row in (0..grid_size).rev() {
        let y = -2.0 + 4.0 * row as f64 / (grid_size - 1) as f64;
        let mut line = String::new();
        for col in 0..grid_size * 2 {
            let x = -2.0 + 4.0 * col as f64 / (grid_size * 2 - 1) as f64;
            let pred = forward_all(&[x, y]);

            // Check if data point
            let mut is_data = false;
            for (dx, dt) in &data {
                if ((dx[0] - x).abs() < 0.1) && ((dx[1] - y).abs() < 0.1) {
                    line.push(if *dt > 0.5 { 'O' } else { 'I' });
                    is_data = true;
                    break;
                }
            }
            if !is_data {
                if pred > 0.45 && pred < 0.55 {
                    line.push('/');
                } else if pred >= 0.5 {
                    line.push('.');
                } else {
                    line.push(' ');
                }
            }
        }
        if row == grid_size - 1 || row == grid_size / 2 || row == 0 {
            println!("  {:>4.1}|{}", y, line);
        } else {
            println!("      |{}", line);
        }
    }
    println!("      +{}", "-".repeat(grid_size * 2));
    println!("  Legend: I=inner(class 0), O=outer(class 1), /=boundary");

    println!();
    println!("Key insight: Hidden layers learn representations where the task becomes easy.");
    println!("Visualizing activations reveals what each neuron detects and how");
    println!("layers compose simple features into complex decision boundaries.");
}
```

---

## Key Takeaways

- Hidden layer activations reveal what each neuron has learned to detect — visualizing them transforms neural networks from black boxes into interpretable models.
- Each neuron is a feature detector: it responds to specific patterns in its input. Different neurons specialize in different aspects of the data.
- Deeper layers learn more abstract features by combining the simpler features detected by earlier layers.
- The class separation in hidden layer space shows whether the representation is good: well-separated classes mean the output layer has an easy classification task.
