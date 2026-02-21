# Activation Functions

> Phase 3 — Artificial Neural Networks | Kata 3.3

---

## Concept & Intuition

### What problem are we solving?

Activation functions introduce non-linearity into neural networks. Without them, a multi-layer network would collapse into a single linear transformation (since the composition of linear functions is linear). The activation function is what gives neural networks their power to model complex, non-linear relationships.

Different activation functions have different properties. Sigmoid squashes values to (0,1), which is good for probabilities but causes vanishing gradients. Tanh squashes to (-1,1), which is zero-centered (better than sigmoid for training). ReLU (max(0,x)) is the most popular: it is fast, does not saturate for positive values, and works well in practice. Leaky ReLU fixes ReLU's "dying neuron" problem by allowing a small gradient for negative values.

Choosing the right activation function for each layer is a practical skill. Hidden layers typically use ReLU (or variants like Leaky ReLU, ELU, GELU). Output layers use sigmoid for binary classification, softmax for multi-class classification, or no activation (identity) for regression. The activation function determines the range and behavior of each layer's output.

### Why naive approaches fail

Using sigmoid or tanh in deep hidden layers causes vanishing gradients: the derivative is always less than 1, so gradients get exponentially smaller as they flow through many layers. This makes training deep networks nearly impossible. ReLU solved this problem by having a derivative of exactly 1 for positive values — gradients flow through without shrinking.

### Mental models

- **Activation as a gatekeeper**: Each neuron decides how much signal to pass through. Sigmoid is a soft gate (0 to 1). ReLU is a hard gate (pass or block). The gate shape determines the network's behavior.
- **Non-linearity as bending**: Linear layers can only rotate and scale the feature space. Activation functions bend it, allowing the network to wrap decision boundaries around complex patterns.
- **Gradient flow as a river**: Gradients must flow from output to input during backpropagation. Sigmoid is a narrow channel (gradients shrink). ReLU is a wide channel (gradients flow freely for positive values).

### Visual explanations

```
  Common activation functions:

  Sigmoid:        Tanh:           ReLU:           Leaky ReLU:
  y│   ___        y│   ___        y│      /       y│      /
   │  /           │  /            │     /         │     /
   │ /            │/              │    /          │    /
   │/         ────┤           ────┤   /       ────┤  /
   │          ────│╱              │  /            │ /
   │              │╱              │ /             │/ (slope=0.01)
   └──── x        └──── x        └──── x        └──── x
   range: (0,1)   range: (-1,1)  range: [0,∞)   range: (-∞,∞)
```

---

## Hands-on Exploration

1. Implement several activation functions and their derivatives.
2. Visualize each function and its gradient using ASCII plots.
3. Demonstrate how activation choice affects learning on a simple problem.

---

## Live Code

```rust
fn main() {
    // === Activation Functions ===
    // The non-linearity that gives neural networks their power.

    println!("=== Activation Functions: Values and Derivatives ===\n");

    // Define activation functions and their derivatives
    let activations: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        ("Sigmoid",
         Box::new(|z: f64| 1.0 / (1.0 + (-z).exp())),
         Box::new(|z: f64| {
             let s = 1.0 / (1.0 + (-z).exp());
             s * (1.0 - s)
         })),
        ("Tanh",
         Box::new(|z: f64| z.tanh()),
         Box::new(|z: f64| 1.0 - z.tanh() * z.tanh())),
        ("ReLU",
         Box::new(|z: f64| if z > 0.0 { z } else { 0.0 }),
         Box::new(|z: f64| if z > 0.0 { 1.0 } else { 0.0 })),
        ("Leaky ReLU",
         Box::new(|z: f64| if z > 0.0 { z } else { 0.01 * z }),
         Box::new(|z: f64| if z > 0.0 { 1.0 } else { 0.01 })),
        ("ELU",
         Box::new(|z: f64| if z > 0.0 { z } else { (z.exp() - 1.0) }),
         Box::new(|z: f64| if z > 0.0 { 1.0 } else { z.exp() })),
    ];

    // === Value table ===
    let test_inputs = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];

    for (name, func, deriv) in &activations {
        println!("  {}:", name);
        println!("    {:>6} {:>8} {:>8}", "z", "f(z)", "f'(z)");
        println!("    {:->6} {:->8} {:->8}", "", "", "");
        for &z in &test_inputs {
            println!("    {:>6.1} {:>8.4} {:>8.4}", z, func(z), deriv(z));
        }
        println!();
    }

    // === ASCII plots ===
    println!("=== ASCII Visualization ===\n");

    let plot_height = 11;
    let plot_width = 41;
    let x_range = (-4.0, 4.0);
    let y_range = (-1.5, 3.0);

    for (name, func, _) in &activations {
        println!("  {}:", name);

        let mut grid = vec![vec![' '; plot_width]; plot_height];

        // Draw axes
        let y_zero_row = ((y_range.1 - 0.0) / (y_range.1 - y_range.0) * (plot_height - 1) as f64) as usize;
        let x_zero_col = ((0.0 - x_range.0) / (x_range.1 - x_range.0) * (plot_width - 1) as f64) as usize;
        if y_zero_row < plot_height {
            for c in 0..plot_width { grid[y_zero_row][c] = '-'; }
        }
        if x_zero_col < plot_width {
            for r in 0..plot_height { grid[r][x_zero_col] = '|'; }
        }
        if y_zero_row < plot_height && x_zero_col < plot_width {
            grid[y_zero_row][x_zero_col] = '+';
        }

        // Plot function
        for col in 0..plot_width {
            let x = x_range.0 + (x_range.1 - x_range.0) * col as f64 / (plot_width - 1) as f64;
            let y = func(x);
            let row = ((y_range.1 - y) / (y_range.1 - y_range.0) * (plot_height - 1) as f64) as i32;
            if row >= 0 && (row as usize) < plot_height {
                grid[row as usize][col] = '*';
            }
        }

        for row in &grid {
            let line: String = row.iter().collect();
            println!("    {}", line);
        }
        println!();
    }

    // === Gradient magnitude comparison ===
    println!("=== Gradient Magnitudes (Critical for Training) ===\n");
    println!("  How much gradient passes through at different input values:\n");

    println!("  {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "z", "Sigmoid'", "Tanh'", "ReLU'", "LeakyReLU'", "ELU'");
    println!("  {:->12} {:->10} {:->10} {:->10} {:->10} {:->10}",
        "", "", "", "", "", "");

    for &z in &[-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let sig_d = { let s = 1.0 / (1.0 + (-z).exp()); s * (1.0 - s) };
        let tanh_d = 1.0 - z.tanh() * z.tanh();
        let relu_d = if z > 0.0 { 1.0 } else { 0.0 };
        let leaky_d = if z > 0.0 { 1.0 } else { 0.01 };
        let elu_d = if z > 0.0 { 1.0 } else { z.exp() };

        println!("  {:>12.1} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            z, sig_d, tanh_d, relu_d, leaky_d, elu_d);
    }

    println!();
    println!("  Key observations:");
    println!("  - Sigmoid gradient max is 0.25 (at z=0) — always < 1, causes vanishing");
    println!("  - Tanh gradient max is 1.0 (at z=0) — better but still saturates");
    println!("  - ReLU gradient is exactly 1 for z>0 — no vanishing! But 0 for z<0");
    println!("  - Leaky ReLU never has zero gradient — fixes dying neuron problem");

    // === Demonstrate effect on training ===
    println!("\n=== Effect on Training: 3-Layer Network ===\n");

    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // Simple dataset: XOR
    let data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    // Train a small network with different activations
    let activation_names = ["Sigmoid", "Tanh", "ReLU"];

    for act_name in &activation_names {
        // Simple 2 → 4 → 1 network
        seed = 42; // reset seed for fair comparison

        let mut w1: Vec<Vec<f64>> = (0..4).map(|_| (0..2).map(|_| rand_f64() * 0.5).collect()).collect();
        let mut b1 = vec![0.0; 4];
        let mut w2: Vec<f64> = (0..4).map(|_| rand_f64() * 0.5).collect();
        let mut b2 = 0.0;

        let act = |z: f64, name: &str| -> f64 {
            match name {
                "Sigmoid" => 1.0 / (1.0 + (-z).exp()),
                "Tanh" => z.tanh(),
                "ReLU" => if z > 0.0 { z } else { 0.01 * z },
                _ => z,
            }
        };
        let act_d = |z: f64, name: &str| -> f64 {
            match name {
                "Sigmoid" => { let s = 1.0 / (1.0 + (-z).exp()); s * (1.0 - s) },
                "Tanh" => 1.0 - z.tanh() * z.tanh(),
                "ReLU" => if z > 0.0 { 1.0 } else { 0.01 },
                _ => 1.0,
            }
        };

        let lr = 0.5;
        let mut loss_at_100 = 0.0;
        let mut loss_at_500 = 0.0;

        for epoch in 0..1000 {
            let mut total_loss = 0.0;

            for (x, target) in &data {
                // Forward
                let z1: Vec<f64> = (0..4).map(|i|
                    w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                ).collect();
                let h1: Vec<f64> = z1.iter().map(|&z| act(z, act_name)).collect();

                let z2: f64 = w2.iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2;
                let pred = 1.0 / (1.0 + (-z2).exp()); // always sigmoid output

                let p = pred.max(1e-15).min(1.0 - 1e-15);
                total_loss += -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());

                // Backward
                let d_out = pred - target;

                for i in 0..4 {
                    let d_h1i = d_out * w2[i];
                    let d_z1i = d_h1i * act_d(z1[i], act_name);
                    w1[i][0] -= lr * d_z1i * x[0];
                    w1[i][1] -= lr * d_z1i * x[1];
                    b1[i] -= lr * d_z1i;
                    w2[i] -= lr * d_out * h1[i];
                }
                b2 -= lr * d_out;
            }

            total_loss /= data.len() as f64;
            if epoch == 99 { loss_at_100 = total_loss; }
            if epoch == 499 { loss_at_500 = total_loss; }
        }

        // Final predictions
        let mut final_loss = 0.0;
        let mut predictions = Vec::new();
        for (x, target) in &data {
            let z1: Vec<f64> = (0..4).map(|i|
                w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
            ).collect();
            let h1: Vec<f64> = z1.iter().map(|&z| act(z, act_name)).collect();
            let z2: f64 = w2.iter().zip(h1.iter()).map(|(w, h)| w * h).sum::<f64>() + b2;
            let pred = 1.0 / (1.0 + (-z2).exp());
            let p = pred.max(1e-15).min(1.0 - 1e-15);
            final_loss += -(target * p.ln() + (1.0 - target) * (1.0 - p).ln());
            predictions.push(pred);
        }
        final_loss /= data.len() as f64;

        println!("  {} hidden activation:", act_name);
        println!("    loss@100: {:.4}, loss@500: {:.4}, loss@1000: {:.4}",
            loss_at_100, loss_at_500, final_loss);
        println!("    predictions: [{:.3}, {:.3}, {:.3}, {:.3}]",
            predictions[0], predictions[1], predictions[2], predictions[3]);
        println!();
    }

    // === When to use what ===
    println!("=== Activation Function Guidelines ===\n");
    println!("  Layer type          │ Recommended activation");
    println!("  ────────────────────┼──────────────────────────────────");
    println!("  Hidden layers       │ ReLU (default), Leaky ReLU, GELU");
    println!("  Binary output       │ Sigmoid (output in [0, 1])");
    println!("  Multi-class output  │ Softmax (outputs sum to 1)");
    println!("  Regression output   │ Identity / Linear (no activation)");
    println!("  Recurrent layers    │ Tanh (bounded, zero-centered)");

    println!();
    println!("Key insight: Activation functions enable non-linearity.");
    println!("ReLU is the default choice for hidden layers due to stable gradients.");
}
```

---

## Key Takeaways

- Activation functions introduce non-linearity — without them, any deep network collapses to a single linear transformation.
- Sigmoid and tanh saturate for large inputs, causing vanishing gradients that make deep networks hard to train.
- ReLU (and variants like Leaky ReLU) solve the vanishing gradient problem by providing a constant gradient of 1 for positive inputs.
- The choice of output activation depends on the task: sigmoid for binary classification, softmax for multi-class, identity for regression.
