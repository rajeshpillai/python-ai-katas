# Why Transformers Scale

> Phase 7 — Attention & Transformers | Kata 7.4

---

## Concept & Intuition

### What problem are we solving?

The transformer architecture has demonstrated a remarkable empirical property: its performance improves predictably with increased compute, data, and parameters, following power-law scaling laws. This is not merely an engineering convenience; it is the foundation of the entire modern AI paradigm. Understanding why transformers scale so well while previous architectures hit performance ceilings is essential for understanding the trajectory of the field.

Three properties of transformers combine to enable scaling. First, parallelizability: unlike RNNs, all positions in a transformer layer are computed simultaneously, meaning that doubling hardware approximately halves training time. This allows transformers to be trained on vastly more data than recurrent models in the same wall-clock time. Second, the attention mechanism provides a flexible, learnable routing of information that does not impose a fixed computational path. As models grow larger, they can learn more nuanced routing patterns. Third, the residual connections and layer normalization in transformers create well-behaved gradient flows that remain stable even in models with hundreds of layers.

The scaling laws discovered by Kaplan et al. (2020) show that loss decreases as a power law in model size, dataset size, and compute budget, with no sign of saturation across many orders of magnitude. This predictability allows researchers to forecast the performance of models they have not yet trained, making large-scale training a calculated investment rather than a gamble.

### Why naive approaches fail

Simply making an RNN larger does not yield predictable improvements. The sequential bottleneck means that training time scales linearly with sequence length regardless of available hardware. A 10x larger RNN takes 10x longer to process each sequence, whereas a 10x larger transformer can distribute the additional computation across parallel hardware.

Increasing the depth of feed-forward networks without residual connections leads to optimization difficulties: gradients either vanish or explode, and the loss landscape becomes highly non-convex. Transformers avoid this through layer normalization and residual connections that create smooth loss landscapes amenable to gradient-based optimization even at extreme depth.

### Mental models

- **Scaling as a recipe**: Transformers are like a recipe that scales linearly. Want to serve 10x more people? Use 10x ingredients and 10x ovens. RNNs are like a recipe where one step depends on the previous, so adding ovens does not help.
- **Highway system**: Residual connections are highways that allow gradient traffic to flow freely. Without them, signals get stuck in local streets (vanishing) or cause accidents (exploding).
- **Predictable investment**: Scaling laws turn AI research from prospecting (uncertain payoff) into engineering (predictable return). You can calculate the cost of achieving a target performance level.

### Visual explanations

```
  Scaling laws (schematic):

  Loss
   |
   |\
   | \
   |  \
   |   \
   |    \__
   |       \___
   |           \______
   |                  \_________
   +-------------------------------------> Parameters
   10^6  10^7  10^8  10^9  10^10

  Power law: L(N) = a * N^(-alpha) + L_irreducible

  Parallelization comparison:

  RNN (seq_len = 1000, 4 GPUs):
  GPU 0: [ seq 0: step 1->2->3->...->1000 ]
  GPU 1: [ seq 1: step 1->2->3->...->1000 ]
  GPU 2: [ seq 2: step 1->2->3->...->1000 ]
  GPU 3: [ seq 3: step 1->2->3->...->1000 ]
  Total time: ~1000 steps (sequence-parallel only)

  Transformer (seq_len = 1000, 4 GPUs, model-parallel):
  GPU 0: [ layer 0-5:  all 1000 positions simultaneously ]
  GPU 1: [ layer 6-11: all 1000 positions simultaneously ]
  GPU 2: [ layer 12-17: all 1000 positions simultaneously ]
  GPU 3: [ layer 18-23: all 1000 positions simultaneously ]
  Total time: ~24 steps (pipeline-parallel across layers)

  Residual connections:

  Without residuals:        With residuals:
  Layer 10 ──> Layer 9      Layer 10 ──+──> Layer 9
            ──> Layer 8                ├──> Layer 8  (skip)
            ──> ...                    ├──> ...
            ──> Layer 1                └──> Layer 1  (direct path!)

  Gradient must traverse    Gradient has a highway
  ALL intermediate layers   directly to early layers
```

---

## Hands-on Exploration

1. Simulate scaling laws by training simple models of different sizes on the same task.
2. Demonstrate how parallelization benefits scale with model size.
3. Show the effect of residual connections on gradient flow through many layers.
4. Compute throughput comparisons between sequential and parallel architectures.

---

## Live Code

```rust
fn main() {
    println!("=== Why Transformers Scale ===\n");

    // 1. Simulate scaling laws
    println!("--- Simulated Scaling Laws ---\n");

    // Simple function approximation: more parameters = better fit
    // Target: fit y = sin(x) * cos(2x) on [0, 2*PI]
    let n_samples = 50;
    let target: Vec<(f64, f64)> = (0..n_samples)
        .map(|i| {
            let x = i as f64 / n_samples as f64 * 2.0 * std::f64::consts::PI;
            let y = x.sin() * (2.0 * x).cos();
            (x, y)
        })
        .collect();

    // "Train" polynomial models of increasing order (proxy for model size)
    let model_sizes = [2, 4, 8, 16, 32];
    println!("{:>12} {:>12} {:>15}", "Parameters", "MSE Loss", "Log10(Loss)");
    println!("{}", "-".repeat(42));

    let mut losses = Vec::new();
    for &size in &model_sizes {
        let mse = fit_polynomial(&target, size);
        losses.push((size as f64, mse));
        println!(
            "{:>12} {:>12.6} {:>15.3}",
            size,
            mse,
            mse.log10()
        );
    }

    // Check power-law relationship
    println!("\n  If scaling is power-law: log(Loss) ~ -alpha * log(Params) + const");
    if losses.len() >= 2 {
        let (log_n1, log_l1) = (losses[0].0.ln(), losses[0].1.ln());
        let (log_n2, log_l2) = (losses[losses.len()-1].0.ln(), losses[losses.len()-1].1.ln());
        let alpha = -(log_l2 - log_l1) / (log_n2 - log_n1);
        println!("  Estimated alpha (scaling exponent): {:.3}", alpha);
        println!("  Loss halves every {:.1}x increase in parameters", 2.0_f64.powf(1.0/alpha));
    }

    // 2. Parallelization advantage
    println!("\n--- Parallelization Advantage ---\n");

    let seq_len = 512;
    let n_layers = 12;
    let gpu_counts = [1, 4, 16, 64, 256];

    println!("{:>6} {:>15} {:>15} {:>10}", "GPUs", "RNN time", "Transformer", "Speedup");
    println!("{}", "-".repeat(50));

    for &gpus in &gpu_counts {
        // RNN: sequence is sequential, can only batch-parallelize
        // Each GPU processes different sequences, but each sequence takes seq_len steps
        let rnn_time = seq_len as f64; // Sequential, regardless of GPUs (per sequence)

        // Transformer: layers are sequential, but within a layer all positions parallel
        // With model parallelism, can split across GPUs
        let positions_per_gpu = (seq_len as f64 / gpus as f64).max(1.0);
        let transformer_time = n_layers as f64 * positions_per_gpu;

        let speedup = rnn_time / transformer_time;
        println!(
            "{:>6} {:>15.0} {:>15.0} {:>9.1}x",
            gpus, rnn_time, transformer_time, speedup
        );
    }

    // 3. Residual connection effect on gradient flow
    println!("\n--- Residual Connections & Gradient Flow ---\n");

    let n_layers_test = 50;
    let layer_dim = 8;

    // Without residuals: gradient passes through each layer's Jacobian
    println!("Gradient magnitude after N layers:");
    println!("{:>8} {:>18} {:>18}", "Layers", "Without Residual", "With Residual");
    println!("{}", "-".repeat(48));

    let mut grad_no_res = 1.0_f64;
    let mut grad_with_res = 1.0_f64;

    // Simulate with random layer Jacobians
    let jacobian_scale = 0.8; // Each layer slightly shrinks the gradient

    for layer in 1..=n_layers_test {
        // Without residual: gradient *= jacobian
        grad_no_res *= jacobian_scale;

        // With residual: gradient = gradient_through_layer + gradient_skip
        // = jacobian_scale * grad + grad
        // After normalization: effectively grad *= sqrt(jacobian_scale^2 + 1)
        // Simplified model: residual provides additive path
        let layer_grad = jacobian_scale;
        // The residual connection means total gradient has both paths
        // In practice, the skip connection provides a floor
        grad_with_res = (grad_with_res * layer_grad).max(0.5_f64.powi(layer as i32 / 10));

        if layer % 10 == 0 || layer == 1 {
            println!(
                "{:>8} {:>18.8} {:>18.8}",
                layer, grad_no_res, grad_with_res
            );
        }
    }

    // 4. Compute efficiency at different scales
    println!("\n--- Compute Efficiency at Scale ---\n");

    // Transformer FLOPs: ~6 * N * D (forward + backward, per token)
    // Where N = total parameters, D = dataset tokens
    let params_list = [1e6, 1e7, 1e8, 1e9, 1e10];
    let tokens = 1e10; // 10B tokens

    println!("{:>12} {:>15} {:>12} {:>12}",
        "Params", "FLOPs", "GPU-hours*", "Est. Loss");
    println!("{}", "-".repeat(55));

    for &n in &params_list {
        let flops = 6.0 * n * tokens;
        let gpu_hours = flops / (3e14 * 3600.0); // A100 ~300 TFLOPS

        // Scaling law: L = 1.5 * N^(-0.076) (simplified Chinchilla-like)
        let loss = 1.5 * n.powf(-0.076);

        println!(
            "{:>12.0} {:>12.2e} {:>12.0} {:>12.4}",
            n, flops, gpu_hours, loss
        );
    }
    println!("  * Approximate A100 GPU-hours (theoretical peak)");

    // 5. Why other architectures don't scale as well
    println!("\n--- Why Other Architectures Plateau ---\n");

    let factors = [
        ("Parallelization",
         "RNN: O(seq_len) serial | Transformer: O(1) parallel per layer"),
        ("Gradient flow",
         "RNN: vanishes over long sequences | Transformer: residual highways"),
        ("Information routing",
         "RNN: fixed sequential path | Transformer: dynamic attention routing"),
        ("Hardware utilization",
         "RNN: ~5% GPU util | Transformer: ~50%+ GPU util (matrix ops)"),
        ("Training data efficiency",
         "RNN: slow per sample | Transformer: fast, can see more data"),
    ];

    for (factor, comparison) in &factors {
        println!("  {}", factor);
        println!("    {}\n", comparison);
    }
}

fn fit_polynomial(data: &[(f64, f64)], degree: usize) -> f64 {
    // Simple polynomial fit using normal equations approximation
    // Build feature matrix
    let n = data.len();
    let mut best_mse = f64::MAX;

    // Use gradient descent to fit polynomial coefficients
    let mut coeffs = vec![0.0_f64; degree + 1];
    let lr = 0.001 / (degree as f64 + 1.0);

    for _epoch in 0..500 {
        let mut grad = vec![0.0; degree + 1];
        let mut mse = 0.0;

        for (x, y) in data {
            // Evaluate polynomial
            let mut pred = 0.0;
            let mut x_pow = 1.0;
            for c in &coeffs {
                pred += c * x_pow;
                x_pow *= x;
            }
            let err = pred - y;
            mse += err * err;

            // Compute gradient
            let mut x_pow = 1.0;
            for (i, g) in grad.iter_mut().enumerate() {
                *g += 2.0 * err * x_pow / n as f64;
                x_pow *= x;
            }
        }
        mse /= n as f64;

        // Gradient descent step with clipping
        for (c, g) in coeffs.iter_mut().zip(grad.iter()) {
            let clipped = g.max(-10.0).min(10.0);
            *c -= lr * clipped;
        }

        if mse < best_mse {
            best_mse = mse;
        }
    }

    best_mse
}
```

---

## Key Takeaways

- Transformers scale predictably with compute, data, and parameters following power-law relationships, making large-scale training a calculated engineering decision rather than a gamble.
- Full parallelization across sequence positions enables transformers to utilize modern GPU hardware efficiently, training on vastly more data in the same time compared to sequential architectures.
- Residual connections and layer normalization maintain healthy gradient flow even through hundreds of layers, preventing the optimization difficulties that plague deep networks without these mechanisms.
- The combination of parallelizability, dynamic information routing via attention, and stable gradient flow creates a virtuous cycle where adding more resources consistently yields better performance.
