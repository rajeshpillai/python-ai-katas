# Bayesian Optimization

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.5

---

## Concept & Intuition

### What problem are we solving?

**Bayesian Optimization** is a strategy for optimizing expensive-to-evaluate functions. In machine learning, the "expensive function" is typically model performance as a function of hyperparameters. Training a model with a given set of hyperparameters might take hours, so we cannot afford to try thousands of combinations (as grid search does). Bayesian optimization builds a probabilistic model (surrogate) of the objective function and uses it to intelligently choose the next hyperparameters to evaluate.

The core idea: maintain a **surrogate model** (typically a Gaussian Process) that predicts the objective function's value and uncertainty at any point. Use an **acquisition function** to decide where to evaluate next, balancing exploitation (points where the surrogate predicts good performance) with exploration (points where the surrogate is uncertain). After each evaluation, update the surrogate and repeat.

Common acquisition functions include: **Expected Improvement (EI)** -- how much improvement over the current best do we expect? **Upper Confidence Bound (UCB)** -- optimistic estimate considering uncertainty. **Probability of Improvement (PI)** -- likelihood of beating the current best.

### Why naive approaches fail

Grid search evaluates every combination, which is exponentially expensive as the number of hyperparameters grows. Random search is better but does not learn from previous evaluations. Bayesian optimization uses all past evaluations to make informed decisions about what to try next, often finding good solutions in 10-50 evaluations where grid search needs thousands.

### Mental models

- **Surrogate as a map**: after a few evaluations, you have a rough "map" of the hyperparameter landscape. The map tells you where performance is likely high and where you have not explored.
- **Acquisition as a treasure hunter**: the acquisition function decides whether to dig where the map says treasure is likely (exploit) or explore unknown territory (explore).
- **EI as expected gain**: "if I evaluate here, how much better than my current best do I expect to get, on average?"

### Visual explanations

```
Bayesian Optimization Loop:
  1. Evaluate f(x) at a few initial points
  2. Fit surrogate model to observations
  3. Maximize acquisition function to find next x
  4. Evaluate f(x_next)
  5. Update surrogate with new observation
  6. Repeat 3-5 until budget exhausted

Surrogate Model:
  x-axis: hyperparameter value
  y-axis: predicted performance

      predicted mean
  1.0 |    .---.
      |   /     \   uncertainty band
      |  / ..... \  (wide where few samples)
  0.5 | / .     . \
      |/ .       . \
  0.0 |.           .\
      +--*----*----*---> x
        observed points

  Acquisition (EI):
      high where: predicted mean is high AND/OR uncertainty is high
```

---

## Hands-on Exploration

1. Implement a simple surrogate model using a kernel-based approach. Fit it to a few observed points and show predictions with uncertainty.
2. Implement Expected Improvement and Upper Confidence Bound acquisition functions.
3. Optimize a synthetic 1D function using Bayesian optimization. Compare the number of evaluations needed vs random search.
4. Apply Bayesian optimization to tune hyperparameters (learning rate, regularization) of a simple model.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };

    // --- Objective function (expensive to evaluate in practice) ---
    // Synthetic: f(x) = sin(3x) * (1 - x)^2 + 0.5*x, x in [0, 2]
    let objective = |x: f64| -> f64 {
        (3.0 * x).sin() * (1.0 - x).powi(2) + 0.5 * x
    };

    // Find true optimum by dense sampling
    let n_dense = 1000;
    let mut true_best_x = 0.0;
    let mut true_best_y = f64::NEG_INFINITY;
    for i in 0..=n_dense {
        let x = 2.0 * i as f64 / n_dense as f64;
        let y = objective(x);
        if y > true_best_y { true_best_y = y; true_best_x = x; }
    }

    println!("=== Bayesian Optimization ===\n");
    println!("Objective: f(x) = sin(3x) * (1-x)^2 + 0.5*x, x in [0, 2]");
    println!("True optimum: x={:.4}, f(x)={:.4}\n", true_best_x, true_best_y);

    // --- Simple Kernel-Based Surrogate ---
    // Using RBF kernel regression as surrogate
    let rbf_kernel = |x1: f64, x2: f64, length_scale: f64| -> f64 {
        (-(x1 - x2).powi(2) / (2.0 * length_scale * length_scale)).exp()
    };

    // Predict mean and variance at a point using kernel regression
    let predict = |x: f64, obs_x: &[f64], obs_y: &[f64], ls: f64, noise: f64| -> (f64, f64) {
        let n = obs_x.len();
        if n == 0 { return (0.0, 1.0); }

        // Compute kernel matrix K + noise*I
        let mut k_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i][j] = rbf_kernel(obs_x[i], obs_x[j], ls);
                if i == j { k_mat[i][j] += noise; }
            }
        }

        // Compute k_star: kernel between x and observed points
        let k_star: Vec<f64> = obs_x.iter().map(|&xi| rbf_kernel(x, xi, ls)).collect();

        // Solve K * alpha = y using Cholesky-like approach (simplified: use iterative)
        // Simple approach: alpha = K^-1 * y via Gauss-Seidel
        let mut alpha = vec![0.0; n];
        for _ in 0..100 {
            for i in 0..n {
                let mut s = obs_y[i];
                for j in 0..n {
                    if j != i { s -= k_mat[i][j] * alpha[j]; }
                }
                alpha[i] = s / k_mat[i][i];
            }
        }

        // Mean prediction
        let mean: f64 = k_star.iter().zip(&alpha).map(|(k, a)| k * a).sum();

        // Variance prediction (simplified)
        // v = K^-1 * k_star
        let mut v = vec![0.0; n];
        for _ in 0..100 {
            for i in 0..n {
                let mut s = k_star[i];
                for j in 0..n {
                    if j != i { s -= k_mat[i][j] * v[j]; }
                }
                v[i] = s / k_mat[i][i];
            }
        }
        let var = 1.0 - k_star.iter().zip(&v).map(|(k, vi)| k * vi).sum::<f64>();
        let var = var.max(0.001);

        (mean, var)
    };

    // --- Acquisition functions ---
    // Standard normal CDF approximation
    let norm_cdf = |x: f64| -> f64 {
        0.5 * (1.0 + (x / 2.0_f64.sqrt()).tanh() * 1.0_f64.min(1.0))
        // Better approximation:
    };
    let norm_cdf = |x: f64| -> f64 {
        if x < -6.0 { return 0.0; }
        if x > 6.0 { return 1.0; }
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = (1.0 / (2.0 * pi).sqrt()) * (-x * x / 2.0).exp();
        let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))));
        if x >= 0.0 { 1.0 - p } else { p }
    };
    let norm_pdf = |x: f64| -> f64 {
        (1.0 / (2.0 * pi).sqrt()) * (-x * x / 2.0).exp()
    };

    // Expected Improvement
    let expected_improvement = |mean: f64, var: f64, best_y: f64| -> f64 {
        let std = var.sqrt();
        if std < 1e-10 { return 0.0; }
        let z = (mean - best_y) / std;
        (mean - best_y) * norm_cdf(z) + std * norm_pdf(z)
    };

    // Upper Confidence Bound
    let ucb = |mean: f64, var: f64, kappa: f64| -> f64 {
        mean + kappa * var.sqrt()
    };

    // --- Bayesian Optimization Loop ---
    let budget = 15;
    let length_scale = 0.3;
    let noise = 0.001;
    let n_candidates = 200;

    // Initial observations
    let mut obs_x: Vec<f64> = vec![0.0, 1.0, 2.0];
    let mut obs_y: Vec<f64> = obs_x.iter().map(|&x| objective(x)).collect();

    println!("=== Bayesian Optimization Progress ===\n");
    println!("{:>4} {:>8} {:>10} {:>10} {:>10} {:>12}",
        "Iter", "x_new", "f(x_new)", "Best x", "Best f(x)", "Acq(EI)");
    println!("{}", "-".repeat(58));

    // Print initial points
    let mut best_idx = 0;
    for i in 1..obs_y.len() {
        if obs_y[i] > obs_y[best_idx] { best_idx = i; }
    }
    println!("{:>4} {:>8} {:>10} {:>10} {:>10} {:>12}",
        0, "-", "-",
        format!("{:.4}", obs_x[best_idx]),
        format!("{:.4}", obs_y[best_idx]), "init");

    for iter in 0..budget {
        let best_y = obs_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Find point that maximizes acquisition function (EI)
        let mut best_acq = f64::NEG_INFINITY;
        let mut best_x_new = 0.0;

        for i in 0..n_candidates {
            let x = 2.0 * i as f64 / (n_candidates - 1) as f64;
            let (mean, var) = predict(x, &obs_x, &obs_y, length_scale, noise);
            let ei = expected_improvement(mean, var, best_y);
            if ei > best_acq { best_acq = ei; best_x_new = x; }
        }

        // Evaluate objective at chosen point
        let y_new = objective(best_x_new);
        obs_x.push(best_x_new);
        obs_y.push(y_new);

        let best_y_now = obs_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let best_x_now = obs_x[obs_y.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0];

        println!("{:>4} {:>8.4} {:>10.4} {:>10.4} {:>10.4} {:>12.6}",
            iter + 1, best_x_new, y_new, best_x_now, best_y_now, best_acq);
    }

    let bo_best_y = obs_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bo_best_x = obs_x[obs_y.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0];

    // --- Compare with Random Search ---
    println!("\n=== Random Search Comparison ===\n");

    let n_random_trials = 10;
    let mut random_best_avg = 0.0;

    for trial in 0..n_random_trials {
        let mut rand_best = f64::NEG_INFINITY;
        // Same total budget: 3 initial + 15 = 18 evaluations
        for _ in 0..18 {
            let x = rand_f64(&mut rng) * 2.0;
            let y = objective(x);
            if y > rand_best { rand_best = y; }
        }
        random_best_avg += rand_best;
    }
    random_best_avg /= n_random_trials as f64;

    println!("{:<25} {:>10} {:>10}", "Method", "Best f(x)", "Evals");
    println!("{}", "-".repeat(47));
    println!("{:<25} {:>10.4} {:>10}", "Bayesian Optimization", bo_best_y, obs_x.len());
    println!("{:<25} {:>10.4} {:>10}", "Random Search (avg)", random_best_avg, 18);
    println!("{:<25} {:>10.4} {:>10}", "True Optimum", true_best_y, n_dense);

    // --- Surrogate quality ---
    println!("\n=== Surrogate Model Quality ===\n");
    println!("{:>6} {:>10} {:>10} {:>10} {:>10}",
        "x", "True f(x)", "Predicted", "Std", "Error");
    println!("{}", "-".repeat(48));

    let mut total_error = 0.0;
    let n_test = 10;
    for i in 0..n_test {
        let x = 2.0 * i as f64 / (n_test - 1) as f64;
        let true_y = objective(x);
        let (pred_mean, pred_var) = predict(x, &obs_x, &obs_y, length_scale, noise);
        let pred_std = pred_var.sqrt();
        let error = (true_y - pred_mean).abs();
        total_error += error;
        println!("{:>6.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            x, true_y, pred_mean, pred_std, error);
    }
    println!("Mean absolute error: {:.4}", total_error / n_test as f64);

    // --- Hyperparameter tuning example ---
    println!("\n=== Hyperparameter Tuning Example ===\n");
    println!("Task: Find optimal learning rate and regularization for linear regression\n");

    // Simulate model performance as a function of 2 hyperparameters
    // hp1 = learning rate (0.001 to 1.0), hp2 = regularization (0.001 to 10.0)
    let model_score = |lr: f64, reg: f64| -> f64 {
        // Simulated validation accuracy (peaks around lr=0.1, reg=0.5)
        let lr_score = -(lr.ln() - 0.1_f64.ln()).powi(2) / 2.0;
        let reg_score = -(reg.ln() - 0.5_f64.ln()).powi(2) / 4.0;
        0.9 + 0.08 * (lr_score + reg_score).exp() / 1.0_f64.exp()
    };

    // Simple 1D BO on each parameter alternately
    let mut best_lr = 0.1;
    let mut best_reg = 1.0;
    let mut best_score = model_score(best_lr, best_reg);

    println!("{:>4} {:>10} {:>10} {:>10}", "Iter", "LR", "Reg", "Score");
    println!("{}", "-".repeat(38));

    for iter in 0..10 {
        // Try random perturbations (simplified BO)
        let lr = (best_lr * (1.0 + 0.5 * (rand_f64(&mut rng) * 2.0 - 1.0))).max(0.001).min(1.0);
        let reg = (best_reg * (1.0 + 0.5 * (rand_f64(&mut rng) * 2.0 - 1.0))).max(0.001).min(10.0);
        let score = model_score(lr, reg);
        if score > best_score {
            best_score = score;
            best_lr = lr;
            best_reg = reg;
        }
        println!("{:>4} {:>10.4} {:>10.4} {:>10.4}", iter + 1, best_lr, best_reg, best_score);
    }
    println!("\nBest hyperparameters: lr={:.4}, reg={:.4}, score={:.4}", best_lr, best_reg, best_score);

    let gap = (true_best_y - bo_best_y).abs();
    println!();
    println!("kata_metric(\"bo_best_y\", {:.4})", bo_best_y);
    println!("kata_metric(\"random_best_y\", {:.4})", random_best_avg);
    println!("kata_metric(\"optimality_gap\", {:.4})", gap);
}
```

---

## Key Takeaways

- **Bayesian optimization efficiently optimizes expensive functions** by building a surrogate model and using acquisition functions to decide where to evaluate next.
- **The surrogate model provides both predictions and uncertainty**, enabling intelligent exploration of the search space.
- **Expected Improvement (EI) balances exploitation and exploration**: it favors points where the predicted value is high or the uncertainty is large.
- **Bayesian optimization typically finds good solutions in far fewer evaluations than random or grid search**, making it ideal for hyperparameter tuning where each evaluation is costly.
