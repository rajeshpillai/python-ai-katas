# Gaussian Mixture Models

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.4

---

## Concept & Intuition

### What problem are we solving?

**Gaussian Mixture Models (GMMs)** assume the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance. The goal is to discover these hidden clusters: which Gaussian generated each data point, and what are the parameters (means, variances, mixing weights) of each component? GMMs are a probabilistic generalization of K-means clustering that provides soft assignments (probabilities of belonging to each cluster) rather than hard assignments.

The model is: P(x) = Sum over k of [pi_k * N(x | mu_k, sigma_k^2)], where pi_k are the mixing weights (must sum to 1), and each N is a Gaussian with its own mean and variance. Unlike K-means, GMMs can model clusters of different sizes, shapes, and densities.

The parameters are learned via the **Expectation-Maximization (EM) algorithm**. In the E-step, we compute the probability that each data point belongs to each cluster (responsibilities). In the M-step, we update the parameters using these responsibilities as soft weights. EM alternates between these steps until convergence, guaranteed to increase the log-likelihood at each iteration.

### Why naive approaches fail

K-means assigns each point to exactly one cluster, which fails when clusters overlap significantly. A point between two clusters gets assigned to one arbitrarily, with no indication of uncertainty. GMMs provide a probability for each cluster, giving a richer and more honest representation. Additionally, K-means assumes spherical clusters of equal size, while GMMs can model elliptical clusters of varying sizes.

Direct optimization of the GMM log-likelihood is intractable because of the sum inside the logarithm. EM cleverly decomposes this into tractable steps by introducing latent variables (cluster assignments).

### Mental models

- **GMMs as soft K-means**: instead of "this point belongs to cluster 2," GMMs say "this point belongs to cluster 2 with probability 0.7 and cluster 1 with probability 0.3."
- **EM as alternating optimization**: E-step asks "given current parameters, which cluster likely generated each point?" M-step asks "given these soft assignments, what are the best parameters?"
- **Mixing weights as priors**: pi_k represents how likely a random point is to come from cluster k, before seeing the point's value.

### Visual explanations

```
Gaussian Mixture Model:
  P(x) = pi_1 * N(x|mu_1, sig_1) + pi_2 * N(x|mu_2, sig_2)

  Component 1: mu=-2, sig=1.0, pi=0.4
  Component 2: mu=+3, sig=0.5, pi=0.6

  Distribution:
    .                           ...
   . .                        .   .
  .   .                      .     .
 .     .      overlap       .       .
.       .       here       .         .
  Cluster 1    ^^^^     Cluster 2

EM Algorithm:
  E-step: Compute responsibilities
    r_ik = pi_k * N(x_i|mu_k, sig_k) / Sum_j[pi_j * N(x_i|mu_j, sig_j)]

  M-step: Update parameters
    mu_k = Sum_i[r_ik * x_i] / Sum_i[r_ik]
    sig_k^2 = Sum_i[r_ik * (x_i - mu_k)^2] / Sum_i[r_ik]
    pi_k = Sum_i[r_ik] / N
```

---

## Hands-on Exploration

1. Generate data from a known mixture of 3 Gaussians. Implement the EM algorithm to recover the parameters.
2. Track the log-likelihood at each EM iteration and verify it is monotonically increasing.
3. Compare GMM soft assignments with K-means hard assignments. Show cases where soft assignments are more informative.
4. Experiment with different numbers of components and use BIC (Bayesian Information Criterion) for model selection.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Generate data from a mixture of 3 Gaussians ---
    let true_means = [-3.0, 1.0, 5.0];
    let true_stds = [0.8, 1.2, 0.6];
    let true_weights = [0.3, 0.45, 0.25];
    let k = true_means.len();
    let n = 200;

    let mut data: Vec<f64> = Vec::new();
    let mut true_labels: Vec<usize> = Vec::new();

    for _ in 0..n {
        let u = rand_f64(&mut rng);
        let component = if u < true_weights[0] { 0 }
            else if u < true_weights[0] + true_weights[1] { 1 }
            else { 2 };
        let x = true_means[component] + true_stds[component] * rand_normal(&mut rng);
        data.push(x);
        true_labels.push(component);
    }

    println!("=== Gaussian Mixture Models (EM Algorithm) ===\n");
    println!("True parameters:");
    for c in 0..k {
        println!("  Component {}: mean={:.1}, std={:.1}, weight={:.2}",
            c, true_means[c], true_stds[c], true_weights[c]);
    }
    println!("Data: {} samples\n", n);

    // --- Gaussian PDF ---
    let gaussian_pdf = |x: f64, mean: f64, std: f64| -> f64 {
        let var = std * std;
        (1.0 / (2.0 * pi * var).sqrt()) * (-0.5 * (x - mean).powi(2) / var).exp()
    };

    // --- EM Algorithm ---
    // Initialize with K-means++ style
    let mut means = vec![-1.0, 2.0, 6.0]; // slightly off
    let mut stds = vec![1.0, 1.0, 1.0];
    let mut weights = vec![1.0 / k as f64; k];
    let mut responsibilities = vec![vec![0.0; k]; n];

    let max_iter = 50;
    let mut ll_history: Vec<f64> = Vec::new();

    println!("=== EM Iterations ===\n");
    println!("{:>4} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Iter", "Log-Lik", "mu_0", "mu_1", "mu_2", "sig_0", "sig_1", "sig_2");
    println!("{}", "-".repeat(82));

    for iter in 0..max_iter {
        // --- E-step: compute responsibilities ---
        for i in 0..n {
            let mut total = 0.0;
            for c in 0..k {
                responsibilities[i][c] = weights[c] * gaussian_pdf(data[i], means[c], stds[c]);
                total += responsibilities[i][c];
            }
            if total > 1e-300 {
                for c in 0..k {
                    responsibilities[i][c] /= total;
                }
            }
        }

        // --- M-step: update parameters ---
        for c in 0..k {
            let n_k: f64 = responsibilities.iter().map(|r| r[c]).sum();
            if n_k < 1e-10 { continue; }

            // Update mean
            means[c] = responsibilities.iter().enumerate()
                .map(|(i, r)| r[c] * data[i]).sum::<f64>() / n_k;

            // Update variance
            let var: f64 = responsibilities.iter().enumerate()
                .map(|(i, r)| r[c] * (data[i] - means[c]).powi(2)).sum::<f64>() / n_k;
            stds[c] = var.sqrt().max(0.01);

            // Update weight
            weights[c] = n_k / n as f64;
        }

        // Compute log-likelihood
        let ll: f64 = data.iter().map(|&x| {
            let p: f64 = (0..k).map(|c| weights[c] * gaussian_pdf(x, means[c], stds[c])).sum();
            p.max(1e-300).ln()
        }).sum();
        ll_history.push(ll);

        if iter < 5 || iter == 9 || iter == 19 || iter == max_iter - 1 {
            println!("{:>4} {:>12.2} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                iter + 1, ll, means[0], means[1], means[2], stds[0], stds[1], stds[2]);
        }
    }

    // --- Sort components by mean for comparison ---
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap());

    println!("\n=== Learned vs True Parameters ===\n");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Comp", "True mu", "Est mu", "True sig", "Est sig", "True w", "Est w");
    println!("{}", "-".repeat(67));
    for (i, &c) in order.iter().enumerate() {
        println!("{:>5} {:>10.2} {:>10.3} {:>10.2} {:>10.3} {:>10.2} {:>10.3}",
            i, true_means[i], means[c], true_stds[i], stds[c],
            true_weights[i], weights[c]);
    }

    // --- Soft vs Hard Assignments ---
    println!("\n=== Soft Assignments (GMM) vs Hard Assignments (K-means) ===\n");

    // K-means
    let mut km_means = vec![-1.0, 2.0, 6.0];
    for _ in 0..50 {
        let mut sums = vec![0.0; k];
        let mut counts = vec![0_usize; k];
        for &x in &data {
            let closest = (0..k).min_by(|&a, &b| {
                (x - km_means[a]).abs().partial_cmp(&(x - km_means[b]).abs()).unwrap()
            }).unwrap();
            sums[closest] += x;
            counts[closest] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 { km_means[c] = sums[c] / counts[c] as f64; }
        }
    }

    // Show some points in the overlap region
    println!("Points near cluster boundaries (soft assignments more informative):\n");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
        "x", "P(c=0)", "P(c=1)", "P(c=2)", "K-means");
    println!("{}", "-".repeat(52));

    let mut shown = 0;
    for i in 0..n {
        // Find points with ambiguous assignments
        let max_r = responsibilities[i].iter().cloned().fold(0.0_f64, f64::max);
        if max_r < 0.85 && shown < 10 {
            let km_cluster = (0..k).min_by(|&a, &b| {
                (data[i] - km_means[a]).abs().partial_cmp(&(data[i] - km_means[b]).abs()).unwrap()
            }).unwrap();
            println!("{:>8.2} {:>10.3} {:>10.3} {:>10.3} {:>10}",
                data[i],
                responsibilities[i][order[0]],
                responsibilities[i][order[1]],
                responsibilities[i][order[2]],
                format!("c={}", km_cluster));
            shown += 1;
        }
    }
    println!("\nGMM provides probability of each cluster; K-means gives only hard assignment.");

    // --- Model Selection with BIC ---
    println!("\n=== Model Selection (BIC) ===\n");
    println!("{:>4} {:>12} {:>12} {:>10}", "K", "Log-Lik", "BIC", "Chosen?");
    println!("{}", "-".repeat(42));

    let mut best_bic = f64::MAX;
    let mut best_k = 1;

    for n_comp in 1..=5 {
        // Run EM with n_comp components
        let mut m: Vec<f64> = (0..n_comp).map(|i| {
            data.iter().sum::<f64>() / n as f64 + (i as f64 - n_comp as f64 / 2.0) * 2.0
        }).collect();
        let mut s = vec![2.0; n_comp];
        let mut w = vec![1.0 / n_comp as f64; n_comp];
        let mut resp = vec![vec![0.0; n_comp]; n];

        for _ in 0..30 {
            // E-step
            for i in 0..n {
                let mut total = 0.0;
                for c in 0..n_comp {
                    resp[i][c] = w[c] * gaussian_pdf(data[i], m[c], s[c]);
                    total += resp[i][c];
                }
                if total > 1e-300 {
                    for c in 0..n_comp { resp[i][c] /= total; }
                }
            }
            // M-step
            for c in 0..n_comp {
                let n_k: f64 = resp.iter().map(|r| r[c]).sum();
                if n_k < 1e-10 { continue; }
                m[c] = resp.iter().enumerate().map(|(i, r)| r[c] * data[i]).sum::<f64>() / n_k;
                let var: f64 = resp.iter().enumerate()
                    .map(|(i, r)| r[c] * (data[i] - m[c]).powi(2)).sum::<f64>() / n_k;
                s[c] = var.sqrt().max(0.01);
                w[c] = n_k / n as f64;
            }
        }

        let ll: f64 = data.iter().map(|&x| {
            let p: f64 = (0..n_comp).map(|c| w[c] * gaussian_pdf(x, m[c], s[c])).sum();
            p.max(1e-300).ln()
        }).sum();

        // BIC = -2 * LL + k * ln(n), where k = number of free parameters
        let n_params = n_comp * 3 - 1; // means + variances + (weights - 1)
        let bic = -2.0 * ll + n_params as f64 * (n as f64).ln();

        let chosen = if bic < best_bic { best_bic = bic; best_k = n_comp; "*" } else { "" };
        println!("{:>4} {:>12.2} {:>12.2} {:>10}", n_comp, ll, bic, chosen);
    }
    println!("\nBIC selects K={} (lowest BIC, penalizing complexity).", best_k);

    // --- Log-likelihood convergence ---
    println!("\n=== Log-Likelihood Convergence ===\n");
    let mut monotonic = true;
    for i in 1..ll_history.len() {
        if ll_history[i] < ll_history[i - 1] - 1e-10 { monotonic = false; break; }
    }
    println!("Log-likelihood monotonically increasing: {}", monotonic);
    println!("Initial LL: {:.2}", ll_history[0]);
    println!("Final LL:   {:.2}", ll_history[ll_history.len() - 1]);

    let mean_err: f64 = order.iter().enumerate()
        .map(|(i, &c)| (means[c] - true_means[i]).abs()).sum::<f64>() / k as f64;

    println!();
    println!("kata_metric(\"mean_estimation_error\", {:.4})", mean_err);
    println!("kata_metric(\"best_k\", {})", best_k);
    println!("kata_metric(\"final_log_likelihood\", {:.2})", ll_history[ll_history.len() - 1]);
}
```

---

## Key Takeaways

- **GMMs model data as a mixture of Gaussians**, providing soft cluster assignments (probabilities) rather than the hard assignments of K-means.
- **The EM algorithm alternates between E-step (compute responsibilities) and M-step (update parameters)**, guaranteed to increase the log-likelihood at each step.
- **Soft assignments are more informative than hard assignments**, especially for points near cluster boundaries where uncertainty is high.
- **BIC (Bayesian Information Criterion) helps choose the number of components** by penalizing model complexity, preventing overfitting.
