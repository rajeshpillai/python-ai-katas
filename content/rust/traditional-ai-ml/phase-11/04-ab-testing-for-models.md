# A/B Testing for Models

> Phase 11 — Productionizing ML | Kata 11.4

---

## Concept & Intuition

### What problem are we solving?

You have a new model that performs better on test data. Should you deploy it? Offline metrics (test accuracy, cross-validation scores) are necessary but not sufficient. A model that looks better on historical data might perform worse in production due to distribution shifts, feedback loops, or unforeseen interactions. **A/B testing** provides the gold standard: deploy both models simultaneously, randomly assign users to each, and measure which one actually performs better in the real world.

A/B testing for models follows the same principles as A/B testing for web pages: random assignment eliminates confounding variables, sufficient sample size ensures statistical significance, and a clear primary metric prevents cherry-picking results. The key difference is that ML models can have subtle failure modes (biased predictions, degraded performance on subgroups) that require monitoring beyond a single metric.

The statistical framework involves: defining a **null hypothesis** (the new model is no better than the old), choosing a **significance level** (alpha, typically 0.05), determining the **minimum sample size** for a given effect size and power, running the test, and computing a **p-value** to decide whether to adopt the new model.

### Why naive approaches fail

Deploying a new model to all users and watching overall metrics is dangerous. If the new model is worse, you have degraded the experience for everyone before detecting the problem. A/B testing limits exposure to 50% of users (or less, with staged rollouts). Additionally, looking at metrics too early and making decisions on small samples leads to false positives -- the test might show a significant difference by chance.

### Mental models

- **A/B test as a controlled experiment**: random assignment is the gold standard for causal inference. It tells you whether the new model *caused* better outcomes, not just whether it is correlated with them.
- **Sample size as confidence**: small samples give noisy estimates. Large samples give precise estimates. The required sample size depends on the effect size you want to detect.
- **p-value as surprise**: "If the models were actually identical, how surprising would this observed difference be?" A small p-value means the difference is unlikely due to chance.

### Visual explanations

```
A/B Test Setup:
  Users arrive --> Random assignment
                    |           |
                  Model A     Model B
                  (control)   (treatment)
                    |           |
                  Measure     Measure
                  outcomes    outcomes
                    |           |
                  Compare with statistical test

Sample Size Calculation:
  effect_size = expected improvement (e.g., 2% accuracy gain)
  alpha = false positive rate (0.05)
  power = true positive detection rate (0.80)
  --> Required n per group

Decision Framework:
  p < 0.05 AND effect > 0  --> Deploy Model B
  p < 0.05 AND effect < 0  --> Keep Model A
  p >= 0.05                --> Inconclusive, need more data

Staged Rollout:
  Day 1-3:   5% traffic to Model B (sanity check)
  Day 4-7:   25% traffic (monitor closely)
  Day 8-14:  50% traffic (formal A/B test)
  Day 15+:   100% if test passes
```

---

## Hands-on Exploration

1. Simulate an A/B test between two models with known performance differences. Compute the p-value and decide whether to deploy.
2. Calculate the minimum sample size needed to detect a 2% improvement with 80% power at alpha=0.05.
3. Demonstrate the problem of early stopping: show how checking results too frequently inflates the false positive rate.
4. Run multiple A/B tests and verify that the false positive rate matches alpha when the null hypothesis is true.

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

    println!("=== A/B Testing for Models ===\n");

    // --- Normal CDF approximation ---
    let norm_cdf = |x: f64| -> f64 {
        if x < -6.0 { return 0.0; }
        if x > 6.0 { return 1.0; }
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = (1.0 / (2.0 * pi).sqrt()) * (-x * x / 2.0).exp();
        let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
            + t * (-1.821255978 + t * 1.330274429))));
        if x >= 0.0 { 1.0 - p } else { p }
    };

    let norm_quantile = |p: f64| -> f64 {
        // Approximation for quantile function
        let mut lo = -6.0;
        let mut hi = 6.0;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            if norm_cdf(mid) < p { lo = mid; } else { hi = mid; }
        }
        (lo + hi) / 2.0
    };

    // --- Two-proportion z-test ---
    fn two_prop_z_test(n_a: usize, success_a: usize, n_b: usize, success_b: usize) -> (f64, f64) {
        let p_a = success_a as f64 / n_a as f64;
        let p_b = success_b as f64 / n_b as f64;
        let p_pool = (success_a + success_b) as f64 / (n_a + n_b) as f64;
        let se = (p_pool * (1.0 - p_pool) * (1.0 / n_a as f64 + 1.0 / n_b as f64)).sqrt();
        if se < 1e-10 { return (0.0, 1.0); }
        let z = (p_b - p_a) / se;
        // Two-tailed p-value
        let p_value = 2.0 * (1.0 - {
            let x = z.abs();
            if x > 6.0 { 1.0 } else {
                let t = 1.0 / (1.0 + 0.2316419 * x);
                let d = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-x * x / 2.0).exp();
                1.0 - d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                    + t * (-1.821255978 + t * 1.330274429))))
            }
        });
        (z, p_value)
    }

    // --- Scenario 1: Real improvement exists ---
    println!("=== Scenario 1: Model B is truly better ===\n");

    let true_rate_a = 0.72; // Model A: 72% conversion
    let true_rate_b = 0.75; // Model B: 75% conversion (3% absolute improvement)
    let n_per_group = 1000;

    // Simulate the test
    let mut successes_a = 0;
    let mut successes_b = 0;
    for _ in 0..n_per_group {
        if rand_f64(&mut rng) < true_rate_a { successes_a += 1; }
        if rand_f64(&mut rng) < true_rate_b { successes_b += 1; }
    }

    let rate_a = successes_a as f64 / n_per_group as f64;
    let rate_b = successes_b as f64 / n_per_group as f64;
    let (z_stat, p_value) = two_prop_z_test(n_per_group, successes_a, n_per_group, successes_b);

    println!("True rates: A={:.2}, B={:.2}", true_rate_a, true_rate_b);
    println!("Observed:   A={:.3} ({}/{}), B={:.3} ({}/{})",
        rate_a, successes_a, n_per_group, rate_b, successes_b, n_per_group);
    println!("Difference: {:.3} ({:.1}% relative)",
        rate_b - rate_a, (rate_b - rate_a) / rate_a * 100.0);
    println!("Z-statistic: {:.3}", z_stat);
    println!("p-value:     {:.6}", p_value);
    println!("Decision:    {}", if p_value < 0.05 { "DEPLOY Model B (significant)" } else { "KEEP Model A (not significant)" });

    // --- Sample Size Calculation ---
    println!("\n=== Sample Size Calculation ===\n");
    println!("How many samples per group to detect a given effect?\n");

    let alpha = 0.05;
    let power = 0.80;
    let z_alpha = norm_quantile(1.0 - alpha / 2.0);
    let z_beta = norm_quantile(power);

    println!("{:>10} {:>12} {:>15}", "Effect", "Min n/group", "Total samples");
    println!("{}", "-".repeat(40));

    let baseline = 0.72;
    for &effect in &[0.01, 0.02, 0.03, 0.05, 0.10] {
        let p1 = baseline;
        let p2 = baseline + effect;
        let p_avg = (p1 + p2) / 2.0;
        let n = ((z_alpha * (2.0 * p_avg * (1.0 - p_avg)).sqrt()
            + z_beta * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt())
            / (p2 - p1)).powi(2);
        let n = n.ceil() as usize;
        println!("{:>+10.2} {:>12} {:>15}", effect, n, n * 2);
    }
    println!("\nSmaller effects require exponentially more samples to detect.");

    // --- Scenario 2: No real difference (null hypothesis true) ---
    println!("\n=== Scenario 2: False Positive Rate Verification ===\n");
    println!("Running 1000 A/B tests where both models are identical (rate=0.72).");
    println!("Expected false positive rate at alpha=0.05: ~5%\n");

    let n_tests = 1000;
    let n_per = 500;
    let mut false_positives = 0;

    for _ in 0..n_tests {
        let mut sa = 0;
        let mut sb = 0;
        for _ in 0..n_per {
            if rand_f64(&mut rng) < true_rate_a { sa += 1; }
            if rand_f64(&mut rng) < true_rate_a { sb += 1; } // same rate!
        }
        let (_, p) = two_prop_z_test(n_per, sa, n_per, sb);
        if p < 0.05 { false_positives += 1; }
    }

    let fp_rate = false_positives as f64 / n_tests as f64;
    println!("False positives: {}/{} = {:.1}%", false_positives, n_tests, fp_rate * 100.0);
    println!("Expected:        ~5.0%");
    println!("Match:           {}", if (fp_rate - 0.05).abs() < 0.03 { "Yes (within expected range)" } else { "No" });

    // --- Scenario 3: Early stopping problem ---
    println!("\n=== Danger of Early Stopping (Peeking) ===\n");
    println!("If you check results daily and stop when p < 0.05,");
    println!("the actual false positive rate inflates well above 5%.\n");

    let n_peek_tests = 1000;
    let max_days = 20;
    let users_per_day = 50;
    let mut early_stop_fp = 0;

    for _ in 0..n_peek_tests {
        let mut sa = 0;
        let mut sb = 0;
        let mut total = 0;
        let mut stopped_early = false;

        for _day in 0..max_days {
            for _ in 0..users_per_day {
                if rand_f64(&mut rng) < true_rate_a { sa += 1; }
                if rand_f64(&mut rng) < true_rate_a { sb += 1; }
                total += 1;
            }
            // Peek at results
            let (_, p) = two_prop_z_test(total, sa, total, sb);
            if p < 0.05 {
                early_stop_fp += 1;
                stopped_early = true;
                break;
            }
        }
    }

    let peek_fp_rate = early_stop_fp as f64 / n_peek_tests as f64;
    println!("{:>25} {:>12}", "Method", "FP Rate");
    println!("{}", "-".repeat(40));
    println!("{:>25} {:>11.1}%", "Fixed sample (correct)", fp_rate * 100.0);
    println!("{:>25} {:>11.1}%", "Daily peeking (wrong)", peek_fp_rate * 100.0);
    println!("\nPeeking inflates false positives by {:.1}x!", peek_fp_rate / fp_rate);
    println!("Solution: pre-specify sample size and only test once,");
    println!("or use sequential testing methods that control for peeking.");

    // --- Scenario 4: Power analysis ---
    println!("\n=== Power Analysis ===\n");
    println!("With n={} per group, what effects can we reliably detect?\n", n_per_group);

    println!("{:>10} {:>10} {:>12}", "Effect", "Power", "Detectable?");
    println!("{}", "-".repeat(35));

    for &effect in &[0.01, 0.02, 0.03, 0.05, 0.10] {
        let p1 = baseline;
        let p2 = baseline + effect;

        // Simulate power
        let n_sim = 500;
        let mut detected = 0;
        for _ in 0..n_sim {
            let mut sa = 0;
            let mut sb = 0;
            for _ in 0..n_per_group {
                if rand_f64(&mut rng) < p1 { sa += 1; }
                if rand_f64(&mut rng) < p2 { sb += 1; }
            }
            let (_, p) = two_prop_z_test(n_per_group, sa, n_per_group, sb);
            if p < 0.05 { detected += 1; }
        }
        let sim_power = detected as f64 / n_sim as f64;
        let detectable = if sim_power >= 0.80 { "Yes" } else { "No" };
        println!("{:>+10.2} {:>9.1}% {:>12}", effect, sim_power * 100.0, detectable);
    }
    println!("\nPower >= 80% is the standard threshold for reliable detection.");

    // --- Staged rollout ---
    println!("\n=== Staged Rollout Plan ===\n");
    println!("Phase 1: Canary (5% traffic)  - check for errors, latency");
    println!("Phase 2: Ramp   (25% traffic) - monitor key metrics");
    println!("Phase 3: Test   (50% traffic) - formal A/B test");
    println!("Phase 4: Ship   (100% traffic) - if test passes");

    let decision = if p_value < 0.05 && rate_b > rate_a { "Deploy Model B" } else { "Keep Model A" };
    println!();
    println!("kata_metric(\"p_value\", {:.6})", p_value);
    println!("kata_metric(\"observed_effect\", {:.4})", rate_b - rate_a);
    println!("kata_metric(\"false_positive_rate\", {:.3})", fp_rate);
    println!("kata_metric(\"peeking_fp_rate\", {:.3})", peek_fp_rate);
}
```

---

## Key Takeaways

- **A/B testing is the gold standard for comparing models in production.** Offline metrics can be misleading; only real-world experiments provide causal evidence of improvement.
- **Sample size matters.** Small effects require large samples to detect reliably. Always calculate the required sample size before starting a test.
- **Never peek at results and stop early.** Checking results repeatedly and stopping when significant inflates the false positive rate far above the nominal alpha. Pre-specify your sample size or use sequential testing methods.
- **The false positive rate should match alpha when the null hypothesis is true.** This is a fundamental validation: if you run 1000 tests where the models are identical, about 50 should show significance at alpha=0.05.
- **Staged rollouts limit risk.** Start with a small percentage of traffic (canary), ramp up gradually, and only commit to 100% after the formal test confirms improvement.
