# Multi-Armed Bandits

> Phase 9 — Reinforcement Learning | Kata 9.6

---

## Concept & Intuition

### What problem are we solving?

The multi-armed bandit problem is the purest form of the exploration-exploitation dilemma. Imagine a row of slot machines (bandits), each with an unknown probability of paying out. You want to maximize your total winnings over many pulls. The challenge: you must balance **exploring** different machines to estimate their payouts with **exploiting** the machine you currently think is best.

This is a simplified RL problem with only one state and multiple actions. It strips away the complexity of sequential decision-making to focus entirely on the exploration-exploitation trade-off. Despite its simplicity, the multi-armed bandit framework has profound real-world applications: A/B testing (which website design is better?), clinical trials (which treatment is more effective?), recommendation systems (which ad to show?), and resource allocation.

The key strategies are: **epsilon-greedy** (explore randomly with probability epsilon), **Upper Confidence Bound (UCB)** (explore actions with high uncertainty), and **Thompson Sampling** (maintain a probability distribution over each arm's quality and sample from it). Each has different theoretical guarantees and practical strengths.

### Why naive approaches fail

Pure exploration (always try randomly) wastes resources. Pure exploitation (always pick the current best) locks onto a suboptimal arm if early estimates are misleading. The challenge is that you never know the true payout probabilities -- you only have noisy estimates from past pulls. An arm that looked bad in 3 pulls might actually be the best.

### Mental models

- **Explore-exploit as information gathering**: exploration is an investment in information that pays off later through better exploitation.
- **UCB as optimism**: "give each arm the benefit of the doubt -- the less you know about it, the more optimistic you should be."
- **Thompson Sampling as probability matching**: "how confident am I that each arm is the best? Pull arms in proportion to that confidence."

### Visual explanations

```
3-Armed Bandit:
  Arm 0: true p = 0.3  (bad)
  Arm 1: true p = 0.5  (okay)
  Arm 2: true p = 0.7  (best -- unknown to agent)

  After 10 pulls:
  Arm 0: 4 pulls, 1 win  -> estimate = 0.25
  Arm 1: 4 pulls, 3 wins -> estimate = 0.75  <-- looks best!
  Arm 2: 2 pulls, 1 win  -> estimate = 0.50

  Problem: Arm 1 looks best due to luck.
           Arm 2 has fewer samples so estimate is unreliable.
           UCB would prioritize Arm 2 (high uncertainty).

UCB formula: score = estimated_value + c * sqrt(ln(total_pulls) / arm_pulls)
             ^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             exploitation              exploration bonus (high for rarely-pulled arms)
```

---

## Hands-on Exploration

1. Create a 5-armed bandit with known payout probabilities. Implement epsilon-greedy and track cumulative regret.
2. Implement UCB1 and compare its regret to epsilon-greedy. UCB should have lower regret.
3. Implement Thompson Sampling and compare all three strategies.
4. Plot the percentage of time each strategy selects the optimal arm over time.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_int = |s: &mut u64, max: usize| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as usize) % max
    };

    // --- 5-Armed Bandit ---
    let true_probs = [0.20, 0.35, 0.55, 0.70, 0.45];
    let n_arms = true_probs.len();
    let best_arm = 3; // true_probs[3] = 0.70
    let n_rounds = 2000;

    let pull = |arm: usize, rng: &mut u64| -> f64 {
        if rand_f64(rng) < true_probs[arm] { 1.0 } else { 0.0 }
    };

    println!("=== Multi-Armed Bandits ===\n");
    println!("Arms: {}, True probabilities: {:?}", n_arms, true_probs);
    println!("Best arm: {} (p={}), Rounds: {}\n", best_arm, true_probs[best_arm], n_rounds);

    // --- Strategy 1: Epsilon-Greedy ---
    let epsilon = 0.1;
    let mut eg_counts = vec![0_usize; n_arms];
    let mut eg_rewards = vec![0.0_f64; n_arms];
    let mut eg_cumulative_regret = 0.0;
    let mut eg_regret_history = Vec::new();
    let mut eg_optimal_pulls = 0;

    for t in 0..n_rounds {
        let arm = if rand_f64(&mut rng) < epsilon {
            rand_int(&mut rng, n_arms)
        } else {
            let mut best = 0;
            let mut best_est = f64::NEG_INFINITY;
            for a in 0..n_arms {
                let est = if eg_counts[a] > 0 { eg_rewards[a] / eg_counts[a] as f64 } else { f64::MAX };
                if est > best_est { best_est = est; best = a; }
            }
            best
        };

        let reward = pull(arm, &mut rng);
        eg_counts[arm] += 1;
        eg_rewards[arm] += reward;
        eg_cumulative_regret += true_probs[best_arm] - true_probs[arm];
        eg_regret_history.push(eg_cumulative_regret);
        if arm == best_arm { eg_optimal_pulls += 1; }
    }

    // --- Strategy 2: UCB1 ---
    let mut ucb_counts = vec![0_usize; n_arms];
    let mut ucb_rewards = vec![0.0_f64; n_arms];
    let mut ucb_cumulative_regret = 0.0;
    let mut ucb_regret_history = Vec::new();
    let mut ucb_optimal_pulls = 0;

    for t in 0..n_rounds {
        let arm = if t < n_arms {
            t // pull each arm once
        } else {
            let mut best = 0;
            let mut best_score = f64::NEG_INFINITY;
            for a in 0..n_arms {
                let mean = ucb_rewards[a] / ucb_counts[a] as f64;
                let exploration = (2.0 * ((t + 1) as f64).ln() / ucb_counts[a] as f64).sqrt();
                let score = mean + exploration;
                if score > best_score { best_score = score; best = a; }
            }
            best
        };

        let reward = pull(arm, &mut rng);
        ucb_counts[arm] += 1;
        ucb_rewards[arm] += reward;
        ucb_cumulative_regret += true_probs[best_arm] - true_probs[arm];
        ucb_regret_history.push(ucb_cumulative_regret);
        if arm == best_arm { ucb_optimal_pulls += 1; }
    }

    // --- Strategy 3: Thompson Sampling ---
    // Beta distribution sampling via approximation
    let mut ts_alpha = vec![1.0_f64; n_arms]; // prior: Beta(1,1) = Uniform
    let mut ts_beta = vec![1.0_f64; n_arms];
    let mut ts_cumulative_regret = 0.0;
    let mut ts_regret_history = Vec::new();
    let mut ts_optimal_pulls = 0;

    // Simple Beta sampling using the inverse CDF approximation
    let sample_beta = |a: f64, b: f64, rng: &mut u64| -> f64 {
        // Use the fact that Gamma(a,1)/[Gamma(a,1)+Gamma(b,1)] ~ Beta(a,b)
        // Simple approximation for small a,b: use Kumaraswamy distribution
        // or just use mean + noise proportional to variance
        let mean = a / (a + b);
        let var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
        let std = var.sqrt();
        // Use uniform noise scaled by std as rough approximation
        let u = rand_f64(rng) * 2.0 - 1.0;
        (mean + u * std * 3.0).max(0.001).min(0.999)
    };

    for _ in 0..n_rounds {
        // Sample from each arm's posterior
        let mut best_sample = f64::NEG_INFINITY;
        let mut arm = 0;
        for a in 0..n_arms {
            let sample = sample_beta(ts_alpha[a], ts_beta[a], &mut rng);
            if sample > best_sample {
                best_sample = sample;
                arm = a;
            }
        }

        let reward = pull(arm, &mut rng);
        if reward > 0.5 {
            ts_alpha[arm] += 1.0;
        } else {
            ts_beta[arm] += 1.0;
        }
        ts_cumulative_regret += true_probs[best_arm] - true_probs[arm];
        ts_regret_history.push(ts_cumulative_regret);
        if arm == best_arm { ts_optimal_pulls += 1; }
    }

    // --- Results ---
    println!("=== Final Results (after {} rounds) ===\n", n_rounds);

    println!("{:<20} {:>12} {:>12} {:>15}", "Strategy", "Cum.Regret", "Optimal%", "Arm Pulls");
    println!("{}", "-".repeat(62));

    let eg_pct = eg_optimal_pulls as f64 / n_rounds as f64 * 100.0;
    let ucb_pct = ucb_optimal_pulls as f64 / n_rounds as f64 * 100.0;
    let ts_pct = ts_optimal_pulls as f64 / n_rounds as f64 * 100.0;

    println!("{:<20} {:>12.1} {:>11.1}% {:>15?}", "Epsilon-Greedy",
        eg_cumulative_regret, eg_pct, eg_counts);
    println!("{:<20} {:>12.1} {:>11.1}% {:>15?}", "UCB1",
        ucb_cumulative_regret, ucb_pct, ucb_counts);
    println!("{:<20} {:>12.1} {:>11.1}%", "Thompson Sampling",
        ts_cumulative_regret, ts_pct);

    // --- Estimated vs true values ---
    println!("\n=== Learned Arm Values ===\n");
    println!("{:>4} {:>8} {:>12} {:>12} {:>12}", "Arm", "True", "E-Greedy", "UCB1", "Thompson");
    println!("{}", "-".repeat(52));
    for a in 0..n_arms {
        let eg_est = if eg_counts[a] > 0 { eg_rewards[a] / eg_counts[a] as f64 } else { 0.0 };
        let ucb_est = if ucb_counts[a] > 0 { ucb_rewards[a] / ucb_counts[a] as f64 } else { 0.0 };
        let ts_est = ts_alpha[a] / (ts_alpha[a] + ts_beta[a]);
        let marker = if a == best_arm { " <-- best" } else { "" };
        println!("{:>4} {:>8.2} {:>12.3} {:>12.3} {:>12.3}{}",
            a, true_probs[a], eg_est, ucb_est, ts_est, marker);
    }

    // --- Regret over time ---
    println!("\n=== Cumulative Regret Over Time ===");
    println!("{:>6} {:>12} {:>12} {:>12}", "Round", "E-Greedy", "UCB1", "Thompson");
    for &t in &[100, 500, 1000, 1500, 2000] {
        if t <= n_rounds {
            println!("{:>6} {:>12.1} {:>12.1} {:>12.1}",
                t, eg_regret_history[t - 1], ucb_regret_history[t - 1],
                ts_regret_history[t - 1]);
        }
    }

    println!();
    println!("kata_metric(\"eg_regret\", {:.1})", eg_cumulative_regret);
    println!("kata_metric(\"ucb_regret\", {:.1})", ucb_cumulative_regret);
    println!("kata_metric(\"ts_regret\", {:.1})", ts_cumulative_regret);
}
```

---

## Key Takeaways

- **The multi-armed bandit is the fundamental exploration-exploitation problem.** All RL exploration strategies build on ideas developed for bandits.
- **Epsilon-greedy is simple but wastes exploration.** It explores uniformly at random, even pulling arms that are clearly suboptimal.
- **UCB uses optimism under uncertainty.** Arms with fewer pulls get a bonus, ensuring they are tried enough to get reliable estimates.
- **Thompson Sampling uses Bayesian probability matching.** It naturally explores uncertain arms and converges to the best arm, often with the lowest regret.
