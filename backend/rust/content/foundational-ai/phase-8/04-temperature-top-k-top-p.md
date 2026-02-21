# Temperature, Top-k, Top-p

> Phase 8 — LLMs | Kata 8.4

---

## Concept & Intuition

### What problem are we solving?

Temperature, top-k, and top-p are the three primary knobs for controlling the randomness of language model generation. While we introduced these concepts briefly in the sampling strategies kata, here we explore their mechanics in depth, understand how they interact, and develop intuition for when and how to tune each one. In practice, these parameters are often combined, and understanding their interplay is essential for effective prompt engineering and application development.

Temperature is applied before softmax: dividing logits by temperature T changes the shape of the probability distribution. T=1.0 gives the model's natural distribution. T < 1.0 sharpens it (making high-probability tokens more dominant), and T > 1.0 flattens it (making the distribution more uniform). As T approaches 0, sampling becomes greedy; as T approaches infinity, sampling becomes uniform random. Temperature operates on the logits, not the probabilities, which means it interacts with the model's confidence in a nonlinear way.

Top-k and top-p act as filters after the probability distribution is computed. Top-k is a hard cutoff: keep exactly k tokens. Top-p (nucleus sampling) is an adaptive cutoff: keep the fewest tokens whose cumulative probability exceeds p. The key insight is that these three controls operate at different stages and can be combined. A common configuration is to apply temperature first (to calibrate the overall randomness), then top-p (to filter the tail), creating a pipeline that gives fine-grained control over generation behavior.

### Why naive approaches fail

Using temperature alone is insufficient because even a well-calibrated distribution may have a long tail of improbable tokens that occasionally get selected. Temperature controls the overall shape but not the tail behavior. A high temperature makes all tokens more likely, including absurd ones.

Using top-k alone with a fixed k is suboptimal because the "right" number of tokens to consider varies dramatically by context. After "The capital of France is", there is essentially one correct answer, and k=50 is far too many. After "I feel", there are many valid continuations, and k=5 is too few. Top-p adapts to this automatically.

Using only top-p without temperature adjustment means you cannot independently control how peaked the distribution is within the nucleus. Two distributions might include the same tokens in their top-p nucleus but have very different internal rankings.

### Mental models

- **Three-stage pipeline**: Temperature adjusts the raw scores (logits), top-p/top-k filters the resulting distribution, and then sampling selects from the filtered distribution. Each stage has a distinct role.
- **Temperature as zoom**: Low temperature zooms in on the peak of the distribution; high temperature zooms out to see the full landscape. Top-p/top-k draws a boundary around what you can sample from.
- **Confidence calibration**: Temperature corrects the model's confidence. If the model is overconfident (too peaked), raise temperature. If underconfident (too flat), lower it.

### Visual explanations

```
  Temperature effect on logits [3.0, 2.0, 1.0, 0.0, -1.0]:

  T=0.5 (sharp):  logits/T = [6.0, 4.0, 2.0, 0.0, -2.0]
                   probs    = [0.84, 0.12, 0.02, 0.00, 0.00]

  T=1.0 (normal): logits/T = [3.0, 2.0, 1.0, 0.0, -1.0]
                   probs    = [0.64, 0.24, 0.09, 0.03, 0.01]

  T=2.0 (flat):   logits/T = [1.5, 1.0, 0.5, 0.0, -0.5]
                   probs    = [0.33, 0.20, 0.12, 0.07, 0.04]

  Combined pipeline:

  Logits -> [/T] -> softmax -> [top-p filter] -> renormalize -> sample

  Example: logits=[3,2,1,0,-1], T=1.0, top-p=0.9
  After softmax:     [0.64, 0.24, 0.09, 0.03, 0.01]
  Cumulative:         0.64  0.88  0.97  ...
  Top-p=0.9 keeps:   [0.64, 0.24, 0.09]  (sum=0.97 >= 0.9)
  Renormalized:       [0.66, 0.25, 0.09]  -> sample from these 3
```

---

## Hands-on Exploration

1. Implement temperature scaling and visualize its effect on probability distributions.
2. Show how top-k and top-p interact with temperature.
3. Measure entropy (randomness) of the distribution under different parameter combinations.
4. Generate text under various parameter settings and analyze quality.

---

## Live Code

```rust
fn main() {
    println!("=== Temperature, Top-k, Top-p ===\n");

    // Example logits from a language model
    let vocab = [
        "Paris", "London", "Berlin", "Rome", "Madrid",
        "Tokyo", "cat", "the", "42", "banana",
    ];
    let logits = vec![
        4.5, 2.8, 2.2, 1.8, 1.5,
        0.8, -1.0, -1.5, -2.0, -3.0,
    ];

    println!("Context: 'The capital of France is ___'\n");
    println!("Raw logits:");
    for (w, l) in vocab.iter().zip(logits.iter()) {
        println!("  {:>10}: {:.1}", w, l);
    }

    // 1. Temperature sweep
    println!("\n--- Temperature Effect ---\n");
    let temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0];

    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>7}",
        "T", "Paris", "London", "Berlin", "Rome", "Madrid", "Entropy"
    );
    println!("{}", "-".repeat(62));

    for &t in &temperatures {
        let probs = softmax_temp(&logits, t);
        let entropy = shannon_entropy(&probs);
        println!(
            "{:>6.1} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}  {:>7.3}",
            t, probs[0], probs[1], probs[2], probs[3], probs[4], entropy
        );
    }

    println!("\n  T->0: greedy (Paris=1.0)");
    println!("  T->inf: uniform (all=0.10)");
    println!("  Max entropy for 10 tokens: {:.3}", (10.0_f64).ln());

    // 2. Top-k with temperature
    println!("\n--- Top-k + Temperature Interaction ---\n");

    let combos = vec![
        (1.0, 1), (1.0, 3), (1.0, 5),
        (0.5, 3), (1.0, 3), (2.0, 3),
    ];

    println!(
        "{:>5} {:>5} {:>35}  {:>7}",
        "T", "k", "Distribution (non-zero probs)", "Entropy"
    );
    println!("{}", "-".repeat(60));

    for (t, k) in &combos {
        let probs = softmax_temp(&logits, *t);
        let filtered = top_k(&probs, *k);
        let entropy = shannon_entropy(&filtered);

        let nonzero: Vec<String> = filtered
            .iter()
            .zip(vocab.iter())
            .filter(|(p, _)| **p > 0.001)
            .map(|(p, w)| format!("{}={:.3}", w, p))
            .collect();

        println!(
            "{:>5.1} {:>5} {:>35}  {:>7.3}",
            t, k,
            nonzero.join(", "),
            entropy
        );
    }

    // 3. Top-p adaptive behavior
    println!("\n--- Top-p Adapts to Confidence ---\n");

    // Confident context (capital of France)
    let confident_logits = vec![5.0, 2.0, 1.5, 1.0, 0.5, -0.5, -1.5, -2.5, -3.0, -4.0];
    let confident_probs = softmax_temp(&confident_logits, 1.0);

    // Uncertain context (something that could be anything)
    let uncertain_logits = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    let uncertain_probs = softmax_temp(&uncertain_logits, 1.0);

    for p_threshold in &[0.7, 0.8, 0.9, 0.95] {
        let conf_filtered = top_p(&confident_probs, *p_threshold);
        let conf_count = conf_filtered.iter().filter(|p| **p > 0.001).count();

        let unc_filtered = top_p(&uncertain_probs, *p_threshold);
        let unc_count = unc_filtered.iter().filter(|p| **p > 0.001).count();

        println!(
            "  p={:.2}: confident context={} tokens, uncertain context={} tokens",
            p_threshold, conf_count, unc_count
        );
    }
    println!("\n  Top-p naturally includes fewer tokens when the model is confident.");

    // 4. Full pipeline: Temperature + Top-p
    println!("\n--- Combined Pipeline: Temperature + Top-p ---\n");

    let pipeline_configs = vec![
        (0.3, 0.9, "Low creativity (factual answers)"),
        (0.7, 0.9, "Moderate creativity (general writing)"),
        (1.0, 0.95, "High creativity (brainstorming)"),
        (1.5, 0.95, "Maximum creativity (poetry)"),
        (0.7, 0.5, "Focused, moderate temp"),
        (1.0, 0.7, "Balanced"),
    ];

    for (t, p, desc) in &pipeline_configs {
        let probs = softmax_temp(&logits, *t);
        let filtered = top_p(&probs, *p);
        let entropy = shannon_entropy(&filtered);
        let n_tokens = filtered.iter().filter(|x| **x > 0.001).count();

        let top_choices: Vec<String> = filtered
            .iter()
            .zip(vocab.iter())
            .filter(|(prob, _)| **prob > 0.001)
            .take(5)
            .map(|(prob, word)| format!("{}={:.2}", word, prob))
            .collect();

        println!(
            "  T={:.1}, p={:.2}: {} tokens, entropy={:.3}  [{}]",
            t, p, n_tokens, entropy,
            top_choices.join(", ")
        );
        println!("    Use case: {}\n", desc);
    }

    // 5. Visualization: probability mass coverage
    println!("--- Probability Mass Visualization ---\n");
    let probs_default = softmax_temp(&logits, 1.0);

    println!("  Cumulative probability coverage (T=1.0):\n");
    let mut sorted_idx: Vec<(usize, f64)> = probs_default.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    sorted_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    for (rank, (idx, prob)) in sorted_idx.iter().enumerate() {
        cumulative += prob;
        let bar = "#".repeat((prob * 50.0) as usize);
        let cum_bar = "=".repeat((cumulative * 30.0) as usize);
        println!(
            "  {:>2}. {:>8} p={:.4} | {} | cum={:.3} {}",
            rank + 1,
            vocab[*idx],
            prob,
            bar,
            cumulative,
            cum_bar
        );
    }

    // Practical recommendations
    println!("\n--- Practical Recommendations ---\n");
    let recommendations = [
        ("Code generation",   "T=0.0-0.2, top-p=0.95",  "Precision is critical"),
        ("Factual QA",        "T=0.0-0.3, top-p=0.90",  "One correct answer"),
        ("Conversational",    "T=0.7, top-p=0.90",       "Natural variety"),
        ("Creative writing",  "T=0.9-1.0, top-p=0.95",  "Diverse vocabulary"),
        ("Brainstorming",     "T=1.0-1.5, top-p=0.98",  "Maximum diversity"),
    ];

    println!("{:<22} {:<28} {}", "Task", "Settings", "Rationale");
    println!("{}", "-".repeat(72));
    for (task, settings, rationale) in &recommendations {
        println!("{:<22} {:<28} {}", task, settings, rationale);
    }
}

fn softmax_temp(logits: &[f64], temperature: f64) -> Vec<f64> {
    let t = temperature.max(1e-10);
    let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn top_k(probs: &[f64], k: usize) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0.0; probs.len()];
    let mut total = 0.0;
    for (idx, prob) in indexed.iter().take(k) {
        filtered[*idx] = *prob;
        total += prob;
    }
    if total > 0.0 {
        for p in &mut filtered { *p /= total; }
    }
    filtered
}

fn top_p(probs: &[f64], p_threshold: f64) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0.0; probs.len()];
    let mut cumulative = 0.0;
    for (idx, prob) in &indexed {
        filtered[*idx] = *prob;
        cumulative += prob;
        if cumulative >= p_threshold { break; }
    }
    let total: f64 = filtered.iter().sum();
    if total > 0.0 {
        for p in &mut filtered { *p /= total; }
    }
    filtered
}

fn shannon_entropy(probs: &[f64]) -> f64 {
    -probs
        .iter()
        .filter(|p| **p > 1e-10)
        .map(|p| p * p.ln())
        .sum::<f64>()
}
```

---

## Key Takeaways

- Temperature controls the overall sharpness of the probability distribution before sampling: low values make it peaked (deterministic), high values make it flat (random).
- Top-k applies a hard cutoff at exactly k tokens, while top-p adapts the cutoff based on cumulative probability, naturally including fewer tokens when the model is confident.
- These controls operate at different stages (temperature on logits, top-k/top-p on probabilities) and are typically combined in a pipeline for fine-grained control.
- The optimal settings depend on the task: precision-critical tasks demand low temperature and tight filtering, while creative tasks benefit from higher temperature and wider nucleus.
