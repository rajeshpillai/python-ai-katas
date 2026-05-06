# Sampling Strategies

> Phase 8 — LLMs | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

When a language model predicts a probability distribution over the next token, we must choose how to select a token from that distribution. This choice, the sampling strategy, has an outsized impact on the quality and character of generated text. The simplest approach, greedy decoding (always pick the most probable token), produces repetitive, dull text. Pure random sampling from the full distribution produces creative but often incoherent text. The challenge is finding the sweet spot between these extremes.

The core tension is between diversity and coherence. Greedy decoding maximizes local probability at each step but often leads to degenerate repetition: "the the the the..." or boring loops. This happens because the globally optimal sequence is not necessarily the one that picks the locally most probable token at each step. On the other hand, sampling from the full distribution occasionally selects very low-probability tokens (typos, off-topic words, grammatical errors) that derail the generation.

Different tasks demand different strategies. Creative writing benefits from higher diversity. Code generation demands high coherence. Factual question-answering needs precision. Understanding the landscape of sampling strategies and their behavioral properties is essential for effectively deploying language models.

### Why naive approaches fail

Greedy decoding fails because language has many valid continuations at each point, and always choosing the single most likely one produces repetitive, mode-collapsed text. The word "the" is the most common word in English, so greedy decoding overproduces it.

Beam search (tracking the top-B most probable sequences) is better than greedy but still tends toward high-probability, generic text. It also has a length bias: shorter sequences accumulate less negative log-probability and are preferred, requiring length normalization heuristics.

Pure sampling fails because even a small probability assigned to an absurd token (say, 0.1% chance of a random emoji in formal text) will eventually be selected during long generation, and once selected, it can cascade into completely incoherent text because the model has never seen such context during training.

### Mental models

- **DJ mixing board**: Sampling strategy is like a mixing board controlling the balance between "safe/boring" (greedy) and "wild/creative" (random). Temperature is the master knob; top-k and top-p are filters that cut out the noise floor.
- **Curated randomness**: The goal is not to eliminate randomness but to curate it. We want the model to choose among reasonable options, not all conceivable options.
- **Explore-exploit tradeoff**: Each token selection is a miniature explore-exploit decision. Too much exploitation (greedy) gets stuck; too much exploration (random) wanders off.

### Visual explanations

```
  Probability distribution for next token:

  Token:     the    cat    dog    sat    .     xyz    !!!
  Prob:     0.35   0.25   0.20   0.10  0.05  0.03  0.02

  Greedy:   Always picks "the" (0.35)
            -> Repetitive, boring

  Full sampling: Can pick ANY token, including "xyz" (0.03)
            -> Occasionally incoherent

  Top-k (k=3): Only consider {the, cat, dog}
  Renormalize: 0.35/0.80=0.44, 0.25/0.80=0.31, 0.20/0.80=0.25
            -> Sample from these three only

  Top-p (p=0.80): Include tokens until cumulative prob >= 0.80
  Include: the(0.35) + cat(0.25) + dog(0.20) = 0.80
            -> Same as top-3 here, but adapts to distribution shape

  Temperature (T=0.5, on original logits):
  Sharpens distribution: the=0.52, cat=0.27, dog=0.15, sat=0.04, ...
            -> More deterministic

  Temperature (T=2.0):
  Flattens distribution: the=0.22, cat=0.20, dog=0.19, sat=0.16, ...
            -> More random
```

---

## Hands-on Exploration

1. Implement greedy, pure random, top-k, and top-p sampling.
2. Generate text with each strategy and compare coherence and diversity.
3. Visualize how each strategy reshapes the probability distribution.
4. Measure repetition rate and vocabulary diversity for different strategies.

---

## Live Code

```rust
fn main() {
    println!("=== Sampling Strategies ===\n");

    // Simulated probability distribution (logits before softmax)
    let vocab = vec![
        "the", "cat", "dog", "sat", "on", "mat", "big",
        "small", "happy", "ran", "ate", "fish", ".",
        "!", "xyz", "qqq",
    ];

    // Logits (pre-softmax scores)
    let logits = vec![
        2.5, 2.0, 1.8, 1.2, 0.8, 0.5, 0.3,
        0.2, 0.1, -0.1, -0.2, -0.3, -0.5,
        -1.0, -2.0, -3.0,
    ];

    let probs = softmax_with_temp(&logits, 1.0);

    println!("--- Original Distribution ---\n");
    print_distribution(&vocab, &probs);

    // 1. Greedy
    println!("\n--- Strategy 1: Greedy ---\n");
    let greedy_idx = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("  Always picks: '{}' (prob={:.3})", vocab[greedy_idx], probs[greedy_idx]);
    println!("  Pro: Maximum local probability");
    println!("  Con: No diversity, leads to repetition\n");

    // 2. Pure random
    println!("--- Strategy 2: Pure Random Sampling ---\n");
    println!("  Sampling 10 tokens from full distribution:");
    let mut selections: Vec<&str> = Vec::new();
    for seed in 0..10 {
        let idx = sample_from_dist(&probs, seed);
        selections.push(vocab[idx]);
    }
    println!("  [{}]", selections.join(", "));
    println!("  Note: can include low-prob tokens like 'xyz' or 'qqq'\n");

    // 3. Top-k sampling
    println!("--- Strategy 3: Top-k Sampling (k=4) ---\n");
    let k = 4;
    let topk_probs = top_k_filter(&probs, k);
    print_distribution(&vocab, &topk_probs);
    println!("  Only top {} tokens are kept; rest set to 0.\n", k);

    // 4. Top-p (nucleus) sampling
    println!("--- Strategy 4: Top-p Sampling (p=0.80) ---\n");
    let p = 0.80;
    let topp_probs = top_p_filter(&probs, p);
    print_distribution(&vocab, &topp_probs);
    println!("  Tokens included until cumulative probability >= {:.2}.\n", p);

    // Compare sampling results
    println!("--- Comparison: 20 Samples from Each Strategy ---\n");

    let strategies: Vec<(&str, Vec<f64>)> = vec![
        ("Full random", probs.clone()),
        ("Top-k (k=3)", top_k_filter(&probs, 3)),
        ("Top-k (k=6)", top_k_filter(&probs, 6)),
        ("Top-p (p=0.7)", top_p_filter(&probs, 0.7)),
        ("Top-p (p=0.9)", top_p_filter(&probs, 0.9)),
    ];

    for (name, dist) in &strategies {
        let mut samples = Vec::new();
        let mut unique = std::collections::HashSet::new();
        for seed in 0..20 {
            let idx = sample_from_dist(dist, seed * 7 + 13);
            samples.push(vocab[idx]);
            unique.insert(vocab[idx]);
        }
        let top_count = samples.iter().filter(|s| **s == vocab[greedy_idx]).count();
        println!(
            "  {:<15}: unique={:>2}/20, top_token_freq={:>2}/20",
            name,
            unique.len(),
            top_count
        );
        println!("    [{}]", samples[..10].join(", "));
    }

    // Show how distribution shape affects top-p behavior
    println!("\n--- Top-p Adapts to Distribution Shape ---\n");

    // Peaked distribution (confident prediction)
    let peaked_logits = vec![5.0, 1.0, 0.5, 0.2, 0.1, -0.5, -1.0, -2.0];
    let peaked_probs = softmax_with_temp(&peaked_logits, 1.0);
    let peaked_topp = top_p_filter(&peaked_probs, 0.9);
    let peaked_count = peaked_topp.iter().filter(|p| **p > 0.0).count();

    // Flat distribution (uncertain prediction)
    let flat_logits = vec![0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
    let flat_probs = softmax_with_temp(&flat_logits, 1.0);
    let flat_topp = top_p_filter(&flat_probs, 0.9);
    let flat_count = flat_topp.iter().filter(|p| **p > 0.0).count();

    println!(
        "  Peaked dist (confident): top-p=0.9 includes {} / {} tokens",
        peaked_count,
        peaked_logits.len()
    );
    println!(
        "  Flat dist (uncertain):   top-p=0.9 includes {} / {} tokens",
        flat_count,
        flat_logits.len()
    );
    println!("  Top-p adapts: fewer tokens when confident, more when uncertain.");

    // Diversity vs Coherence metrics
    println!("\n--- Repetition Analysis ---\n");
    println!("Generating 50 tokens with each strategy:\n");

    let gen_strategies: Vec<(&str, Vec<f64>)> = vec![
        ("Greedy", greedy_dist(&probs)),
        ("Top-k=2", top_k_filter(&probs, 2)),
        ("Top-k=5", top_k_filter(&probs, 5)),
        ("Top-p=0.8", top_p_filter(&probs, 0.8)),
        ("Full random", probs.clone()),
    ];

    for (name, dist) in &gen_strategies {
        let mut tokens = Vec::new();
        for seed in 0..50 {
            let idx = sample_from_dist(dist, seed * 31 + 7);
            tokens.push(vocab[idx]);
        }

        // Compute repetition rate (consecutive duplicates)
        let repetitions = tokens.windows(2).filter(|w| w[0] == w[1]).count();
        let unique_count = {
            let mut u: Vec<&str> = tokens.clone();
            u.sort();
            u.dedup();
            u.len()
        };

        println!(
            "  {:<12}: unique={:>2}/{}, consecutive_repeats={:>2}, top_token%={:.0}%",
            name,
            unique_count,
            vocab.len(),
            repetitions,
            tokens.iter().filter(|t| **t == vocab[greedy_idx]).count() as f64 / 50.0 * 100.0
        );
    }
}

fn softmax_with_temp(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn top_k_filter(probs: &[f64], k: usize) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0.0; probs.len()];
    let mut total = 0.0;
    for (idx, prob) in indexed.iter().take(k) {
        filtered[*idx] = *prob;
        total += prob;
    }
    // Renormalize
    if total > 0.0 {
        for p in &mut filtered {
            *p /= total;
        }
    }
    filtered
}

fn top_p_filter(probs: &[f64], p_threshold: f64) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut filtered = vec![0.0; probs.len()];
    let mut cumulative = 0.0;
    for (idx, prob) in &indexed {
        filtered[*idx] = *prob;
        cumulative += prob;
        if cumulative >= p_threshold {
            break;
        }
    }
    // Renormalize
    let total: f64 = filtered.iter().sum();
    if total > 0.0 {
        for p in &mut filtered {
            *p /= total;
        }
    }
    filtered
}

fn greedy_dist(probs: &[f64]) -> Vec<f64> {
    let max_idx = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let mut dist = vec![0.0; probs.len()];
    dist[max_idx] = 1.0;
    dist
}

fn sample_from_dist(probs: &[f64], seed: u64) -> usize {
    // Simple deterministic pseudo-random sampling
    let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (state >> 33) as f64 / (1u64 << 31) as f64;

    let mut cumulative = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

fn print_distribution(vocab: &[&str], probs: &[f64]) {
    let max_bar = 25;
    for (word, prob) in vocab.iter().zip(probs.iter()) {
        if *prob > 0.001 {
            let bar = "#".repeat((prob * max_bar as f64) as usize);
            println!("  {:>8}: {:.3} {}", word, prob, bar);
        }
    }
}
```

---

## Key Takeaways

- Sampling strategy determines how tokens are selected from the model's predicted distribution and dramatically affects the quality and character of generated text.
- Greedy decoding is deterministic but produces repetitive text; pure random sampling is diverse but incoherent. Practical strategies operate between these extremes.
- Top-k sampling restricts to the k most probable tokens, while top-p (nucleus) sampling adapts the cutoff based on the distribution shape, including fewer tokens when the model is confident and more when it is uncertain.
- The choice of sampling strategy should match the task: creative writing benefits from diversity (higher top-k/p), while factual generation benefits from precision (lower values or greedy).
