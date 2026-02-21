# Next-Token Prediction

> Phase 8 — LLMs | Kata 8.2

---

## Concept & Intuition

### What problem are we solving?

The core training objective of modern large language models is deceptively simple: given a sequence of tokens, predict the next token. This is called causal language modeling or autoregressive modeling. The model reads tokens [t_1, t_2, ..., t_n] and must output a probability distribution over the entire vocabulary for what t_{n+1} should be. Training maximizes the log-probability of the actual next token across all positions in the training corpus.

Despite its simplicity, this objective is remarkably powerful. To predict the next token well, the model must learn grammar (what word forms are valid), semantics (what words mean), world knowledge (facts about the world), reasoning patterns (logical and mathematical structure), and even theory of mind (what a character in a story might think). The loss function does not explicitly target any of these capabilities; they all emerge as necessary skills for reducing prediction error.

Autoregressive generation leverages the trained model to produce text: sample a token from the predicted distribution, append it to the context, and repeat. This process can generate coherent paragraphs, translate languages, answer questions, or write code, all from the simple foundation of next-token prediction. The quality of generation depends critically on both the model's prediction quality and the sampling strategy used.

### Why naive approaches fail

A simple bigram model (predicting the next token from only the previous token) captures local patterns but produces incoherent text beyond a few words because it lacks context about what was said earlier. Each prediction is made in isolation from the broader narrative.

A bag-of-words model that considers all previous tokens but ignores their order can choose plausible topics but produces grammatically incorrect sequences. Word order carries essential information: "dog bites man" and "man bites dog" have the same bag of words but very different meanings.

Maximizing the average probability of any next token (rather than the specific correct one) leads to models that always predict the most common word (like "the" or "a"). The cross-entropy loss specifically penalizes the model for assigning low probability to the actual next token, forcing it to distribute probability mass appropriately across the vocabulary.

### Mental models

- **Ultimate reading comprehension**: To predict the next word, you must truly understand everything that came before. The better your understanding, the lower your prediction error.
- **Lossy compression**: A language model is a compressed representation of its training data. It cannot memorize everything, so it must learn the underlying patterns and rules that generated the text.
- **Autoregressive unrolling**: Generation is like a snowball rolling downhill. Each predicted token becomes context for the next prediction, and the model's choices compound, for better or worse.

### Visual explanations

```
  Next-token prediction training:

  Input:  [The] [cat] [sat] [on]  [the] [mat]
  Target:  [cat] [sat] [on]  [the] [mat] [EOS]

  At each position, model predicts a distribution:

  Position 3 input: [The, cat, sat]
  Model output:     P(on)=0.25, P(down)=0.15, P(up)=0.10, ...
  Actual next:      "on"
  Loss:            -log(0.25) = 1.39

  Position 4 input: [The, cat, sat, on]
  Model output:     P(the)=0.40, P(a)=0.20, P(his)=0.10, ...
  Actual next:      "the"
  Loss:            -log(0.40) = 0.92

  Total loss = average of all position losses

  Autoregressive generation:

  Start: [The]
  Step 1: P(next|The) -> sample "cat" -> [The, cat]
  Step 2: P(next|The, cat) -> sample "sat" -> [The, cat, sat]
  Step 3: P(next|The, cat, sat) -> sample "on" -> [The, cat, sat, on]
  ...
```

---

## Hands-on Exploration

1. Build a simple next-token predictor using frequency-based conditional probabilities.
2. Train it on a small corpus and compute per-token cross-entropy loss.
3. Generate text autoregressively and observe coherence.
4. Compare perplexity of different context lengths to show why more context helps.

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    println!("=== Next-Token Prediction ===\n");

    // Training corpus
    let corpus = "\
        the cat sat on the mat . \
        the dog sat on the rug . \
        the cat ate the fish . \
        the dog ate the bone . \
        a big cat sat on a small mat . \
        a small dog sat on a big rug . \
        the cat chased the dog . \
        the dog chased the cat . \
        the big cat ate a small fish . \
        the small dog ate a big bone .";

    let tokens: Vec<&str> = corpus.split_whitespace().collect();

    // Build trigram model (context of 2)
    let mut bigram_model: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    let mut trigram_model: HashMap<(&str, &str), HashMap<&str, usize>> = HashMap::new();

    for w in tokens.windows(2) {
        *bigram_model.entry(w[0]).or_default().entry(w[1]).or_default() += 1;
    }
    for w in tokens.windows(3) {
        *trigram_model.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 1;
    }

    // Show prediction distributions
    println!("--- Prediction Distributions ---\n");

    let contexts = vec![
        vec!["the"],
        vec!["the", "cat"],
        vec!["the", "dog"],
        vec!["sat", "on"],
        vec!["ate", "the"],
    ];

    for ctx in &contexts {
        println!("Context: [{}]", ctx.join(", "));

        if ctx.len() == 1 {
            if let Some(dist) = bigram_model.get(ctx[0]) {
                print_distribution(dist);
            }
        } else if ctx.len() == 2 {
            if let Some(dist) = trigram_model.get(&(ctx[0], ctx[1])) {
                print_distribution(dist);
            }
        }
        println!();
    }

    // Compute cross-entropy loss on the training data
    println!("--- Cross-Entropy Loss ---\n");

    let test_sentences = vec![
        "the cat sat on the mat .",
        "the dog ate the bone .",
        "a big cat chased the dog .",
        "the fish sat on the cat .",  // Unusual sentence
    ];

    for sentence in &test_sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut total_loss = 0.0;
        let mut count = 0;

        for i in 1..words.len() {
            let prob = if i >= 2 {
                // Use trigram
                get_prob_trigram(&trigram_model, words[i-2], words[i-1], words[i])
                    .unwrap_or_else(|| get_prob_bigram(&bigram_model, words[i-1], words[i])
                        .unwrap_or(0.01))
            } else {
                get_prob_bigram(&bigram_model, words[i-1], words[i]).unwrap_or(0.01)
            };

            let loss = -(prob.max(1e-10)).ln();
            total_loss += loss;
            count += 1;
        }

        let avg_loss = total_loss / count as f64;
        let perplexity = avg_loss.exp();
        println!("  \"{}\"", sentence);
        println!(
            "    avg loss={:.3}, perplexity={:.2}\n",
            avg_loss, perplexity
        );
    }

    // Autoregressive generation
    println!("--- Autoregressive Generation ---\n");

    // Deterministic: always pick most likely
    println!("Greedy generation (always most likely):");
    for start in &["the", "a"] {
        let text = generate(&bigram_model, &trigram_model, start, 12, false);
        println!("  Start='{}': {}", start, text);
    }

    println!("\nStochastic generation (sample from distribution):");
    for seed in 0..4 {
        let text = generate_stochastic(
            &bigram_model,
            &trigram_model,
            "the",
            12,
            seed,
        );
        println!("  Seed {}: {}", seed, text);
    }

    // Show why context length matters
    println!("\n--- Context Length Impact ---\n");

    // Unigram vs bigram vs trigram predictions
    let test_context = ["the", "cat", "sat", "on", "the"];

    // Unigram: P(next) regardless of context
    let mut unigram: HashMap<&str, usize> = HashMap::new();
    for t in &tokens {
        *unigram.entry(t).or_default() += 1;
    }
    let total_tokens: usize = unigram.values().sum();

    println!("Predicting next token after [{}]:", test_context.join(", "));

    // Unigram prediction
    let uni_top = get_top_n(&unigram, 5);
    print!("  Unigram (no context):  ");
    for (w, c) in &uni_top {
        print!("{}={:.3} ", w, *c as f64 / total_tokens as f64);
    }
    println!();

    // Bigram prediction
    let last = test_context[test_context.len() - 1];
    if let Some(dist) = bigram_model.get(last) {
        let top = get_top_n(dist, 5);
        print!("  Bigram  (last 1):      ");
        let total: usize = dist.values().sum();
        for (w, c) in &top {
            print!("{}={:.3} ", w, *c as f64 / total as f64);
        }
        println!();
    }

    // Trigram prediction
    let last2 = (
        test_context[test_context.len() - 2],
        test_context[test_context.len() - 1],
    );
    if let Some(dist) = trigram_model.get(&last2) {
        let top = get_top_n(dist, 5);
        print!("  Trigram (last 2):      ");
        let total: usize = dist.values().sum();
        for (w, c) in &top {
            print!("{}={:.3} ", w, *c as f64 / total as f64);
        }
        println!();
    }
    println!("\n  More context -> sharper, more accurate predictions.");
}

fn get_prob_bigram(model: &HashMap<&str, HashMap<&str, usize>>, context: &str, target: &str) -> Option<f64> {
    model.get(context).and_then(|dist| {
        let total: usize = dist.values().sum();
        dist.get(target).map(|c| *c as f64 / total as f64)
    })
}

fn get_prob_trigram(
    model: &HashMap<(&str, &str), HashMap<&str, usize>>,
    w1: &str, w2: &str, target: &str,
) -> Option<f64> {
    model.get(&(w1, w2)).and_then(|dist| {
        let total: usize = dist.values().sum();
        dist.get(target).map(|c| *c as f64 / total as f64)
    })
}

fn print_distribution(dist: &HashMap<&str, usize>) {
    let total: usize = dist.values().sum();
    let mut entries: Vec<(&&str, &usize)> = dist.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    for (word, count) in entries.iter().take(6) {
        let prob = **count as f64 / total as f64;
        let bar = "#".repeat((prob * 30.0) as usize);
        println!("    {:>8}: {:.3} {}", word, prob, bar);
    }
}

fn generate(
    bigram: &HashMap<&str, HashMap<&str, usize>>,
    trigram: &HashMap<(&str, &str), HashMap<&str, usize>>,
    start: &str,
    max_tokens: usize,
    _stochastic: bool,
) -> String {
    let mut result = vec![start.to_string()];

    for _ in 0..max_tokens {
        let next = if result.len() >= 2 {
            let w1 = result[result.len() - 2].as_str();
            let w2 = result[result.len() - 1].as_str();
            // Try trigram first
            trigram.get(&(w1, w2))
                .and_then(|dist| dist.iter().max_by_key(|(_, c)| *c).map(|(w, _)| w.to_string()))
                .or_else(|| {
                    bigram.get(w2)
                        .and_then(|dist| dist.iter().max_by_key(|(_, c)| *c).map(|(w, _)| w.to_string()))
                })
        } else {
            let w = result.last().unwrap().as_str();
            bigram.get(w)
                .and_then(|dist| dist.iter().max_by_key(|(_, c)| *c).map(|(w, _)| w.to_string()))
        };

        match next {
            Some(word) => {
                let is_end = word == ".";
                result.push(word);
                if is_end { break; }
            }
            None => break,
        }
    }

    result.join(" ")
}

fn generate_stochastic(
    bigram: &HashMap<&str, HashMap<&str, usize>>,
    trigram: &HashMap<(&str, &str), HashMap<&str, usize>>,
    start: &str,
    max_tokens: usize,
    seed: u64,
) -> String {
    let mut result = vec![start.to_string()];
    let mut rng_state = seed;

    for _ in 0..max_tokens {
        let dist: Option<Vec<(String, f64)>> = if result.len() >= 2 {
            let w1 = result[result.len() - 2].as_str();
            let w2 = result[result.len() - 1].as_str();
            trigram.get(&(w1, w2)).map(|d| {
                let total: f64 = d.values().sum::<usize>() as f64;
                d.iter().map(|(w, c)| (w.to_string(), *c as f64 / total)).collect()
            }).or_else(|| {
                bigram.get(w2).map(|d| {
                    let total: f64 = d.values().sum::<usize>() as f64;
                    d.iter().map(|(w, c)| (w.to_string(), *c as f64 / total)).collect()
                })
            })
        } else {
            let w = result.last().unwrap().as_str();
            bigram.get(w).map(|d| {
                let total: f64 = d.values().sum::<usize>() as f64;
                d.iter().map(|(w, c)| (w.to_string(), *c as f64 / total)).collect()
            })
        };

        match dist {
            Some(d) => {
                // Simple pseudo-random sampling
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                let mut cumulative = 0.0;
                let mut chosen = d[0].0.clone();
                for (word, prob) in &d {
                    cumulative += prob;
                    if r < cumulative {
                        chosen = word.clone();
                        break;
                    }
                }
                let is_end = chosen == ".";
                result.push(chosen);
                if is_end { break; }
            }
            None => break,
        }
    }

    result.join(" ")
}

fn get_top_n<'a>(dist: &'a HashMap<&str, usize>, n: usize) -> Vec<(&'a str, usize)> {
    let mut entries: Vec<(&str, usize)> = dist.iter().map(|(w, c)| (*w, *c)).collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(n);
    entries
}
```

---

## Key Takeaways

- Next-token prediction is the deceptively simple objective behind LLMs: predict the probability distribution over the next token given all preceding tokens.
- This objective implicitly requires the model to learn grammar, semantics, world knowledge, and reasoning, as all are necessary for accurate prediction.
- Cross-entropy loss and perplexity measure prediction quality; lower perplexity means the model is less "surprised" by the actual text.
- Autoregressive generation leverages the trained predictor to produce text token by token, with each new token becoming context for the next prediction, making sampling strategy critical for output quality.
