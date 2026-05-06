# N-grams

> Phase 6 — Sequence Models | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

Language is sequential: the meaning and likelihood of each word depends heavily on the words that came before it. N-grams are the simplest model of this sequential dependence. An N-gram model estimates the probability of a word given the previous N-1 words by counting occurrences in a training corpus. A bigram (N=2) model predicts each word based on the immediately preceding word; a trigram (N=3) uses the two preceding words.

Despite their simplicity, N-grams reveal fundamental properties of language. Word sequences follow power-law distributions (Zipf's law): a few sequences are very common while most are rare. This means that even large corpora leave most possible N-grams unobserved. The tension between capturing longer context (larger N) and having enough data to estimate probabilities reliably (smaller N) is a central challenge that motivates all subsequent sequence modeling techniques.

N-gram models also introduce key concepts that persist throughout NLP: conditional probability, perplexity as a language model evaluation metric, and the curse of dimensionality in discrete sequence spaces. Understanding N-grams provides the foundation for appreciating why neural approaches to language modeling were developed and what advantages they offer.

### Why naive approaches fail

The fundamental problem is sparsity. With a vocabulary of 50,000 words, there are 50,000^3 = 1.25 x 10^14 possible trigrams. No corpus is large enough to observe more than a tiny fraction of these. Unseen N-grams receive zero probability, which is catastrophic: a single unseen trigram in a sentence makes the entire sentence probability zero.

Smoothing techniques (Laplace, Kneser-Ney) partially address this by redistributing probability mass to unseen N-grams, but they cannot solve the fundamental problem: N-gram models treat each word as an atomic symbol with no notion of similarity. The model cannot infer that "the cat sat" and "the dog sat" are analogous because "cat" and "dog" are as different as "cat" and "bicycle" in the N-gram framework.

### Mental models

- **Autocomplete from memory**: N-gram models are like a person who completes sentences by recalling the most common continuations they have seen. They have no understanding of meaning, just statistical patterns.
- **Markov blanket**: Each word depends only on its N-1 predecessors and is independent of everything before that. This is a strong (and often wrong) assumption that limits the model's ability to capture long-range dependencies.
- **Frequency table lookup**: The entire model is just a giant lookup table mapping context sequences to probability distributions over next words.

### Visual explanations

```
  Training corpus: "the cat sat on the mat the cat ate the fish"

  Bigram counts:                    Bigram probabilities:
  (the, cat)  -> 2                  P(cat | the)  = 2/4 = 0.50
  (the, mat)  -> 1                  P(mat | the)  = 1/4 = 0.25
  (the, fish) -> 1                  P(fish | the) = 1/4 = 0.25
  (cat, sat)  -> 1                  P(sat | cat)  = 1/2 = 0.50
  (cat, ate)  -> 1                  P(ate | cat)  = 1/2 = 0.50
  (sat, on)   -> 1                  P(on | sat)   = 1/1 = 1.00
  (on, the)   -> 1                  P(the | on)   = 1/1 = 1.00
  (ate, the)  -> 1                  P(the | ate)  = 1/1 = 1.00

  Sparsity problem:
  Vocabulary: V words
  Possible bigrams:   V^2
  Possible trigrams:  V^3
  Possible 5-grams:   V^5

  V=10,000:  bigrams=10^8  trigrams=10^12  5-grams=10^20
  Most will NEVER appear in any training corpus!
```

---

## Hands-on Exploration

1. Build a bigram and trigram model from a small corpus.
2. Compute conditional probabilities and generate text by sampling.
3. Observe the sparsity problem as N increases.
4. Implement simple Laplace smoothing and see how it affects probabilities.

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    println!("=== N-grams ===\n");

    let corpus = "the cat sat on the mat the cat ate the fish \
                  the dog sat on the rug the dog ate the bone \
                  a cat sat on a mat a dog sat on a rug \
                  the cat sat on the floor the dog ran to the park";

    let words: Vec<&str> = corpus.split_whitespace().collect();
    println!("Corpus: {} words, {} unique", words.len(), {
        let mut unique: Vec<&str> = words.clone();
        unique.sort();
        unique.dedup();
        unique.len()
    });

    // Build bigram model
    let mut bigram_counts: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    for window in words.windows(2) {
        *bigram_counts
            .entry(window[0])
            .or_default()
            .entry(window[1])
            .or_default() += 1;
    }

    println!("\n--- Bigram Model ---");
    let contexts_to_show = vec!["the", "cat", "dog", "sat", "on"];
    for ctx in &contexts_to_show {
        if let Some(next_counts) = bigram_counts.get(ctx) {
            let total: usize = next_counts.values().sum();
            let mut entries: Vec<(&&str, &usize)> = next_counts.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1));
            let probs: Vec<String> = entries
                .iter()
                .take(5)
                .map(|(w, c)| format!("{}={:.2}", w, **c as f64 / total as f64))
                .collect();
            println!("  P(? | {:>5}) -> {}", ctx, probs.join(", "));
        }
    }

    // Build trigram model
    let mut trigram_counts: HashMap<(&str, &str), HashMap<&str, usize>> = HashMap::new();
    for window in words.windows(3) {
        *trigram_counts
            .entry((window[0], window[1]))
            .or_default()
            .entry(window[2])
            .or_default() += 1;
    }

    println!("\n--- Trigram Model ---");
    let tri_contexts = vec![("the", "cat"), ("the", "dog"), ("sat", "on"), ("on", "the")];
    for (w1, w2) in &tri_contexts {
        if let Some(next_counts) = trigram_counts.get(&(*w1, *w2)) {
            let total: usize = next_counts.values().sum();
            let mut entries: Vec<(&&str, &usize)> = next_counts.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1));
            let probs: Vec<String> = entries
                .iter()
                .map(|(w, c)| format!("{}={:.2}", w, **c as f64 / total as f64))
                .collect();
            println!("  P(? | {} {}) -> {}", w1, w2, probs.join(", "));
        }
    }

    // Sparsity analysis
    println!("\n--- Sparsity Analysis ---");
    let mut vocab: Vec<&str> = words.clone();
    vocab.sort();
    vocab.dedup();
    let v = vocab.len();

    let observed_bigrams: usize = bigram_counts.values().map(|m| m.len()).sum();
    let possible_bigrams = v * v;

    let observed_trigrams: usize = trigram_counts.values().map(|m| m.len()).sum();
    let possible_trigrams = v * v * v;

    println!("  Vocabulary size: {}", v);
    println!(
        "  Bigrams:  observed {} / possible {} ({:.2}%)",
        observed_bigrams,
        possible_bigrams,
        observed_bigrams as f64 / possible_bigrams as f64 * 100.0
    );
    println!(
        "  Trigrams: observed {} / possible {} ({:.4}%)",
        observed_trigrams,
        possible_trigrams,
        observed_trigrams as f64 / possible_trigrams as f64 * 100.0
    );

    // Generate text using bigram model with simple sampling
    println!("\n--- Text Generation (bigram, deterministic: most likely) ---");
    let mut current = "the";
    let mut generated = vec![current];
    for _ in 0..12 {
        if let Some(next_counts) = bigram_counts.get(current) {
            let mut entries: Vec<(&&str, &usize)> = next_counts.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1));
            current = entries[0].0;
            generated.push(current);
        } else {
            break;
        }
    }
    println!("  {}", generated.join(" "));

    // Laplace smoothing demo
    println!("\n--- Laplace Smoothing ---");
    let context = "cat";
    let unseen_word = "jumped";
    if let Some(next_counts) = bigram_counts.get(context) {
        let total: usize = next_counts.values().sum();

        // Without smoothing
        let p_unseen = 0.0;
        println!("  P({} | {}) without smoothing: {:.4}", unseen_word, context, p_unseen);

        // With Laplace smoothing (add-1)
        let alpha = 1.0;
        let p_smoothed = alpha / (total as f64 + alpha * v as f64);
        println!(
            "  P({} | {}) with Laplace (alpha=1): {:.4}",
            unseen_word, context, p_smoothed
        );

        // Show effect on seen words
        for (word, count) in next_counts.iter() {
            let p_orig = *count as f64 / total as f64;
            let p_smooth = (*count as f64 + alpha) / (total as f64 + alpha * v as f64);
            println!(
                "  P({} | {}) : original={:.3}, smoothed={:.4}",
                word, context, p_orig, p_smooth
            );
        }
    }

    // Perplexity calculation
    println!("\n--- Perplexity ---");
    let test = "the cat sat on the mat";
    let test_words: Vec<&str> = test.split_whitespace().collect();
    let mut log_prob_sum = 0.0_f64;
    let mut count = 0;
    for window in test_words.windows(2) {
        if let Some(next_counts) = bigram_counts.get(window[0]) {
            let total: usize = next_counts.values().sum();
            if let Some(c) = next_counts.get(window[1]) {
                let p = *c as f64 / total as f64;
                log_prob_sum += p.ln();
                count += 1;
            }
        }
    }
    if count > 0 {
        let perplexity = (-log_prob_sum / count as f64).exp();
        println!("  Test: \"{}\"", test);
        println!("  Bigram perplexity: {:.2}", perplexity);
        println!("  (Lower is better; 1.0 means perfect prediction)");
    }
}
```

---

## Key Takeaways

- N-gram models estimate word probabilities by counting sequences in a corpus, capturing local statistical patterns in language.
- The sparsity problem is fundamental: as N grows, the number of possible N-grams explodes exponentially, leaving most unobserved even in large corpora.
- Smoothing techniques redistribute probability mass but cannot address the deeper limitation that N-grams treat words as atomic symbols with no notion of semantic similarity.
- N-grams introduce foundational concepts (conditional probability, perplexity, the bias-variance tradeoff in context length) that remain central to all language modeling approaches.
