# Tokenization

> Phase 8 — LLMs | Kata 8.1

---

## Concept & Intuition

### What problem are we solving?

Language models operate on sequences of discrete tokens, but raw text is a stream of characters. Tokenization is the process of converting text into a sequence of integer token IDs from a fixed vocabulary. The choice of tokenization scheme profoundly impacts model behavior, efficiency, and capabilities. Character-level tokenization gives a tiny vocabulary (around 256 entries) but requires the model to process very long sequences. Word-level tokenization gives manageable sequence lengths but cannot handle unseen words and produces enormous vocabularies.

Modern LLMs use subword tokenization, most commonly Byte Pair Encoding (BPE) or its variants. BPE starts with individual characters and iteratively merges the most frequent adjacent pair into a new token. After thousands of merges, the vocabulary contains common words as single tokens ("the", "and"), frequent subwords ("ing", "tion", "un"), and individual characters as fallback. This elegantly balances vocabulary size, sequence length, and open-vocabulary coverage: any possible text can be tokenized, common patterns are efficient, and rare words are decomposed into recognizable subwords.

Tokenization has surprisingly deep consequences. Languages with different scripts get different compression ratios (English text often tokenizes more efficiently than Chinese or Arabic, affecting model fairness and cost). The granularity of tokens affects what patterns the model can learn: a model that sees "unhappiness" as ["un", "happiness"] can leverage morphological structure, while one that sees it as a single token cannot.

### Why naive approaches fail

Character-level tokenization requires the model to implicitly learn spelling, word boundaries, and morphology from scratch. A sentence of 50 words might become 250+ characters, dramatically increasing sequence length and computational cost (attention is O(n^2) in sequence length). The model must use many layers just to assemble characters into word-level meanings before it can reason about semantics.

Word-level tokenization fails on out-of-vocabulary words. If "cryptocurrency" was not in the training vocabulary, it becomes an UNK token, losing all information. Even with a vocabulary of 100,000 words, the model cannot handle misspellings, neologisms, technical jargon, or morphological variants it has not seen. And storing embeddings for 100,000+ words requires significant memory.

### Mental models

- **Compression scheme**: Tokenization is like text compression. Frequent sequences get short codes (single tokens), rare sequences get longer codes (multiple tokens). The vocabulary is the codebook.
- **LEGO bricks**: Subword tokens are like LEGO bricks. Common objects (words) come pre-assembled, but unusual constructions can always be built from smaller pieces (characters).
- **Efficient alphabet**: BPE creates an alphabet where the "letters" range from single characters to full words, optimized so that the most common texts require the fewest letters.

### Visual explanations

```
  BPE tokenization process:

  Start: "l o w e r" "l o w e s t" "n e w e r" "n e w e s t"
         (character level, spaces between chars)

  Step 1: Most frequent pair = (e, r) -> merge to "er"
          "l o w er" "l o w e s t" "n e w er" "n e w e s t"

  Step 2: Most frequent pair = (e, s) -> merge to "es"
          "l o w er" "l o w es t" "n e w er" "n e w es t"

  Step 3: Most frequent pair = (es, t) -> merge to "est"
          "l o w er" "l o w est" "n e w er" "n e w est"

  Step 4: Most frequent pair = (l, o) -> merge to "lo"
          "lo w er" "lo w est" "n e w er" "n e w est"

  Step 5: Most frequent pair = (lo, w) -> merge to "low"
          "low er" "low est" "n e w er" "n e w est"

  Step 6: Most frequent pair = (n, e) -> merge to "ne"
          "low er" "low est" "ne w er" "ne w est"

  Step 7: Most frequent pair = (ne, w) -> merge to "new"
          "low er" "low est" "new er" "new est"

  Final vocabulary: {l,o,w,e,r,s,t,n, er, es, est, lo, low, ne, new}
```

---

## Hands-on Exploration

1. Implement BPE from scratch on a small corpus.
2. Observe how the vocabulary evolves as merges are applied.
3. Tokenize new text using the learned BPE vocabulary and measure compression ratio.
4. Compare character-level, word-level, and BPE tokenization on the same text.

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    println!("=== Tokenization ===\n");

    // Training corpus (each word appears with a frequency)
    let corpus = vec![
        ("low", 5),
        ("lower", 2),
        ("lowest", 1),
        ("new", 6),
        ("newer", 3),
        ("newest", 2),
        ("show", 4),
        ("showing", 2),
        ("showed", 1),
    ];

    // Initialize: split each word into characters plus end-of-word marker
    let mut vocab: Vec<(Vec<String>, usize)> = corpus
        .iter()
        .map(|(word, freq)| {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            (chars, *freq)
        })
        .collect();

    println!("--- BPE Training ---\n");
    println!("Initial vocabulary (character level):");
    print_vocab(&vocab);

    let num_merges = 12;
    let mut merge_rules: Vec<(String, String, String)> = Vec::new();

    for step in 0..num_merges {
        // Count adjacent pairs
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for (tokens, freq) in &vocab {
            for window in tokens.windows(2) {
                *pair_counts
                    .entry((window[0].clone(), window[1].clone()))
                    .or_default() += freq;
            }
        }

        if pair_counts.is_empty() {
            break;
        }

        // Find most frequent pair
        let (best_pair, count) = pair_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .unwrap();

        let merged = format!("{}{}", best_pair.0, best_pair.1);
        println!(
            "Step {:>2}: merge '{}' + '{}' -> '{}' (count={})",
            step + 1, best_pair.0, best_pair.1, merged, count
        );

        merge_rules.push((best_pair.0.clone(), best_pair.1.clone(), merged.clone()));

        // Apply merge to vocabulary
        for (tokens, _) in &mut vocab {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    println!("\nFinal vocabulary after {} merges:", num_merges);
    print_vocab(&vocab);

    // Collect all unique tokens
    let mut all_tokens: Vec<String> = Vec::new();
    for (tokens, _) in &vocab {
        for t in tokens {
            if !all_tokens.contains(t) {
                all_tokens.push(t.clone());
            }
        }
    }
    all_tokens.sort();
    println!(
        "Unique tokens ({}): [{}]\n",
        all_tokens.len(),
        all_tokens.join(", ")
    );

    // Tokenize new text
    println!("--- Tokenizing New Text ---\n");
    let test_words = vec!["lower", "newest", "showing", "unknown", "newer"];

    for word in &test_words {
        let tokens = bpe_tokenize(word, &merge_rules);
        let char_tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        println!("  '{}' -> BPE: [{}]", word, tokens.join("|"));
        println!(
            "  {}   char:  {} tokens, BPE: {} tokens, compression: {:.1}x",
            " ".repeat(word.len()),
            char_tokens.len(),
            tokens.len(),
            char_tokens.len() as f64 / tokens.len() as f64
        );
    }

    // Comparison of tokenization approaches
    println!("\n--- Tokenization Approach Comparison ---\n");

    let test_text = "the newest lower showing";
    let words_in_text: Vec<&str> = test_text.split_whitespace().collect();
    let char_count: usize = test_text.chars().filter(|c| !c.is_whitespace()).count();

    let bpe_count: usize = words_in_text
        .iter()
        .map(|w| bpe_tokenize(w, &merge_rules).len())
        .sum();

    println!("Text: \"{}\"", test_text);
    println!("  Character-level: {} tokens", char_count);
    println!("  Word-level:      {} tokens", words_in_text.len());
    println!("  BPE:             {} tokens", bpe_count);

    // Demonstrate the vocabulary size vs sequence length tradeoff
    println!("\n--- Vocabulary Size vs Sequence Length Tradeoff ---\n");
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Method", "Vocab size", "Avg tokens", "OOV handling"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Character", "~256", "~5x words", "Always"
    );
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "BPE (32k)", "32,000", "~1.3x words", "Always"
    );
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "BPE (50k)", "50,000", "~1.1x words", "Always"
    );
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Word-level", "~100,000+", "1x words", "Fails (UNK)"
    );

    // Token frequency distribution (Zipf-like)
    println!("\n--- Token Frequency Distribution ---\n");
    let mut token_freqs: HashMap<String, usize> = HashMap::new();
    for (word, freq) in &corpus {
        let tokens = bpe_tokenize(word, &merge_rules);
        for t in tokens {
            *token_freqs.entry(t).or_default() += freq;
        }
    }

    let mut freq_list: Vec<(String, usize)> = token_freqs.into_iter().collect();
    freq_list.sort_by(|a, b| b.1.cmp(&a.1));

    println!("{:>12} {:>8} {}", "Token", "Freq", "Bar");
    for (token, freq) in freq_list.iter().take(15) {
        let bar = "#".repeat(freq.min(&30));
        println!("{:>12} {:>8} {}", token, freq, bar);
    }
}

fn bpe_tokenize(word: &str, merge_rules: &[(String, String, String)]) -> Vec<String> {
    let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    for (left, right, merged) in merge_rules {
        let mut i = 0;
        while i + 1 < tokens.len() {
            if tokens[i] == *left && tokens[i + 1] == *right {
                tokens[i] = merged.clone();
                tokens.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    tokens
}

fn print_vocab(vocab: &[(Vec<String>, usize)]) {
    for (tokens, freq) in vocab {
        println!("  [{}] x{}", tokens.join("|"), freq);
    }
    println!();
}
```

---

## Key Takeaways

- Tokenization converts raw text into discrete token IDs, and the choice of scheme profoundly impacts model efficiency, coverage, and what linguistic patterns can be learned.
- BPE iteratively merges the most frequent adjacent pairs, creating a vocabulary that balances compression (common words as single tokens) with coverage (any text can be tokenized via character fallback).
- The vocabulary size vs sequence length tradeoff is fundamental: smaller vocabularies produce longer sequences (more computation) while larger vocabularies require more embedding parameters.
- Tokenization is not a neutral preprocessing step; it encodes assumptions about language structure and can create systematic biases across different languages, scripts, and text styles.
