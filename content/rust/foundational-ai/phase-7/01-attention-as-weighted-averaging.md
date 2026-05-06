# Attention as Weighted Averaging

> Phase 7 — Attention & Transformers | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

The attention mechanism is arguably the most important innovation in modern deep learning. At its core, it is surprisingly simple: attention is weighted averaging where the weights are computed dynamically based on the content of the inputs rather than being fixed or learned as parameters. Given a query (what we are looking for), a set of keys (what is available), and corresponding values (the information we want to retrieve), attention computes a similarity between the query and each key, normalizes these similarities into weights using softmax, and returns the weighted sum of the values.

This solves the fundamental bottleneck of recurrent networks. Instead of forcing all information to flow through a sequential chain of hidden states, attention allows any position to directly access information from any other position. A word at position 100 can directly "look at" a word at position 1, without the information degrading through 99 intermediate steps. The computational cost is O(n^2) in sequence length (since every position attends to every other position), but this computation is fully parallelizable.

The key insight is that the attention weights are not learned parameters but are computed on the fly from the input data. This means the network can dynamically decide which information to focus on, adapting its information routing based on the actual content rather than relying on fixed connectivity patterns. This dynamic routing is what gives attention its extraordinary flexibility and power.

### Why naive approaches fail

A fixed weighted average (averaging all positions equally) discards the information about which positions are relevant to the current query. If you are trying to resolve the pronoun "it" in a sentence, equal weighting of all words is useless; you need to specifically focus on the antecedent.

Using a learned attention matrix (fixed weights for each position pair) fails because the relevant positions change depending on the input content. In "The cat sat on the mat because it was tired," "it" refers to "cat." But in "The cat sat on the mat because it was dirty," "it" refers to "mat." The attention pattern must change based on the semantic content of the sentence, which requires content-based (dynamic) attention.

### Mental models

- **Library lookup**: The query is your search question, keys are the index cards in the card catalog, and values are the books themselves. You compare your question to each index card, and the best-matching books contribute most to your answer.
- **Spotlight**: Attention shines a variable spotlight on different parts of the input. The spotlight can be broad (attending to many positions) or focused (attending to one position), and its position depends on what you are looking for.
- **Selective averaging**: Regular averaging treats all items equally. Attention is like asking "which of these items are relevant to my question?" and averaging only the relevant ones.

### Visual explanations

```
  Attention mechanism:

  Query:  "What does 'it' refer to?"
              |
              v
  Keys:   [The] [cat] [sat] [on] [the] [mat] [because] [it]
  Scores:  0.1   0.7   0.1  0.0   0.0   0.1    0.0     0.0
              |     |     |    |     |     |      |       |
              v     v     v    v     v     v      v       v
  Values: [v_0] [v_1] [v_2] [v_3] [v_4] [v_5]  [v_6]  [v_7]

  Output = 0.1*v_0 + 0.7*v_1 + 0.1*v_2 + ... + 0.0*v_7
         ≈ v_1 (the "cat" representation)

  The math:

  scores = Query . Keys^T     (dot products)
  weights = softmax(scores)   (normalize to sum to 1)
  output  = weights . Values  (weighted sum)
```

---

## Hands-on Exploration

1. Implement scaled dot-product attention from scratch.
2. Show how different queries produce different attention weight distributions.
3. Demonstrate that attention can selectively retrieve specific information from a sequence.
4. Compare attention output quality to simple averaging and RNN-style compression.

---

## Live Code

```rust
fn main() {
    println!("=== Attention as Weighted Averaging ===\n");

    // Simulate a sequence of 6 "word" embeddings (dim=4)
    // Each word has a semantic vector
    let words = ["The", "cat", "sat", "on", "the", "mat"];
    let embeddings: Vec<Vec<f64>> = vec![
        vec![0.1, 0.9, 0.0, 0.1],  // The (article)
        vec![0.8, 0.1, 0.3, 0.9],  // cat (noun, animal)
        vec![0.2, 0.1, 0.9, 0.2],  // sat (verb)
        vec![0.1, 0.5, 0.1, 0.0],  // on (preposition)
        vec![0.1, 0.9, 0.0, 0.1],  // the (article)
        vec![0.7, 0.2, 0.2, 0.8],  // mat (noun, object)
    ];

    let dim = embeddings[0].len();
    let scale = (dim as f64).sqrt();

    // Query 1: Looking for a noun/animal (similar to "cat")
    let query1 = vec![0.8, 0.0, 0.2, 0.9];
    println!("Query 1: Looking for noun/animal-like words");
    let (weights1, output1) = attention(&query1, &embeddings, &embeddings, scale);
    print_attention(&words, &weights1);
    println!("  Output: {}\n", fmt_vec(&output1));

    // Query 2: Looking for a verb (similar to "sat")
    let query2 = vec![0.2, 0.0, 0.9, 0.2];
    println!("Query 2: Looking for verb-like words");
    let (weights2, output2) = attention(&query2, &embeddings, &embeddings, scale);
    print_attention(&words, &weights2);
    println!("  Output: {}\n", fmt_vec(&output2));

    // Query 3: Looking for articles (similar to "the")
    let query3 = vec![0.1, 0.9, 0.0, 0.1];
    println!("Query 3: Looking for articles");
    let (weights3, output3) = attention(&query3, &embeddings, &embeddings, scale);
    print_attention(&words, &weights3);
    println!("  Output: {}\n", fmt_vec(&output3));

    // Compare with simple averaging
    println!("--- Comparison: Attention vs Simple Average ---\n");

    let simple_avg: Vec<f64> = (0..dim)
        .map(|d| embeddings.iter().map(|e| e[d]).sum::<f64>() / embeddings.len() as f64)
        .collect();

    // How close is each method's output to the "cat" embedding?
    let cat_emb = &embeddings[1];
    let attn_sim = cosine_similarity(&output1, cat_emb);
    let avg_sim = cosine_similarity(&simple_avg, cat_emb);

    println!("  Looking for 'cat'-like content:");
    println!("  Attention output similarity to 'cat': {:.4}", attn_sim);
    println!("  Simple average similarity to 'cat':   {:.4}", avg_sim);
    println!("  Attention is much more precise at retrieval!\n");

    // Full self-attention: every position queries every other position
    println!("--- Self-Attention Matrix ---\n");
    println!("Each position attends to all positions:");
    println!(
        "{:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "", words[0], words[1], words[2], words[3], words[4], words[5]
    );

    for (i, query) in embeddings.iter().enumerate() {
        let (weights, _) = attention(query, &embeddings, &embeddings, scale);
        let w_strs: Vec<String> = weights.iter().map(|w| format!("{:.3}", w)).collect();
        println!(
            "{:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
            words[i], w_strs[0], w_strs[1], w_strs[2], w_strs[3], w_strs[4], w_strs[5]
        );
    }

    println!("\nObserve: similar words attend strongly to each other.");
    println!("'The' and 'the' attend to each other (same type).");
    println!("'cat' and 'mat' attend to each other (both nouns).");

    // Demonstrate temperature/scaling effect
    println!("\n--- Effect of Scaling ---\n");
    for temp in &[0.5, 1.0, 2.0, 4.0] {
        let (weights, _) = attention(&query1, &embeddings, &embeddings, *temp);
        let entropy = -weights
            .iter()
            .filter(|w| **w > 1e-10)
            .map(|w| w * w.ln())
            .sum::<f64>();
        let w_strs: Vec<String> = weights.iter().map(|w| format!("{:.3}", w)).collect();
        println!(
            "  scale={:.1}: weights=[{}]  entropy={:.3}",
            temp,
            w_strs.join(", "),
            entropy
        );
    }
    println!("\n  Lower scale -> sharper attention (more focused)");
    println!("  Higher scale -> flatter attention (more uniform)");
}

fn attention(
    query: &[f64],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    // Compute scaled dot-product scores
    let scores: Vec<f64> = keys
        .iter()
        .map(|key| {
            let dot: f64 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
            dot / scale
        })
        .collect();

    // Softmax
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

    // Weighted sum of values
    let dim = values[0].len();
    let output: Vec<f64> = (0..dim)
        .map(|d| {
            weights
                .iter()
                .zip(values.iter())
                .map(|(w, v)| w * v[d])
                .sum()
        })
        .collect();

    (weights, output)
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}

fn print_attention(words: &[&str], weights: &[f64]) {
    let max_bar = 30;
    for (word, weight) in words.iter().zip(weights.iter()) {
        let bar_len = (weight * max_bar as f64) as usize;
        let bar = "#".repeat(bar_len);
        println!("  {:>6}: {:.3} {}", word, weight, bar);
    }
}

fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}
```

---

## Key Takeaways

- Attention is dynamically computed weighted averaging: the weights depend on the content of the query and keys, not on fixed positions or learned parameters.
- Different queries produce different attention distributions over the same set of keys, enabling selective information retrieval from any position.
- Scaling the dot products controls the sharpness of attention: smaller scale produces more focused (peaky) distributions while larger scale produces more uniform distributions.
- Self-attention (where queries, keys, and values all come from the same sequence) allows every position to directly attend to every other position, eliminating the sequential bottleneck of recurrence.
