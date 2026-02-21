# Self-Attention Visualization

> Phase 7 — Attention & Transformers | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

Self-attention is the specific case where the query, key, and value vectors all derive from the same input sequence. Each position generates its own query ("what am I looking for?"), key ("what do I contain?"), and value ("what information do I provide?") by linearly projecting its embedding through learned weight matrices W_Q, W_K, and W_V. The full self-attention computation produces a new representation for each position that is a weighted combination of all positions' values.

Visualizing the attention weight matrix reveals how the network routes information. In a well-trained transformer, these patterns are highly interpretable: pronouns attend to their antecedents, verbs attend to their subjects, and adjectives attend to the nouns they modify. Multi-head attention extends this by running multiple independent attention operations in parallel, each with its own W_Q, W_K, W_V matrices. Different heads can specialize in different types of relationships: one head might track syntactic dependencies while another tracks semantic similarity.

Understanding how to read and interpret self-attention patterns is crucial for debugging, interpreting, and improving transformer models. The attention matrix is one of the few windows we have into how transformers process information internally, making it an invaluable diagnostic tool.

### Why naive approaches fail

Using the raw embeddings directly as queries and keys (without the linear projections) severely limits what relationships can be captured. The projections allow the model to create specialized representations for "what to look for" versus "what to advertise" versus "what to provide," which is far more flexible than using the same vector for all three purposes.

A single attention head can only produce one attention pattern per layer. Some positions might need to attend to both their syntactic governor and a semantically related word simultaneously. Without multiple heads, the single attention pattern must compromise, averaging between these different needs. Multi-head attention solves this by giving each head independent parameters.

### Mental models

- **Multiple reading glasses**: Each attention head is like a pair of reading glasses with different colored lenses. One reveals syntactic structure, another reveals coreference chains, another reveals semantic similarity. Together, they give a complete picture.
- **Conference call**: Each position announces its key ("I'm a noun referring to an animal"), broadcasts its value ("here's my full representation"), and makes queries ("I need to find the verb associated with me"). The attention mechanism routes information based on key-query matches.
- **Heatmap of relevance**: The attention matrix is a heatmap where cell (i,j) shows how much position i relies on position j for its updated representation. Bright spots reveal the information flow pattern.

### Visual explanations

```
  Self-attention with Q, K, V projections:

  Input:  [x_0, x_1, x_2, x_3]
              |     |     |     |
           ┌──┴──┐──┴──┐──┴──┐──┴──┐
           │ W_Q │ W_K │ W_V │     │  (shared across positions)
           └──┬──┘──┬──┘──┬──┘     │
              v     v     v
  Queries: [q_0, q_1, q_2, q_3]
  Keys:    [k_0, k_1, k_2, k_3]
  Values:  [v_0, v_1, v_2, v_3]

  Attention matrix:
           k_0   k_1   k_2   k_3
  q_0  [ 0.05  0.80  0.10  0.05 ]   pos 0 attends mostly to pos 1
  q_1  [ 0.70  0.10  0.15  0.05 ]   pos 1 attends mostly to pos 0
  q_2  [ 0.10  0.10  0.10  0.70 ]   pos 2 attends mostly to pos 3
  q_3  [ 0.05  0.05  0.70  0.20 ]   pos 3 attends mostly to pos 2

  Multi-head attention:
  Head 1 (syntax):     Head 2 (semantics):  Combined:
  subj->verb           noun->noun           Both patterns
  [.1 .8 .1 .0]       [.1 .1 .1 .7]       captured!
  [.7 .1 .1 .1]       [.1 .1 .1 .7]
  ...                  ...
```

---

## Hands-on Exploration

1. Implement the full Q, K, V projection pipeline for self-attention.
2. Visualize the attention matrix as an ASCII heatmap.
3. Implement multi-head attention and observe how different heads learn different patterns.
4. Show how the output representation at each position changes after self-attention.

---

## Live Code

```rust
fn main() {
    println!("=== Self-Attention Visualization ===\n");

    let words = ["The", "cat", "chased", "the", "mouse", "quickly"];
    let seq_len = words.len();
    let embed_dim = 8;
    let head_dim = 4;
    let num_heads = 2;

    // Create embeddings with semantic structure
    let embeddings: Vec<Vec<f64>> = vec![
        vec![0.1, 0.8, 0.0, 0.1, 0.2, 0.1, 0.0, 0.3],  // The (article)
        vec![0.9, 0.1, 0.3, 0.8, 0.1, 0.7, 0.2, 0.1],  // cat (noun, agent)
        vec![0.2, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.1],  // chased (verb)
        vec![0.1, 0.8, 0.0, 0.1, 0.2, 0.1, 0.0, 0.3],  // the (article)
        vec![0.8, 0.2, 0.2, 0.9, 0.1, 0.6, 0.3, 0.2],  // mouse (noun, patient)
        vec![0.1, 0.1, 0.7, 0.1, 0.6, 0.0, 0.8, 0.0],  // quickly (adverb)
    ];

    // Initialize W_Q, W_K, W_V for each head
    let heads: Vec<AttentionHead> = (0..num_heads)
        .map(|h| AttentionHead {
            w_q: init_matrix(head_dim, embed_dim, 0.4, h * 3),
            w_k: init_matrix(head_dim, embed_dim, 0.4, h * 3 + 1),
            w_v: init_matrix(head_dim, embed_dim, 0.4, h * 3 + 2),
        })
        .collect();

    let w_o = init_matrix(embed_dim, head_dim * num_heads, 0.3, 100);

    for (h_idx, head) in heads.iter().enumerate() {
        println!("=== Head {} ===\n", h_idx + 1);

        // Project to Q, K, V
        let queries: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_q, e)).collect();
        let keys: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_k, e)).collect();
        let values: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_v, e)).collect();

        // Compute attention matrix
        let scale = (head_dim as f64).sqrt();
        let mut attn_matrix = vec![vec![0.0; seq_len]; seq_len];

        for i in 0..seq_len {
            let scores: Vec<f64> = keys
                .iter()
                .map(|k| dot(&queries[i], k) / scale)
                .collect();
            let weights = softmax(&scores);
            attn_matrix[i] = weights;
        }

        // Print attention matrix
        print!("{:>10}", "");
        for w in &words {
            print!("{:>8}", w);
        }
        println!();

        for (i, row) in attn_matrix.iter().enumerate() {
            print!("{:>10}", words[i]);
            for w in row {
                print!("{:>8.3}", w);
            }
            println!();
        }
        println!();

        // ASCII heatmap
        println!("  Heatmap (darker = stronger attention):");
        print!("{:>10}", "");
        for w in &words {
            print!(" {:>3}", &w[..w.len().min(3)]);
        }
        println!();

        for (i, row) in attn_matrix.iter().enumerate() {
            print!("{:>10}", words[i]);
            for w in row {
                let symbol = if *w > 0.4 {
                    " ###"
                } else if *w > 0.25 {
                    " ==="
                } else if *w > 0.15 {
                    " ..."
                } else {
                    "    "
                };
                print!("{}", symbol);
            }
            println!();
        }
        println!();

        // Analyze what each position attends to most
        println!("  Strongest attention for each position:");
        for (i, row) in attn_matrix.iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            println!(
                "    '{}' attends most to '{}' (weight={:.3})",
                words[i], words[max_idx], row[max_idx]
            );
        }
        println!();
    }

    // Multi-head combination
    println!("=== Multi-Head Output ===\n");

    // Compute outputs from all heads
    let mut all_head_outputs: Vec<Vec<Vec<f64>>> = Vec::new();

    for head in &heads {
        let queries: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_q, e)).collect();
        let keys: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_k, e)).collect();
        let values: Vec<Vec<f64>> = embeddings.iter().map(|e| mat_vec(&head.w_v, e)).collect();

        let scale = (head_dim as f64).sqrt();
        let mut head_output = vec![vec![0.0; head_dim]; seq_len];

        for i in 0..seq_len {
            let scores: Vec<f64> = keys.iter().map(|k| dot(&queries[i], k) / scale).collect();
            let weights = softmax(&scores);
            for d in 0..head_dim {
                head_output[i][d] = weights
                    .iter()
                    .zip(values.iter())
                    .map(|(w, v)| w * v[d])
                    .sum();
            }
        }
        all_head_outputs.push(head_output);
    }

    // Concatenate heads and project
    println!("Concatenation + output projection:");
    for (i, word) in words.iter().enumerate() {
        let mut concat = Vec::new();
        for head_out in &all_head_outputs {
            concat.extend_from_slice(&head_out[i]);
        }
        let final_out = mat_vec(&w_o, &concat);

        // Compare to original embedding
        let sim = cosine_similarity(&embeddings[i], &final_out);
        println!(
            "  '{}': original_sim={:.3} (new repr incorporates context from attended positions)",
            word, sim
        );
    }

    println!("\nAfter self-attention, each position's representation");
    println!("is enriched with information from related positions.");
    println!("'cat' now carries verb and subject context.");
    println!("'chased' now carries agent and patient context.");
}

struct AttentionHead {
    w_q: Vec<Vec<f64>>,
    w_k: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
}

fn mat_vec(matrix: &[Vec<f64>], vec_in: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vec_in.iter()).map(|(w, x)| w * x).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn init_matrix(rows: usize, cols: usize, scale: f64, seed: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| ((i * cols + j + seed * 77) as f64 * 1.618).sin() * scale)
                .collect()
        })
        .collect()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 1e-10 && nb > 1e-10 { d / (na * nb) } else { 0.0 }
}
```

---

## Key Takeaways

- Self-attention uses learned W_Q, W_K, W_V projections to create specialized query, key, and value representations from the same input, decoupling "what to look for" from "what to advertise" from "what to provide."
- The attention matrix reveals information flow patterns: which positions each word relies on for its updated representation.
- Multi-head attention runs multiple independent attention operations in parallel, allowing different heads to capture different types of relationships (syntactic, semantic, positional).
- After self-attention, each position's representation is enriched with context from all positions it attends to, enabling the model to build context-aware representations in a single parallel step.
