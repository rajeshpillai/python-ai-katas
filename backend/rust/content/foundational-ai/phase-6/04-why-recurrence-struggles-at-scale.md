# Why Recurrence Struggles at Scale

> Phase 6 — Sequence Models | Kata 6.4

---

## Concept & Intuition

### What problem are we solving?

As we scale sequence models to handle longer sequences, larger datasets, and more parameters, the fundamental design of recurrent architectures becomes a critical bottleneck. The sequential dependency h_t = f(h_{t-1}, x_t) means that computing the hidden state at position t requires having already computed all hidden states at positions 1 through t-1. This creates three interrelated scaling problems that become increasingly severe as we push toward the capabilities needed for modern NLP.

First, training time scales linearly with sequence length because backpropagation through time (BPTT) must traverse the entire sequence. On modern GPUs with thousands of parallel cores, this means most hardware sits idle while the recurrent computation proceeds one step at a time. Second, memory requirements grow linearly with sequence length because BPTT requires storing all intermediate hidden states. For a 10,000-token sequence with a 1024-dimensional hidden state, that is 40 MB of activations per layer per training example. Third, the effective information bandwidth of the hidden state is fixed: no matter how long the sequence, the entire history must be compressed into a single fixed-size vector. As sequences grow longer, this bottleneck becomes increasingly severe.

These scaling limitations are not merely engineering inconveniences; they represent fundamental architectural constraints. The transformer architecture, introduced in 2017, addressed all three by replacing recurrence with self-attention, enabling parallel computation across all positions while allowing each position to directly access any other position regardless of distance.

### Why naive approaches fail

Increasing the hidden state size helps with information capacity but worsens the computational bottleneck. Doubling the hidden size from 512 to 1024 quadruples the per-step computation (matrix multiplications scale as hidden_size^2) while doing nothing to enable parallelism.

Truncated BPTT (limiting the number of time steps we backpropagate through) reduces memory requirements but fundamentally caps the range of learnable dependencies. If you truncate at 256 steps, the model cannot learn any dependency longer than 256 tokens, regardless of the actual sequence length.

Processing sequences in segments and passing the final hidden state as initialization for the next segment preserves some continuity but still funnels all information through a fixed-size bottleneck at the segment boundary. The model has no way to selectively retrieve specific information from earlier segments.

### Mental models

- **Single-threaded program**: An RNN is like a single-threaded program that must execute instructions one at a time. A transformer is like a parallel program where thousands of threads process different positions simultaneously.
- **Bottleneck hourglass**: The hidden state is the narrow neck of an hourglass. All information from the past must flow through this small opening. As the sand (tokens) piles up, more and more information is lost.
- **Telephone game**: Recurrence is like the telephone game. A message passed through 100 people degrades predictably. Attention is like broadcasting to everyone simultaneously.

### Visual explanations

```
  Sequential vs Parallel computation:

  RNN (sequential):
  Step: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
  Time: ========================================  (8 sequential steps)
  GPU utilization: [##                          ]  (one core busy)

  Transformer (parallel):
  Step: 1  2  3  4  5  6  7  8  (all at once!)
  Time: ====                                      (1 parallel step)
  GPU utilization: [############################]  (all cores busy)

  Memory bottleneck:

  Sequence: [The] [treaty] [signed] [in] [1648] ... (1000 words) ... [ended] [the] [war]
                                                      |
                                                    h_1000 must encode ALL of this
                                                    in a fixed-size vector!

  Information capacity vs sequence length:

  Hidden dim = 512 (fixed)

  Seq len  10: 512 dims / 10 tokens  = 51.2 dims per token (comfortable)
  Seq len 100: 512 dims / 100 tokens = 5.1 dims per token (compressed)
  Seq len 1K:  512 dims / 1000       = 0.5 dims per token (extreme loss)
```

---

## Hands-on Exploration

1. Measure computation time scaling of RNN forward pass with sequence length.
2. Demonstrate the information bottleneck by encoding sequences of increasing length into a fixed hidden state.
3. Show that retrieval accuracy for early tokens degrades as sequence length grows.
4. Compare sequential vs parallel computation to illustrate the GPU utilization problem.

---

## Live Code

```rust
use std::time::Instant;

fn main() {
    println!("=== Why Recurrence Struggles at Scale ===\n");

    let input_size = 8;
    let hidden_size = 16;

    let w_h = init_matrix(hidden_size, hidden_size, 0.2, 1);
    let w_x = init_matrix(hidden_size, input_size, 0.3, 2);
    let bias = vec![0.0; hidden_size];

    // 1. Computation time scaling
    println!("--- Computation Time vs Sequence Length ---\n");
    let lengths = [50, 100, 200, 500, 1000, 2000, 5000];

    for &seq_len in &lengths {
        let sequence: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                (0..input_size)
                    .map(|i| ((t * input_size + i) as f64 * 0.7).sin() * 0.5)
                    .collect()
            })
            .collect();

        let start = Instant::now();
        let mut h = vec![0.0; hidden_size];
        for x in &sequence {
            h = rnn_step(&h, x, &w_h, &w_x, &bias);
        }
        let elapsed = start.elapsed();

        println!(
            "  Seq len {:>5}: {:>8.1} us  ({:.1} us/step)",
            seq_len,
            elapsed.as_micros() as f64,
            elapsed.as_micros() as f64 / seq_len as f64
        );
    }

    // 2. Information bottleneck experiment
    println!("\n--- Information Bottleneck ---\n");
    println!("Encoding unique tokens, then measuring how well we can");
    println!("distinguish which tokens were in the sequence.\n");

    let num_unique_tokens = 20;
    let tokens: Vec<Vec<f64>> = (0..num_unique_tokens)
        .map(|t| {
            let mut v = vec![0.0; input_size];
            v[t % input_size] = 1.0;
            v[(t + 1) % input_size] = 0.5;
            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Encode increasing prefixes and check if we can identify the first token
    let test_lengths = [2, 5, 10, 15, 20];
    let first_token = &tokens[0];

    for &len in &test_lengths {
        let mut h = vec![0.0; hidden_size];

        for t in 0..len {
            h = rnn_step(&h, &tokens[t], &w_h, &w_x, &bias);
        }

        // Check: which original token is the hidden state most similar to?
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_idx = 0;
        let mut first_sim = 0.0;
        for (i, tok) in tokens.iter().enumerate().take(len) {
            // Project hidden state toward token space using W_x transpose (approximation)
            let sim = cosine_similarity(&h, tok);
            if i == 0 {
                first_sim = sim;
            }
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        println!(
            "  Encoded {} tokens: first_token_sim={:.4}, most_similar=token_{} (sim={:.4})",
            len, first_sim, best_idx, best_sim
        );
    }
    println!("\n  As more tokens are encoded, similarity to the first token decreases.");
    println!("  The hidden state is dominated by recent inputs.\n");

    // 3. Selective retrieval failure
    println!("--- Selective Retrieval Failure ---\n");
    println!("An RNN cannot selectively retrieve a specific past token.");
    println!("It can only forward its running summary.\n");

    // Encode a sequence, then try to answer "what was token at position K?"
    let seq_len = 15;
    let mut h = vec![0.0; hidden_size];
    let mut hidden_states = Vec::new();

    for t in 0..seq_len {
        h = rnn_step(&h, &tokens[t % num_unique_tokens], &w_h, &w_x, &bias);
        hidden_states.push(h.clone());
    }

    // From the final hidden state, try to recover each position's token
    let final_h = &hidden_states[seq_len - 1];
    println!("  From final hidden state, similarity to token at each position:");
    for pos in [0, 3, 7, 10, 14] {
        let original = &tokens[pos % num_unique_tokens];
        let sim = cosine_similarity(final_h, original);
        let dist = seq_len - 1 - pos;
        println!(
            "    Position {:>2} (distance {:>2} from end): sim = {:.4}",
            pos, dist, sim
        );
    }
    println!("\n  Recent tokens have stronger representation than distant ones.");

    // 4. Parallel computation comparison
    println!("--- Parallelization Comparison ---\n");
    println!("{:>10} {:>15} {:>15} {:>10}", "Seq len", "RNN (serial)", "Attn (parallel)", "Speedup");
    println!("{}", "-".repeat(55));
    let gpu_cores = 1024;
    for &len in &[64, 256, 1024, 4096] {
        let rnn_steps = len as u64; // sequential
        let attn_work = (len * len) as u64; // total attention computation
        let attn_parallel_steps = attn_work / gpu_cores as u64 + 1; // parallelized

        let speedup = rnn_steps as f64 / attn_parallel_steps as f64;
        println!(
            "{:>10} {:>15} {:>15} {:>9.1}x",
            len, rnn_steps, attn_parallel_steps, speedup
        );
    }
    println!("\n  Attention has more total work (O(n^2)) but can be parallelized.");
    println!("  On GPUs with {} cores, it often finishes faster than sequential RNN.", gpu_cores);

    // 5. Memory requirements
    println!("\n--- BPTT Memory Requirements ---\n");
    let hidden_dims = [256, 512, 1024];
    let seq_lens = [128, 512, 2048, 8192];
    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "Seq len", "h=256", "h=512", "h=1024"
    );
    for &sl in &seq_lens {
        let mems: Vec<String> = hidden_dims
            .iter()
            .map(|&hd| {
                let bytes = sl * hd * 4; // f32
                if bytes > 1_000_000 {
                    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
                } else {
                    format!("{:.0} KB", bytes as f64 / 1_000.0)
                }
            })
            .collect();
        println!("{:>8} {:>10} {:>10} {:>10}", sl, mems[0], mems[1], mems[2]);
    }
    println!("\n  Memory scales as O(seq_len * hidden_size) per layer.");
    println!("  With 24 layers and batch size 32, multiply accordingly.");
}

fn rnn_step(h_prev: &[f64], x: &[f64], w_h: &[Vec<f64>], w_x: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    let n = h_prev.len();
    let mut h_new = vec![0.0; n];
    for i in 0..n {
        let mut sum = bias[i];
        for j in 0..h_prev.len() { sum += w_h[i][j] * h_prev[j]; }
        for j in 0..x.len() { sum += w_x[i][j] * x[j]; }
        h_new[i] = sum.tanh();
    }
    h_new
}

fn init_matrix(rows: usize, cols: usize, scale: f64, seed: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|i| {
        (0..cols).map(|j| {
            ((i * cols + j + seed * 100) as f64 * 1.618).sin() * scale
        }).collect()
    }).collect()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}
```

---

## Key Takeaways

- Recurrent processing is inherently sequential: computing h_t requires h_{t-1}, preventing any parallelization across time steps and leaving most GPU cores idle.
- The fixed-size hidden state creates an information bottleneck that becomes more severe as sequence length increases, with recent tokens dominating and early tokens being forgotten.
- Memory for backpropagation through time scales linearly with sequence length, and truncating BPTT caps the maximum learnable dependency range.
- These scaling limitations directly motivated the development of the transformer architecture, which replaces sequential processing with parallelizable self-attention, allowing direct connections between any two positions.
