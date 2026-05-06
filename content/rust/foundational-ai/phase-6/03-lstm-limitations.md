# LSTM Limitations

> Phase 6 — Sequence Models | Kata 6.3

---

## Concept & Intuition

### What problem are we solving?

Long Short-Term Memory (LSTM) networks were designed to address the vanishing gradient problem of vanilla RNNs. They introduce a cell state that acts as an information highway, plus three gates (forget, input, output) that control what information to retain, add, or expose at each step. The cell state can carry information across many time steps with minimal transformation, allowing gradients to flow more freely during backpropagation.

However, LSTMs have significant limitations that motivate the move to attention-based architectures. While they alleviate the vanishing gradient problem, they do not eliminate it entirely. The cell state is still updated sequentially, meaning information must pass through every intermediate step to travel from position 1 to position 100. Each gate introduces a multiplicative factor that, over many steps, still causes degradation. In practice, LSTMs struggle with dependencies beyond a few hundred tokens.

More fundamentally, LSTMs process tokens strictly sequentially. Each hidden state depends on the previous one, making parallelization across time steps impossible. This sequential bottleneck limits training speed on modern GPUs, which excel at parallel computation. As datasets and model sizes have grown, this inability to parallelize has become the dominant practical limitation of recurrent architectures.

### Why naive approaches fail

Simply making the cell state larger does not fix the fundamental sequential bottleneck. A 2048-dimensional cell state is still updated one step at a time, and the information capacity is still limited by how many multiplicative gate operations the signal must pass through.

Stacking multiple LSTM layers (deep LSTMs) helps capture hierarchical features but does not address the sequential processing constraint. Each layer still processes the sequence left-to-right, and the total computation time scales linearly with sequence length times the number of layers.

Bidirectional LSTMs process the sequence in both directions and concatenate the hidden states, which helps capture both past and future context. But they still cannot directly connect distant positions and require twice the computation.

### Mental models

- **Leaky pipe**: The cell state is a pipe carrying water (information). The gates are valves that control flow. Even with good valves, over a very long pipe, friction causes loss. This is why LSTMs still struggle with very long sequences.
- **Single-lane highway**: Information must travel sequentially through every position. There is no way to "skip ahead" from position 1 to position 100. This is the fundamental limitation that attention solves.
- **Sequential assembly line**: Each worker (time step) must finish before the next can start. You cannot add more workers to speed up a single sequence, only process multiple sequences simultaneously.

### Visual explanations

```
  LSTM Cell:

  x_t ──────────────────────────┐
                                 │
  h_{t-1} ──┐                   │
             ├──[forget gate]──┐ │
             ├──[input gate]───┤ │
             ├──[output gate]──┤ │
             │                  │ │
  c_{t-1} ──┼────( x forget )──┘ │
             │    ( + input*g )───┘
             │         │
             │     c_t ─────── cell state (information highway)
             │         │
             └────( x output )
                       │
                   h_t ─────── hidden state (exposed output)

  The sequential bottleneck:

  Position:  1 ──> 2 ──> 3 ──> ... ──> 98 ──> 99 ──> 100
             │     │     │              │      │       │
  Cell:      c1 -> c2 -> c3 -> ... -> c98 -> c99 -> c100

  To connect pos 1 to pos 100: signal passes through 99 gates!
  Even if each gate passes 99% of the signal:
  0.99^99 = 0.37 (63% of information lost!)
```

---

## Hands-on Exploration

1. Implement a simplified LSTM cell with forget, input, and output gates.
2. Demonstrate how the cell state preserves information better than a vanilla RNN.
3. Show that even LSTM cell state degrades over very long sequences.
4. Measure the sequential processing time and show it scales linearly with sequence length.

---

## Live Code

```rust
fn main() {
    println!("=== LSTM Limitations ===\n");

    let input_size = 3;
    let hidden_size = 4;

    // Initialize LSTM weights
    let w_f = init_matrix(hidden_size, input_size + hidden_size, 0.3, 1);
    let b_f = vec![1.0; hidden_size]; // Bias forget gate high to start (common practice)
    let w_i = init_matrix(hidden_size, input_size + hidden_size, 0.3, 2);
    let b_i = vec![0.0; hidden_size];
    let w_c = init_matrix(hidden_size, input_size + hidden_size, 0.3, 3);
    let b_c = vec![0.0; hidden_size];
    let w_o = init_matrix(hidden_size, input_size + hidden_size, 0.3, 4);
    let b_o = vec![0.0; hidden_size];

    let lstm = LSTMWeights {
        w_f, b_f, w_i, b_i, w_c, b_c, w_o, b_o,
        hidden_size, input_size,
    };

    // Comparison: Store a signal and try to retrieve it after N steps

    println!("--- Information Retention: LSTM vs Vanilla RNN ---\n");

    // Store a specific signal in step 0
    let signal_input = vec![1.0, 0.5, -0.3];
    let blank_input = vec![0.0, 0.0, 0.0];

    // LSTM retention test
    let mut h = vec![0.0; hidden_size];
    let mut c = vec![0.0; hidden_size];

    // Inject signal
    let (h_new, c_new) = lstm_step(&h, &c, &signal_input, &lstm);
    h = h_new;
    c = c_new;
    let h_after_signal = h.clone();
    let c_after_signal = c.clone();

    println!("LSTM cell state after signal:  {}", fmt_vec(&c));
    println!("LSTM hidden state after signal: {}", fmt_vec(&h));

    // Run blank inputs for many steps
    println!("\nLSTM retention over blank steps:");
    for step in 1..=50 {
        let (h_new, c_new) = lstm_step(&h, &c, &blank_input, &lstm);
        h = h_new;
        c = c_new;

        if step <= 5 || step % 10 == 0 {
            let c_sim = cosine_similarity(&c_after_signal, &c);
            let h_sim = cosine_similarity(&h_after_signal, &h);
            println!(
                "  Step {:>3}: cell_sim={:.4}, hidden_sim={:.4}",
                step, c_sim, h_sim
            );
        }
    }

    // Vanilla RNN retention test for comparison
    let w_h_rnn = init_matrix(hidden_size, hidden_size, 0.3, 5);
    let w_x_rnn = init_matrix(hidden_size, input_size, 0.5, 6);
    let b_rnn = vec![0.0; hidden_size];

    let mut h_rnn = vec![0.0; hidden_size];
    h_rnn = rnn_step(&h_rnn, &signal_input, &w_h_rnn, &w_x_rnn, &b_rnn);
    let h_rnn_after_signal = h_rnn.clone();

    println!("\nVanilla RNN retention over blank steps:");
    for step in 1..=50 {
        h_rnn = rnn_step(&h_rnn, &blank_input, &w_h_rnn, &w_x_rnn, &b_rnn);

        if step <= 5 || step % 10 == 0 {
            let sim = cosine_similarity(&h_rnn_after_signal, &h_rnn);
            println!("  Step {:>3}: hidden_sim={:.4}", step, sim);
        }
    }

    // Gate analysis: signal degradation through forget gates
    println!("\n--- Forget Gate Degradation Analysis ---\n");
    let forget_rates = [0.99, 0.95, 0.90, 0.80];
    println!("Signal retained after N steps with constant forget gate value:");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}", "Steps", "f=0.99", "f=0.95", "f=0.90", "f=0.80");
    for steps in [10, 50, 100, 200, 500, 1000] {
        let retentions: Vec<String> = forget_rates
            .iter()
            .map(|f| format!("{:.6}", f.powi(steps)))
            .collect();
        println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
            steps, retentions[0], retentions[1], retentions[2], retentions[3]);
    }

    // Sequential bottleneck demonstration
    println!("\n--- Sequential Processing Bottleneck ---\n");
    let seq_lengths = [10, 50, 100, 500, 1000];
    println!("Operations required (sequential, cannot parallelize):");
    for len in &seq_lengths {
        let ops = len * (4 * hidden_size * (input_size + hidden_size)); // 4 gates
        println!("  Length {:>5}: {:>10} ops (must be sequential)", len, ops);
    }
    println!("\nTransformer self-attention for comparison (parallelizable):");
    for len in &seq_lengths {
        let ops = len * len * hidden_size; // attention is O(n^2 * d) but PARALLEL
        println!(
            "  Length {:>5}: {:>10} ops (but fully parallelizable across {} positions)",
            len, ops, len
        );
    }
}

struct LSTMWeights {
    w_f: Vec<Vec<f64>>, b_f: Vec<f64>,
    w_i: Vec<Vec<f64>>, b_i: Vec<f64>,
    w_c: Vec<Vec<f64>>, b_c: Vec<f64>,
    w_o: Vec<Vec<f64>>, b_o: Vec<f64>,
    hidden_size: usize,
    input_size: usize,
}

fn lstm_step(
    h_prev: &[f64], c_prev: &[f64], x: &[f64], w: &LSTMWeights,
) -> (Vec<f64>, Vec<f64>) {
    // Concatenate h_prev and x
    let mut combined = Vec::with_capacity(w.hidden_size + w.input_size);
    combined.extend_from_slice(h_prev);
    combined.extend_from_slice(x);

    let f_gate = sigmoid_vec(&affine(&combined, &w.w_f, &w.b_f));
    let i_gate = sigmoid_vec(&affine(&combined, &w.w_i, &w.b_i));
    let c_tilde = tanh_vec(&affine(&combined, &w.w_c, &w.b_c));
    let o_gate = sigmoid_vec(&affine(&combined, &w.w_o, &w.b_o));

    // New cell state: c_t = f * c_prev + i * c_tilde
    let c_new: Vec<f64> = c_prev.iter().zip(f_gate.iter()).zip(i_gate.iter().zip(c_tilde.iter()))
        .map(|((c, f), (i, ct))| f * c + i * ct)
        .collect();

    // New hidden state: h_t = o * tanh(c_t)
    let h_new: Vec<f64> = o_gate.iter().zip(c_new.iter())
        .map(|(o, c)| o * c.tanh())
        .collect();

    (h_new, c_new)
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

fn affine(input: &[f64], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    weights.iter().zip(bias.iter()).map(|(w_row, b)| {
        w_row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f64>() + b
    }).collect()
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn sigmoid_vec(v: &[f64]) -> Vec<f64> { v.iter().map(|x| sigmoid(*x)).collect() }
fn tanh_vec(v: &[f64]) -> Vec<f64> { v.iter().map(|x| x.tanh()).collect() }

fn init_matrix(rows: usize, cols: usize, scale: f64, seed: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|i| {
        (0..cols).map(|j| {
            let s = ((i * cols + j + seed * 100) as f64 * 1.618).sin() * scale;
            s
        }).collect()
    }).collect()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}

fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:+.4}", x)).collect();
    format!("[{}]", parts.join(", "))
}
```

---

## Key Takeaways

- LSTMs use gates and a cell state highway to preserve information better than vanilla RNNs, but the forget gate still causes exponential signal decay over long sequences.
- Even with a forget gate of 0.99, only 37% of a signal survives after 100 steps, making reliable long-range dependency learning difficult.
- The strict sequential processing of LSTMs creates a computational bottleneck that prevents parallelization across time steps, limiting training speed on modern hardware.
- These limitations -- residual vanishing gradients and sequential processing -- directly motivated the development of attention mechanisms and transformer architectures.
