# RNN Intuition

> Phase 6 — Sequence Models | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

Recurrent Neural Networks (RNNs) solve the fixed-context limitation of N-grams by maintaining a hidden state that is updated at each time step. Instead of conditioning on the last N-1 words, an RNN theoretically conditions on the entire history of the sequence. At each step, the RNN reads one input, combines it with its current hidden state, and produces a new hidden state. This hidden state is a compressed summary of everything the network has seen so far.

The recurrence relation is elegant: h_t = f(W_h * h_{t-1} + W_x * x_t + b), where h_t is the hidden state at time t, x_t is the input at time t, and f is a nonlinear activation (typically tanh). The same weight matrices W_h and W_x are shared across all time steps, which means the model has a fixed parameter count regardless of sequence length. This parameter sharing is analogous to how convolutional filters share weights across spatial positions.

RNNs bridge the gap between N-gram counting and modern neural language models. They demonstrate that neural networks can process sequences of variable length, learn distributed representations of context, and generate text one token at a time. However, they also introduce the vanishing gradient problem that motivates LSTMs, GRUs, and eventually transformers.

### Why naive approaches fail

Using a fixed-size window (like N-grams) means you can never capture dependencies longer than N. The sentence "The cat that the dog chased ran away" requires connecting "cat" to "ran" across an intervening clause. No finite N is guaranteed sufficient for all such dependencies.

Using a simple bag-of-words approach (treating all previous words as an unordered set) destroys sequential information. "Dog bites man" and "Man bites dog" have the same bag-of-words representation despite opposite meanings. RNNs preserve order because the hidden state at each position depends on the specific sequence of previous updates.

### Mental models

- **Running summary**: The hidden state is like a running summary you update as you read a book. After each sentence, you revise your mental model of the story. You do not remember every word, but you carry forward the essential context.
- **Conveyor belt**: Inputs arrive one at a time on a conveyor belt. At each station, a worker (the recurrent cell) reads the new item, checks their notes from the previous item, and writes updated notes for the next station.
- **Feedback loop**: The output of the network feeds back as input to itself, creating a loop that allows information to persist across time steps.

### Visual explanations

```
  RNN unrolled through time:

  x_0         x_1         x_2         x_3
   |           |           |           |
   v           v           v           v
  [RNN] ---> [RNN] ---> [RNN] ---> [RNN] ---> ...
  h_0   h_1  h_1   h_2  h_2   h_3  h_3   h_4
   |           |           |           |
   v           v           v           v
  y_0         y_1         y_2         y_3

  Same weights (W_h, W_x) at every step!

  Hidden state as compressed history:

  h_0 = f(W_x * "The")
  h_1 = f(W_h * h_0 + W_x * "cat")       // knows "The cat"
  h_2 = f(W_h * h_1 + W_x * "sat")       // knows "The cat sat"
  h_3 = f(W_h * h_2 + W_x * "on")        // knows "The cat sat on"
  ...

  Each h_t is a fixed-size vector summarizing the entire history.
```

---

## Hands-on Exploration

1. Implement a minimal RNN cell with tanh activation.
2. Process a sequence of input vectors step by step, observing hidden state evolution.
3. Show how the hidden state carries information from earlier inputs.
4. Demonstrate the vanishing gradient problem by tracking gradient magnitudes through many steps.

---

## Live Code

```rust
fn main() {
    println!("=== RNN Intuition ===\n");

    // Hyperparameters
    let input_size = 4;
    let hidden_size = 3;

    // Initialize weights with deterministic pseudo-random values
    let w_h = init_matrix(hidden_size, hidden_size, 0.3);
    let w_x = init_matrix(hidden_size, input_size, 0.5);
    let bias = vec![0.0; hidden_size];

    // Create a sequence of 6 input vectors (like word embeddings)
    let sequence: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0, 0.0], // "The"
        vec![0.0, 1.0, 0.0, 0.0], // "cat"
        vec![0.0, 0.0, 1.0, 0.0], // "sat"
        vec![0.0, 0.0, 0.0, 1.0], // "on"
        vec![1.0, 0.0, 0.0, 0.0], // "the"
        vec![0.0, 0.0, 1.0, 1.0], // "mat"
    ];
    let labels = ["The", "cat", "sat", "on", "the", "mat"];

    println!("Processing sequence: {}\n", labels.join(" -> "));

    // Forward pass
    let mut h = vec![0.0_f64; hidden_size]; // initial hidden state
    let mut hidden_states = Vec::new();

    for (t, (x, label)) in sequence.iter().zip(labels.iter()).enumerate() {
        h = rnn_step(&h, x, &w_h, &w_x, &bias);
        hidden_states.push(h.clone());

        let h_str: Vec<String> = h.iter().map(|v| format!("{:+.4}", v)).collect();
        println!("  t={} ({:>4}): h = [{}]", t, label, h_str.join(", "));
    }

    // Show that hidden state carries history
    println!("\n--- Hidden State Carries History ---\n");
    println!("Processing the SAME word 'the' at different positions:");
    let h_at_0: Vec<String> = hidden_states[0].iter().map(|v| format!("{:+.4}", v)).collect();
    let h_at_4: Vec<String> = hidden_states[4].iter().map(|v| format!("{:+.4}", v)).collect();
    println!("  h after 'The' (position 0): [{}]", h_at_0.join(", "));
    println!("  h after 'the' (position 4): [{}]", h_at_4.join(", "));
    println!("  Different! Because position 4 carries context of 'The cat sat on'.");

    // Demonstrate vanishing gradients
    println!("\n--- Vanishing Gradient Problem ---\n");
    println!("Tracking how gradients shrink through time steps:");

    // The gradient of h_t with respect to h_0 involves multiplying
    // the Jacobian (derivative of tanh) times W_h at each step.
    // For tanh, the derivative is (1 - tanh(x)^2), which is <= 1.
    // If the spectral norm of W_h < 1, gradients vanish exponentially.

    let mut grad_magnitude = 1.0_f64;
    let spectral_norm_approx = matrix_spectral_norm_approx(&w_h);
    println!("  Approximate spectral norm of W_h: {:.4}", spectral_norm_approx);
    println!();

    for t in 0..20 {
        // Each step multiplies by approximately spectral_norm * tanh_derivative
        // tanh derivative for typical hidden values is around 0.3-0.8
        let tanh_deriv = 0.5; // average for moderate activations
        grad_magnitude *= spectral_norm_approx * tanh_deriv;

        if t % 4 == 0 || t == 19 {
            let bar_len = (grad_magnitude * 50.0).min(50.0).max(0.0) as usize;
            let bar = "#".repeat(bar_len);
            println!(
                "  t={:>2}: gradient magnitude = {:.8}  {}",
                t, grad_magnitude, bar
            );
        }
    }

    println!(
        "\n  After 20 steps, gradient has shrunk by factor {:.2}",
        1.0 / grad_magnitude.max(1e-15)
    );
    println!("  This means the network cannot learn long-range dependencies!");

    // Compare information retention
    println!("\n--- Information Retention Test ---\n");
    println!("How similar is the hidden state to the FIRST input after N steps?");

    let mut h_test = vec![0.0_f64; hidden_size];
    let first_input = &sequence[0];
    h_test = rnn_step(&h_test, first_input, &w_h, &w_x, &bias);
    let h_after_first = h_test.clone();

    // Now feed many zero inputs (noise-free, just decay)
    let zero_input = vec![0.0; input_size];
    for step in 1..=15 {
        h_test = rnn_step(&h_test, &zero_input, &w_h, &w_x, &bias);
        let similarity = cosine_similarity(&h_after_first, &h_test);
        if step <= 5 || step % 5 == 0 {
            println!("  After {} blank steps: similarity = {:.4}", step, similarity);
        }
    }
    println!("\n  The initial signal fades as the hidden state is overwritten.");
}

fn rnn_step(
    h_prev: &[f64],
    x: &[f64],
    w_h: &[Vec<f64>],
    w_x: &[Vec<f64>],
    bias: &[f64],
) -> Vec<f64> {
    let hidden_size = h_prev.len();
    let mut h_new = vec![0.0; hidden_size];

    for i in 0..hidden_size {
        let mut sum = bias[i];
        for j in 0..h_prev.len() {
            sum += w_h[i][j] * h_prev[j];
        }
        for j in 0..x.len() {
            sum += w_x[i][j] * x[j];
        }
        h_new[i] = sum.tanh();
    }
    h_new
}

fn init_matrix(rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    let seed = (i * cols + j) as f64;
                    (seed * 1.618 + 0.5).sin() * scale
                })
                .collect()
        })
        .collect()
}

fn matrix_spectral_norm_approx(m: &[Vec<f64>]) -> f64 {
    // Power iteration approximation
    let n = m.len();
    let mut v = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..20 {
        let mut mv = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mv[i] += m[i][j] * v[j];
            }
        }
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v = mv.iter().map(|x| x / norm).collect();
        }
    }
    // Compute Mv norm
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += m[i][j] * v[j];
        }
    }
    mv.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}
```

---

## Key Takeaways

- RNNs process sequences by maintaining a hidden state that is updated at each time step, theoretically capturing unbounded context.
- The same weight matrices are shared across all time steps, giving RNNs a fixed parameter count regardless of sequence length.
- The hidden state at each position encodes different information even for the same input, because it incorporates the full history of preceding inputs.
- Vanishing gradients cause the influence of early inputs to decay exponentially, severely limiting the practical ability of vanilla RNNs to learn long-range dependencies.
