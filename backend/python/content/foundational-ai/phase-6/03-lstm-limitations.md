# LSTM Limitations

> Phase 6 — Sequence Models | Kata 6.3

---

## Concept & Intuition

### What problem are we solving?

Vanilla RNNs suffer from the vanishing gradient problem: when backpropagating through many time steps, gradients are repeatedly multiplied by the same weight matrix, causing them to shrink exponentially. This means vanilla RNNs effectively cannot learn dependencies that span more than about 10-20 time steps. The Long Short-Term Memory (LSTM) architecture was designed specifically to solve this problem by introducing a **cell state** -- a dedicated memory highway that can carry information across many time steps without degradation.

The LSTM achieves this through three **gates**: the forget gate decides what to discard from the cell state, the input gate decides what new information to write, and the output gate decides what to expose as the hidden state. These gates are themselves small neural networks that learn when to remember, when to forget, and when to output. The cell state flows through time with only element-wise operations (multiply and add), avoiding the repeated matrix multiplications that cause gradients to vanish.

However, LSTMs are not a perfect solution. While they dramatically extend the effective memory range from ~10 steps to ~100-200 steps, they still struggle with very long sequences (thousands of steps). The gates can only partially mitigate gradient decay over such distances, and the sequential nature of processing means errors accumulate. Understanding both the power and the limitations of LSTMs is essential for appreciating why attention mechanisms and transformers were needed.

### Why naive approaches fail

The vanilla RNN uses a single hidden state `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)`. The problem is structural: at each time step, the hidden state is fully overwritten by a nonlinear transformation. There is no mechanism to preserve specific pieces of information selectively. It is like rewriting your entire notebook every time you learn something new -- inevitably, old information gets lost.

You might try to increase the hidden state size to "make more room" for memories, but this does not solve the gradient problem. The vanishing gradient is about the path of gradient flow through time, not the width of the hidden state. A 1000-dimensional vanilla RNN still cannot learn that step 1 affects step 100, because the gradient signal between them passes through 99 matrix multiplications.

### Mental models

- **Highway with on-ramps.** The cell state is a highway running through time. The forget gate is a toll booth that can block old traffic. The input gate is an on-ramp that adds new traffic. The output gate is an off-ramp that lets some traffic exit to be used. The highway itself carries information smoothly without the stop-and-go of surface streets (matrix multiplications).
- **Filing cabinet.** The cell state is a filing cabinet. The forget gate shreds old files, the input gate adds new files, and the output gate determines which files to show when someone asks. Each gate independently decides what to do, unlike a vanilla RNN which dumps the entire cabinet and rebuilds it every step.
- **Leaky bucket with a tap.** Information in a vanilla RNN is like water in a bucket with holes -- it drains quickly. The LSTM adds a tap (input gate) and plugs (forget gate), so you can control the flow. But even with the best plumbing, some leakage occurs over very long time spans.
- **Still reading one word at a time.** Despite its improved memory, an LSTM still processes tokens strictly left-to-right (or right-to-left). It cannot look ahead, and each step must finish before the next begins. This sequential bottleneck fundamentally limits its capability.

### Visual explanations

```
LSTM Cell Architecture:

                    cell state (c_t)
        c_{t-1} ---[x]----(+)-------------> c_t
                    |       |
                  forget   input
                  gate     gate        [x] = element-wise multiply
                  (f_t)    (i_t)       (+) = element-wise add
                    |       |
                    |    [tanh]--+
                    |       |   |
                    |      new  |
                    |    candidate
                    |    (c_hat_t)
                    |               output gate
        h_{t-1} ---+               (o_t)
                   |                  |
            +------+------+     +----+----+
            | concat      |     | tanh(c_t)|
            | [h_{t-1},x_t]     |    [x]   |
            +------+------+     +----+----+
                   |                  |
                   v                  v
             all 3 gates            h_t
             computed here        (hidden state output)

Gate equations:
  f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)    # forget gate
  i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)    # input gate
  c_hat = tanh(W_c * [h_{t-1}, x_t] + b_c)     # candidate
  c_t = f_t * c_{t-1} + i_t * c_hat             # new cell state
  o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)    # output gate
  h_t = o_t * tanh(c_t)                         # hidden state

Memory range comparison:
  Vanilla RNN: |###..............................| (~10 steps)
  LSTM:        |###########################.....| (~100-200 steps)
  Transformer: |################################| (thousands of steps)
```

---

## Hands-on Exploration

1. Implement a simplified LSTM cell from scratch with forget, input, and output gates, and trace the cell state and gate values as a sequence is processed.
2. Compare vanilla RNN vs LSTM on a "remember the first token" task with increasing sequence lengths, showing how the LSTM retains information much longer.
3. Push the LSTM to very long sequences to demonstrate that even it eventually fails -- the cell state gradually drifts and the signal from early tokens fades.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

# --- Simplified LSTM cell ---
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        d = input_dim + hidden_dim
        s = 0.3
        self.Wf = np.random.randn(d, hidden_dim) * s  # forget gate
        self.Wi = np.random.randn(d, hidden_dim) * s  # input gate
        self.Wc = np.random.randn(d, hidden_dim) * s  # candidate
        self.Wo = np.random.randn(d, hidden_dim) * s  # output gate
        self.bf = np.ones(hidden_dim)   # bias forget gate high (remember by default)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([h_prev, x])
        f = sigmoid(concat @ self.Wf + self.bf)   # forget gate
        i = sigmoid(concat @ self.Wi + self.bi)   # input gate
        c_hat = np.tanh(concat @ self.Wc + self.bc)  # candidate
        c = f * c_prev + i * c_hat                 # cell state
        o = sigmoid(concat @ self.Wo + self.bo)   # output gate
        h = o * np.tanh(c)                         # hidden state
        return h, c, {'f': f, 'i': i, 'o': o}

# --- Vanilla RNN cell for comparison ---
class RNNCell:
    def __init__(self, input_dim, hidden_dim):
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.3
        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.3
        self.b = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev):
        h = np.tanh(h_prev @ self.Wh + x @ self.Wx + self.b)
        return h

# --- Task: Remember the signal from step 0 after many steps ---
hidden_dim = 8
input_dim = 3

print("=== LSTM vs Vanilla RNN: 'Remember the first token' ===\n")

for seq_len in [5, 20, 50, 100, 200]:
    np.random.seed(42)
    lstm = LSTMCell(input_dim, hidden_dim)
    rnn = RNNCell(input_dim, hidden_dim)

    # First token is a strong signal; rest are noise
    signal = np.array([1.0, 0.5, -0.5])
    noise_scale = 0.01

    # LSTM forward pass
    h_lstm = np.zeros(hidden_dim)
    c_lstm = np.zeros(hidden_dim)
    h_lstm, c_lstm, _ = lstm.forward(signal, h_lstm, c_lstm)
    c_after_signal = c_lstm.copy()
    for t in range(1, seq_len):
        noise = np.random.randn(input_dim) * noise_scale
        h_lstm, c_lstm, gates = lstm.forward(noise, h_lstm, c_lstm)

    # RNN forward pass
    h_rnn = np.zeros(hidden_dim)
    h_rnn = rnn.forward(signal, h_rnn)
    h_rnn_after_signal = h_rnn.copy()
    for t in range(1, seq_len):
        noise = np.random.randn(input_dim) * noise_scale
        h_rnn = rnn.forward(noise, h_rnn)

    # Measure how much of the original signal is retained
    lstm_retention = np.linalg.norm(c_lstm) / (np.linalg.norm(c_after_signal) + 1e-8)
    rnn_retention = np.linalg.norm(h_rnn) / (np.linalg.norm(h_rnn_after_signal) + 1e-8)

    lstm_bar = "#" * int(min(lstm_retention * 20, 30))
    rnn_bar = "#" * int(min(rnn_retention * 20, 30))

    print(f"  Seq length: {seq_len:3d}")
    print(f"    RNN  retention: {rnn_retention:.4f} |{rnn_bar}")
    print(f"    LSTM retention: {lstm_retention:.4f} |{lstm_bar}")
    print()

# --- Trace gate values over time ---
print("=== Gate Activity Trace (sequence length 20) ===\n")
np.random.seed(42)
lstm = LSTMCell(input_dim, hidden_dim)
h = np.zeros(hidden_dim)
c = np.zeros(hidden_dim)
print(f"  Step | Forget Gate  | Input Gate   | Output Gate  | Cell Norm")
print(f"  -----+--------------+--------------+--------------+----------")
for t in range(20):
    x = signal if t == 0 else np.random.randn(input_dim) * noise_scale
    h, c, gates = lstm.forward(x, h, c)
    f_avg = np.mean(gates['f'])
    i_avg = np.mean(gates['i'])
    o_avg = np.mean(gates['o'])
    c_norm = np.linalg.norm(c)
    marker = " <-- signal" if t == 0 else ""
    print(f"  {t:4d} |    {f_avg:.4f}    |    {i_avg:.4f}    |"
          f"    {o_avg:.4f}    | {c_norm:.4f}{marker}")

print("\n=== Key Observations ===")
print("  1. LSTM retains signal MUCH longer than vanilla RNN")
print("  2. Forget gate stays high (near 1.0) => cell state preserved")
print("  3. Input gate stays low for noise => junk is filtered out")
print("  4. But even LSTM signal fades over very long sequences")
```

---

## Key Takeaways

- **Gates give LSTMs selective memory.** The forget gate controls what to discard, the input gate controls what to store, and the output gate controls what to expose. This selectivity is what vanilla RNNs lack.
- **The cell state is a gradient highway.** Information flows through the cell state via element-wise operations (multiply and add), avoiding the repeated matrix multiplications that cause vanishing gradients in vanilla RNNs.
- **LSTMs extend, but do not eliminate, memory limits.** They handle dependencies of ~100-200 steps well, compared to ~10 steps for vanilla RNNs. But at sequence lengths of thousands, even LSTMs struggle.
- **Forget gate initialization matters.** Initializing the forget gate bias to a positive value (so the gate starts near 1.0) is a well-known trick that dramatically improves LSTM training by defaulting to "remember."
- **Sequential processing remains the bottleneck.** Despite better memory, LSTMs still process tokens one at a time. This limitation motivated the development of attention mechanisms and the transformer architecture.
