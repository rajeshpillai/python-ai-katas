# Why Recurrence Struggles at Scale

> Phase 6 — Sequence Models | Kata 6.4

---

## Concept & Intuition

### What problem are we solving?

Recurrent neural networks -- including LSTMs and GRUs -- process sequences one token at a time. Each step depends on the previous step's output. This strict sequential dependency creates three fundamental scaling problems that become severe as sequences grow longer and models grow larger.

First, **sequential computation prevents parallelism**. On modern GPUs with thousands of cores, a fully connected layer can process an entire batch simultaneously. But an RNN processing a 1000-token sequence must do 1000 sequential steps, each waiting for the previous one to finish. Those GPU cores sit idle. Processing time scales linearly with sequence length, regardless of hardware parallelism. A transformer processes the same sequence in a single parallel step.

Second, **gradient flow degrades over distance**. Even LSTMs, which were designed to combat vanishing gradients, struggle when the distance between relevant tokens exceeds a few hundred steps. The gradient signal from token 1000 that needs to reach token 1 must travel through 999 sequential operations. At each step, some signal is lost. By contrast, a transformer's self-attention connects every token to every other token directly, so the gradient path between any two tokens is always length 1.

Third, **the hidden state becomes a bottleneck**. All information about the sequence so far must be compressed into a fixed-size hidden state vector. For a 512-dimensional hidden state processing a 10,000-token document, that is roughly 20 tokens worth of information per hidden dimension. The network must decide at each step what to keep and what to discard, and inevitably important information is lost. Attention mechanisms solve this by letting the model look back at the full sequence of past states rather than relying on a single compressed summary.

### Why naive approaches fail

The obvious response to "RNNs are slow" is to try to parallelize them. But this is structurally impossible: `h_t` depends on `h_{t-1}`, which depends on `h_{t-2}`, and so on. You cannot compute step 50 without first computing steps 1 through 49. This is not an engineering limitation -- it is a mathematical dependency chain. Techniques like truncated backpropagation-through-time (TBPTT) reduce memory by only backpropagating through a window of steps, but this explicitly discards long-range gradient information, making the long-distance learning problem worse.

You might try to compensate by making the hidden state larger. But a wider hidden state means more parameters in the recurrent weight matrix, which makes each sequential step slower and does not solve the gradient flow problem. The bottleneck is not the width of the pipe -- it is the length of the chain. No amount of widening can substitute for the direct connections that attention provides.

### Mental models

- **Assembly line vs. crowd.** An RNN is like an assembly line: each worker must wait for the previous worker to finish. Attention is like a crowd where everyone can talk to everyone simultaneously. As the group grows, the assembly line gets proportionally slower, but the crowd processes everything in parallel.
- **Telephone game.** In an RNN, information from early tokens is passed through a chain of processing steps, like the telephone game. Each handoff introduces noise and loses fidelity. Attention lets every player hear the original speaker directly.
- **Packing a single suitcase.** The hidden state is like packing for a trip with one suitcase of fixed size. A short trip is fine, but for a year-long journey, you must leave behind things you will later need. Attention is like having access to your entire wardrobe at all times.
- **O(n) vs O(1) path length.** The fundamental issue is path length. In an RNN, the gradient path between token i and token j is |i-j| steps long. In a transformer, it is always 1 step. Shorter paths mean stronger, cleaner gradient signals.

### Visual explanations

```
Sequential vs Parallel Processing:

  RNN (sequential - must wait):
  Time: t=0    t=1    t=2    t=3    t=4    ...    t=999
        [x0]->[x1]->[x2]->[x3]->[x4]-> ... ->[x999]
        h0     h1     h2     h3     h4          h999
        ^^^    waits  waits  waits  waits       waits
                                         Total: 1000 steps

  Transformer (parallel - all at once):
  Time: t=0
        [x0] [x1] [x2] [x3] [x4] ... [x999]
          \   |   /  |   \  ... /
           ATTENTION (all pairs computed simultaneously)
          /   |   \  |   /  ... \
        [y0] [y1] [y2] [y3] [y4] ... [y999]
                                         Total: 1 step

Gradient path length:

  RNN (token 0 -> token 100):
  t=0 -> t=1 -> t=2 -> ... -> t=99 -> t=100
  |____________ 100 steps ______________|
  gradient must survive 100 multiplications

  Transformer (token 0 -> token 100):
  t=0 --------direct attention--------> t=100
  |_____________ 1 step ________________|
  gradient travels through 1 operation

Hidden state bottleneck:

  RNN:     entire history compressed into h_t
           [===h_t===]  (fixed size, e.g., 512 floats)
           must represent ALL of:  x_0, x_1, ..., x_{t-1}

  Attention: can look at ALL previous states
           [h_0] [h_1] [h_2] ... [h_{t-1}]  (full history accessible)
           query selects what is relevant NOW
```

---

## Hands-on Exploration

1. Measure the wall-clock time to process sequences of increasing length with an RNN, confirming the linear scaling of processing time.
2. Compute the gradient magnitude flowing from the last token back to the first token for different sequence lengths, showing exponential decay.
3. Measure the hidden state's "information retention" -- how much information about early tokens survives to the end of the sequence -- and contrast this with direct access (simulated attention).

---

## Live Code

```python
import numpy as np
import time

np.random.seed(42)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

# --- Simple RNN ---
class SimpleRNN:
    def __init__(self, input_dim, hidden_dim):
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.5 / np.sqrt(hidden_dim)
        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.5 / np.sqrt(input_dim)
        self.hidden_dim = hidden_dim

    def forward_sequence(self, xs):
        h = np.zeros(self.hidden_dim)
        for x in xs:
            h = np.tanh(h @ self.Wh + x @ self.Wx)
        return h

# --- 1. Processing time scales linearly ---
print("=== 1. Processing Time vs Sequence Length ===\n")
rnn = SimpleRNN(input_dim=8, hidden_dim=32)
lengths = [50, 100, 200, 500, 1000, 2000]
times = []
for L in lengths:
    xs = [np.random.randn(8) * 0.1 for _ in range(L)]
    start = time.time()
    for _ in range(5):  # average over 5 runs
        rnn.forward_sequence(xs)
    elapsed = (time.time() - start) / 5
    times.append(elapsed)
    bar = "#" * int(elapsed * 3000)
    print(f"  Length {L:5d}: {elapsed*1000:7.2f} ms {bar}")

if times[-1] > 0 and times[0] > 0:
    ratio = times[-1] / times[0]
    length_ratio = lengths[-1] / lengths[0]
    print(f"\n  Length grew {length_ratio:.0f}x, time grew {ratio:.1f}x")
    print(f"  Confirms: processing time is O(n) -- linear in sequence length")

# --- 2. Gradient decay over distance ---
print("\n=== 2. Gradient Magnitude vs Distance ===\n")
hidden_dim = 16
np.random.seed(42)
Wh = np.random.randn(hidden_dim, hidden_dim) * 0.9 / np.sqrt(hidden_dim)

print(f"  Distance | Gradient Norm | Visualization")
print(f"  ---------+---------------+---------------------------")
for distance in [1, 5, 10, 20, 50, 100, 200]:
    # Gradient of h_T w.r.t. h_0 = product of Jacobians
    # For tanh RNN, Jacobian at each step ~ diag(1-tanh^2(z)) @ Wh
    # We simulate by multiplying representative Jacobians
    grad = np.eye(hidden_dim)
    for t in range(distance):
        # Approximate Jacobian: derivative of tanh times Wh
        tanh_deriv = np.diag(np.random.uniform(0.3, 1.0, hidden_dim))
        J = tanh_deriv @ Wh
        grad = J @ grad
    norm = np.linalg.norm(grad)
    bar = "#" * min(int(np.log10(norm + 1e-30) + 15), 30)
    if norm > 0:
        bar = "#" * max(1, min(int(np.log10(norm + 1e-30) + 10), 30))
    print(f"  {distance:7d}  | {norm:13.6f} | {bar}")

print(f"\n  Gradient decays roughly exponentially with distance!")
print(f"  This is why RNNs cannot learn long-range dependencies.")

# --- 3. Hidden state bottleneck ---
print("\n=== 3. Hidden State Information Bottleneck ===\n")
np.random.seed(42)
rnn_small = SimpleRNN(input_dim=4, hidden_dim=16)

# Store a target signal at step 0, then process noise
target_signal = np.array([1.0, -1.0, 0.5, -0.5])

print(f"  Stored signal: {target_signal}")
print(f"  Hidden dim: 16 | Measuring retention after N noise steps\n")
print(f"  Steps | h dot target | Retention | RNN vs Direct Access")
print(f"  ------+--------------+-----------+---------------------")

for n_steps in [0, 5, 10, 25, 50, 100, 200]:
    np.random.seed(42)
    rnn_test = SimpleRNN(input_dim=4, hidden_dim=16)
    h = np.zeros(16)
    h = np.tanh(h @ rnn_test.Wh + target_signal @ rnn_test.Wx)
    h_after_signal = h.copy()

    for t in range(n_steps):
        noise = np.random.randn(4) * 0.01
        h = np.tanh(h @ rnn_test.Wh + noise @ rnn_test.Wx)

    # How much of original signal direction is retained?
    if np.linalg.norm(h_after_signal) > 0:
        cos_sim = np.dot(h, h_after_signal) / (
            np.linalg.norm(h) * np.linalg.norm(h_after_signal) + 1e-8)
    else:
        cos_sim = 0.0
    direct = 1.0  # attention always has cos_sim = 1.0 (direct access)
    rnn_bar = "#" * max(0, int(abs(cos_sim) * 20))
    att_bar = "#" * 20
    print(f"  {n_steps:5d} | {cos_sim:+11.4f}  | {rnn_bar:20s} | {att_bar}")

# --- Summary ---
print("\n=== Why Transformers Won ===")
print(f"  +-------------------+--------+-------------+")
print(f"  | Property          |  RNN   | Transformer |")
print(f"  +-------------------+--------+-------------+")
print(f"  | Parallelizable    |   No   |     Yes     |")
print(f"  | Gradient path     |  O(n)  |     O(1)    |")
print(f"  | Memory access     | Bottl. |    Direct   |")
print(f"  | Time complexity   |  O(n)  |   O(n^2)*   |")
print(f"  +-------------------+--------+-------------+")
print(f"  * O(n^2) attention is expensive, but parallelizable!")
print(f"  Parallelism on modern GPUs > sequential efficiency.")
```

---

## Key Takeaways

- **Sequential processing prevents parallelism.** Each RNN step depends on the previous one, so processing time scales linearly with sequence length. Modern GPUs with thousands of cores cannot help -- the computation is inherently serial.
- **Gradient paths grow linearly with distance.** For an RNN to learn that token 1 affects token 1000, the gradient must flow through 999 sequential operations. Even with LSTMs, the signal degrades exponentially over such distances.
- **The hidden state is a bottleneck.** All sequence history must be compressed into a fixed-size vector. For long sequences, this forces the network to discard information that may be needed later. Attention mechanisms eliminate this bottleneck by providing direct access to all past states.
- **These are structural limitations, not engineering ones.** No amount of clever optimization, larger hidden states, or better hardware can overcome the fundamental sequential dependency chain. The architecture itself must change.
- **Transformers solve all three problems.** Self-attention processes all positions in parallel, provides O(1) gradient paths between any two tokens, and lets the model attend directly to any past state. This is why transformers have replaced RNNs as the dominant sequence architecture.
