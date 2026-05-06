# Why Transformers Scale

> Phase 7 — Attention & Transformers | Kata 7.4

---

## Concept & Intuition

### What problem are we solving?

The deep learning revolution required not just better algorithms but better hardware utilization. RNNs process sequences one token at a time -- each step depends on the previous step's output. This inherently sequential nature means that even on a GPU with thousands of cores, an RNN processing a 1000-word sentence must perform 1000 serial steps. Transformers fundamentally changed this by processing all positions in parallel. Every word computes its attention to every other word simultaneously, making the operation a large matrix multiplication that GPUs excel at.

This architectural advantage is the key reason Transformers, not RNNs or CNNs, became the foundation of large language models. When you double your compute budget, a Transformer can process sequences proportionally faster or handle proportionally larger models. The scaling laws discovered by Kaplan et al. (2020) showed that Transformer performance improves predictably as a power law with model size, dataset size, and compute -- a property that does not hold nearly as cleanly for sequential architectures.

The practical consequence is staggering. Training GPT-3 with 175 billion parameters would have been effectively impossible with an RNN architecture, not because the math is different, but because the computation cannot be parallelized efficiently enough. Transformers turned the scaling knob from "limited by sequential depth" to "limited by how many GPUs you can afford."

### Why naive approaches fail

RNNs suffer from a fundamental parallelism bottleneck: hidden state h_t depends on h_{t-1}, which depends on h_{t-2}, and so on. You cannot compute h_100 without first computing h_1 through h_99. This creates a critical path of length O(n) that no amount of hardware can shorten. Even bidirectional RNNs just run two sequential passes instead of one.

CNNs applied to sequences can parallelize across positions, but each layer only captures local context within its kernel width. To capture dependencies across a 100-word span, you need many stacked layers (O(log n) with dilated convolutions, O(n/k) with kernel size k). This means deep, expensive networks for long-range dependencies. Transformers capture all pairwise dependencies in a single layer with O(1) sequential operations.

### Mental models

- **Assembly line vs parallel workstations:** An RNN is an assembly line where each worker must wait for the previous one to finish. A Transformer is a factory floor where all workers operate simultaneously, each consulting the others as needed via radio.
- **Phone tree vs group call:** Passing a message through an RNN is like a phone tree -- each person calls the next. Attention is a group conference call where everyone hears everyone else simultaneously.
- **Sequential cooking vs buffet preparation:** An RNN cooks courses one at a time (appetizer must finish before entree starts). A Transformer prepares all courses simultaneously, with chefs glancing at each other's stations to coordinate.

### Visual explanations

```
  RNN: SEQUENTIAL PROCESSING
  ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐
  │ x1 │-->│ x2 │-->│ x3 │-->│ x4 │-->│ x5 │
  └────┘   └────┘   └────┘   └────┘   └────┘
  step 1   step 2   step 3   step 4   step 5

  Time: O(n) sequential steps
  x5 sees x1 through a chain of 4 transformations

  TRANSFORMER: PARALLEL PROCESSING
  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
  │ x1 │ │ x2 │ │ x3 │ │ x4 │ │ x5 │
  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
     │      │      │      │      │
     └──────┴──────┴──────┴──────┘
     All-to-all attention (ONE parallel step)
     │      │      │      │      │
  ┌──┴─┐ ┌──┴─┐ ┌──┴─┐ ┌──┴─┐ ┌──┴─┐
  │ y1 │ │ y2 │ │ y3 │ │ y4 │ │ y5 │
  └────┘ └────┘ └────┘ └────┘ └────┘

  Time: O(1) sequential steps (but O(n^2) parallel work)
  Every position directly attends to every other

  SCALING COMPARISON:
  Sequence length:  10    100    1000   10000
  RNN steps:        10    100    1000   10000
  Transformer:       1      1       1       1  (sequential)
  Transformer:     100  10000   10^6    10^8  (parallel ops)
                   ^--- GPUs handle this efficiently!
```

---

## Hands-on Exploration

1. Simulate RNN sequential processing and time the critical-path length as sequence length grows
2. Simulate Transformer parallel processing and compare the sequential depth
3. Measure how computation scales with sequence length for both architectures
4. Observe how maximum information distance (path length between distant tokens) differs
5. Demonstrate the scaling law: Transformer loss decreasing predictably with model size

---

## Live Code

```python
import numpy as np
import time

np.random.seed(42)

def rnn_forward(X, W_h, W_x, b):
    """RNN: sequential processing -- h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)."""
    seq_len, d_in = X.shape
    d_h = W_h.shape[0]
    h = np.zeros(d_h)
    steps = 0
    for t in range(seq_len):
        h = np.tanh(W_h @ h + W_x @ X[t] + b)
        steps += 1
    return h, steps

def transformer_forward(X, W_Q, W_K, W_V):
    """Transformer: parallel attention -- all positions at once."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attn = e / e.sum(axis=1, keepdims=True)
    output = attn @ V
    sequential_steps = 1  # all done in one parallel step
    return output, sequential_steps

d_model = 16
d_hidden = 16

# RNN parameters
W_h = np.random.randn(d_hidden, d_hidden) * 0.3
W_x = np.random.randn(d_hidden, d_model) * 0.3
b = np.zeros(d_hidden)

# Transformer parameters
W_Q = np.random.randn(d_model, d_hidden) * 0.3
W_K = np.random.randn(d_model, d_hidden) * 0.3
W_V = np.random.randn(d_model, d_hidden) * 0.3

print("=" * 60)
print("WHY TRANSFORMERS SCALE")
print("=" * 60)

# --- Compare sequential steps ---
print("\n--- Sequential Steps (Critical Path Length) ---\n")
print(f"{'Seq Len':>8s} {'RNN Steps':>10s} {'Tfmr Steps':>11s} {'Speedup':>8s}")
print("-" * 42)
for seq_len in [4, 16, 64, 256, 512]:
    X = np.random.randn(seq_len, d_model) * 0.5
    _, rnn_steps = rnn_forward(X, W_h, W_x, b)
    _, tfmr_steps = transformer_forward(X, W_Q, W_K, W_V)
    print(f"{seq_len:>8d} {rnn_steps:>10d} {tfmr_steps:>11d} "
          f"{rnn_steps/tfmr_steps:>7.0f}x")

# --- Information path length ---
print(f"\n{'=' * 60}")
print("INFORMATION PATH LENGTH (token 0 -> token n)")
print("=" * 60)
print("\nHow many transformations must information travel through?\n")
print(f"{'Seq Len':>8s} {'RNN Path':>10s} {'Tfmr Path':>10s}")
print("-" * 32)
for n in [10, 50, 100, 500, 1000]:
    rnn_path = n        # must traverse n hidden states
    tfmr_path = 1       # direct attention connection
    print(f"{n:>8d} {rnn_path:>10d} {tfmr_path:>10d}")

# --- Computation cost comparison ---
print(f"\n{'=' * 60}")
print("TOTAL OPERATIONS (Parallel Work)")
print("=" * 60)
print(f"\n{'Seq Len':>8s} {'RNN O(nd^2)':>12s} {'Tfmr O(n^2d)':>13s} {'Ratio':>8s}")
print("-" * 45)
d = d_model
for n in [16, 64, 256, 1024]:
    rnn_ops = n * d * d
    tfmr_ops = n * n * d
    ratio = tfmr_ops / rnn_ops
    winner = "<-- Tfmr wins" if ratio < 1 else ""
    print(f"{n:>8d} {rnn_ops:>12,d} {tfmr_ops:>13,d} {ratio:>7.1f}x {winner}")
print(f"\n  Note: Transformer wins when n < d (common for d=768,4096)")

# --- Simulate scaling law ---
print(f"\n{'=' * 60}")
print("SCALING LAW: Loss vs Model Size")
print("=" * 60)
print("\nSimulating: L(N) = a * N^(-alpha) + L_infinity")
print("(Kaplan et al. 2020: loss follows a power law)\n")

a = 5.0
alpha = 0.076
L_inf = 1.69  # irreducible loss

model_sizes = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
labels = ["1M", "10M", "100M", "1B", "10B", "100B"]

print(f"{'Model Size':>12s} {'Loss':>8s}  Visual")
print("-" * 50)
losses = []
for size, label in zip(model_sizes, labels):
    loss = a * (size ** -alpha) + L_inf
    losses.append(loss)
    bar_len = int((loss - L_inf) * 80)
    bar = "#" * bar_len
    print(f"{label:>12s} {loss:>8.4f}  |{bar}")
print(f"{'Irreducible':>12s} {L_inf:>8.4f}  |")

# --- Wall clock comparison ---
print(f"\n{'=' * 60}")
print("WALL CLOCK: RNN vs Transformer (simulated)")
print("=" * 60)
seq_lengths = [32, 128, 512]
for sl in seq_lengths:
    X = np.random.randn(sl, d_model)
    t0 = time.perf_counter()
    for _ in range(20):
        rnn_forward(X, W_h, W_x, b)
    rnn_time = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        transformer_forward(X, W_Q, W_K, W_V)
    tfmr_time = (time.perf_counter() - t0) / 20
    print(f"\n  Seq len={sl}:")
    print(f"    RNN:         {rnn_time*1000:8.3f} ms")
    print(f"    Transformer: {tfmr_time*1000:8.3f} ms")
```

---

## Key Takeaways

- **Transformers trade sequential depth for parallel breadth.** One attention layer connects all positions directly, while RNNs require O(n) sequential steps.
- **GPUs thrive on parallelism.** Matrix multiplications (the core of attention) map perfectly to GPU architectures, while sequential RNN steps leave most cores idle.
- **Scaling laws make Transformers predictable.** Loss decreases as a power law with compute, data, and parameters -- this predictability enabled the strategic scaling that produced GPT-3, GPT-4, and beyond.
- **Information path length is O(1) in Transformers.** Any token can attend to any other token in a single step, eliminating the vanishing gradient problem for long-range dependencies.
- **The O(n^2) cost of attention is the key tradeoff.** Transformers pay for parallelism with quadratic memory and compute in sequence length, motivating research into efficient attention variants.
