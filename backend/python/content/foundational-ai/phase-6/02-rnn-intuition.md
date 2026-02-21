# RNN Intuition

> Phase 6 — Sequence Models | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

N-grams have a hard limit: they can only look back n-1 steps. A bigram model processing "the cat that chased the dog sat on the ___" has no idea that "cat" was the subject many words ago. We need a model with a **memory** that persists across the entire sequence.

A Recurrent Neural Network (RNN) solves this by maintaining a **hidden state** -- a vector that gets updated at every time step. At each step, the RNN takes two inputs: the current input (e.g., the current character or word) and the previous hidden state. It combines them through a weight matrix and a nonlinearity to produce a new hidden state. This new hidden state encodes information about everything the network has seen so far -- in theory.

The key equation is: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)`. The hidden state `h_{t-1}` carries the past, the input `x_t` brings the present, and the weights learn how to blend them. The hidden state is the RNN's "memory" -- a compressed summary of the sequence so far, encoded as a fixed-size vector.

### Why naive approaches fail

You might think: "why not just feed the entire sequence into a regular dense network?" The problem is that sequences have variable length. A sentence might be 5 words or 50. A dense network needs fixed-size input. You could pad to a maximum length, but then short sequences waste computation and you can never handle anything longer than your maximum.

More fundamentally, a dense network treats each position independently. It doesn't know that position 3 comes after position 2. An RNN processes positions in order, building up its understanding step by step, naturally handling any length sequence with the same parameters.

### Mental models

- **Reading a book one page at a time**: You don't remember every word, but your understanding (hidden state) evolves with each page. After page 50, your mental state summarizes the plot so far.
- **A conveyor belt in a factory**: Each station (time step) receives the item (hidden state) from the previous station, modifies it based on new information (current input), and passes it along.
- **Telephone game with notes**: Each person hears the message (input), combines it with their notes from the previous person (hidden state), writes new notes, and passes them forward.

### Visual explanations

```
RNN unrolled through time:

  x_0         x_1         x_2         x_3
   |           |           |           |
   v           v           v           v
 [RNN] ----> [RNN] ----> [RNN] ----> [RNN] ---> ...
   |    h_0    |    h_1    |    h_2    |    h_3
   v           v           v           v
  y_0         y_1         y_2         y_3

Same weights at every step!

Inside one RNN cell:
  +-----------------------------------------+
  |  h_new = tanh(W_hh * h_prev + W_xh * x + b)  |
  |                                         |
  |  h_prev --->[W_hh]--+                  |
  |                      +--> [+] -> [tanh] -> h_new
  |  x -------->[W_xh]--+       ^            |
  |                              |            |
  |                             [b]           |
  +-----------------------------------------+
```

---

## Hands-on Exploration

1. Initialize random weights and a zero hidden state, then manually step through a short sequence to see how the hidden state evolves
2. Run the same input through multiple times and observe that identical inputs produce identical hidden states (deterministic)
3. Change one early input in the sequence and observe how it affects all subsequent hidden states (the memory effect)

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Dimensions ---
input_size = 4    # e.g., one-hot encoded vocabulary of 4 chars
hidden_size = 3   # small hidden state for visibility

# --- Initialize weights (normally these are learned) ---
W_xh = np.random.randn(hidden_size, input_size) * 0.5   # input -> hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5  # hidden -> hidden
b_h = np.zeros(hidden_size)

print("RNN weights:")
print(f"  W_xh (hidden x input): {W_xh.shape}")
print(f"  W_hh (hidden x hidden): {W_hh.shape}\n")

# --- RNN cell: the core computation ---
def rnn_cell(x, h_prev):
    """One step of an RNN: combines input and previous hidden state."""
    return np.tanh(W_hh @ h_prev + W_xh @ x + b_h)

# --- Create a sequence of one-hot inputs ---
vocab = ['a', 'b', 'c', 'd']
sequence = "abcdb"
print(f"Input sequence: '{sequence}'\n")

def one_hot(char):
    vec = np.zeros(input_size)
    vec[vocab.index(char)] = 1.0
    return vec

# --- Process sequence step by step ---
h = np.zeros(hidden_size)  # initial hidden state
print("Step-by-step processing:")
print(f"  {'Step':<6} {'Input':<6} {'Hidden State':>36}")
print("  " + "-" * 50)
print(f"  {'init':<6} {'---':<6} {np.array2string(h, precision=3, floatmode='fixed'):>36}")

hidden_states = [h.copy()]
for t, char in enumerate(sequence):
    x = one_hot(char)
    h = rnn_cell(x, h)
    hidden_states.append(h.copy())
    print(f"  {t:<6} {char!r:<6} {np.array2string(h, precision=3, floatmode='fixed'):>36}")

# --- Show the memory effect: change one early input ---
print("\n--- Memory effect: changing input at step 0 ---")
print("Original sequence: 'abcdb'")
print("Modified sequence: 'dbcdb'  (changed first char)\n")

h_orig = np.zeros(hidden_size)
h_mod = np.zeros(hidden_size)

for t, (c_orig, c_mod) in enumerate(zip("abcdb", "dbcdb")):
    h_orig = rnn_cell(one_hot(c_orig), h_orig)
    h_mod = rnn_cell(one_hot(c_mod), h_mod)
    diff = np.abs(h_orig - h_mod).sum()
    marker = " <-- changed input" if t == 0 else ""
    print(f"  Step {t}: diff = {diff:.4f}{marker}")

print("\n  One change at step 0 ripples through ALL future hidden states!")

# --- Weight sharing: same weights at every step ---
print("\n--- Weight sharing demonstration ---")
print("Processing 'a' at different positions produces different h")
print("because h_prev differs, even though x and weights are the same:")
for t in [0, 2, 4]:
    print(f"  After step {t}: h = {np.array2string(hidden_states[t+1], precision=3, floatmode='fixed')}"
          if sequence[t] == 'a' else f"  Step {t}: input='{sequence[t]}' (not 'a')")
```

---

## Key Takeaways

- **An RNN maintains a hidden state that acts as memory.** This fixed-size vector is updated at each time step, encoding a summary of all past inputs.
- **The same weights are shared across all time steps.** This is what makes RNNs handle variable-length sequences -- one set of parameters works for any sequence length.
- **Each hidden state depends on all previous inputs.** Changing an early input ripples forward through every subsequent hidden state, giving the network a form of memory.
- **The hidden state is a lossy compression.** A fixed-size vector cannot perfectly encode an arbitrarily long history -- information from early steps gradually fades (more on this in Kata 6.3).
- **RNNs process sequences one step at a time.** This sequential nature is both their strength (natural ordering) and their weakness (no parallelism, as we'll see in Kata 6.4).
