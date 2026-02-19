# Attention as Weighted Averaging

> Phase 7 — Attention & Transformers | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

When a neural network processes a sequence -- like a sentence of words or a series of measurements -- it needs to decide which parts of the input are most relevant for producing each part of the output. Consider translating "The cat sat on the mat" to French. When generating the French word for "cat," the model needs to focus heavily on the English word "cat" and less on "the" or "on." This selective focus is exactly what attention provides.

Before attention, sequence models like RNNs tried to compress an entire input sequence into a single fixed-size vector. This is like trying to summarize an entire book into a single sentence -- inevitably, crucial details get lost. The longer the sequence, the worse this bottleneck becomes. Attention solves this by allowing the model to look back at all input positions and dynamically decide which ones matter most for the current computation.

At its mathematical core, attention is surprisingly simple: it computes relevance scores between a query (what am I looking for?) and a set of keys (what is available?), then uses those scores to take a weighted average of the corresponding values (what information do I extract?). This query-key-value framework is the foundation of modern AI.

### Why naive approaches fail

The simplest approach to combining information from a sequence is to just average all the vectors equally. But equal averaging treats every word as equally important -- "the" gets the same weight as "cat." This washes out the signal. You could also try using only the last hidden state (as vanilla RNNs do), but then early information must survive through many processing steps, leading to the vanishing gradient problem.

Fixed-window approaches (only look at the 3 nearest words) fail because relevance is not determined by proximity alone. In "The cat that I saw yesterday sat on the mat," the verb "sat" needs to attend to "cat" despite many intervening words.

### Mental models

- **Spotlight at a concert:** Attention is like a spotlight operator who can illuminate different performers depending on what the audience needs to see. The spotlight (query) searches for the most relevant performer (keys), and the audience sees the illuminated content (values).
- **Library research:** You have a research question (query). You scan book titles and summaries (keys) to find relevant ones. You then read the actual content (values) of the most relevant books, spending more time on the highly relevant ones.
- **Weighted voting:** Each input position casts a "vote" on what the output should be. Attention weights determine how much each vote counts -- relevant positions get louder voices.

### Visual explanations

```
  THE ATTENTION MECHANISM

  Query: "What French word should I produce now?"

  Keys & Values (input words):
  ┌─────┬─────┬─────┬─────┬─────┬─────┐
  │ The │ cat │ sat │ on  │ the │ mat │
  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
     │     │     │     │     │     │
  Score: 0.05  0.70  0.10  0.05  0.03  0.07   <-- attention weights
     │     │     │     │     │     │
     └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
        │     │     │     │     │
        v     v     v     v     v
     ┌──────────────────────────────┐
     │  Weighted sum of all values  │ --> output
     └──────────────────────────────┘

  QUERY-KEY-VALUE DECOMPOSITION

  Input x ──┬──> W_Q * x = Query  (what am I looking for?)
            ├──> W_K * x = Key    (what do I contain?)
            └──> W_V * x = Value  (what info do I provide?)

  score(i,j) = Query_i . Key_j / sqrt(d_k)
  weight(i,j) = softmax(score(i,:))
  output_i    = sum_j( weight(i,j) * Value_j )
```

---

## Hands-on Exploration

1. Start with simple equal-weight averaging of word vectors and observe how meaning gets diluted
2. Implement dot-product scoring between a query and multiple keys to compute relevance
3. Apply softmax to convert raw scores into proper attention weights that sum to one
4. Use the weights to compute a weighted average of value vectors
5. Vary the query and observe how attention weights shift to different keys

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()

# --- Setup: simulate word embeddings (dim=4) for a short sentence ---
words = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 4  # embedding dimension

# Random word embeddings (in practice these are learned)
embeddings = np.random.randn(len(words), d_model)

# --- Learned projection matrices for Q, K, V ---
d_k = 4  # key/query dimension
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_k) * 0.5

# --- Project embeddings into queries, keys, values ---
Q = embeddings @ W_Q  # (6, 4)
K = embeddings @ W_K  # (6, 4)
V = embeddings @ W_V  # (6, 4)

# --- Compute attention scores: Q . K^T / sqrt(d_k) ---
scores = Q @ K.T / np.sqrt(d_k)  # (6, 6)

print("=" * 55)
print("ATTENTION AS WEIGHTED AVERAGING")
print("=" * 55)

# --- Compare naive average vs attention for word index 1 ("cat") ---
query_idx = 1
print(f"\nQuery word: '{words[query_idx]}'")
print(f"\nRaw attention scores from '{words[query_idx]}' to all words:")
for i, w in enumerate(words):
    print(f"  {w:>4s}: {scores[query_idx, i]:+.3f}")

# Apply softmax to get attention weights
attn_weights = softmax(scores[query_idx])
print(f"\nAttention weights (after softmax):")
for i, w in enumerate(words):
    bar = "#" * int(attn_weights[i] * 40)
    print(f"  {w:>4s}: {attn_weights[i]:.3f} |{bar}")

# --- Naive equal averaging ---
naive_output = np.mean(V, axis=0)

# --- Attention-weighted output ---
attn_output = attn_weights @ V  # weighted sum of values

print(f"\nNaive average output (all words equal): {naive_output.round(3)}")
print(f"Attention-weighted output:              {attn_output.round(3)}")
print(f"Difference norm: {np.linalg.norm(attn_output - naive_output):.4f}")

# --- Full attention matrix for all words ---
print(f"\n{'=' * 55}")
print("FULL ATTENTION MATRIX (who attends to whom)")
print("=" * 55)
header = "      " + "".join(f"{w:>6s}" for w in words)
print(header)
print("      " + "-" * (6 * len(words)))

for i in range(len(words)):
    weights_i = softmax(scores[i])
    row = f"{words[i]:>5s} |"
    for j in range(len(words)):
        row += f" {weights_i[j]:.2f} "
    print(row)

# --- Show that attention output is just a weighted average ---
print(f"\n{'=' * 55}")
print("VERIFYING: attention = weighted average of values")
print("=" * 55)
manual_sum = np.zeros(d_k)
for j in range(len(words)):
    manual_sum += attn_weights[j] * V[j]
    print(f"  + {attn_weights[j]:.3f} * V('{words[j]}')")
print(f"\n  Manual weighted sum:  {manual_sum.round(3)}")
print(f"  Matrix computation:   {attn_output.round(3)}")
print(f"  Match: {np.allclose(manual_sum, attn_output)}")
```

---

## Key Takeaways

- **Attention is dynamic relevance weighting.** Instead of treating all inputs equally, attention computes how relevant each input is to the current query and weights accordingly.
- **Query-Key-Value is a retrieval framework.** The query asks a question, keys are matched against it, and values provide the answer -- it is soft database lookup.
- **Softmax creates a probability distribution.** Raw dot-product scores are converted into weights that sum to one, ensuring the output is a proper weighted average.
- **Scaling by sqrt(d_k) prevents saturation.** Without scaling, large dot products push softmax into regions where gradients vanish, making learning difficult.
- **Attention replaces the information bottleneck.** Instead of compressing everything into one vector, attention lets the model access any input position directly.
