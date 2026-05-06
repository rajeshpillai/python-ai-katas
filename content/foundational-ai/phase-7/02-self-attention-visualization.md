# Self-Attention Visualization

> Phase 7 — Attention & Transformers | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 7.1, we saw attention as a mechanism where a query attends to separate keys and values. Self-attention is the special case where queries, keys, and values all come from the same sequence. Every word in a sentence attends to every other word (including itself) to build a richer representation that captures context. When processing the word "bank" in "I sat by the river bank," self-attention lets "bank" look at "river" and update its representation to reflect the waterside meaning rather than the financial one.

Self-attention is the core operation inside a Transformer. It runs in parallel across all positions -- every word simultaneously queries every other word. The result is that each word's representation gets enriched with information from the entire sequence. This is what allows Transformers to capture long-range dependencies that RNNs struggle with.

Visualizing the attention matrix reveals fascinating patterns. Some words attend primarily to themselves. Function words like "the" and "of" often spread attention broadly. Content words tend to form sharp attention patterns to semantically related words. Understanding these patterns gives us insight into what the model has learned about language structure.

### Why naive approaches fail

Without self-attention, each word's representation is computed in isolation (bag-of-words) or only from its local neighborhood (CNNs with small kernels). Bag-of-words completely ignores word order and context -- "dog bites man" and "man bites dog" produce identical representations. Local windows miss long-range dependencies: in "The trophy doesn't fit in the suitcase because it is too big," understanding what "it" refers to requires connecting words far apart.

Even bidirectional RNNs, which process the sequence in both directions, struggle because information must flow step-by-step through the chain. By the time distant context reaches a word, it has been compressed and distorted through many transformations.

### Mental models

- **Cocktail party:** At a party, you hear many conversations but selectively tune in to relevant ones. Self-attention is every person simultaneously listening to every other person and deciding whose words matter most to them right now.
- **Wikipedia hyperlinks:** Each word in a sentence is like a Wikipedia article with links to related articles. Self-attention computes which links are strongest and pulls in information from linked articles.
- **Group discussion:** In a meeting, each person updates their understanding by listening to everyone else. The attention weights represent how much each person values what each other person said.

### Visual explanations

```
  SELF-ATTENTION: every word attends to every word

  Input:  "The  cat  sat  on  the  mat"
           |    |    |    |    |    |
           v    v    v    v    v    v
         ┌────────────────────────────┐
         │  Project to Q, K, V        │
         │  Q = X @ W_Q               │
         │  K = X @ W_K               │
         │  V = X @ W_V               │
         └────────────────────────────┘
           |    |    |    |    |    |
           v    v    v    v    v    v
         ┌────────────────────────────┐
         │ Attention matrix A = QK^T  │
         │                            │
         │      The cat sat on the mat│
         │  The [.1  .1  .1  .2  .3 .2]
         │  cat [.1  .3  .3  .1  .0 .2]  <-- "cat" attends
         │  sat [.1  .3  .2  .1  .1 .2]      to "cat" and "sat"
         │   on [.2  .1  .1  .2  .2 .2]
         │  the [.1  .1  .1  .2  .3 .2]
         │  mat [.1  .2  .1  .1  .2 .3]
         └────────────────────────────┘
           |    |    |    |    |    |
           v    v    v    v    v    v
         ┌────────────────────────────┐
         │  Output = A @ V            │
         │  (context-enriched vectors)│
         └────────────────────────────┘
```

---

## Hands-on Exploration

1. Create simple word vectors for a short sentence and compute the full self-attention matrix
2. Visualize the attention matrix as a text-based heatmap to see which words attend to which
3. Observe patterns: do words attend mostly to themselves, to adjacent words, or to semantically related ones?
4. Modify one word's embedding and watch how the entire attention pattern shifts
5. Compare the input embeddings to the attention-enriched output embeddings

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def softmax(x, axis=-1):
    """Row-wise softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# --- Sentence with semantic relationships ---
words = ["The", "cat", "chased", "the", "mouse", "quickly"]
n = len(words)
d_model = 8

# Create embeddings with some semantic structure
embeddings = np.random.randn(n, d_model) * 0.5
# Make "cat" and "mouse" somewhat similar (both animals)
embeddings[1] += np.array([1, 1, 0, 0, 0.5, 0, 0, 0])
embeddings[4] += np.array([1, 1, 0, 0, -0.5, 0, 0, 0])
# Make "chased" an action word
embeddings[2] += np.array([0, 0, 1, 1, 0, 0, 0, 0])
# Make "quickly" related to the action
embeddings[5] += np.array([0, 0, 0.8, 0.6, 0, 0, 0, 0])

# --- Projection matrices ---
d_k = 6
W_Q = np.random.randn(d_model, d_k) * 0.3
W_K = np.random.randn(d_model, d_k) * 0.3
W_V = np.random.randn(d_model, d_k) * 0.3

Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V

# --- Self-attention computation ---
scores = Q @ K.T / np.sqrt(d_k)
attn_matrix = softmax(scores, axis=1)

print("=" * 60)
print("SELF-ATTENTION VISUALIZATION")
print("=" * 60)

# --- Display attention matrix as heatmap ---
print("\nAttention Matrix (each row shows where that word looks):\n")
header = "         " + "".join(f"{w:>9s}" for w in words)
print(header)
print("         " + "-" * (9 * n))

heat_chars = " .:-=+*#@"  # brightness levels

for i in range(n):
    row = f"{words[i]:>8s} |"
    for j in range(n):
        w = attn_matrix[i, j]
        row += f"  {w:.3f}  "
    print(row)

# --- Visual heatmap with block characters ---
print("\nHeatmap (darker = stronger attention):\n")
print("         " + "".join(f"{w:>9s}" for w in words))
print("         " + "-" * (9 * n))
for i in range(n):
    row = f"{words[i]:>8s} |"
    for j in range(n):
        w = attn_matrix[i, j]
        level = min(int(w * len(heat_chars) * 2.5), len(heat_chars) - 1)
        block = heat_chars[level] * 5
        row += f"  {block}  "
    print(row)
print(f"\n  Legend: ' '=0.0  '.'=low  '#'=med  '@'=high attention")

# --- Analyze strongest attention for each word ---
print(f"\n{'=' * 60}")
print("WHERE EACH WORD LOOKS (top 2 attention targets)")
print("=" * 60)
for i in range(n):
    top2 = np.argsort(attn_matrix[i])[::-1][:2]
    print(f"\n  '{words[i]}' attends to:")
    for idx in top2:
        bar = "#" * int(attn_matrix[i, idx] * 30)
        print(f"    -> '{words[idx]}': {attn_matrix[i, idx]:.3f} |{bar}")

# --- Show how self-attention enriches representations ---
print(f"\n{'=' * 60}")
print("REPRESENTATION ENRICHMENT")
print("=" * 60)
output = attn_matrix @ V  # context-enriched representations

for i in range(n):
    sim_before = np.dot(embeddings[i], embeddings[i])
    mixed_info = []
    for j in range(n):
        if j != i and attn_matrix[i, j] > 0.18:
            mixed_info.append(words[j])
    info = ", ".join(mixed_info) if mixed_info else "(mostly self)"
    change = np.linalg.norm(output[i] - V[i])
    print(f"  '{words[i]}' absorbed info from: {info}  "
          f"(change: {change:.3f})")

# --- Cosine similarity before and after attention ---
print(f"\n{'=' * 60}")
print("COSINE SIMILARITY: cat vs mouse")
print("=" * 60)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_before = cosine_sim(embeddings[1], embeddings[4])
sim_after = cosine_sim(output[1], output[4])
print(f"  Before self-attention: {sim_before:.4f}")
print(f"  After self-attention:  {sim_after:.4f}")
print(f"  Context changed the relationship by: {abs(sim_after - sim_before):.4f}")
```

---

## Key Takeaways

- **Self-attention is attention where Q, K, V come from the same sequence.** Every word attends to every other word, building context-aware representations.
- **The attention matrix reveals linguistic structure.** Patterns show which words the model considers related -- verbs attend to their subjects, modifiers attend to what they modify.
- **Representations become context-dependent after self-attention.** The same word "bank" will have different output representations in "river bank" vs "bank account" because it attends to different context words.
- **Self-attention is permutation-equivariant.** The operation itself does not know word order -- it only sees vector similarities. This is why positional encoding (Kata 7.3) is needed.
- **Computational cost is O(n^2) in sequence length.** Every word attends to every other word, creating an n-by-n attention matrix. This is the main scalability challenge of Transformers.
