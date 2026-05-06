# Positional Encoding

> Phase 7 — Attention & Transformers | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

Self-attention (Kata 7.2) has a fundamental blind spot: it is completely indifferent to the order of its inputs. If you shuffle the words in a sentence, self-attention produces the exact same attention weights (just in a different order). But word order is crucial to meaning -- "dog bites man" and "man bites dog" mean very different things. We need a way to inject position information into the model so it can distinguish first from last, nearby from distant.

Positional encoding solves this by adding a unique position-dependent signal to each word's embedding before it enters the self-attention layers. The original Transformer paper by Vaswani et al. (2017) proposed sinusoidal positional encodings: each position gets a vector computed from sine and cosine functions at different frequencies. This elegant approach requires no learned parameters and has a beautiful mathematical property -- the encoding of any position can be expressed as a linear transformation of any other position's encoding, making it easy for the model to learn relative position relationships.

The key insight is that by using sinusoids at many different frequencies, you create a unique "fingerprint" for each position. Low-frequency components capture coarse position (beginning vs end of sequence), while high-frequency components capture fine position (this exact spot). Together, they give the model everything it needs to reason about both absolute and relative positions.

### Why naive approaches fail

The simplest approach -- just adding the integer position (0, 1, 2, ...) to each embedding -- fails because these numbers have unbounded magnitude. Position 500 would dominate the embedding values, drowning out the actual word meaning. You could normalize by dividing by sequence length, but then the same position gets different encodings for different sequence lengths, and the model cannot generalize.

One-hot position vectors (a separate dimension for each position) waste parameters and cannot generalize to sequences longer than what was seen during training. Learned position embeddings work well in practice but also fix a maximum sequence length and lack the elegant extrapolation properties of sinusoidal encodings.

### Mental models

- **Clock hands:** A clock uses two hands (hour and minute) rotating at different speeds to uniquely identify any time. Sinusoidal encodings use many "hands" (sine/cosine pairs) at different frequencies to uniquely identify any position.
- **Musical tuning fork set:** Each position is like striking multiple tuning forks simultaneously. The unique combination of vibrations (frequencies) creates a distinct "chord" for each position, and nearby positions sound similar.
- **Binary counting with smooth transitions:** Binary numbers give each position a unique code, but transitions are abrupt. Sinusoidal encoding is like smooth binary -- each dimension oscillates between -1 and +1, with different dimensions cycling at different rates.

### Visual explanations

```
  SINUSOIDAL POSITIONAL ENCODING

  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Position 0:  sin(0)=0.00  cos(0)=1.00  sin(0)=0.00  cos(0)=1.00 ...
  Position 1:  sin(1)=0.84  cos(1)=0.54  sin(.01)=.01  cos(.01)=1.0 ...
  Position 2:  sin(2)=0.91  cos(2)=-.42  sin(.02)=.02  cos(.02)=1.0 ...

  Dimension:    high freq <-----------> low freq
                ▓▓▓▓▓▓▓▓               ░░░░░░░░

  WHY NEARBY POSITIONS ARE SIMILAR:

  Position 5:  [0.96, 0.28, 0.05, 1.00, 0.01, 1.00, ...]
  Position 6:  [0.28,-0.96, 0.06, 1.00, 0.01, 1.00, ...]
  Position 50: [-0.26,0.96, 0.48, 0.88, 0.05, 1.00, ...]

  Positions 5 and 6 share similar low-frequency components.
  Position 50 differs from 5 in both high and low frequencies.

  ADDING POSITIONAL ENCODING TO WORD EMBEDDINGS:

  word_embedding("cat")   = [0.5, -0.3, 0.8, 0.1]
  positional_encoding(2)  = [0.9, -0.4, 0.0, 1.0]
  ─────────────────────────────────────────────────
  input to transformer    = [1.4, -0.7, 0.8, 1.1]   (element-wise sum)
```

---

## Hands-on Exploration

1. Implement the sinusoidal positional encoding formula for a range of positions
2. Visualize the encoding matrix to see the wave patterns across positions and dimensions
3. Compute dot-product similarity between position encodings to verify that nearby positions are more similar
4. Show that the distance between positions 5 and 6 is the same as between positions 100 and 101 (translation invariance)
5. Add positional encodings to word embeddings and observe how identical words at different positions become distinguishable

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def positional_encoding(max_len, d_model):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    PE[:, 0::2] = np.sin(position / div_term)  # even dims
    PE[:, 1::2] = np.cos(position / div_term)  # odd dims
    return PE

max_len = 50
d_model = 16
PE = positional_encoding(max_len, d_model)

print("=" * 60)
print("POSITIONAL ENCODING")
print("=" * 60)

# --- Visualize the encoding pattern ---
print("\nEncoding heatmap (positions 0-9, dims 0-15):")
print("         " + "".join(f" d{d:<3d}" for d in range(d_model)))
print("         " + "-" * (5 * d_model))
chars = " .:-=+*#@"
for pos in range(10):
    row = f"  pos {pos:2d} |"
    for dim in range(d_model):
        val = PE[pos, dim]
        level = int((val + 1) / 2 * (len(chars) - 1))
        level = max(0, min(len(chars) - 1, level))
        row += f"  {chars[level]}  "
    print(row)
print(f"  Legend: ' '=-1.0   '='=0.0   '@'=+1.0")

# --- Similarity between positions ---
print(f"\n{'=' * 60}")
print("SIMILARITY BETWEEN POSITIONS (dot product)")
print("=" * 60)
positions_to_check = [0, 1, 2, 5, 10, 25, 49]
header = "      " + "".join(f" p{p:<4d}" for p in positions_to_check)
print(header)
print("      " + "-" * (6 * len(positions_to_check)))
for i in positions_to_check:
    row = f"p{i:<4d}|"
    for j in positions_to_check:
        sim = np.dot(PE[i], PE[j]) / d_model
        row += f" {sim:+.2f}"
    print(row)

# --- Nearby positions are more similar ---
print(f"\n{'=' * 60}")
print("NEARBY POSITIONS ARE MORE SIMILAR")
print("=" * 60)
ref_pos = 10
print(f"\nDistance from position {ref_pos} to others:")
for offset in [0, 1, 2, 5, 10, 20, 39]:
    other = ref_pos + offset
    if other < max_len:
        dist = np.linalg.norm(PE[ref_pos] - PE[other])
        bar = "#" * int(dist * 5)
        print(f"  pos {other:2d} (offset {offset:+3d}): "
              f"dist={dist:.3f} |{bar}")

# --- Translation invariance ---
print(f"\n{'=' * 60}")
print("TRANSLATION INVARIANCE")
print("=" * 60)
pairs = [(0, 1), (10, 11), (25, 26), (48, 49)]
print("\nDistance between consecutive positions at different locations:")
for a, b in pairs:
    dist = np.linalg.norm(PE[a] - PE[b])
    print(f"  |PE({a:2d}) - PE({b:2d})| = {dist:.4f}")

# --- Effect on word embeddings ---
print(f"\n{'=' * 60}")
print("SAME WORD, DIFFERENT POSITIONS")
print("=" * 60)
word_vec = np.random.randn(d_model) * 0.5  # embedding for "cat"
pos3 = word_vec + PE[3]
pos20 = word_vec + PE[20]
cos_sim = np.dot(pos3, pos20) / (np.linalg.norm(pos3) * np.linalg.norm(pos20))
print(f"\n  'cat' at position 3 vs 'cat' at position 20:")
print(f"  Cosine similarity of raw embedding with itself:  1.0000")
print(f"  Cosine similarity after adding pos encoding:     {cos_sim:.4f}")
print(f"  The model can now distinguish the two positions!")
```

---

## Key Takeaways

- **Transformers need explicit position information.** Self-attention is permutation-equivariant by default, so without positional encoding, word order is invisible.
- **Sinusoidal encoding uses frequency to create unique position fingerprints.** Low-frequency components encode coarse position, high-frequency components encode fine position.
- **Nearby positions have similar encodings.** The smooth sinusoidal functions ensure that positions close together have high dot-product similarity, encoding a natural notion of proximity.
- **The encoding is translation-invariant in relative distance.** The relationship between positions 5 and 7 is the same as between 100 and 102, which helps the model learn relative position patterns.
- **Positional encoding is added, not concatenated.** The position signal is summed with the word embedding, which means position and meaning share the same vector space and can interact through attention.
