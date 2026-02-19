# Temperature, Top-k, Top-p

> Phase 8 — Large Language Models (LLMs) | Kata 8.4

---

## Concept & Intuition

### What problem are we solving?

In Kata 8.3, we saw that how we sample from a language model's output distribution matters enormously. This kata dives deep into three specific controls that shape the distribution before sampling: temperature, top-k, and top-p. These are the "knobs" that practitioners turn when deploying language models, and understanding their mechanics is essential for getting good results.

Temperature modifies the softmax distribution by dividing the logits (raw model outputs) by a temperature parameter T before applying softmax. When T is low (e.g., 0.1), the distribution becomes sharply peaked -- the model is very "confident" and almost always picks the top token. When T is high (e.g., 2.0), the distribution flattens out -- all tokens become more equally likely, increasing randomness. At T=1.0, the distribution is unchanged from the model's original output.

Top-k and top-p are filtering strategies that zero out low-probability tokens before sampling. Top-k keeps only the k highest-probability tokens. Top-p (nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds p. The key difference is that top-k uses a fixed number of candidates regardless of the distribution shape, while top-p adapts -- using fewer candidates when the model is confident and more when it is uncertain.

### Why naive approaches fail

Using temperature alone has a problem: even at low temperature, the model might still have some probability mass on clearly wrong tokens. Temperature reshapes the distribution but does not remove bad options entirely. A slightly more probable but still inappropriate token can occasionally be sampled.

Top-k alone has the problem of using a fixed window. If the model assigns 95% probability to one token, using k=50 means 49 of those candidates are very low probability and might produce bad output. Conversely, if the model is genuinely uncertain among 100 plausible options, k=10 cuts off many good choices. Top-p solves this by adapting the candidate set to the distribution shape, but in practice, combining all three techniques often yields the best results.

### Mental models

- **Temperature as a thermostat:** Low temperature freezes the distribution into a sharp spike (one clear winner). High temperature melts it into a flat puddle (everything equally likely). Room temperature (T=1) is the model's natural output.
- **Top-k as a VIP list:** Only the top k candidates get into the club, regardless of how qualified others might be. Simple but rigid.
- **Top-p as a budget:** You have a probability "budget" of p. You invite candidates starting from the most likely until you have spent your budget. Popular events (confident predictions) have few attendees; uncertain ones have many.

### Visual explanations

```
  TEMPERATURE EFFECT ON DISTRIBUTION

  Original logits: [3.0, 2.0, 1.0, 0.5, 0.1, -1.0]

  T=0.5 (sharp):     ████████████████████  0.73
                      ████████              0.18
                      ███                   0.07
                      ██                    0.02
                                            0.01
                                            0.00

  T=1.0 (original):  ██████████████        0.38
                      █████████             0.14
                      █████                 0.05
                      ████                  0.03
                      ███                   0.02
                      █                     0.01

  T=2.0 (flat):      ████████████          0.26
                      ██████████            0.20
                      ████████              0.15
                      ███████               0.13
                      ███████               0.12
                      █████                 0.09

  TOP-K FILTERING (k=3):
  Keep top 3 tokens, zero out the rest, renormalize.

  Before:  [0.38, 0.14, 0.05, 0.03, 0.02, 0.01]
  Mask:    [keep, keep, keep, zero, zero, zero]
  After:   [0.67, 0.25, 0.09, 0.00, 0.00, 0.00]

  TOP-P FILTERING (p=0.9):
  Keep tokens until cumulative prob >= 0.9.

  Sorted:  0.38 + 0.14 + 0.05 + 0.03 + ... = 0.90 >= p
  Keep:    [0.38, 0.14, 0.05, 0.03]  (4 tokens)
  Renorm:  [0.63, 0.23, 0.08, 0.05]
```

---

## Hands-on Exploration

1. Start with raw logits and apply softmax at different temperatures to see the distribution shape change
2. Implement top-k filtering and observe how it truncates the tail
3. Implement top-p filtering and observe how it adapts the number of candidates
4. Combine temperature with top-k and top-p to see the compound effect
5. Generate text sequences with different parameter combinations and compare quality

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def softmax_with_temperature(logits, temperature=1.0):
    """Apply temperature scaling then softmax."""
    scaled = logits / temperature
    e = np.exp(scaled - np.max(scaled))
    return e / e.sum()

def top_k_filter(probs, k):
    """Keep only top-k tokens, zero out rest, renormalize."""
    filtered = np.zeros_like(probs)
    top_k_idx = np.argsort(probs)[::-1][:k]
    filtered[top_k_idx] = probs[top_k_idx]
    return filtered / filtered.sum()

def top_p_filter(probs, p):
    """Keep smallest set of tokens with cumulative prob >= p."""
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    filtered = np.zeros_like(probs)
    filtered[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
    return filtered / filtered.sum()

# --- Vocabulary and logits ---
tokens = ["the", "cat", "sat", "mat", "dog", "ran", "big", "on",
          "hat", "red", "was", "and", "it", "very", "a", "nice"]
logits = np.array([3.5, 2.8, 1.5, 1.0, 0.8, 0.5, 0.2, 0.0,
                   -0.3, -0.5, -0.8, -1.0, -1.5, -2.0, -2.5, -3.0])

print("=" * 60)
print("TEMPERATURE, TOP-K, TOP-P")
print("=" * 60)

# --- Temperature comparison ---
print("\n--- Temperature Effect ---\n")
for T in [0.3, 0.7, 1.0, 1.5, 3.0]:
    probs = softmax_with_temperature(logits, T)
    top3_idx = np.argsort(probs)[::-1][:3]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    print(f"  T={T:.1f}  entropy={entropy:.2f}  "
          f"top: {tokens[top3_idx[0]]}={probs[top3_idx[0]]:.3f}, "
          f"{tokens[top3_idx[1]]}={probs[top3_idx[1]]:.3f}, "
          f"{tokens[top3_idx[2]]}={probs[top3_idx[2]]:.3f}")
    bar = ""
    for i in range(min(8, len(tokens))):
        idx = np.argsort(probs)[::-1][i]
        bar += "#" * max(1, int(probs[idx] * 40)) + " "
    print(f"         [{bar.strip()}]")

# --- Top-k comparison ---
print(f"\n--- Top-k Filtering ---\n")
base_probs = softmax_with_temperature(logits, 1.0)
print(f"  Original distribution (top 8):")
for i, idx in enumerate(np.argsort(base_probs)[::-1][:8]):
    bar = "#" * int(base_probs[idx] * 50)
    print(f"    {tokens[idx]:>5s}: {base_probs[idx]:.4f} |{bar}")

for k in [3, 5, 10]:
    filtered = top_k_filter(base_probs, k)
    nonzero = np.sum(filtered > 0)
    ent = -np.sum(filtered * np.log(filtered + 1e-10))
    print(f"\n  top-k={k} ({nonzero} tokens kept, entropy={ent:.2f}):")
    for idx in np.argsort(filtered)[::-1][:k]:
        if filtered[idx] > 0:
            bar = "#" * int(filtered[idx] * 50)
            print(f"    {tokens[idx]:>5s}: {filtered[idx]:.4f} |{bar}")

# --- Top-p comparison ---
print(f"\n--- Top-p Filtering ---\n")
for p in [0.5, 0.7, 0.9, 0.95]:
    filtered = top_p_filter(base_probs, p)
    nonzero = np.sum(filtered > 0)
    ent = -np.sum(filtered * np.log(filtered + 1e-10))
    kept = [tokens[i] for i in np.argsort(filtered)[::-1] if filtered[i] > 0]
    print(f"  top-p={p:.2f}: {nonzero} tokens kept, "
          f"entropy={ent:.2f}, tokens={kept}")

# --- Combined: Temperature + Top-p ---
print(f"\n{'=' * 60}")
print("COMBINING: Temperature + Top-p")
print("=" * 60)
combos = [(0.5, 0.9), (1.0, 0.9), (1.5, 0.9),
          (1.0, 0.5), (1.0, 0.7), (1.0, 0.95)]
for T, p in combos:
    probs = softmax_with_temperature(logits, T)
    filtered = top_p_filter(probs, p)
    nonzero = np.sum(filtered > 0)
    top_token = tokens[np.argmax(filtered)]
    top_prob = np.max(filtered)
    ent = -np.sum(filtered * np.log(filtered + 1e-10))
    label = f"T={T:.1f}, p={p:.2f}"
    bar = "#" * int(ent * 8)
    print(f"  {label:>14s} -> {nonzero:2d} tokens, "
          f"top='{top_token}'({top_prob:.2f}), entropy={ent:.2f} |{bar}")

# --- Generate sequences with different settings ---
print(f"\n{'=' * 60}")
print("GENERATED SEQUENCES (5 tokens each)")
print("=" * 60)
# Simple bigram transition logits
trans_logits = np.random.randn(len(tokens), len(tokens)) * 1.5
for i in range(len(tokens)):
    trans_logits[i, i] -= 1.0  # discourage self-loops

configs = [("Greedy (T->0)", 0.01, 1.0, 16),
           ("Cold (T=0.3)", 0.3, 0.9, 16),
           ("Normal (T=1)", 1.0, 0.9, 16),
           ("Hot (T=2.0)", 2.0, 0.95, 16),
           ("Focused (k=3)", 1.0, 1.0, 3)]

for name, T, p, k in configs:
    print(f"\n  {name}:")
    for trial in range(3):
        seq = [0]  # start with "the"
        for _ in range(5):
            raw = trans_logits[seq[-1]]
            probs = softmax_with_temperature(raw, T)
            probs = top_k_filter(probs, k)
            probs = top_p_filter(probs, p)
            seq.append(np.random.choice(len(tokens), p=probs))
        text = " ".join(tokens[t] for t in seq)
        print(f"    [{trial+1}] {text}")
```

---

## Key Takeaways

- **Temperature controls distribution sharpness.** T<1 makes the model more deterministic (sharper peaks), T>1 makes it more random (flatter distribution), T=1 is the model's natural output.
- **Top-k filters by rank.** It keeps exactly k candidates regardless of the probability distribution shape, which can be too many or too few depending on model confidence.
- **Top-p filters by cumulative probability.** It dynamically adjusts the candidate set size based on the distribution, keeping fewer options when confident and more when uncertain.
- **These controls compose.** In practice, temperature is applied first (reshaping the distribution), then top-k or top-p filtering removes unlikely candidates, and finally sampling picks from the remaining set.
- **There is no universally best setting.** Creative writing benefits from higher temperature and broader sampling. Factual Q&A benefits from low temperature and tight filtering. The right settings depend on the task.
