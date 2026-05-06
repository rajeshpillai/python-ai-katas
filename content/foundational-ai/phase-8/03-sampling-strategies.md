# Sampling Strategies

> Phase 8 — Large Language Models (LLMs) | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

A language model outputs a probability distribution over the entire vocabulary for the next token. The critical question is: how do we choose which token to actually generate? This choice dramatically affects the quality, creativity, and coherence of the output. The same model can produce boring repetitive text or creative diverse text depending solely on how we sample from its output distribution.

Consider a model that predicts the next word after "The cat sat on the". It might assign 30% probability to "mat", 15% to "floor", 10% to "roof", 8% to "table", and smaller probabilities to thousands of other words. Greedy decoding always picks "mat" -- safe but predictable. Pure random sampling might pick "elephant" (0.001% probability) -- creative but nonsensical. The art of text generation lies in finding the sweet spot between these extremes.

Nucleus (top-p) sampling emerged as a principled solution. Instead of a fixed number of candidates, it dynamically selects the smallest set of tokens whose cumulative probability exceeds a threshold p. For confident predictions (one token at 95%), it acts nearly greedy. For uncertain predictions (many tokens at 5% each), it allows broad exploration. This adaptive behavior matches human intuition about when to be predictable versus creative.

### Why naive approaches fail

Greedy decoding (always pick the highest-probability token) seems safe but produces degenerate text. It tends to get stuck in repetitive loops: "The cat sat on the mat. The cat sat on the mat. The cat sat..." This happens because once the model enters a high-probability pattern, greedy decoding keeps reinforcing it. Human language is not a sequence of locally optimal choices -- we sometimes pick surprising words that lead to interesting sentences.

Pure random sampling from the full distribution has the opposite problem. Most words in the vocabulary have tiny but nonzero probabilities. Sampling from the full distribution means occasionally picking wildly inappropriate words, producing text like "The cat sat on the refrigerator quantum Belgium." The long tail of low-probability tokens contains mostly noise, and sampling from it produces incoherent text.

### Mental models

- **Restaurant ordering:** Greedy is always ordering the most popular dish -- safe but boring. Random sampling is blindly pointing at the menu -- you might get dessert as your main course. Nucleus sampling is choosing from the chef's recommended dishes -- a curated set of good options.
- **Music improvisation:** A greedy jazz musician always plays the most expected note -- technically correct but lifeless. A fully random player hits random keys -- chaos. A skilled improviser chooses from a dynamic set of notes that "fit" -- sometimes the expected resolution, sometimes a surprising but consonant choice.
- **Exploring a city:** Greedy always takes the main road. Random walks in any direction including into walls. Nucleus sampling follows interesting-looking streets that are actually navigable.

### Visual explanations

```
  PROBABILITY DISTRIBUTION FOR NEXT TOKEN:

  Token:    mat   floor  roof  table chair  ...  banana  quantum
  Prob:    0.30   0.15  0.10  0.08  0.07   ...  0.001   0.0001

  GREEDY (always pick max):
  ████████████████████████████████ mat (0.30)
  Always picks "mat". Repetitive and boring.

  RANDOM (sample from full distribution):
  Any token can be picked, weighted by probability.
  Sometimes picks "quantum" (0.0001). Incoherent.

  NUCLEUS / TOP-P (p=0.7):
  ████████████████████████████████ mat    (0.30)  ─┐
  ████████████████                 floor  (0.15)   │ cumulative
  ██████████                       roof   (0.10)   ├ sum = 0.70
  █████████                        table  (0.08)   │ >= p
  ████████                         chair  (0.07)  ─┘
  - - - - - cutoff - - - - - - - - - - - - - - - -
  ██████                           bed    (0.05)  ← excluded
  ...                                             ← excluded

  Only sample from the top tokens that sum to >= p.
  Renormalize their probabilities and sample.
```

---

## Hands-on Exploration

1. Create a simulated vocabulary with realistic probability distributions (some peaked, some flat)
2. Generate sequences using greedy decoding and observe the repetition problem
3. Generate sequences using pure random sampling and observe the incoherence problem
4. Implement nucleus (top-p) sampling and observe the improved quality
5. Compare the diversity and quality of outputs across all three strategies

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Simulated vocabulary and transition probabilities ---
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "big",
         "red", "hat", "and", "then", "a", "very", "fast", "slow"]
V = len(vocab)

# Transition probabilities (simplified bigram model)
# Each row: probability of next word given current word
transitions = np.random.dirichlet(np.ones(V) * 0.3, size=V)
# Make some transitions more peaked (realistic)
for i in range(V):
    transitions[i][np.argmax(transitions[i])] += 0.3
    transitions[i] /= transitions[i].sum()

def greedy_decode(start_idx, length):
    """Always pick the most probable next token."""
    tokens = [start_idx]
    for _ in range(length):
        probs = transitions[tokens[-1]]
        tokens.append(np.argmax(probs))
    return tokens

def random_sample(start_idx, length):
    """Sample from the full distribution."""
    tokens = [start_idx]
    for _ in range(length):
        probs = transitions[tokens[-1]]
        tokens.append(np.random.choice(V, p=probs))
    return tokens

def nucleus_sample(start_idx, length, p=0.7):
    """Top-p (nucleus) sampling."""
    tokens = [start_idx]
    for _ in range(length):
        probs = transitions[tokens[-1]]
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        # Find cutoff: smallest set with cumsum >= p
        cutoff = np.searchsorted(cumsum, p) + 1
        # Keep only nucleus tokens
        nucleus_idx = sorted_idx[:cutoff]
        nucleus_probs = probs[nucleus_idx]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        chosen = np.random.choice(nucleus_idx, p=nucleus_probs)
        tokens.append(chosen)
    return tokens

def tokens_to_text(tokens):
    return " ".join(vocab[t] for t in tokens)

def measure_diversity(tokens):
    """Unique bigrams / total bigrams."""
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return len(set(bigrams)) / len(bigrams)

print("=" * 60)
print("SAMPLING STRATEGIES COMPARISON")
print("=" * 60)

start = 0  # start with "the"
length = 15

# --- Greedy ---
print("\n--- Greedy Decoding ---")
for trial in range(3):
    tokens = greedy_decode(start, length)
    print(f"  [{trial+1}] {tokens_to_text(tokens)}")
    print(f"      Diversity: {measure_diversity(tokens):.2f}")

# --- Random ---
print("\n--- Random Sampling ---")
for trial in range(3):
    tokens = random_sample(start, length)
    print(f"  [{trial+1}] {tokens_to_text(tokens)}")
    print(f"      Diversity: {measure_diversity(tokens):.2f}")

# --- Nucleus ---
print("\n--- Nucleus Sampling (p=0.7) ---")
for trial in range(3):
    tokens = nucleus_sample(start, length, p=0.7)
    print(f"  [{trial+1}] {tokens_to_text(tokens)}")
    print(f"      Diversity: {measure_diversity(tokens):.2f}")

# --- Detailed view of one step ---
print(f"\n{'=' * 60}")
print("DETAILED: One Sampling Step")
print("=" * 60)
current_word = 1  # "cat"
probs = transitions[current_word]
sorted_idx = np.argsort(probs)[::-1]

print(f"\nAfter '{vocab[current_word]}', next word probabilities:")
cumsum = 0
p_threshold = 0.7
cutoff_shown = False
for rank, idx in enumerate(sorted_idx[:10]):
    cumsum += probs[idx]
    marker = ""
    if cumsum >= p_threshold and not cutoff_shown:
        marker = " <-- nucleus cutoff (p=0.7)"
        cutoff_shown = True
    bar = "#" * int(probs[idx] * 60)
    print(f"  {vocab[idx]:>6s}: {probs[idx]:.3f} (cum:{cumsum:.3f}) "
          f"|{bar}{marker}")

# --- Repetition analysis ---
print(f"\n{'=' * 60}")
print("REPETITION ANALYSIS (100 trials, length=30)")
print("=" * 60)
n_trials = 100
for name, sampler in [("Greedy", lambda: greedy_decode(start, 30)),
                       ("Random", lambda: random_sample(start, 30)),
                       ("Nucleus p=0.7", lambda: nucleus_sample(start, 30, 0.7))]:
    diversities = []
    for _ in range(n_trials):
        tokens = sampler()
        diversities.append(measure_diversity(tokens))
    mean_div = np.mean(diversities)
    std_div = np.std(diversities)
    bar = "#" * int(mean_div * 40)
    print(f"  {name:>15s}: diversity={mean_div:.3f} +/- {std_div:.3f} |{bar}")
```

---

## Key Takeaways

- **Greedy decoding is deterministic and repetitive.** Always picking the most likely token leads to loops and boring, predictable text.
- **Random sampling includes too much noise.** The long tail of low-probability tokens produces incoherent text when sampled.
- **Nucleus (top-p) sampling adapts to model confidence.** It selects from a dynamic set of plausible tokens, being conservative when the model is confident and exploratory when it is uncertain.
- **Diversity and coherence are in tension.** Sampling strategies navigate the tradeoff between generating varied text and maintaining logical consistency.
- **The sampling strategy is not part of the model.** The same trained model can produce vastly different outputs depending on how we decode from its probability distributions.
