# Next-Token Prediction

> Phase 8 — Large Language Models | Kata 8.2

---

## Concept & Intuition

### What problem are we solving?

The core task behind language models is deceptively simple: given a sequence of tokens, predict the next one. "The cat sat on the ___" -- most humans would guess "mat" or "floor." A language model does the same thing, but it assigns a probability to every token in its vocabulary. This is called **next-token prediction**, and it is the single training objective behind GPT and similar models.

A **bigram model** is the simplest language model: it predicts the next token using only the previous one. P("mat" | "the") is estimated by counting how often "mat" follows "the" in training data. This is crude -- it ignores all context beyond one token -- but it illustrates the fundamental idea. More powerful models (trigrams, RNNs, transformers) use longer contexts, but the objective is identical: maximize the probability of the actual next token.

We evaluate language models using **perplexity**: the exponential of the average negative log-probability assigned to each token. A perplexity of 10 means the model is as uncertain as choosing uniformly among 10 options. Lower perplexity means better predictions. Perplexity drops as we give the model more context, which is why transformers with thousands of tokens of context outperform bigram models.

### Why naive approaches fail

A uniform random model assigns equal probability to every token -- it has maximum perplexity and generates gibberish. A bigram model captures some structure ("qu" is usually followed by a vowel) but misses long-range dependencies ("The doctor who treated my grandmother's neighbor's cat prescribed ___" -- a bigram model has no idea a medical context is relevant). The lesson: more context enables better prediction, which is exactly what attention mechanisms in transformers provide.

### Mental models

- **Autocomplete on your phone**: it predicts the next word based on what you've typed so far
- **Fill in the blank**: language modeling is a continuous fill-in-the-blank exercise
- **Perplexity as surprise**: low perplexity = the model is rarely surprised by what comes next
- **Context window**: imagine reading a book through a tiny keyhole (bigram) vs a large window (transformer)

### Visual explanations

```
Next-token prediction:

  Input:   [The] [cat] [sat] [on] [the] [?]
                                         |
                                    Probability distribution:
                                    mat    0.25  ████████
                                    floor  0.15  █████
                                    chair  0.10  ███
                                    ...    ...
                                    zebra  0.001

Bigram model (context = 1):        Longer context (context = 5):
  P(next | "the") -->               P(next | "the cat sat on the") -->
  Knows only one word.              Knows full sentence.
  Many things follow "the".         "mat" is much more likely.

Perplexity comparison:
  Uniform random:  |████████████████████████| 1000 (vocab size)
  Bigram model:    |████████████|             120
  5-gram model:    |██████|                    45
  Transformer:     |██|                        15
```

---

## Hands-on Exploration

1. Build a bigram transition matrix from a text corpus and sample from it
2. Compute perplexity for bigram vs uniform models on the same text
3. Extend to a trigram model and observe the perplexity improvement

---

## Live Code

```python
import numpy as np
np.random.seed(42)

# --- Build a simple corpus ---
corpus = "the cat sat on the mat the cat ate the rat the dog sat on the mat"
words = corpus.split()
vocab = sorted(set(words))
w2i = {w: i for i, w in enumerate(vocab)}
ids = [w2i[w] for w in words]
V = len(vocab)
print(f"Corpus: '{corpus}'")
print(f"Vocab ({V}): {vocab}")

# --- Bigram model: count transitions ---
bigram_counts = np.zeros((V, V))
for i in range(len(ids) - 1):
    bigram_counts[ids[i], ids[i + 1]] += 1

# Add smoothing and normalize to probabilities
bigram_probs = (bigram_counts + 0.1) / (bigram_counts + 0.1).sum(axis=1, keepdims=True)

print("\n=== Bigram Transition Probabilities ===")
print(f"{'From':<8}", end="")
for w in vocab:
    print(f"{w:>8}", end="")
print()
for i, w in enumerate(vocab):
    print(f"{w:<8}", end="")
    for j in range(V):
        print(f"{bigram_probs[i,j]:>8.2f}", end="")
    print()

# --- Generate text with bigram model ---
def generate_bigram(start_word, length=8):
    idx = w2i[start_word]
    result = [start_word]
    for _ in range(length):
        idx = np.random.choice(V, p=bigram_probs[idx])
        result.append(vocab[idx])
    return " ".join(result)

print("\n=== Generated Text (Bigram) ===")
for _ in range(3):
    print(f"  {generate_bigram('the')}")

# --- Compute perplexity ---
def perplexity(probs_matrix, token_ids):
    log_prob_sum = 0
    n = 0
    for i in range(len(token_ids) - 1):
        p = probs_matrix[token_ids[i], token_ids[i + 1]]
        log_prob_sum += np.log2(p + 1e-10)
        n += 1
    return 2 ** (-log_prob_sum / n)

# Uniform model: equal probability for everything
uniform_probs = np.ones((V, V)) / V

bigram_ppl = perplexity(bigram_probs, ids)
uniform_ppl = perplexity(uniform_probs, ids)

print(f"\n=== Perplexity Comparison ===")
print(f"Uniform model:  {uniform_ppl:.1f}  (as confused as picking from {V} words)")
print(f"Bigram model:   {bigram_ppl:.1f}  (much less confused!)")
print(f"Perfect model:  1.0  (always knows the answer)")

# --- Show how context helps ---
print("\n=== More Context = Better Prediction ===")
print("After 'the':     could be cat, dog, mat, rat  (4 options)")
the_next = bigram_probs[w2i['the']]
top = np.argsort(the_next)[::-1][:4]
for i in top:
    print(f"  P({vocab[i]:>5} | the) = {the_next[i]:.2f}")
```

---

## Key Takeaways

- **Language modeling is next-token prediction.** Given context, assign probabilities to every possible next token.
- **Bigram models are the simplest language model.** They predict using only the previous token, which captures local patterns but misses long-range meaning.
- **Perplexity measures prediction quality.** Lower perplexity means the model is less "surprised" by the actual text.
- **More context reduces perplexity.** The jump from bigrams to transformers is largely about using longer context windows.
- **All LLMs share this objective.** GPT-2, GPT-4, and LLaMA all train on the same core task -- they just differ in how much context they can use and how they parameterize the probability distribution.
