# N-Grams

> Phase 6 — Sequence Models | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

Language has **order**. The sentence "the cat sat on the mat" makes sense, but "mat the on sat cat the" doesn't. If we want a model that can generate or predict text, it needs to understand which characters or words are likely to follow others. N-gram models are the simplest way to capture this sequential structure.

An n-gram is a contiguous sequence of n items from text. A bigram is 2 items, a trigram is 3. The core idea is beautifully simple: to predict the next character, just look at the previous n-1 characters and see what has historically followed that context. If 'th' is frequently followed by 'e', then after seeing 'th' we should predict 'e' with high probability.

This is the **Markov assumption**: the future depends only on the recent past, not the entire history. A bigram model says "only the last character matters." A trigram says "only the last two characters matter." This is obviously wrong for real language (the word at the end of a paragraph may depend on the first sentence), but it's a powerful simplification that works surprisingly well for local patterns.

### Why naive approaches fail

Without n-grams, you might try predicting the next character by looking at overall character frequencies: 'e' is common in English, so always predict 'e'. But this ignores context entirely. After 'q', the next letter is almost always 'u', not 'e'. Character frequencies without context are nearly useless for generation.

You could try to condition on the entire history of text seen so far, but with a vocabulary of even 26 letters, the number of possible histories grows exponentially. After 10 characters, there are 26^10 possible contexts -- you'd never see most of them in training data. N-grams strike a practical balance: short enough contexts that you can estimate probabilities reliably, long enough to capture useful patterns.

### Mental models

- **Autocomplete on your phone**: it looks at the last 1-2 words you typed and suggests what's most likely next. That's an n-gram model.
- **Finishing someone's sentence**: if someone says "how are..." you predict "you" because that trigram is incredibly common.
- **Cooking by pattern**: after "peanut butter and..." you know "jelly" is coming. You don't need the whole recipe, just the last few words.

### Visual explanations

```
Text: "the cat sat on the mat"

Bigrams (character-level):         Count
  "th" -> "e"                        2
  "he" -> " "                        2
  "e " -> "c", "m"                   1 each
  "ca" -> "t"                        1
  "at" -> " "                        2
  " s" -> "a"                        1
  ...

Probability calculation:
  P(next='e' | prev='th') = count("the") / count("th") = 2/2 = 1.0
  P(next=' ' | prev='at') = count("at ") / count("at") = 2/3 = 0.67

Generation (sampling chain):
  Start: "t" -> "h" (100%) -> "e" (100%) -> " " (100%) -> ...
         Each step: look up distribution, sample next character
```

---

## Hands-on Exploration

1. Count all character bigrams in a sample text and find the most common transitions
2. Convert raw counts to probability distributions and verify they sum to 1.0
3. Generate new text by sampling from the learned distributions and observe how it captures local patterns but produces nonsense globally

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Training text ---
text = "the cat sat on the mat the cat ate the rat the bat sat on the hat"
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"Text: '{text}'")
print(f"Vocabulary ({vocab_size} chars): {chars}\n")

# --- Build bigram counts ---
bigram_counts = np.zeros((vocab_size, vocab_size))
for i in range(len(text) - 1):
    c1 = char_to_idx[text[i]]
    c2 = char_to_idx[text[i + 1]]
    bigram_counts[c1][c2] += 1

# --- Convert counts to probabilities ---
bigram_probs = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)
bigram_probs = np.nan_to_num(bigram_probs)  # handle zero rows

# --- Show top bigram transitions ---
print("Top 10 bigram transitions:")
print(f"  {'Bigram':<10} {'Count':>6} {'Prob':>6}")
print("  " + "-" * 24)
pairs = []
for i in range(vocab_size):
    for j in range(vocab_size):
        if bigram_counts[i][j] > 0:
            pairs.append((chars[i], chars[j], bigram_counts[i][j], bigram_probs[i][j]))
pairs.sort(key=lambda x: -x[2])
for c1, c2, count, prob in pairs[:10]:
    display = f"'{c1}{c2}'"
    print(f"  {display:<10} {count:>6.0f} {prob:>6.2f}")

# --- Build trigram model ---
trigram_counts = {}
for i in range(len(text) - 2):
    context = text[i:i+2]
    next_char = text[i+2]
    if context not in trigram_counts:
        trigram_counts[context] = {}
    trigram_counts[context][next_char] = trigram_counts[context].get(next_char, 0) + 1

print("\nTrigram examples (context -> next char distribution):")
for ctx in ["th", "at", "he", " t", "on"]:
    if ctx in trigram_counts:
        total = sum(trigram_counts[ctx].values())
        dist = {c: f"{n/total:.0%}" for c, n in trigram_counts[ctx].items()}
        print(f"  '{ctx}' -> {dist}")

# --- Generate text with bigram model ---
def generate_bigram(start_char, length):
    result = [start_char]
    for _ in range(length - 1):
        idx = char_to_idx[result[-1]]
        probs = bigram_probs[idx]
        if probs.sum() == 0:
            break
        next_idx = np.random.choice(vocab_size, p=probs)
        result.append(chars[next_idx])
    return "".join(result)

print("\nGenerated text (bigram, 5 samples):")
for i in range(5):
    print(f"  '{generate_bigram('t', 30)}'")

# --- The Markov assumption visualized ---
print("\nMarkov assumption: only the LAST character matters for bigrams")
print(f"  After 't': {dict(zip([chars[j] for j in range(vocab_size) if bigram_probs[char_to_idx['t']][j] > 0], [f'{bigram_probs[char_to_idx[\"t\"]][j]:.0%}' for j in range(vocab_size) if bigram_probs[char_to_idx['t']][j] > 0]))}")
print("  Doesn't matter if 't' appears in 'cat', 'bat', or 'the' -- same distribution!")
```

---

## Key Takeaways

- **N-grams capture local sequential patterns.** By counting which characters follow which, we learn the statistical texture of a language at a local level.
- **The Markov assumption is a powerful simplification.** Assuming only the last n-1 characters matter makes estimation tractable, but limits how far back the model can "see."
- **Counts become probabilities through normalization.** Dividing each row by its sum turns raw frequency counts into proper probability distributions.
- **Longer n-grams capture more context but need exponentially more data.** Trigrams are better than bigrams, but you need far more text to estimate them reliably (the curse of dimensionality).
- **N-grams set the stage for neural sequence models.** Their fundamental limitation -- fixed, short context windows -- motivates the RNNs we'll build next.
