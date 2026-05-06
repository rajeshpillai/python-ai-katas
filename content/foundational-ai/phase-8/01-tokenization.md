# Tokenization

> Phase 8 — Large Language Models | Kata 8.1

---

## Concept & Intuition

### What problem are we solving?

Language models don't understand text -- they understand numbers. Before any neural network can process "The cat sat on the mat", we need to convert those characters into a sequence of integers. This conversion is called **tokenization**. The choice of how to split text into tokens fundamentally shapes what a model can learn.

Character-level tokenization splits text into individual characters. It has a tiny vocabulary (just ~100 characters) but produces very long sequences. Word-level tokenization uses whole words, giving shorter sequences but a massive vocabulary (100,000+ entries) and it can't handle unknown words. **Subword tokenization** (like BPE) finds the sweet spot: it merges frequently co-occurring character pairs into tokens, building a vocabulary of common subwords. "unhappiness" might become ["un", "happi", "ness"] -- three meaningful pieces instead of 11 characters or 1 opaque word.

The vocabulary size is a critical design choice. Too small means long sequences that are expensive to process. Too large means a sparse embedding table that's hard to learn. Modern LLMs typically use 32k-100k tokens, balancing sequence length against vocabulary coverage.

### Why naive approaches fail

Using whole words fails because language is infinitely productive. People constantly create new words, misspellings, and technical terms. A word-level tokenizer would map all of these to a single "[UNKNOWN]" token, losing all information. Character-level tokenization works for any text but makes sequences so long that the model struggles to learn long-range dependencies -- "cat" as three separate characters is harder to learn than "cat" as one token.

### Mental models

- **Character-level** = spelling out every word letter by letter. Flexible but slow.
- **Word-level** = a dictionary lookup. Fast but breaks on new words.
- **BPE** = learning common abbreviations. "because" becomes "bc" if you've seen it enough. The most common patterns get their own shorthand.
- **Vocabulary size tradeoff** = a dictionary: too small and you can't express ideas, too large and you can't memorize it all.

### Visual explanations

```
Input: "the cat sat"

Character-level:  [t] [h] [e] [ ] [c] [a] [t] [ ] [s] [a] [t]
                   11 tokens, vocab size ~70

Word-level:       [the] [cat] [sat]
                   3 tokens, vocab size ~100,000+

BPE (subword):    [the] [_cat] [_sat]
                   3 tokens, vocab size ~8,000

BPE on rare word: "unhappiness"
                  [un] [happi] [ness]
                   3 tokens -- broken into meaningful pieces!

Vocabulary size vs sequence length:
  Small vocab -----> Long sequences  (slow to process)
  Large vocab -----> Short sequences (hard to train embeddings)
  BPE sweet spot --> Moderate both   (practical for LLMs)
```

---

## Hands-on Exploration

1. Tokenize a sentence character-by-character and count the vocabulary size
2. Implement BPE: find the most common adjacent pair, merge it, repeat
3. Compare sequence lengths and vocabulary sizes across tokenization strategies

---

## Live Code

```python
import numpy as np
np.random.seed(42)

# --- Character-level tokenization ---
text = "the cat sat on the mat the cat ate the rat"
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
char_tokens = [char_to_id[c] for c in text]

print("=== Character-level Tokenization ===")
print(f"Text: '{text}'")
print(f"Vocab: {chars}")
print(f"Vocab size: {len(chars)}")
print(f"Tokens: {char_tokens[:20]}...")
print(f"Sequence length: {len(char_tokens)}")

# --- Simple BPE-style tokenization ---
print("\n=== BPE-style Tokenization ===")
tokens = list(text)  # start with characters
num_merges = 10

for step in range(num_merges):
    # Count adjacent pairs
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    if not pairs:
        break
    # Find most frequent pair
    best = max(pairs, key=pairs.get)
    merged = best[0] + best[1]
    # Merge all occurrences
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
            new_tokens.append(merged)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = new_tokens
    if step < 6:
        print(f"Merge {step+1}: '{best[0]}'+'{best[1]}' -> '{merged}'"
              f"  (count={pairs[best]}, seq_len={len(tokens)})")

vocab = sorted(set(tokens))
print(f"\nFinal BPE tokens: {tokens}")
print(f"BPE vocab size: {len(vocab)}")
print(f"BPE sequence length: {len(tokens)}")

# --- Tradeoff comparison ---
print("\n=== Vocabulary Size vs Sequence Length ===")
words = text.split()
word_vocab = sorted(set(words))
print(f"{'Method':<15} {'Vocab Size':>10} {'Seq Length':>10} {'Ratio':>8}")
print("-" * 47)
print(f"{'Character':<15} {len(chars):>10} {len(char_tokens):>10}"
      f" {len(char_tokens)/len(chars):>8.1f}")
print(f"{'BPE (10 merge)':<15} {len(vocab):>10} {len(tokens):>10}"
      f" {len(tokens)/len(vocab):>8.1f}")
print(f"{'Word':<15} {len(word_vocab):>10} {len(words):>10}"
      f" {len(words)/len(word_vocab):>8.1f}")
```

---

## Key Takeaways

- **Tokenization converts text to numbers.** Models see integer sequences, not characters or words.
- **BPE finds a middle ground.** It merges frequent character pairs to build a subword vocabulary that handles rare words gracefully.
- **Vocabulary size is a tradeoff.** Smaller vocab means longer sequences; larger vocab means sparser embeddings.
- **Tokenization shapes model behavior.** The same text tokenized differently leads to different model capabilities and biases.
- **No token is sacred.** Words like "unhappiness" may split into ["un", "happi", "ness"], and that is by design -- each piece carries meaning.
