# Scratchpads

> Phase 9 — Reasoning Models | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

A Transformer model has a fixed computational budget per token: L layers, each with a fixed hidden dimension d. This means there is a hard ceiling on how much computation the model can perform between reading input and producing output. For problems that require tracking many variables, performing multi-digit arithmetic, or holding intermediate results in memory, this fixed budget is simply insufficient. The model needs a way to externalize its computation -- to "write things down" and refer back to them.

Scratchpads provide exactly this capability. Instead of requiring the model to solve a problem entirely within its internal activations (its "head"), a scratchpad allows the model to write intermediate computations into its output tokens and then read them back via attention in subsequent steps. This is directly analogous to how humans use paper for long division: our working memory can hold about 7 items, but with paper, we can solve problems of arbitrary complexity by writing down and looking up intermediate results.

The scratchpad concept was formalized in research showing that Transformers with scratchpads can solve problems that are provably impossible for fixed-depth Transformers without them. For example, adding two n-digit numbers requires O(n) sequential carry operations. A fixed-depth Transformer cannot perform O(n) sequential computations, but a model generating n scratch tokens can. The scratchpad effectively turns a bounded-depth circuit into an unbounded-depth computation.

### Why naive approaches fail

Without a scratchpad, a model must perform all computation in its fixed-depth forward pass. Consider adding 3847 + 2965. The model must propagate carries (7+5=12, carry 1; 4+6+1=11, carry 1; etc.) which requires a sequential chain of operations. A Transformer with L layers can only perform L sequential operations, so if the numbers have more digits than the model has layers, carries cannot propagate correctly. The model may get small additions right (memorized or shallow computation) but fail systematically on larger numbers.

Increasing model size (more layers, wider hidden states) helps, but is wasteful. Adding 100-digit numbers does not require a smarter model -- it requires more steps. A scratchpad decouples problem difficulty from model depth, letting a small model solve large problems by using more output tokens.

### Mental models

- **Paper for long division:** You can do 15 divided by 3 in your head. But 847293 divided by 371? You need paper to track partial quotients and remainders. The scratchpad is the model's paper.
- **Programmer's debug log:** When debugging complex code, you print intermediate values. This does not make you a better programmer -- it lets you track state that exceeds your working memory.
- **Chalkboard in a lecture:** A professor does not derive a complex proof purely in their head. They write each step on the board, refer back to earlier steps, and build the proof incrementally.

### Visual explanations

```
  WITHOUT SCRATCHPAD (fixed computation depth):

  Input: "3847 + 2965 = ?"
         │
         ▼
  ┌──────────────────────────┐
  │  Layer 1: process input  │
  │  Layer 2: ???            │   Model must compute entire
  │  Layer 3: ???            │   addition in L layers.
  │  ...                     │   Carries can't propagate
  │  Layer L: produce answer │   far enough!
  └──────────────────────────┘
         │
         ▼
  Output: "6812"  (CORRECT)  ... or "6702" (WRONG - lost a carry)

  WITH SCRATCHPAD (unbounded computation):

  Input: "3847 + 2965 = ?"
         │
         ▼
  Generate scratch tokens:
  "7+5=12, write 2, carry 1"   ← step 1 (uses L layers)
  "4+6+1=11, write 1, carry 1" ← step 2 (uses L layers, READS step 1)
  "8+9+1=18, write 8, carry 1" ← step 3 (uses L layers, READS steps 1-2)
  "3+2+1=6, write 6"           ← step 4 (uses L layers, READS steps 1-3)
         │
         ▼
  "Answer: 6812"  (CORRECT - carries propagated via scratchpad)

  Each step uses the full L layers, AND can attend back to
  all previous scratch tokens. Total depth: steps * L layers.
```

---

## Hands-on Exploration

1. Implement a simulated model that tries to add multi-digit numbers without a scratchpad
2. Implement the same model with a scratchpad for intermediate carry tracking
3. Compare accuracy as the number of digits increases
4. Show that scratchpad accuracy stays high while direct accuracy collapses
5. Demonstrate that the scratchpad approach generalizes to other sequential computations

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def model_without_scratchpad(a, b, model_depth=3):
    """Simulate a fixed-depth model adding two numbers.
    Can only propagate carries 'model_depth' digits."""
    a_digits = [int(d) for d in str(a)][::-1]
    b_digits = [int(d) for d in str(b)][::-1]
    max_len = max(len(a_digits), len(b_digits)) + 1
    # Pad
    a_digits += [0] * (max_len - len(a_digits))
    b_digits += [0] * (max_len - len(b_digits))
    result = []
    carry = 0
    for i in range(max_len):
        if i < model_depth:
            # Within model depth: correct computation
            s = a_digits[i] + b_digits[i] + carry
            result.append(s % 10)
            carry = s // 10
        else:
            # Beyond model depth: carry gets "lost" with some probability
            s = a_digits[i] + b_digits[i]
            # Model "forgets" the carry from distant digits
            if np.random.random() < 0.3:
                s += carry  # sometimes remembers
            result.append(s % 10)
            carry = s // 10 if np.random.random() < 0.3 else 0
    # Remove leading zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return int("".join(str(d) for d in result[::-1]))

def model_with_scratchpad(a, b):
    """Simulate a model that uses a scratchpad for carries.
    Each digit addition is a separate 'generation step'."""
    a_digits = [int(d) for d in str(a)][::-1]
    b_digits = [int(d) for d in str(b)][::-1]
    max_len = max(len(a_digits), len(b_digits)) + 1
    a_digits += [0] * (max_len - len(a_digits))
    b_digits += [0] * (max_len - len(b_digits))
    scratchpad = []
    result = []
    carry = 0
    for i in range(max_len):
        s = a_digits[i] + b_digits[i] + carry
        digit = s % 10
        carry = s // 10
        step = f"  pos {i}: {a_digits[i]}+{b_digits[i]}+carry({carry if i==0 else scratchpad[-1]['carry']})={s}, write {digit}, carry {carry}"
        scratchpad.append({"step": step, "carry": carry})
        result.append(digit)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return int("".join(str(d) for d in result[::-1])), scratchpad

print("=" * 60)
print("SCRATCHPADS: External Working Memory")
print("=" * 60)

# --- Detailed example ---
a, b = 3847, 2965
correct = a + b
print(f"\nProblem: {a} + {b} = {correct}")

no_scratch = model_without_scratchpad(a, b, model_depth=3)
with_scratch, pad = model_with_scratchpad(a, b)

print(f"\n  Without scratchpad: {no_scratch} "
      f"{'(CORRECT)' if no_scratch == correct else '(WRONG)'}")
print(f"  With scratchpad:    {with_scratch} "
      f"{'(CORRECT)' if with_scratch == correct else '(WRONG)'}")

print(f"\n  Scratchpad contents:")
for entry in pad:
    print(f"  {entry['step']}")

# --- Accuracy vs number of digits ---
print(f"\n{'=' * 60}")
print("ACCURACY vs NUMBER OF DIGITS (100 trials each)")
print("=" * 60)
print(f"\n{'Digits':>7s} {'No Scratch':>11s} {'Scratchpad':>11s}")
print("-" * 33)

for n_digits in [2, 3, 4, 6, 8, 10, 15]:
    no_scratch_correct = 0
    scratch_correct = 0
    trials = 100

    for _ in range(trials):
        a = np.random.randint(10**(n_digits-1), 10**n_digits)
        b = np.random.randint(10**(n_digits-1), 10**n_digits)
        correct = a + b

        ns_result = model_without_scratchpad(a, b, model_depth=3)
        s_result, _ = model_with_scratchpad(a, b)

        if ns_result == correct:
            no_scratch_correct += 1
        if s_result == correct:
            scratch_correct += 1

    ns_acc = no_scratch_correct / trials
    s_acc = scratch_correct / trials
    ns_bar = "#" * int(ns_acc * 25)
    s_bar = "#" * int(s_acc * 25)
    print(f"{n_digits:>7d} {ns_acc:>10.0%} |{ns_bar:<25s}")
    print(f"{'':>7s} {s_acc:>10.0%} |{s_bar:<25s} (scratch)")

# --- Scratchpad for sequential computation ---
print(f"\n{'=' * 60}")
print("GENERALIZATION: Cumulative Sum with Scratchpad")
print("=" * 60)
sequence = [14, 27, 33, 8, 45, 19, 56, 11, 62, 5]
print(f"\nSequence: {sequence}")
print(f"\nScratchpad trace:")
running = 0
for i, val in enumerate(sequence):
    running += val
    print(f"  Step {i:2d}: running_sum + {val:2d} = {running:4d}  "
          f"|{'#' * (running // 8)}")
print(f"\nFinal sum: {running} (correct: {sum(sequence)})")

# --- Working memory capacity comparison ---
print(f"\n{'=' * 60}")
print("WORKING MEMORY CAPACITY")
print("=" * 60)
print("\nItems to track vs success rate:\n")
for n_items in [3, 5, 7, 10, 15, 25]:
    # Without scratchpad: success drops as items exceed ~7
    no_sp = min(1.0, max(0.0, 1.0 - 0.15 * max(0, n_items - 4)))
    # With scratchpad: stays high regardless
    with_sp = min(1.0, max(0.0, 1.0 - 0.005 * n_items))
    print(f"  {n_items:2d} items: no-scratch={no_sp:.0%}  "
          f"scratch={with_sp:.0%}  "
          f"[{'#'*int(no_sp*15):<15s}|{'#'*int(with_sp*15):<15s}]")
```

---

## Key Takeaways

- **Scratchpads externalize working memory.** By writing intermediate results into output tokens, the model overcomes the fixed-depth limitation of its forward pass.
- **Fixed-depth models have provable computational limits.** A Transformer with L layers cannot solve problems requiring more than L sequential computation steps, no matter how wide the model.
- **Scratchpads decouple problem size from model size.** A small model with a scratchpad can solve arbitrarily large problems by generating more intermediate tokens.
- **Each scratch token gets the full model depth.** When the model generates a scratch token and then reads it back via attention, the total effective computation depth is (number of tokens) times L layers.
- **Scratchpads are the foundation of reasoning models.** Modern reasoning models like o1 and o3 use extended "thinking" token sequences that function as sophisticated scratchpads, breaking complex problems into manageable steps.
