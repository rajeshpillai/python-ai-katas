# Chain-of-Thought as Latent Variables

> Phase 9 — Reasoning Models | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

Language models are trained to predict the next token, which seems like a simple pattern-matching task. But some problems require genuine multi-step reasoning -- you cannot jump from question to answer without intermediate steps. Consider "What is 47 times 23?" A model that tries to produce the answer directly must somehow compute a multiplication in a single forward pass through its weights. But a model that first generates intermediate steps -- "47 times 20 is 940, 47 times 3 is 141, 940 plus 141 is 1081" -- breaks the problem into manageable pieces, each of which is simple enough for next-token prediction.

Chain-of-thought (CoT) reasoning treats these intermediate steps as latent variables -- hidden computations that bridge the gap between input and output. In probabilistic terms, instead of computing P(answer|question) directly, CoT computes P(answer|question) = sum over all chains P(answer|chain, question) * P(chain|question). The chain of reasoning acts as a scaffold that makes the final answer much easier to reach.

This insight has transformed how we use language models. Simply prompting a model with "Let's think step by step" before it answers can dramatically improve accuracy on math, logic, and common-sense reasoning tasks. The generated reasoning tokens serve as external working memory, allowing the model to build up a solution incrementally rather than trying to compute everything in a single pass through its neural network.

### Why naive approaches fail

Direct answer prediction (no chain of thought) fails on multi-step problems because a single forward pass through a neural network has limited computational depth. A Transformer with L layers can only perform L sequential computation steps. If a problem requires more reasoning steps than the network has layers, the network literally cannot compute the answer in one pass. It is like trying to solve a 10-step math problem while only being allowed to think for 3 steps.

Memorizing input-output pairs (the lookup table approach) fails because the space of possible multi-step problems is combinatorially enormous. A model cannot memorize every possible multiplication -- there are infinitely many. But it can learn the algorithm for multiplication (multiply digit by digit, carry, add partial products), which generalizes to any inputs. Chain-of-thought makes this algorithmic approach expressible through next-token prediction.

### Mental models

- **Showing your work in math class:** A teacher who sees only the final answer cannot tell if you understood the problem or guessed. Showing intermediate steps both proves understanding and makes the correct answer far more likely.
- **GPS navigation:** Instead of teleporting from start to destination, GPS gives turn-by-turn directions. Each turn is simple; the chain of turns solves the complex routing problem.
- **Cooking recipe:** You do not go directly from raw ingredients to finished dish. The recipe (chain of thought) breaks the process into steps, each building on the previous one.

### Visual explanations

```
  DIRECT PREDICTION (no chain of thought):

  "What is 47 * 23?"  ──────────>  "1081"
                         one giant
                         leap          (hard! must compute in
                                        one forward pass)

  CHAIN-OF-THOUGHT PREDICTION:

  "What is 47 * 23?"
         │
         v
  "47 * 20 = 940"        (step 1: easy multiplication)
         │
         v
  "47 * 3 = 141"         (step 2: easy multiplication)
         │
         v
  "940 + 141 = 1081"     (step 3: easy addition)
         │
         v
  "The answer is 1081"   (step 4: copy result)

  Each step is simple enough for next-token prediction!

  PROBABILISTIC VIEW:

  P(answer | question) = sum_chain P(answer | chain) * P(chain | question)

  Without CoT:  P("1081" | "47*23") = ??? (must compute directly)

  With CoT:     P("940"  | "47*20")  = high (simple multiplication)
                P("141"  | "47*3")   = high (simple multiplication)
                P("1081" | "940+141")= high (simple addition)
                Product of easy steps >> direct computation
```

---

## Hands-on Exploration

1. Build a simple tokenized math problem solver that attempts direct answer prediction
2. Observe that direct prediction fails on multi-step problems
3. Implement chain-of-thought decomposition that breaks problems into steps
4. Show that step-by-step solving achieves much higher accuracy
5. Measure how accuracy degrades with problem complexity for both approaches

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def simulate_single_step_accuracy(difficulty, model_capacity):
    """Probability of getting a single reasoning step right.
    Higher capacity and lower difficulty -> higher accuracy."""
    logit = model_capacity - difficulty
    return 1 / (1 + np.exp(-logit))

def direct_prediction(total_difficulty, model_capacity):
    """Model tries to answer in one step."""
    return simulate_single_step_accuracy(total_difficulty, model_capacity)

def chain_of_thought(step_difficulties, model_capacity):
    """Model decomposes into steps. Each step must succeed."""
    step_accs = []
    for d in step_difficulties:
        acc = simulate_single_step_accuracy(d, model_capacity)
        step_accs.append(acc)
    return np.prod(step_accs), step_accs

print("=" * 60)
print("CHAIN-OF-THOUGHT AS LATENT VARIABLES")
print("=" * 60)

# --- Simple example: multi-digit arithmetic ---
model_capacity = 3.0

problems = [
    ("5 + 3",           8.0, [1.0]),
    ("47 + 38",         6.0, [2.0, 1.5]),
    ("47 * 23",         9.0, [2.5, 2.5, 2.0]),
    ("(47*23) + (15*8)", 12.0, [2.5, 2.5, 2.0, 2.0, 2.0, 2.0]),
    ("((47*23)+15) * 8", 14.0, [2.5, 2.5, 2.0, 1.5, 2.5, 2.5, 2.0]),
]

print(f"\nModel capacity: {model_capacity}")
print(f"\n{'Problem':<22s} {'Direct':>8s} {'CoT':>8s} {'Steps':>6s}")
print("-" * 50)

for name, total_diff, step_diffs in problems:
    direct = direct_prediction(total_diff, model_capacity)
    cot, step_accs = chain_of_thought(step_diffs, model_capacity)
    print(f"{name:<22s} {direct:>7.1%} {cot:>7.1%} "
          f"{len(step_diffs):>5d}")

# --- Detailed view of one problem ---
print(f"\n{'=' * 60}")
print("DETAILED: '47 * 23' step-by-step")
print("=" * 60)
steps = [
    ("47 * 20 = 940",   2.5),
    ("47 * 3 = 141",    2.5),
    ("940 + 141 = 1081", 2.0),
]
print(f"\n  Direct attempt (difficulty=9.0):")
direct_acc = direct_prediction(9.0, model_capacity)
print(f"    P(correct) = {direct_acc:.1%}")
bar = "#" * int(direct_acc * 40)
print(f"    |{bar:<40s}|")

print(f"\n  Chain-of-thought decomposition:")
running_product = 1.0
for step_name, diff in steps:
    acc = simulate_single_step_accuracy(diff, model_capacity)
    running_product *= acc
    bar = "#" * int(acc * 40)
    print(f"    Step: {step_name:<20s} P={acc:.1%} |{bar}")
print(f"\n    Overall P(all steps correct) = {running_product:.1%}")
print(f"    Improvement over direct: {running_product/direct_acc:.1f}x")

# --- How accuracy scales with problem complexity ---
print(f"\n{'=' * 60}")
print("SCALING: Accuracy vs Problem Complexity")
print("=" * 60)
print(f"\n{'Num Steps':>10s} {'Direct':>8s} {'CoT':>8s} {'Ratio':>8s}")
print("-" * 38)

for n_steps in [1, 2, 3, 5, 8, 12, 20]:
    total_diff = 2.0 * n_steps + 1.0
    step_diffs = [2.0] * n_steps
    direct = direct_prediction(total_diff, model_capacity)
    cot, _ = chain_of_thought(step_diffs, model_capacity)
    ratio = cot / max(direct, 1e-10)
    d_bar = "#" * int(direct * 30)
    c_bar = "#" * int(cot * 30)
    print(f"{n_steps:>10d} {direct:>7.1%} {cot:>7.1%} {ratio:>7.1f}x")

# --- Monte Carlo simulation ---
print(f"\n{'=' * 60}")
print("SIMULATION: 1000 random problems")
print("=" * 60)
n_trials = 1000
direct_correct = 0
cot_correct = 0

for _ in range(n_trials):
    n_steps = np.random.randint(2, 8)
    step_diffs = np.random.uniform(1.0, 3.5, n_steps)
    total_diff = np.sum(step_diffs) * 0.8

    # Direct: one shot
    p_direct = direct_prediction(total_diff, model_capacity)
    if np.random.random() < p_direct:
        direct_correct += 1

    # CoT: each step independently
    all_correct = True
    for d in step_diffs:
        p_step = simulate_single_step_accuracy(d, model_capacity)
        if np.random.random() >= p_step:
            all_correct = False
            break
    if all_correct:
        cot_correct += 1

print(f"\n  Direct prediction: {direct_correct}/{n_trials} "
      f"= {direct_correct/n_trials:.1%} correct")
print(f"  Chain-of-thought:  {cot_correct}/{n_trials} "
      f"= {cot_correct/n_trials:.1%} correct")
print(f"  CoT advantage: {cot_correct/max(direct_correct,1):.1f}x")
```

---

## Key Takeaways

- **Chain-of-thought decomposes hard problems into easy steps.** Each intermediate reasoning step is simple enough for next-token prediction, even when the overall problem is too complex for direct computation.
- **Intermediate tokens serve as external working memory.** The generated reasoning text allows the model to store and retrieve intermediate results, overcoming the fixed-width bottleneck of a single forward pass.
- **CoT accuracy is the product of step accuracies.** Each step must succeed for the chain to work, so very long chains can still fail. The benefit comes from each step being much easier than the whole problem.
- **Latent variable interpretation provides theoretical grounding.** Chain-of-thought marginalizes over possible reasoning paths, and the model learns to assign high probability to correct paths through training.
- **Prompting alone can unlock CoT.** Simply adding "Let's think step by step" causes models to generate intermediate reasoning, dramatically improving performance without any retraining.
