# Emergence Through Scale

> Phase 8 — Large Language Models (LLMs) | Kata 8.5

---

## Concept & Intuition

### What problem are we solving?

One of the most surprising discoveries in modern AI is emergence: as language models grow larger, they do not just get incrementally better at existing tasks -- they spontaneously develop entirely new capabilities that smaller models lack entirely. A model with 1 billion parameters might score 0% on multi-step arithmetic, while the same architecture with 100 billion parameters suddenly scores 80%. This is not gradual improvement; it is a sharp phase transition, like water suddenly becoming ice at 0 degrees Celsius.

Understanding emergence matters because it means we cannot predict what a model will be capable of just by extrapolating from smaller models. The capabilities of GPT-4 could not have been reliably predicted from GPT-3's performance on the same benchmarks. This has profound implications for AI safety (dangerous capabilities might emerge unexpectedly) and AI development strategy (you might need to scale past a critical threshold before a capability appears at all).

The mechanism behind emergence is debated, but one compelling explanation involves the interaction between model capacity and task complexity. Simple tasks (pattern matching, basic grammar) require relatively few parameters and improve smoothly with scale. Complex tasks (multi-step reasoning, analogical thinking) require the model to simultaneously coordinate many internal representations. Below a critical capacity threshold, the model simply cannot hold enough information in flight to solve these tasks. Above the threshold, all the necessary pieces suddenly click into place.

### Why naive approaches fail

Linearly extrapolating from small model performance fails dramatically. If a 1B parameter model scores 5% on a task and a 10B model scores 7%, you might predict a 100B model will score 9%. But the actual score might be 70%. The relationship between scale and capability is not linear -- it follows a sigmoid or step-function pattern for emergent tasks. This makes traditional scaling analysis misleading.

Similarly, trying to design specific architectures for specific capabilities misses the point. Emergence suggests that general capacity (more parameters, more data, more compute) can substitute for task-specific design. No one designed GPT-3 to do arithmetic or translate between languages -- these capabilities emerged from the simple objective of predicting the next token at sufficient scale.

### Mental models

- **Phase transitions in physics:** Water does not gradually become ice. At exactly 0C, it undergoes a sudden phase transition. Similarly, model capabilities can appear suddenly at a critical scale threshold, not gradually.
- **Critical mass in nuclear physics:** Below a critical mass of fissile material, nothing happens. Above it, you get a chain reaction. Model capabilities have similar thresholds -- below a certain size, a capability is simply absent.
- **Puzzle assembly:** With 30% of puzzle pieces, you see nothing recognizable. At 60%, vague shapes appear. At 90%, the picture suddenly "clicks" and you see the whole image. More capacity lets the model assemble more pieces of the reasoning puzzle.

### Visual explanations

```
  EMERGENT CAPABILITY: ACCURACY vs MODEL SIZE

  Accuracy
  100% |                                    ****
       |                                 ***
       |                               **
   80% |                              *
       |                             *
   60% |                            *    <-- phase transition
       |                           *
   40% |                          *
       |                         *
   20% |                        *
       |  .......................*
    0% |________________________*_____________
       1M   10M  100M   1B   10B  100B  1T
                    Model Parameters

  NON-EMERGENT TASK (smooth improvement):

  Accuracy
  100% |                              *********
       |                       *******
   80% |                  *****
       |              ****
   60% |          ****
       |       ***
   40% |     **
       |   **
   20% |  *
       | *
    0% |*_____________________________________
       1M   10M  100M   1B   10B  100B  1T

  MULTIPLE CAPABILITIES EMERGE AT DIFFERENT THRESHOLDS:

  Grammar    ████████████████████████████████  (small models)
  Translation     ██████████████████████████   (medium models)
  Arithmetic           █████████████████████   (large models)
  Reasoning                  ████████████████  (very large)
  Planning                        ███████████  (largest)
              1M   10M  100M  1B  10B  100B
```

---

## Hands-on Exploration

1. Simulate a simple neural network at different sizes and observe smooth vs emergent capabilities
2. Define tasks of varying complexity and measure performance as network capacity grows
3. Identify the critical thresholds where phase transitions occur
4. Plot the contrast between smooth-scaling tasks and emergent tasks
5. Observe how task composition (chaining multiple steps) creates emergent behavior

---

## Live Code

```python
import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def simulate_capability(model_size, task_complexity, noise=0.05):
    """Simulate accuracy of a model on a task.
    Emergent behavior: accuracy is near-zero until model_size
    exceeds a threshold proportional to task_complexity, then
    rises sharply (sigmoid transition).
    """
    threshold = task_complexity * 1.2
    steepness = 8.0 / task_complexity
    base_acc = sigmoid(steepness * (np.log10(model_size) - threshold))
    return np.clip(base_acc + np.random.randn() * noise, 0, 1)

def simulate_smooth_task(model_size, max_size=1e12, noise=0.03):
    """Non-emergent task: smooth power-law improvement."""
    acc = (np.log10(model_size) / np.log10(max_size)) ** 0.7
    return np.clip(acc + np.random.randn() * noise, 0, 1)

print("=" * 60)
print("EMERGENCE THROUGH SCALE")
print("=" * 60)

# --- Model sizes to test ---
sizes = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9,
         1e10, 3e10, 1e11, 3e11, 1e12]
size_labels = ["1M", "3M", "10M", "30M", "100M", "300M", "1B",
               "3B", "10B", "30B", "100B", "300B", "1T"]

# --- Tasks with different complexity levels ---
tasks = {
    "Grammar":        3.0,   # simple, emerges early
    "Translation":    5.0,   # medium complexity
    "Arithmetic":     7.0,   # requires more capacity
    "Multi-step":     9.0,   # complex reasoning
    "Planning":      11.0,   # highest complexity
}

print("\n--- Emergent Capabilities vs Model Size ---\n")
print(f"{'Task':<14s}", end="")
for label in size_labels:
    print(f"{label:>6s}", end="")
print()
print("-" * (14 + 6 * len(size_labels)))

for task_name, complexity in tasks.items():
    print(f"{task_name:<14s}", end="")
    for size in sizes:
        acc = simulate_capability(size, complexity, noise=0.02)
        if acc < 0.1:
            symbol = "  .  "
        elif acc < 0.3:
            symbol = "  -  "
        elif acc < 0.6:
            symbol = "  +  "
        elif acc < 0.8:
            symbol = "  *  "
        else:
            symbol = "  @  "
        print(f"{symbol}", end=" ")
    print()

print(f"\n  Legend: .=<10%  -=10-30%  +=30-60%  *=60-80%  @=>80%")

# --- Smooth vs emergent task comparison ---
print(f"\n{'=' * 60}")
print("SMOOTH vs EMERGENT (side by side)")
print("=" * 60)
print(f"\n{'Size':>8s}  {'Smooth (grammar)':>17s}  {'Emergent (reasoning)':>21s}")
print("-" * 52)
for size, label in zip(sizes, size_labels):
    smooth = simulate_smooth_task(size, noise=0.01)
    emergent = simulate_capability(size, 8.0, noise=0.01)
    s_bar = "#" * int(smooth * 25)
    e_bar = "#" * int(emergent * 25)
    print(f"{label:>8s}  {smooth:.2f} |{s_bar:<25s}  "
          f"{emergent:.2f} |{e_bar:<25s}")

# --- Phase transition detection ---
print(f"\n{'=' * 60}")
print("PHASE TRANSITION DETECTION")
print("=" * 60)
print("\nFinding critical size for each capability:\n")
for task_name, complexity in tasks.items():
    prev_acc = 0
    critical_size = "N/A"
    for size, label in zip(sizes, size_labels):
        acc = simulate_capability(size, complexity, noise=0.0)
        if prev_acc < 0.3 and acc >= 0.3:
            critical_size = label
            break
        prev_acc = acc
    print(f"  {task_name:<14s} emerges at ~{critical_size}")

# --- Compositionality creates emergence ---
print(f"\n{'=' * 60}")
print("WHY COMPOSITION CREATES EMERGENCE")
print("=" * 60)
print("\nTask: chain N sub-tasks, each needs accuracy > 0.9 to work")
print("Overall accuracy = (sub-task accuracy)^N\n")
sub_tasks = [1, 2, 3, 5, 8]
print(f"{'Sub-acc':>8s}", end="")
for n in sub_tasks:
    print(f"  chain={n}", end="")
print()
print("-" * (8 + 10 * len(sub_tasks)))
for sub_acc in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    print(f"  {sub_acc:.2f}  ", end="")
    for n in sub_tasks:
        chain_acc = sub_acc ** n
        print(f"   {chain_acc:.4f} ", end="")
    print()
print("\n  Notice: at 0.80 sub-accuracy, chain=8 gives only 17%.")
print("  But at 0.95 sub-accuracy, chain=8 gives 66%.")
print("  Small improvement in sub-tasks -> large jump in chains!")
```

---

## Key Takeaways

- **Emergence is a phase transition, not gradual improvement.** Certain capabilities appear suddenly at critical model sizes, going from near-zero to high accuracy over a narrow range.
- **Task complexity determines the emergence threshold.** Simple tasks emerge in small models; complex multi-step tasks require orders of magnitude more parameters.
- **Composition explains sudden jumps.** When a task requires chaining multiple sub-capabilities, a small improvement in each sub-capability can produce a dramatic jump in overall performance.
- **Extrapolation from small models is unreliable.** You cannot predict what a 100B parameter model will do by testing a 1B parameter model -- the capabilities may be qualitatively different.
- **Emergence has safety implications.** If dangerous capabilities (deception, manipulation) follow the same pattern, they could appear unexpectedly in future models, making proactive safety research critical.
