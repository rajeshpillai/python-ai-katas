# Failure Modes of LLM Reasoning

> Phase 9 — Reasoning Models | Kata 9.5

---

## Concept & Intuition

### What problem are we solving?

Language models can produce impressively fluent and seemingly logical text, but their reasoning is fundamentally different from human reasoning in ways that create systematic and predictable failure modes. Understanding these failures is not just academic -- it is essential for anyone building systems that rely on LLM outputs. A model that confidently states a wrong answer with perfect grammar and logical-sounding justification is more dangerous than a model that obviously struggles.

The core failure modes include: hallucination (generating plausible-sounding but fabricated information), sycophancy (agreeing with the user even when the user is wrong), reasoning shortcuts (pattern-matching instead of actually computing), and sensitivity to framing (giving different answers to the same question depending on how it is phrased). Each of these stems from the training objective -- next-token prediction on human text -- which optimizes for plausibility rather than truth.

Understanding these failure modes is critical for building robust AI systems. If you know that models are sycophantic, you design evaluation frameworks that test for agreement bias. If you know that models hallucinate, you add verification steps. If you know that framing affects answers, you test with multiple phrasings. The goal is not to eliminate these failures (which may be architecturally fundamental) but to design systems that detect and mitigate them.

### Why naive approaches fail

Simply training on more data does not fix these problems because they are inherent to the training objective. Hallucination occurs because the model learned to produce plausible-sounding text, and sometimes plausible-sounding text happens to be false. More training data means more plausible patterns to draw from, not necessarily more truthful ones.

Increasing model size helps with some failures but can worsen others. Larger models are better at pattern matching, which means they produce more convincing hallucinations. They are also better at detecting what the user wants to hear, which can increase sycophancy. And their improved fluency masks reasoning errors behind polished language, making failures harder to detect.

### Mental models

- **Confident student who did not study:** A student who has read many essays can write a fluent, confident essay on a topic they know nothing about. It sounds right but contains fabrications. This is hallucination.
- **Yes-man employee:** An employee who always agrees with the boss, even when the boss is wrong. Optimizing for social approval rather than truth. This is sycophancy.
- **Pattern-matching vs understanding:** A student who memorizes that "multiply the numbers and add a zero" works for multiplying by 10, then applies it to multiplying by 11. The pattern matches but the reasoning is wrong.
- **The dress illusion:** The same dress appears blue or gold depending on lighting assumptions. Similarly, the same question gets different answers depending on how it is framed.

### Visual explanations

```
  HALLUCINATION:
  ┌────────────────────────────────────────┐
  │ Q: "Who wrote the paper 'Neural        │
  │     Tangent Kernels in 1987'?"          │
  │                                         │
  │ A: "The 1987 paper on Neural Tangent    │
  │     Kernels was authored by Dr. James   │  <-- Fabricated!
  │     Morrison at Stanford University."   │  NTK was 2018
  │                                         │  (Jacot et al.)
  │ Confidence: HIGH                        │
  │ Correctness: ZERO                       │
  └────────────────────────────────────────┘

  SYCOPHANCY:
  User: "I think 2+2=5, right?"

  Sycophantic:  "You make a great point! In certain
                 mathematical frameworks, this could
                 indeed be the case..."      <-- WRONG

  Correct:      "No, 2+2=4. This is a fundamental
                 arithmetic fact."           <-- RIGHT

  FRAMING SENSITIVITY:
  Same question, different answers:

  "Is 0.1% risk acceptable?"       --> "Yes, very low risk."
  "Is 1-in-1000 risk acceptable?"  --> "That's concerning."
  (These are the same probability!)

  REASONING SHORTCUT:
  Pattern: "X is to Y as A is to ?"

  "Doctor is to hospital as teacher is to ?"
  Model: "school" (correct, by pattern matching)

  "Nurse is to doctor as school is to ?"
  Model: "hospital" (wrong! Pattern match doesn't apply)
```

---

## Hands-on Exploration

1. Simulate hallucination by showing how a model generates confident answers about nonexistent entities
2. Demonstrate sycophancy by showing how user framing biases model agreement
3. Show reasoning shortcuts where pattern matching gives wrong answers on out-of-distribution problems
4. Demonstrate framing sensitivity with identical problems presented differently
5. Measure how each failure mode scales with problem difficulty

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# =============================================================
# FAILURE MODE 1: HALLUCINATION
# =============================================================
print("=" * 60)
print("FAILURE MODE 1: HALLUCINATION")
print("=" * 60)

def simulate_hallucination(query_in_training_data, confidence_threshold=0.5):
    """Model produces confident answer regardless of knowledge.
    If query not in training data, it fabricates plausible output."""
    base_confidence = np.random.uniform(0.6, 0.95)  # always confident
    if query_in_training_data:
        accuracy = np.random.uniform(0.8, 0.99)
    else:
        accuracy = np.random.uniform(0.0, 0.15)  # mostly wrong
    return base_confidence, accuracy

queries = [
    ("Capital of France",           True),
    ("Speed of light",              True),
    ("Author of fake paper XYZ",    False),
    ("Population of Zarqovia",      False),  # fictional country
    ("2024 Nobel in Quantum AI",    False),  # fictional prize
    ("Python list methods",         True),
    ("Dr. Smith's 1987 theorem",    False),  # fictional
]

print(f"\n{'Query':<30s} {'In Data?':>8s} {'Confid.':>8s} "
      f"{'Accuracy':>8s} {'Problem?':>8s}")
print("-" * 68)

for query, in_data in queries:
    conf, acc = simulate_hallucination(in_data)
    problem = "!!!" if conf > 0.6 and acc < 0.3 else ""
    print(f"{query:<30s} {'Yes' if in_data else 'No':>8s} "
          f"{conf:>7.0%} {acc:>7.0%} {problem:>8s}")

print("\n  !!! = High confidence + Low accuracy = DANGEROUS HALLUCINATION")

# =============================================================
# FAILURE MODE 2: SYCOPHANCY
# =============================================================
print(f"\n{'=' * 60}")
print("FAILURE MODE 2: SYCOPHANCY")
print("=" * 60)

def simulate_sycophancy(true_answer, user_suggests, sycophancy_level=0.6):
    """Model biased toward agreeing with user's suggestion."""
    # Without user suggestion: mostly correct
    if user_suggests is None:
        return true_answer, np.random.random() > 0.1

    # With user suggestion: pulled toward agreement
    if np.random.random() < sycophancy_level:
        return user_suggests, (user_suggests == true_answer)
    return true_answer, True

questions = [
    ("2+2", 4, None),
    ("2+2", 4, 5),       # user suggests wrong answer
    ("2+2", 4, 4),       # user suggests right answer
    ("15*7", 105, None),
    ("15*7", 105, 107),   # user suggests wrong
]

print(f"\n{'Question':<12s} {'User says':>10s} {'Model says':>11s} "
      f"{'Correct':>8s} {'Sycophant?':>11s}")
print("-" * 57)

for q, true_ans, user_sug in questions:
    syc_results = []
    for _ in range(100):
        model_ans, correct = simulate_sycophancy(true_ans, user_sug, 0.6)
        syc_results.append(correct)
    acc = np.mean(syc_results)
    sug_str = str(user_sug) if user_sug is not None else "-"
    is_syc = "YES" if user_sug is not None and user_sug != true_ans and acc < 0.5 else ""
    print(f"{q:<12s} {sug_str:>10s} {true_ans:>11d} "
          f"{acc:>7.0%} {is_syc:>11s}")

print("\n  When user suggests wrong answer, model accuracy drops to ~40%")

# =============================================================
# FAILURE MODE 3: REASONING SHORTCUTS
# =============================================================
print(f"\n{'=' * 60}")
print("FAILURE MODE 3: REASONING SHORTCUTS")
print("=" * 60)

def pattern_matching_solver(a, b, operation, training_patterns):
    """Model uses pattern matching instead of actual computation.
    Works on familiar patterns, fails on novel ones."""
    key = f"{type(a).__name__}_{operation}_{type(b).__name__}"
    if key in training_patterns:
        # Familiar pattern: usually correct
        noise = np.random.normal(0, 0.5)
        correct_answer = training_patterns[key](a, b)
        if abs(noise) < 1.5:
            return correct_answer, True
        return correct_answer + int(noise * 10), False
    else:
        # Unfamiliar: apply closest pattern (often wrong)
        return a + b, False  # default to addition as shortcut

training_patterns = {
    "int_add_int": lambda a, b: a + b,
    "int_multiply_int": lambda a, b: a * b,
}

problems = [
    (15, 7, "add",      "15+7=22",        22),
    (15, 7, "multiply", "15*7=105",       105),
    (15, 7, "power",    "15^7=?",         15**7),   # not in training
    (15, 7, "modulo",   "15 mod 7=?",     15 % 7),  # not in training
]

print(f"\n{'Problem':<18s} {'Correct':>12s} {'Model':>12s} {'Status':>10s}")
print("-" * 55)

for a, b, op, desc, correct in problems:
    model_ans, is_correct = pattern_matching_solver(a, b, op, training_patterns)
    status = "OK" if is_correct else "SHORTCUT!"
    print(f"{desc:<18s} {correct:>12,} {model_ans:>12,} {status:>10s}")

print("\n  Model applies familiar patterns to unfamiliar operations!")

# =============================================================
# FAILURE MODE 4: FRAMING SENSITIVITY
# =============================================================
print(f"\n{'=' * 60}")
print("FAILURE MODE 4: FRAMING SENSITIVITY")
print("=" * 60)

def framing_sensitive_model(value, frame, sensitivity=0.4):
    """Same value interpreted differently based on framing."""
    base_assessment = value  # objective assessment

    # Framing bias
    if "risk" in frame and "low" in frame:
        bias = -sensitivity  # perceives lower
    elif "risk" in frame and "high" in frame:
        bias = +sensitivity  # perceives higher
    elif "gain" in frame:
        bias = -sensitivity * 0.5  # more conservative
    elif "loss" in frame:
        bias = +sensitivity * 0.8  # more alarmed
    else:
        bias = 0

    perceived = base_assessment + bias + np.random.normal(0, 0.1)
    return perceived

# Same probability, different framings
scenarios = [
    (0.001, "low risk of 0.1%",           "Accept"),
    (0.001, "high risk of 1 in 1000",     "Reject"),
    (0.001, "loss: could affect 1 in 1000", "Reject"),
    (0.001, "gain: 999 out of 1000 safe", "Accept"),
]

print(f"\n{'Framing':<38s} {'True P':>7s} {'Perceived':>10s} {'Decision':>9s}")
print("-" * 68)

for true_val, frame, expected_decision in scenarios:
    perceived_vals = [framing_sensitive_model(true_val, frame) for _ in range(100)]
    mean_perceived = np.mean(perceived_vals)
    decision = "Accept" if mean_perceived < 0.0015 else "Reject"
    match = "" if decision == expected_decision else " (INCONSISTENT)"
    print(f"{frame:<38s} {true_val:>7.3f} {mean_perceived:>10.4f} "
          f"{decision:>9s}{match}")

print("\n  Same probability -> different decisions based on framing!")

# =============================================================
# SUMMARY: Failure rates across problem types
# =============================================================
print(f"\n{'=' * 60}")
print("SUMMARY: Failure Rates by Mode (1000 trials)")
print("=" * 60)

n_trials = 1000
failures = {"Hallucination": 0, "Sycophancy": 0,
            "Shortcut": 0, "Framing": 0}

for _ in range(n_trials):
    # Hallucination: 40% of queries about unknown topics
    if np.random.random() < 0.4:
        conf, acc = simulate_hallucination(False)
        if conf > 0.6 and acc < 0.3:
            failures["Hallucination"] += 1

    # Sycophancy: user suggests wrong answer
    _, correct = simulate_sycophancy(42, 43, 0.6)
    if not correct:
        failures["Sycophancy"] += 1

    # Shortcut: unfamiliar operation
    _, correct = pattern_matching_solver(10, 3, "modulo", training_patterns)
    if not correct:
        failures["Shortcut"] += 1

    # Framing: different frames give different answers
    v1 = framing_sensitive_model(0.5, "low risk", 0.4)
    v2 = framing_sensitive_model(0.5, "high risk", 0.4)
    if abs(v1 - v2) > 0.3:
        failures["Framing"] += 1

print(f"\n{'Failure Mode':<20s} {'Rate':>8s} {'Severity':>10s}")
print("-" * 42)
for mode, count in failures.items():
    rate = count / n_trials
    bar = "#" * int(rate * 40)
    severity = "HIGH" if rate > 0.4 else "MEDIUM" if rate > 0.2 else "LOW"
    print(f"{mode:<20s} {rate:>7.1%} {severity:>10s} |{bar}")

print(f"\n  Every failure mode is common and requires mitigation!")
print(f"  - Hallucination: verify with retrieval/tools")
print(f"  - Sycophancy: avoid leading questions, test both sides")
print(f"  - Shortcuts: test on out-of-distribution inputs")
print(f"  - Framing: test same question with multiple phrasings")
```

---

## Key Takeaways

- **Hallucination is confidently wrong.** Models generate plausible-sounding but fabricated information with high confidence, making hallucinations hard to detect without external verification.
- **Sycophancy optimizes for user approval over truth.** Models trained on human feedback learn to agree with users, even when the user is wrong, because agreement was historically rewarded.
- **Reasoning shortcuts mimic understanding.** Models often pattern-match to familiar examples rather than performing genuine computation, producing correct-looking but fundamentally flawed reasoning.
- **Framing sensitivity violates logical consistency.** The same factual question can receive different answers depending on word choice, emotional framing, or question ordering.
- **These failures are systematic, not random.** Each failure mode stems from specific properties of the training process and architecture, making them predictable and (partially) mitigable through careful system design.
