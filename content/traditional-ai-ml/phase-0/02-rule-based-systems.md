# Rule-Based Systems

> Phase 0 — What is AI? | Kata 0.2

---

## Concept & Intuition

### What problem are we solving?

Before machine learning became dominant, the primary approach to building "intelligent" systems was to encode human expertise as explicit rules. Rule-based systems (also called expert systems or production systems) operate on a simple but powerful idea: capture domain knowledge as a collection of IF-THEN rules, then apply those rules to new situations to make decisions. These systems powered medical diagnosis, tax preparation, and industrial control for decades.

The appeal is obvious — rules are transparent, auditable, and directly encode human reasoning. When a medical expert system recommends a diagnosis, you can trace exactly which rules fired and why. This explainability remains a significant advantage even today, in an era where many ML models operate as "black boxes."

However, rule-based systems have a fundamental limitation: **brittleness**. The real world is full of exceptions, edge cases, and gradual transitions that resist clean if-then boundaries. As the number of rules grows, interactions between them become unpredictable, maintenance becomes a nightmare, and the system fails catastrophically when encountering situations its designers didn't anticipate. Understanding why rule-based systems fail is essential to appreciating why statistical and learning-based approaches emerged.

### Why naive approaches fail

The naive assumption is that you can always add more rules to handle new cases. In practice, this leads to "rule explosion" — a combinatorial growth in the number of rules needed to cover all situations. A medical diagnosis system might start with 50 rules but quickly need thousands as edge cases accumulate. Worse, new rules can conflict with existing ones, creating contradictions that are hard to detect and resolve.

Another failure mode is the **knowledge acquisition bottleneck**. Extracting rules from human experts is slow, expensive, and often incomplete. Experts frequently rely on intuition they cannot articulate as explicit rules. This gap between what experts know and what can be formalized is one of the key reasons the expert systems boom of the 1980s gave way to the AI winter of the early 1990s.

### Mental models

- **The recipe analogy**: Rules are like cooking recipes — they work perfectly when ingredients and conditions match expectations, but a great chef (general intelligence) can improvise when the recipe doesn't cover the situation
- **The bureaucracy trap**: Adding more rules is like adding more bureaucratic procedures. At some point, the system becomes so complex that it's harder to maintain the rules than to just do the task manually
- **Crisp vs. fuzzy boundaries**: Rules draw sharp lines ("temperature > 100 means fever"), but reality is continuous. A temperature of 99.9 is not meaningfully different from 100.1, yet the rule treats them completely differently
- **The maintenance iceberg**: The visible rules are the tip; below the surface lies a web of interactions, edge cases, and implicit assumptions that grow exponentially

### Visual explanations

```
Rule-Based System Architecture
================================

+------------------+     +------------------+     +------------------+
|   Knowledge Base |     |  Inference Engine |     |   Working Memory |
|   (IF-THEN Rules)|---->|  (Rule Matcher)  |<--->|   (Current Facts)|
|                  |     |                  |     |                  |
| IF fever AND     |     | 1. Match rules   |     | - temp = 102     |
|   cough THEN flu |     | 2. Resolve       |     | - has_cough = T  |
|                  |     |    conflicts      |     | - has_rash = F   |
| IF rash AND      |     | 3. Fire best     |     |                  |
|   fever THEN ... |     |    rule           |     | => CONCLUSION:   |
+------------------+     +------------------+     |    flu           |
                                                   +------------------+

The Brittleness Problem
========================

  Rules:                    Reality:
  ┌─────────┬─────────┐     ╭─────────────────────╮
  │  Cold   │   Hot   │     │  Cold ... Warm . Hot │
  │ (< 60°) │ (>=60°) │     │    ~~~~~~~~~~~       │
  └─────────┴─────────┘     ╰─────────────────────╯
  Sharp boundary at 60°      Gradual transition

  59°F = "Cold"              59°F and 61°F feel
  61°F = "Hot"               almost the same!

Rule Explosion
===============

  2 variables, 2 values each:   2^2 =   4 rules
  5 variables, 3 values each:   3^5 = 243 rules
 10 variables, 4 values each:  4^10 = ~1 million rules!
```

---

## Hands-on Exploration

1. **Build a tiny expert system**: Using the code below, start with the provided medical diagnosis rules. Add 3 new diseases with their symptoms. Notice how quickly the rule set grows and how hard it becomes to handle overlapping symptoms.

2. **Find the edge cases**: Feed the system unusual combinations of symptoms that don't match any rule cleanly. Observe how the system either gives no answer or a wrong answer. Count how many edge cases you can find — this reveals the brittleness problem firsthand.

3. **Conflict resolution**: Add two rules that could both fire for the same set of symptoms but give different conclusions. Experiment with different conflict resolution strategies (first match, most specific, highest priority). See how the choice of strategy changes outcomes.

4. **Compare with a learned approach**: After running the rule-based system, imagine you had 10,000 patient records with symptoms and diagnoses. Would you rather maintain rules or train a classifier? This thought experiment motivates the transition to machine learning.

---

## Live Code

```python
"""
Rule-Based Systems — Building and breaking an expert system.

This code implements a simple medical diagnosis expert system using
IF-THEN rules, then demonstrates its brittleness with edge cases.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Part 1: A Simple Rule-Based Expert System
# ============================================================

class RuleBasedExpertSystem:
    """A forward-chaining rule-based expert system."""

    def __init__(self):
        self.rules = []
        self.facts = {}

    def add_rule(self, conditions, conclusion, confidence=1.0, name=""):
        """Add an IF-THEN rule to the knowledge base."""
        self.rules.append({
            "conditions": conditions,
            "conclusion": conclusion,
            "confidence": confidence,
            "name": name,
        })

    def set_facts(self, facts):
        """Set the current working memory (observed symptoms)."""
        self.facts = facts.copy()

    def evaluate(self):
        """Forward-chain through all rules. Return matching conclusions."""
        results = []
        for rule in self.rules:
            match = True
            for key, expected in rule["conditions"].items():
                if key not in self.facts or self.facts[key] != expected:
                    match = False
                    break
            if match:
                results.append({
                    "conclusion": rule["conclusion"],
                    "confidence": rule["confidence"],
                    "rule_name": rule["name"],
                })
        return results


# Build the expert system
expert = RuleBasedExpertSystem()

# Add medical diagnosis rules
expert.add_rule(
    conditions={"fever": True, "cough": True, "body_ache": True},
    conclusion="Influenza (Flu)",
    confidence=0.85,
    name="R1: Flu rule",
)

expert.add_rule(
    conditions={"fever": True, "sore_throat": True, "cough": False},
    conclusion="Strep Throat",
    confidence=0.80,
    name="R2: Strep rule",
)

expert.add_rule(
    conditions={"sneezing": True, "runny_nose": True, "fever": False},
    conclusion="Common Cold",
    confidence=0.90,
    name="R3: Cold rule",
)

expert.add_rule(
    conditions={"fever": True, "rash": True},
    conclusion="Measles",
    confidence=0.75,
    name="R4: Measles rule",
)

expert.add_rule(
    conditions={"headache": True, "stiff_neck": True, "fever": True},
    conclusion="Meningitis",
    confidence=0.70,
    name="R5: Meningitis rule",
)

# Test cases — including edge cases that expose brittleness
test_cases = [
    {"name": "Classic flu", "symptoms": {"fever": True, "cough": True, "body_ache": True}},
    {"name": "Common cold", "symptoms": {"sneezing": True, "runny_nose": True, "fever": False}},
    {"name": "Ambiguous case", "symptoms": {"fever": True, "cough": True, "rash": True, "body_ache": True}},
    {"name": "No matching rule", "symptoms": {"fatigue": True, "dizziness": True}},
    {"name": "Partial match", "symptoms": {"fever": True, "cough": True, "body_ache": False}},
]

print("=" * 60)
print("RULE-BASED MEDICAL DIAGNOSIS EXPERT SYSTEM")
print("=" * 60)

for case in test_cases:
    expert.set_facts(case["symptoms"])
    results = expert.evaluate()
    print(f"\nPatient: {case['name']}")
    print(f"  Symptoms: {case['symptoms']}")
    if results:
        for r in results:
            print(f"  => Diagnosis: {r['conclusion']} (confidence: {r['confidence']:.0%})")
            print(f"     Matched: {r['rule_name']}")
    else:
        print("  => NO DIAGNOSIS — no rules matched!")
        print("     This is the brittleness problem in action.")

# ============================================================
# Part 2: Visualizing Rule Explosion
# ============================================================

num_variables = np.arange(2, 12)
values_per_var = [2, 3, 4]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: rule explosion
ax = axes[0]
for v in values_per_var:
    num_rules = v ** num_variables
    ax.semilogy(num_variables, num_rules, "o-", label=f"{v} values/variable", linewidth=2)

ax.set_xlabel("Number of Variables", fontsize=12)
ax.set_ylabel("Number of Rules (log scale)", fontsize=12)
ax.set_title("Rule Explosion Problem", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="Practical limit (~1000 rules)")
ax.text(6, 1500, "Practical maintenance limit", color="red", fontsize=9)

# Right plot: coverage vs rules added (diminishing returns)
ax = axes[1]
np.random.seed(42)
rules_added = np.arange(1, 101)
# Simulate diminishing returns: each new rule covers less new territory
coverage = 100 * (1 - np.exp(-0.05 * rules_added))
# Add some noise
coverage += np.random.normal(0, 0.5, len(rules_added))
coverage = np.clip(coverage, 0, 100)

ax.plot(rules_added, coverage, "-", linewidth=2, color="#2ecc71")
ax.fill_between(rules_added, coverage, alpha=0.2, color="#2ecc71")
ax.axhline(y=95, color="red", linestyle="--", alpha=0.5)
ax.text(50, 96, "95% coverage — the last 5% requires enormous effort", fontsize=9, color="red")
ax.set_xlabel("Number of Rules Added", fontsize=12)
ax.set_ylabel("Case Coverage (%)", fontsize=12)
ax.set_title("Diminishing Returns of Adding Rules", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Part 3: The Decision Boundary Problem
# ============================================================

print("\n" + "=" * 60)
print("CRISP BOUNDARIES vs REALITY")
print("=" * 60)

# Show how rule boundaries create artifacts
temps = np.linspace(95, 105, 200)

# Rule-based classification
def rule_classify(temp):
    if temp >= 100.4:
        return 1.0  # "Fever"
    else:
        return 0.0  # "No fever"

# A more realistic probability curve
def realistic_probability(temp):
    return 1 / (1 + np.exp(-2 * (temp - 100.4)))

rule_labels = np.array([rule_classify(t) for t in temps])
realistic = realistic_probability(temps)

fig, ax = plt.subplots(figsize=(10, 5))
ax.step(temps, rule_labels, "r-", linewidth=2, label="Rule-based (sharp cutoff)", where="mid")
ax.plot(temps, realistic, "b-", linewidth=2, label="Realistic (gradual transition)")
ax.axvline(x=100.4, color="gray", linestyle=":", alpha=0.5)
ax.text(100.5, 0.5, "100.4°F threshold", rotation=90, va="center", fontsize=9, color="gray")
ax.set_xlabel("Temperature (°F)", fontsize=12)
ax.set_ylabel("P(Fever)", fontsize=12)
ax.set_title("Rule-Based vs. Probabilistic Classification", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nKey insight: Rules create artificial sharp boundaries.")
print("A patient at 100.3°F and one at 100.5°F are treated completely")
print("differently, even though they are clinically almost identical.")
print("This brittleness is a fundamental limitation of rule-based systems.")
```

---

## Key Takeaways

- **Rule-based systems encode expertise as IF-THEN rules** — they are transparent, auditable, and were the dominant AI paradigm in the 1980s
- **Brittleness is the fatal flaw**: Rule-based systems fail when encountering situations not covered by existing rules, and adding more rules leads to combinatorial explosion
- **The knowledge acquisition bottleneck** makes it expensive and slow to extract and formalize expert knowledge, especially tacit knowledge that experts cannot articulate
- **Sharp decision boundaries** are unrealistic for most real-world problems, where categories blend gradually rather than having crisp cutoffs
- **Rule-based systems still have their place** in safety-critical domains where explainability is mandatory and the problem space is well-bounded (e.g., tax calculations, regulatory compliance)
