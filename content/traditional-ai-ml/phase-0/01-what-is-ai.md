# What is AI?

> Phase 0 — What is AI? | Kata 0.1

---

## Concept & Intuition

### What problem are we solving?

Artificial Intelligence is one of the most transformative fields in computer science, yet its definition has shifted dramatically over the decades. At its core, AI is the study and engineering of systems that can perform tasks typically requiring human intelligence — recognizing patterns, making decisions, understanding language, and navigating complex environments. Understanding the history and scope of AI is essential before diving into specific techniques.

The field was formally born at the 1956 Dartmouth Conference, where researchers like John McCarthy, Marvin Minsky, and Claude Shannon proposed that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." Since then, AI has gone through cycles of excitement ("AI summers") and disillusionment ("AI winters"), each driven by breakthroughs and subsequent realizations about the limits of current approaches.

Today, we distinguish between **Narrow AI** (also called Weak AI), which excels at specific tasks like image classification or playing chess, and **General AI** (also called Strong AI or AGI), which would possess human-level reasoning across all domains. Every practical AI system deployed today — from voice assistants to self-driving cars — is narrow AI. Understanding this distinction prevents unrealistic expectations and guides how we approach building intelligent systems.

### Why naive approaches fail

A common misconception is that AI is simply "programming a computer to be smart." But hard-coding intelligence through exhaustive rules quickly becomes intractable. The real world is messy, ambiguous, and infinitely varied. You cannot write enough if-then statements to handle every possible situation a self-driving car might encounter, or every way a sentence in English can be phrased.

Another naive approach is assuming that more data or faster hardware alone will produce intelligence. While computational resources are important enablers, intelligence also requires the right algorithms, representations, and learning paradigms. The history of AI teaches us that breakthroughs often come from fundamental shifts in how we frame problems, not just from scaling up existing approaches.

### Mental models

- **AI as a spectrum**: Think of AI not as a binary (intelligent or not) but as a spectrum from simple automation (thermostat) to narrow intelligence (chess engine) to hypothetical general intelligence (human-level reasoning)
- **The toolbox analogy**: AI is not one technique but a toolbox — rule-based systems, search algorithms, statistical learning, neural networks, and more. Different problems call for different tools
- **Seasons of AI**: The field cycles through hype and disappointment. Each "winter" pruned overblown claims but left behind real, lasting contributions (expert systems, backpropagation, deep learning)
- **The Turing Test lens**: Alan Turing reframed "Can machines think?" into the more practical "Can a machine behave indistinguishably from a human?" — a shift from philosophy to engineering

### Visual explanations

```
Timeline of AI Milestones
==========================

1950        1960        1970        1980        1990        2000        2010        2020
 |           |           |           |           |           |           |           |
 Turing      Dartmouth   ELIZA      Expert      AI Winter   Deep Blue   Deep        ChatGPT
 Test        Conference  chatbot    Systems     ends        beats       Learning    LLMs
 paper       (AI born)              boom                    Kasparov    revolution  era
 |           |           |           |           |           |           |           |
 +--- AI Spring 1 ------+--- AI Winter 1 ------+--- AI Spring 2 ------+--- AI Spring 3 --->


Types of AI
============

+--------------------------------------------------+
|              Artificial Intelligence              |
|                                                   |
|  +--------------------+  +---------------------+  |
|  |    Narrow AI       |  |   General AI (AGI)  |  |
|  |  (Weak AI)         |  |   (Strong AI)       |  |
|  |                    |  |                     |  |
|  | - Chess engines    |  | - Human-level       |  |
|  | - Image classifiers|  |   reasoning         |  |
|  | - Spam filters     |  | - Transfer across   |  |
|  | - Voice assistants |  |   all domains       |  |
|  | - Recommendation   |  | - Self-awareness?   |  |
|  |   systems          |  |                     |  |
|  |                    |  | STATUS: Hypothetical|  |
|  | STATUS: Deployed   |  +---------------------+  |
|  +--------------------+                           |
+--------------------------------------------------+
```

---

## Hands-on Exploration

1. **Map the AI landscape**: List five products or services you use daily. For each one, identify whether AI is involved, and if so, what type of AI technique it likely uses (rules, search, machine learning, deep learning). This builds intuition for how pervasive narrow AI already is.

2. **The Turing Test experiment**: Find an online chatbot and have a conversation. Note moments where it feels "intelligent" and moments where the illusion breaks. What patterns do you notice in its failures? This reveals the gap between narrow and general AI.

3. **Timeline deep dive**: Pick one AI milestone (e.g., Deep Blue vs. Kasparov, AlphaGo vs. Lee Sedol, GPT-3's release). Research what specific techniques made it possible. Identify whether the breakthrough was primarily algorithmic, data-driven, or compute-driven.

4. **Classify AI approaches**: Run the code below to see a simple taxonomy of AI approaches. Modify it to add new techniques you discover in your research.

---

## Live Code

```python
"""
What is AI? — A visual taxonomy of AI approaches and their relationships.

This code builds an interactive classification of AI methods and prints
a structured overview of the field.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the AI taxonomy as a structured dictionary
ai_taxonomy = {
    "Symbolic AI": {
        "description": "Uses explicit rules and logic",
        "techniques": ["Expert Systems", "Search Algorithms", "Logic Programming", "Knowledge Graphs"],
        "era": "1950s-1980s",
        "strengths": "Explainable, precise",
        "weaknesses": "Brittle, hard to scale",
    },
    "Statistical AI": {
        "description": "Uses probability and statistics",
        "techniques": ["Bayesian Networks", "HMMs", "Naive Bayes", "Regression"],
        "era": "1980s-2000s",
        "strengths": "Handles uncertainty",
        "weaknesses": "Needs feature engineering",
    },
    "Machine Learning": {
        "description": "Learns patterns from data",
        "techniques": ["Decision Trees", "SVMs", "Random Forests", "k-NN"],
        "era": "1990s-present",
        "strengths": "Adapts to data",
        "weaknesses": "Needs labeled data, can overfit",
    },
    "Deep Learning": {
        "description": "Learns hierarchical representations",
        "techniques": ["CNNs", "RNNs", "Transformers", "GANs"],
        "era": "2010s-present",
        "strengths": "Handles raw data, scales well",
        "weaknesses": "Data hungry, black box",
    },
}

# Print the taxonomy
print("=" * 60)
print("TAXONOMY OF AI APPROACHES")
print("=" * 60)
for category, info in ai_taxonomy.items():
    print(f"\n{'─' * 50}")
    print(f"  {category}")
    print(f"  {info['description']}")
    print(f"  Era: {info['era']}")
    print(f"  Strengths: {info['strengths']}")
    print(f"  Weaknesses: {info['weaknesses']}")
    print(f"  Techniques: {', '.join(info['techniques'])}")

# Visualize the evolution of AI approaches over time
fig, ax = plt.subplots(figsize=(12, 6))

categories = list(ai_taxonomy.keys())
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# Create timeline bars
era_ranges = {
    "Symbolic AI": (1950, 1990),
    "Statistical AI": (1980, 2010),
    "Machine Learning": (1990, 2026),
    "Deep Learning": (2010, 2026),
}

for i, (cat, (start, end)) in enumerate(era_ranges.items()):
    ax.barh(i, end - start, left=start, height=0.6, color=colors[i], alpha=0.8, label=cat)
    ax.text(start + (end - start) / 2, i, cat, ha="center", va="center", fontweight="bold", fontsize=10)

# Add milestone markers
milestones = {
    1956: "Dartmouth\nConference",
    1997: "Deep Blue",
    2012: "AlexNet",
    2017: "Transformer",
    2022: "ChatGPT",
}

for year, label in milestones.items():
    ax.axvline(x=year, color="gray", linestyle="--", alpha=0.5)
    ax.text(year, len(categories) - 0.5, f"{label}\n({year})", ha="center", va="bottom", fontsize=8, color="gray")

ax.set_xlabel("Year", fontsize=12)
ax.set_title("Evolution of AI Approaches", fontsize=14, fontweight="bold")
ax.set_yticks(range(len(categories)))
ax.set_yticklabels([""] * len(categories))
ax.set_xlim(1945, 2030)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Simple demonstration: rule-based vs learned classification
print("\n" + "=" * 60)
print("NARROW AI vs GENERAL AI — A Simple Example")
print("=" * 60)

# Rule-based approach (Narrow AI for temperature classification)
def rule_based_classify(temp_celsius):
    """A simple rule-based system — narrow, brittle, but explainable."""
    if temp_celsius < 0:
        return "Freezing"
    elif temp_celsius < 15:
        return "Cold"
    elif temp_celsius < 25:
        return "Comfortable"
    elif temp_celsius < 35:
        return "Hot"
    else:
        return "Extreme heat"

temperatures = [-10, 5, 20, 30, 45]
print("\nRule-based temperature classifier (Narrow AI):")
for t in temperatures:
    print(f"  {t:>4}°C  ->  {rule_based_classify(t)}")

print("\nKey insight: This system is 'intelligent' at one task only.")
print("It cannot classify images, understand text, or do anything else.")
print("This is the essence of Narrow AI — powerful but specialized.")
```

---

## Key Takeaways

- **AI is a broad field** encompassing rule-based systems, search algorithms, statistical methods, machine learning, and deep learning — not just one technique
- **Narrow AI vs General AI**: Every deployed AI system today is narrow (specialized to one task). General AI that reasons across all domains remains a research goal, not a reality
- **History matters**: Understanding AI's cycles of hype and disillusionment helps set realistic expectations and appreciate which ideas have stood the test of time
- **The right tool for the right job**: Different AI problems demand different approaches. A chess engine uses search; a spam filter uses statistical learning; an image classifier uses deep learning
- **Intelligence is not binary**: AI capabilities exist on a spectrum, and even simple rule-based systems can appear "intelligent" within their narrow domain
