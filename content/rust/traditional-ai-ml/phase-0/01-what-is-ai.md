# What is AI?

> Phase 0 — What is AI? | Kata 0.01

---

## Concept & Intuition

### What problem are we solving?

Artificial Intelligence is the study and engineering of systems that can perceive their environment, reason about it, and take actions to achieve goals. At its core, AI asks a deceptively simple question: can we build machines that exhibit intelligent behavior? This encompasses everything from a thermostat that adjusts temperature to a chess engine that defeats grandmasters.

Before diving into machine learning algorithms, it is essential to understand what intelligence means in a computational context. Intelligence is not a single capability but a spectrum: pattern recognition, planning, language understanding, learning from experience, and generalization to new situations. AI research organizes these capabilities into narrow AI (systems that excel at one specific task) and general AI (hypothetical systems with human-level versatility).

In this kata, we will build a simple decision-making agent that classifies inputs based on rules and thresholds. This establishes the foundational pattern that every AI system follows: observe inputs, apply some transformation or reasoning, and produce outputs.

### Why naive approaches fail

A naive approach to building intelligent systems is to hard-code every possible scenario. For a tic-tac-toe game, you might enumerate all board states and optimal moves. But this approach collapses when the problem space grows: chess has roughly 10^44 legal positions. You cannot enumerate them all. This combinatorial explosion is the fundamental reason we need smarter approaches — algorithms that can generalize from limited information rather than memorize every case.

### Mental models

- **AI as function approximation**: Every AI system maps inputs to outputs. The challenge is finding the right mapping function, whether by hand-coding rules or learning from data.
- **The intelligence spectrum**: From simple if-then rules to deep neural networks, AI systems vary in how much structure is provided by the programmer versus discovered by the algorithm.
- **No free lunch**: No single AI approach works best for all problems. Understanding the problem domain is as important as knowing the algorithm.

### Visual explanations

```
  Input Space                    AI System                   Output Space
 +-----------+              +----------------+              +-----------+
 | sensor    |              |                |              | action    |
 | readings, | ----------> |  Rules / Model  | ----------> | decision, |
 | images,   |              |  / Learned fn  |              | label,    |
 | text ...  |              |                |              | value ... |
 +-----------+              +----------------+              +-----------+
                                    ^
                                    |
                            Knowledge / Data
                            (hand-coded or learned)
```

---

## Hands-on Exploration

1. Define a simple classification problem: given a temperature reading, classify it as "cold", "comfortable", or "hot".
2. Implement a threshold-based classifier — the simplest possible AI agent.
3. Test it against a range of inputs and observe how the decision boundaries work.
4. Reflect on what happens when you need to handle more dimensions (temperature + humidity).

---

## Live Code

```rust
fn main() {
    // A simple threshold-based AI agent
    // This is the most basic form of artificial intelligence:
    // mapping inputs to outputs via rules.

    let temperatures = vec![
        -10.0, 0.0, 10.0, 15.0, 20.0, 22.0, 25.0, 30.0, 35.0, 40.0,
    ];

    println!("=== Simple AI Agent: Temperature Classifier ===\n");
    println!("{:<12} {:<15} {:<10}", "Temp (C)", "Classification", "Confidence");
    println!("{}", "-".repeat(37));

    let mut correct = 0;
    let total = temperatures.len();

    for &temp in &temperatures {
        let (label, confidence) = classify_temperature(temp);
        println!("{:<12.1} {:<15} {:<10.2}", temp, label, confidence);

        // Self-check: verify classification is consistent
        let (label2, _) = classify_temperature(temp);
        if label == label2 {
            correct += 1;
        }
    }

    let consistency = correct as f64 / total as f64;
    kata_metric("consistency", consistency);

    // Demonstrate the limitation: what about humidity?
    println!("\n=== Limitation: Multi-dimensional Input ===\n");
    println!("Temperature 30C, Humidity 20% -> feels comfortable (dry heat)");
    println!("Temperature 30C, Humidity 90% -> feels hot (muggy)");
    println!("A single-threshold classifier cannot capture this interaction!\n");

    // A slightly smarter agent: 2D classification
    let scenarios = vec![
        (30.0, 20.0),
        (30.0, 90.0),
        (20.0, 50.0),
        (25.0, 70.0),
    ];

    println!("{:<10} {:<12} {:<15}", "Temp", "Humidity", "Comfort");
    println!("{}", "-".repeat(37));

    for (temp, humidity) in &scenarios {
        let comfort = classify_comfort(*temp, *humidity);
        println!("{:<10.1} {:<12.1} {:<15}", temp, humidity, comfort);
    }

    // Key insight: as dimensions grow, hand-coded rules become unmanageable
    // This motivates machine learning
    let rule_count_1d = 2; // two thresholds for 1D
    let rule_count_2d = 4; // four regions for 2D
    println!("\nRules needed for 1D: {}", rule_count_1d);
    println!("Rules needed for 2D: {}", rule_count_2d);
    println!("Rules needed for 10D: ~{} (combinatorial explosion!)", 2_i64.pow(10));

    kata_metric("rule_count_1d", rule_count_1d as f64);
    kata_metric("rule_count_2d", rule_count_2d as f64);
}

fn classify_temperature(temp: f64) -> (&'static str, f64) {
    // Decision boundaries at 15C and 28C
    if temp < 15.0 {
        let confidence = 1.0 - (temp - 0.0).abs() / 30.0;
        ("cold", confidence.max(0.5))
    } else if temp <= 28.0 {
        let mid = 21.5;
        let confidence = 1.0 - (temp - mid).abs() / 13.0;
        ("comfortable", confidence.max(0.5))
    } else {
        let confidence = (temp - 28.0).min(12.0) / 12.0;
        ("hot", confidence.max(0.5))
    }
}

fn classify_comfort(temp: f64, humidity: f64) -> &'static str {
    // Heat index approximation: perceived temperature
    let heat_index = temp + 0.05 * humidity;
    if heat_index < 20.0 {
        "cold"
    } else if heat_index <= 33.0 {
        "comfortable"
    } else {
        "uncomfortable"
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- AI is fundamentally about mapping inputs to outputs through some form of reasoning or learned function.
- Even simple threshold-based classifiers are a form of AI — they encode knowledge about decision boundaries.
- Hard-coded rules suffer from combinatorial explosion as the input dimensionality grows, motivating data-driven (machine learning) approaches.
- Understanding the limitations of rule-based systems is essential context for appreciating why ML exists.
