# Emergence Through Scale

> Phase 8 — LLMs | Kata 8.5

---

## Concept & Intuition

### What problem are we solving?

As language models scale up in parameters, data, and compute, they exhibit a remarkable phenomenon: qualitatively new capabilities appear that were absent in smaller models. These emergent abilities include few-shot learning (solving tasks from a handful of examples in the prompt), chain-of-thought reasoning (solving multi-step problems by generating intermediate reasoning), and instruction following (correctly interpreting and executing novel instructions). These capabilities were not explicitly trained for; they emerge as a byproduct of the next-token prediction objective applied at sufficient scale.

The term "emergence" is borrowed from complexity science, where it describes properties of a system that are not present in its individual components. A single neuron cannot reason, but billions of neurons with the right connectivity can. Similarly, a small language model trained on next-token prediction learns surface statistics, but a sufficiently large model trained on the same objective appears to develop something resembling understanding.

Understanding emergence matters because it suggests that the most important capabilities of LLMs were not designed by engineers but discovered through scaling. This has profound implications for AI safety (we may not fully understand what a model can do until we test it), for research strategy (scaling may be more productive than architectural innovation for some capabilities), and for forecasting (if we can characterize the scaling laws governing emergence, we can predict what capabilities future models might develop).

### Why naive approaches fail

Attempting to explicitly program each capability (reasoning, translation, summarization, etc.) as a separate module creates an intractable engineering problem. There are too many possible tasks, and hard-coding each one does not generalize to novel tasks. The beauty of emergence is that a single training objective produces a model capable of thousands of tasks.

Extrapolating from small model behavior to predict large model behavior fails because emergence is often nonlinear. A model with 1 billion parameters might score 0% on a particular benchmark, while a model with 10 billion parameters scores 5%, and a model with 100 billion parameters scores 80%. The capability appears to "switch on" at a threshold rather than improving gradually. This makes it difficult to predict capabilities from smaller-scale experiments.

### Mental models

- **Phase transitions**: Emergence is like water freezing. The temperature drops gradually, but the transition from liquid to solid happens suddenly at a threshold. Similarly, capabilities in LLMs can appear suddenly as model size crosses a threshold.
- **Critical mass**: Like a nuclear chain reaction that only sustains above a critical mass of fuel, certain cognitive capabilities may require a critical mass of parameters, data, and compute to manifest.
- **Compositionality threshold**: Small models learn individual skills (grammar, facts, simple patterns). At some scale, they learn to compose these skills, enabling capabilities that are qualitatively different from any individual skill.

### Visual explanations

```
  Emergent capabilities vs model scale (schematic):

  Accuracy
  100%|                                         ____
     |                                    ___/
  80%|                                ___/
     |                            ___/
  60%|                        ___/
     |                    ___/
  40%|               ____/
     |          ____/
  20%|     ____/
     | ___/
   0%|___________________________________________
     10^7    10^8    10^9    10^10   10^11  Parameters
     |--------|--------|---------|---------|
     Small    Medium   Large    Very Large

  Capabilities appearing at different scales:

  10^8  params: Basic grammar, word associations
  10^9  params: Simple QA, sentiment analysis
  10^10 params: Multi-step reasoning, translation
  10^11 params: Chain-of-thought, code generation
  10^12+params: Complex reasoning, instruction following

  The "grokking" analogy:
  Training loss decreases gradually...
  But test accuracy on specific tasks jumps suddenly!

  Loss: ╲╲╲╲╲─────────────── (smooth decrease)
  Acc:  ________╱╱╱╱╱──────── (sudden jump)
```

---

## Hands-on Exploration

1. Simulate emergent behavior using a simple threshold model.
2. Show how composing simple skills creates qualitatively new capabilities.
3. Demonstrate the relationship between model capacity and task complexity.
4. Illustrate why small-scale experiments can fail to predict large-scale behavior.

---

## Live Code

```rust
fn main() {
    println!("=== Emergence Through Scale ===\n");

    // 1. Simulate emergence: models of different sizes on tasks of different difficulty
    println!("--- Simulating Emergent Capabilities ---\n");

    // Each "skill" has a difficulty threshold. A model can perform a compound task
    // only if it has mastered all component skills.
    let skills = vec![
        ("grammar", 0.2),        // Easy, small models can learn
        ("vocabulary", 0.3),     // Easy
        ("facts", 0.5),          // Medium
        ("simple_logic", 0.6),   // Medium
        ("multi_step", 0.8),     // Hard
        ("abstraction", 0.9),    // Very hard
    ];

    // Compound tasks require multiple skills
    let tasks = vec![
        ("Simple QA", vec!["grammar", "vocabulary", "facts"]),
        ("Translation", vec!["grammar", "vocabulary"]),
        ("Multi-step math", vec!["grammar", "simple_logic", "multi_step"]),
        ("Chain-of-thought", vec!["grammar", "facts", "simple_logic", "multi_step"]),
        ("Novel reasoning", vec!["grammar", "facts", "simple_logic", "multi_step", "abstraction"]),
    ];

    // Model "capacity" increases with scale
    let model_scales: Vec<(f64, &str)> = vec![
        (0.1, "10M"),
        (0.3, "100M"),
        (0.5, "1B"),
        (0.65, "10B"),
        (0.8, "100B"),
        (0.95, "1T"),
    ];

    // A model masters a skill if its capacity exceeds the skill's difficulty
    // with some noise (sigmoid transition)
    println!("Skill mastery by model scale:");
    println!("{:>8} {}", "Scale", skills.iter().map(|(n, _)| format!("{:>12}", n)).collect::<Vec<_>>().join(""));
    println!("{}", "-".repeat(8 + 12 * skills.len()));

    for (capacity, label) in &model_scales {
        print!("{:>8}", label);
        for (_, difficulty) in &skills {
            let mastery = sigmoid((*capacity - difficulty) * 15.0);
            let symbol = if mastery > 0.8 {
                "  YES"
            } else if mastery > 0.3 {
                "  ~"
            } else {
                "  -"
            };
            print!("{:>12}", symbol);
        }
        println!();
    }

    // Task performance: product of component skill masteries
    println!("\n\nTask performance (emergent behavior):");
    print!("{:>25}", "Task");
    for (_, label) in &model_scales {
        print!("{:>8}", label);
    }
    println!();
    println!("{}", "-".repeat(25 + 8 * model_scales.len()));

    for (task_name, required_skills) in &tasks {
        print!("{:>25}", task_name);
        for (capacity, _) in &model_scales {
            // Task success = product of skill masteries (all must be high)
            let task_score: f64 = required_skills
                .iter()
                .map(|skill_name| {
                    let difficulty = skills
                        .iter()
                        .find(|(n, _)| n == skill_name)
                        .unwrap()
                        .1;
                    sigmoid((capacity - difficulty) * 15.0)
                })
                .product();

            let display = format!("{:.0}%", task_score * 100.0);
            print!("{:>8}", display);
        }
        println!();
    }

    println!("\nNotice: simple tasks improve gradually, complex tasks 'jump' suddenly.");

    // 2. Composition creates new capabilities
    println!("\n--- Composition Creates New Capabilities ---\n");

    // Simulate: small models learn skills A and B independently but cannot combine them
    // Large models learn to combine skills
    let skill_a_scores = [0.9, 0.95, 0.98, 0.99]; // Near-perfect at all scales
    let skill_b_scores = [0.3, 0.6, 0.85, 0.95]; // Improves with scale
    let composition_ability = [0.0, 0.1, 0.6, 0.95]; // Emerges at large scale

    println!(
        "{:>12} {:>10} {:>10} {:>15} {:>15}",
        "Scale", "Skill A", "Skill B", "Composition", "Combined Task"
    );
    println!("{}", "-".repeat(65));

    let scale_labels = ["100M", "1B", "10B", "100B"];
    for i in 0..4 {
        let combined = skill_a_scores[i] * skill_b_scores[i] * composition_ability[i];
        println!(
            "{:>12} {:>10.0}% {:>10.0}% {:>15.0}% {:>15.0}%",
            scale_labels[i],
            skill_a_scores[i] * 100.0,
            skill_b_scores[i] * 100.0,
            composition_ability[i] * 100.0,
            combined * 100.0,
        );
    }
    println!("\n  Individual skills may exist at small scale, but");
    println!("  the ability to COMPOSE them emerges only at large scale.");

    // 3. Why small-scale experiments mislead
    println!("\n--- Why Small-Scale Experiments Mislead ---\n");

    // Fit a line to small-model data, then extrapolate
    let actual_scores: Vec<(f64, f64)> = vec![
        (7.0, 0.0),   // 10^7 params
        (8.0, 0.0),   // 10^8 params
        (9.0, 2.0),   // 10^9 params
        (10.0, 15.0),  // 10^10 params
        (11.0, 65.0),  // 10^11 params
        (12.0, 85.0),  // 10^12 params
    ];

    println!("Actual performance on a complex reasoning task:\n");
    for (log_params, score) in &actual_scores {
        let bar = "#".repeat(*score as usize / 2);
        println!(
            "  10^{:.0} params: {:>5.1}%  {}",
            log_params, score, bar
        );
    }

    // Linear extrapolation from first 3 points
    println!("\nLinear extrapolation from first 3 data points:");
    let slope = (actual_scores[2].1 - actual_scores[0].1)
        / (actual_scores[2].0 - actual_scores[0].0);
    let intercept = actual_scores[0].1 - slope * actual_scores[0].0;

    for (log_params, actual) in &actual_scores {
        let predicted = (slope * log_params + intercept).max(0.0);
        println!(
            "  10^{:.0}: predicted={:>5.1}%, actual={:>5.1}%  {}",
            log_params,
            predicted,
            actual,
            if (predicted - actual).abs() > 20.0 { "<-- WRONG!" } else { "" }
        );
    }
    println!("\n  Linear extrapolation from small models dramatically");
    println!("  underestimates large model capabilities.");

    // 4. The few-shot learning emergence
    println!("\n--- Few-Shot Learning Emergence ---\n");

    // Simulate: given N examples in prompt, accuracy vs model size
    let n_examples = [0, 1, 2, 5];

    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Scale", "0-shot", "1-shot", "2-shot", "5-shot"
    );
    println!("{}", "-".repeat(52));

    for (i, label) in scale_labels.iter().enumerate() {
        let capacity = [0.3, 0.5, 0.7, 0.95][i];
        let scores: Vec<String> = n_examples
            .iter()
            .map(|n| {
                // Few-shot benefit scales with model capacity
                let base = sigmoid((capacity - 0.5) * 10.0) * 50.0;
                let few_shot_bonus = (*n as f64).ln().max(0.0) * capacity * 30.0;
                let score = (base + few_shot_bonus).min(100.0);
                format!("{:.0}%", score)
            })
            .collect();
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>10}",
            label, scores[0], scores[1], scores[2], scores[3]
        );
    }
    println!("\n  Small models barely improve with more examples.");
    println!("  Large models dramatically improve with just a few examples.");
    println!("  Few-shot learning is an emergent capability of scale.");
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

---

## Key Takeaways

- Emergence refers to qualitatively new capabilities that appear in large language models but are absent in smaller ones, despite identical training objectives.
- Complex tasks require composing multiple skills, and this composition ability appears to develop suddenly at a critical scale rather than improving gradually.
- Small-scale experiments can dramatically underestimate the capabilities of large models because emergent abilities follow nonlinear (often S-shaped) scaling curves.
- The implication is that increasing scale is itself a research methodology: some capabilities cannot be discovered or studied in small models, requiring a "bet" on compute that pays off through emergence.
