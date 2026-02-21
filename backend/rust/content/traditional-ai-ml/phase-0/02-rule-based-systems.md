# Rule-Based Systems

> Phase 0 — What is AI? | Kata 0.02

---

## Concept & Intuition

### What problem are we solving?

Rule-based systems (also called expert systems) were among the first successful AI applications. The idea is straightforward: encode human expert knowledge as a collection of if-then rules, then apply those rules to new situations. A medical diagnosis system might have rules like "IF fever AND cough AND fatigue THEN suspect flu". These systems dominated AI from the 1970s through the 1990s and remain useful for well-understood, structured domains.

The power of rule-based systems lies in their transparency. Every decision can be traced back to specific rules, making them interpretable and auditable. In domains like healthcare, finance, and law, this explainability is not just nice to have — it is often legally required. When a system denies a loan application, you need to explain why.

In this kata, we build a complete rule-based inference engine from scratch. We will define facts, rules, and a forward-chaining mechanism that derives new conclusions from known facts — the same architecture used in classic expert systems like MYCIN and CLIPS.

### Why naive approaches fail

The simplest rule-based approach is a flat list of if-else statements. This works for small problems but quickly becomes unmaintainable. Rules interact with each other: one rule's conclusion becomes another rule's premise. Without a proper inference engine, you end up manually ordering rules and tracking dependencies. When you add a new rule, you must verify it does not conflict with existing ones — a task that grows quadratically with the number of rules.

A proper inference engine separates the knowledge (rules) from the reasoning mechanism (the engine), making the system modular and extensible.

### Mental models

- **Forward chaining**: Start with known facts, apply rules to derive new facts, repeat until no new facts emerge. Like a chain of dominoes falling forward.
- **Knowledge base as a graph**: Facts are nodes, rules are edges. Inference is graph traversal.
- **Separation of concerns**: The inference engine does not know about medicine, finance, or any domain. It only knows how to apply rules to facts. Domain knowledge lives entirely in the rules.

### Visual explanations

```
  Known Facts           Rules                    Derived Facts
 +----------+    +------------------+         +----------------+
 | fever    | -> | IF fever AND     | ------> | possible_flu   |
 | cough    |    |    cough THEN    |         +----------------+
 +----------+    |    possible_flu  |                |
                 +------------------+                v
                 | IF possible_flu  |         +----------------+
                 |    AND fatigue   | ------> | recommend_test |
                 |    THEN          |         +----------------+
                 |    recommend_test|
                 +------------------+

  Forward Chaining: facts -> rules -> new facts -> rules -> ...
```

---

## Hands-on Exploration

1. Define a set of facts (known conditions) and rules (if-then implications).
2. Implement a forward-chaining inference engine that repeatedly applies rules until no new facts are derived.
3. Test the engine with a medical diagnosis scenario.
4. Observe how adding a single rule can trigger a cascade of new inferences.

---

## Live Code

```rust
fn main() {
    println!("=== Rule-Based Expert System: Medical Diagnosis ===\n");

    // Define the knowledge base: rules
    let rules: Vec<Rule> = vec![
        Rule {
            name: "R1: Flu detection",
            conditions: vec!["fever", "cough"],
            conclusion: "possible_flu",
        },
        Rule {
            name: "R2: Flu confirmation",
            conditions: vec!["possible_flu", "fatigue"],
            conclusion: "likely_flu",
        },
        Rule {
            name: "R3: Cold detection",
            conditions: vec!["sneezing", "runny_nose"],
            conclusion: "possible_cold",
        },
        Rule {
            name: "R4: Allergy detection",
            conditions: vec!["sneezing", "itchy_eyes"],
            conclusion: "possible_allergy",
        },
        Rule {
            name: "R5: Rest recommendation",
            conditions: vec!["likely_flu"],
            conclusion: "recommend_rest",
        },
        Rule {
            name: "R6: Test recommendation",
            conditions: vec!["likely_flu", "high_fever"],
            conclusion: "recommend_test",
        },
        Rule {
            name: "R7: Severe case",
            conditions: vec!["recommend_test", "breathing_difficulty"],
            conclusion: "emergency",
        },
    ];

    // Scenario 1: Patient with flu symptoms
    println!("--- Scenario 1: Flu Patient ---");
    let facts1 = vec![
        "fever", "cough", "fatigue", "high_fever",
    ];
    let result1 = forward_chain(&rules, &facts1);
    print_results(&result1);

    // Scenario 2: Patient with cold
    println!("\n--- Scenario 2: Cold Patient ---");
    let facts2 = vec!["sneezing", "runny_nose"];
    let result2 = forward_chain(&rules, &facts2);
    print_results(&result2);

    // Scenario 3: Ambiguous - cold or allergy?
    println!("\n--- Scenario 3: Ambiguous Symptoms ---");
    let facts3 = vec!["sneezing", "runny_nose", "itchy_eyes"];
    let result3 = forward_chain(&rules, &facts3);
    print_results(&result3);

    // Scenario 4: Severe case
    println!("\n--- Scenario 4: Severe Case ---");
    let facts4 = vec![
        "fever", "cough", "fatigue", "high_fever", "breathing_difficulty",
    ];
    let result4 = forward_chain(&rules, &facts4);
    print_results(&result4);

    // Metrics
    let total_rules = rules.len();
    kata_metric("total_rules", total_rules as f64);
    kata_metric("scenario1_derived_facts", (result1.len() - 4) as f64);
    kata_metric("scenario4_derived_facts", (result4.len() - 5) as f64);

    // Demonstrate conflict detection
    println!("\n=== Rule Conflict Analysis ===");
    let conflicts = detect_conflicts(&rules);
    println!("Rules with shared conditions: {}", conflicts);
    kata_metric("potential_conflicts", conflicts as f64);
}

struct Rule {
    name: &'static str,
    conditions: Vec<&'static str>,
    conclusion: &'static str,
}

fn forward_chain(rules: &[Rule], initial_facts: &[&str]) -> Vec<String> {
    let mut facts: Vec<String> = initial_facts.iter().map(|s| s.to_string()).collect();
    let mut changed = true;
    let mut iteration = 0;

    println!("Initial facts: {:?}", initial_facts);

    while changed {
        changed = false;
        iteration += 1;

        for rule in rules {
            // Check if all conditions are met
            let all_met = rule.conditions.iter().all(|cond| {
                facts.iter().any(|f| f == cond)
            });

            // Check if conclusion is not already known
            let already_known = facts.iter().any(|f| f == rule.conclusion);

            if all_met && !already_known {
                println!(
                    "  Iteration {}: {} fires -> {}",
                    iteration, rule.name, rule.conclusion
                );
                facts.push(rule.conclusion.to_string());
                changed = true;
            }
        }
    }

    println!("Inference complete after {} iterations.", iteration);
    facts
}

fn print_results(facts: &[String]) {
    println!("Final knowledge base: {:?}", facts);
    println!(
        "Derived facts: {:?}",
        facts.iter()
            .filter(|f| {
                !["fever", "cough", "fatigue", "high_fever", "sneezing",
                  "runny_nose", "itchy_eyes", "breathing_difficulty"]
                    .contains(&f.as_str())
            })
            .collect::<Vec<_>>()
    );
}

fn detect_conflicts(rules: &[Rule]) -> usize {
    let mut conflicts = 0;
    for i in 0..rules.len() {
        for j in (i + 1)..rules.len() {
            let shared = rules[i].conditions.iter().filter(|c| {
                rules[j].conditions.contains(c)
            }).count();
            if shared > 0 && rules[i].conclusion != rules[j].conclusion {
                println!(
                    "  {} and {} share {} condition(s)",
                    rules[i].name, rules[j].name, shared
                );
                conflicts += 1;
            }
        }
    }
    conflicts
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Rule-based systems encode expert knowledge as if-then rules and use an inference engine to derive conclusions.
- Forward chaining starts from known facts and applies rules iteratively until no new conclusions can be drawn.
- The separation of knowledge (rules) from reasoning (engine) makes the system modular and domain-independent.
- Rule-based systems are transparent and explainable, but they struggle with scaling, uncertainty, and domains where rules are hard to articulate.
