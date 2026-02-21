# Chain-of-Thought as Latent Variables

> Phase 9 — Reasoning Models | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

When a language model is asked "What is 17 * 24?", it must produce the answer in a single forward pass if forced to answer immediately. The model's computation is limited to the fixed number of layers in its architecture, typically 32-96 transformer layers. For problems requiring multi-step reasoning, this fixed compute budget is fundamentally insufficient. Chain-of-thought (CoT) prompting solves this by allowing the model to generate intermediate reasoning steps before the final answer: "17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408."

From a probabilistic perspective, the chain-of-thought tokens act as latent variables. The model does not directly compute P(answer | question). Instead, it computes P(answer | question, reasoning), where the reasoning tokens are generated as an intermediate representation. These tokens effectively give the model additional computation: each generated token triggers another full forward pass through all transformer layers, allowing the model to iteratively refine its internal representation of the problem.

This reframing is powerful because it reveals that chain-of-thought is not just a prompting trick but a fundamental extension of the model's computational capacity. Without CoT, the model has O(L) computation (L = number of layers). With CoT of length T, the model has O(L * T) computation. For problems whose complexity scales with input size (like multi-digit arithmetic or multi-step logical deductions), this additional computation is essential.

### Why naive approaches fail

Asking a model to directly output the answer to a complex question forces it to "compress" multi-step reasoning into a single forward pass. This works for simple pattern matching ("What is the capital of France?") but fails for problems requiring sequential reasoning. The model has no way to store intermediate results or revisit earlier computations.

Increasing model size alone does not solve this. A larger model has more parameters and can store more knowledge, but it still has the same fixed number of layers and thus the same compute budget per token. Making a model 10x larger gives it 10x more knowledge but does not give it 10x more reasoning steps. Chain-of-thought provides additional reasoning steps at inference time without requiring a larger model.

### Mental models

- **Scratch paper**: Chain-of-thought is like allowing a student to use scratch paper during an exam. The intermediate writing does not contain new information, but the act of writing structures the thinking process and allows intermediate results to be stored and referenced.
- **Computation unrolling**: Each CoT token is like an extra "virtual layer" in the network. By generating 100 reasoning tokens, the model effectively runs 100 additional forward passes, massively expanding its compute budget for that specific problem.
- **Marginalization**: The final answer P(answer | question) = sum over all possible reasoning chains of P(answer | question, chain) * P(chain | question). The model implicitly marginalizes over reasoning paths by generating one likely chain.

### Visual explanations

```
  Direct answering (no CoT):

  Question: "What is 17 * 24?"
  Model:    [32 layers of fixed computation]
  Output:   "408" (or often wrong: "388")

  Chain-of-thought answering:

  Question: "What is 17 * 24?"
  Token 1:  "17" [32 layers] -> "17 *"
  Token 2:  "* 24" [32 layers] -> "17 * 24 ="
  Token 3:  "=" [32 layers] -> "17 * 20"
  Token 4:  "20" [32 layers] -> "+ 17 * 4"
  Token 5:  "4" [32 layers] -> "= 340"
  Token 6:  "340" [32 layers] -> "+ 68"
  Token 7:  "68" [32 layers] -> "= 408"
  Output:   "408" (correct!)

  Total computation: 7 * 32 = 224 effective layers!

  Latent variable view:

  P(answer | Q) = Sum_r [ P(answer | Q, r) * P(r | Q) ]

  where r is the reasoning chain (latent variable)

  Without CoT: Must compute P(408 | "17*24") directly
  With CoT:    Compute P(408 | "17*24", "17*20+17*4=340+68") = easy!
```

---

## Hands-on Exploration

1. Implement a multi-step arithmetic task and show how breaking it into steps improves accuracy.
2. Demonstrate that a fixed-depth computation fails on sufficiently complex inputs.
3. Show how intermediate results act as external memory that expands computational capacity.
4. Measure accuracy with and without chain-of-thought decomposition.

---

## Live Code

```rust
fn main() {
    println!("=== Chain-of-Thought as Latent Variables ===\n");

    // Simulate a model with limited computation per "forward pass"
    // It can do simple operations but not compound ones in a single step

    println!("--- Direct vs Chain-of-Thought Computation ---\n");

    // Task: compute multi-step arithmetic expressions
    let problems = vec![
        ("3 + 5", vec!["3 + 5 = 8"]),
        ("17 * 24", vec!["17 * 20 = 340", "17 * 4 = 68", "340 + 68 = 408"]),
        ("(3 + 5) * (2 + 4)", vec!["3 + 5 = 8", "2 + 4 = 6", "8 * 6 = 48"]),
        (
            "23 * 47 + 15",
            vec!["23 * 40 = 920", "23 * 7 = 161", "920 + 161 = 1081", "1081 + 15 = 1096"],
        ),
        (
            "(12 + 8) * (7 - 3) + 5",
            vec!["12 + 8 = 20", "7 - 3 = 4", "20 * 4 = 80", "80 + 5 = 85"],
        ),
    ];

    for (expr, chain) in &problems {
        println!("  Problem: {}", expr);

        // Direct computation: simulate limited model that can only do single ops
        let direct_result = try_direct_compute(expr);
        let cot_result = chain_of_thought_compute(chain);

        println!("    Direct (1 step):         {}", direct_result);
        println!("    Chain-of-thought ({} steps): {}", chain.len(), cot_result);
        println!("    CoT steps:");
        for step in chain {
            println!("      -> {}", step);
        }
        println!();
    }

    // 2. Computational capacity analysis
    println!("--- Computational Capacity ---\n");

    let model_layers = 32;
    println!("Model depth: {} layers\n", model_layers);

    let cot_lengths = [0, 5, 10, 20, 50, 100];
    println!(
        "{:>10} {:>15} {:>15} {:>15}",
        "CoT tokens", "Total layers", "Relative compute", "Max problem complexity"
    );
    println!("{}", "-".repeat(58));

    for &cot_len in &cot_lengths {
        let total_layers = model_layers * (cot_len + 1);
        let relative = total_layers as f64 / model_layers as f64;
        let max_complexity = ((total_layers as f64).ln() * 3.0) as usize;
        println!(
            "{:>10} {:>15} {:>14.1}x {:>15}",
            cot_len, total_layers, relative, max_complexity
        );
    }

    // 3. Accuracy simulation: with and without CoT
    println!("\n--- Accuracy: Direct vs CoT ---\n");

    // Simulate: model accuracy depends on problem complexity vs compute budget
    let problems_by_steps: Vec<(usize, &str)> = vec![
        (1, "Simple addition"),
        (2, "Two-step arithmetic"),
        (3, "Three-step logic"),
        (5, "Five-step reasoning"),
        (8, "Eight-step proof"),
        (12, "Twelve-step derivation"),
    ];

    let model_capacity = 3.0; // Can reliably do ~3 steps in one pass

    println!(
        "{:>30} {:>8} {:>12} {:>12}",
        "Problem", "Steps", "Direct", "With CoT"
    );
    println!("{}", "-".repeat(65));

    for (steps, desc) in &problems_by_steps {
        // Direct: accuracy drops exponentially with steps beyond capacity
        let direct_acc = sigmoid((model_capacity - *steps as f64) * 2.0) * 100.0;

        // CoT: each step is within capacity, compound accuracy
        let per_step_acc = 0.95; // 95% per step with CoT
        let cot_acc = per_step_acc.powi(*steps as i32) * 100.0;

        println!(
            "{:>30} {:>8} {:>11.1}% {:>11.1}%",
            desc, steps, direct_acc, cot_acc
        );
    }

    // 4. Latent variable perspective
    println!("\n--- Latent Variable Perspective ---\n");

    // The reasoning chain is a latent variable that, when marginalized out,
    // gives us P(answer | question)
    // With CoT, we condition on a specific reasoning chain

    let question = "What is 23 * 47?";
    let correct_answer = 1081.0;

    // Simulate different reasoning chains with different accuracies
    let chains = vec![
        ("23*47 = 23*40 + 23*7 = 920 + 161 = 1081", 1081.0, 0.7),
        ("23*47 = 23*50 - 23*3 = 1150 - 69 = 1081", 1081.0, 0.2),
        ("23*47 = 20*47 + 3*47 = 940 + 141 = 1081", 1081.0, 0.08),
        ("23*47 ≈ 1000 (rough estimate)", 1000.0, 0.02),
    ];

    println!("  Question: {}", question);
    println!("  Correct answer: {}\n", correct_answer);
    println!("  Possible reasoning chains (latent variables):");

    let mut total_correct_prob = 0.0;
    for (chain, result, prob) in &chains {
        let is_correct = (*result - correct_answer).abs() < 0.1;
        if is_correct {
            total_correct_prob += prob;
        }
        println!(
            "    P(chain)={:.2}: {} -> {} {}",
            prob,
            chain,
            result,
            if is_correct { "(correct)" } else { "(wrong)" }
        );
    }
    println!(
        "\n  P(correct answer) = sum of P(chain) where chain leads to correct answer"
    );
    println!("  P(1081) = {:.2}", total_correct_prob);
    println!(
        "\n  CoT selects the most likely chain, conditioning on it"
    );
    println!("  to make P(correct | chain) much higher.");

    // 5. Scaling of problem difficulty
    println!("\n--- Problem Complexity Scaling ---\n");

    // Multi-digit multiplication: difficulty scales with digit count
    println!("{:>12} {:>10} {:>15} {:>15}", "Digits", "Steps", "Direct acc", "CoT acc");
    println!("{}", "-".repeat(55));

    for digits in [1, 2, 3, 4, 5, 6] {
        let steps = digits * (digits + 1) / 2; // Roughly proportional
        let direct_acc = sigmoid((model_capacity - steps as f64) * 2.0) * 100.0;
        let cot_acc = 0.95_f64.powi(steps as i32) * 100.0;

        println!(
            "{:>12} {:>10} {:>14.1}% {:>14.1}%",
            digits, steps, direct_acc, cot_acc
        );
    }

    println!("\n  Direct computation fails quickly as difficulty increases.");
    println!("  CoT degrades gracefully because each step is independently reliable.");
}

fn try_direct_compute(expr: &str) -> String {
    // Simulate a model that can only handle simple single-operator expressions
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() == 3 {
        // Simple: a op b
        if let (Ok(a), Ok(b)) = (parts[0].parse::<f64>(), parts[2].parse::<f64>()) {
            let result = match parts[1] {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                _ => return "Cannot compute".to_string(),
            };
            return format!("{}", result);
        }
    }
    "Too complex for single step (would need CoT)".to_string()
}

fn chain_of_thought_compute(chain: &[&str]) -> String {
    // Execute each step of the reasoning chain
    let mut last_result = 0.0;
    for step in chain {
        // Parse "X op Y = Z" format
        let parts: Vec<&str> = step.split('=').collect();
        if parts.len() == 2 {
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                last_result = val;
            }
        }
    }
    format!("{}", last_result)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

---

## Key Takeaways

- Chain-of-thought tokens act as latent variables that expand the model's computational capacity beyond its fixed architecture depth, turning O(L) computation into O(L * T).
- Without CoT, models must compress multi-step reasoning into a single forward pass, which fails for problems whose complexity exceeds the model's layer depth.
- From a probabilistic perspective, CoT conditions the answer on an explicit reasoning chain, making complex answers easy to derive from well-structured intermediate results.
- CoT accuracy degrades gracefully with problem complexity (each step has independent high accuracy), while direct computation accuracy drops catastrophically beyond the model's single-pass capacity.
