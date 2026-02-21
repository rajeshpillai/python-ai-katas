# Scratchpads

> Phase 9 — Reasoning Models | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

A scratchpad is the explicit working memory that a language model creates through its generated tokens. While chain-of-thought provides the general framework for intermediate reasoning, the scratchpad concept emphasizes a specific and critical function: using generated text as external storage for intermediate results that would otherwise be lost. When a model writes "Step 1: 23 * 40 = 920", the token "920" is now in the context window and can be referenced by subsequent tokens. Without this external storage, the model would need to hold this intermediate result entirely in its hidden state activations, which is unreliable and capacity-limited.

The scratchpad concept reveals a deep connection between language models and Turing machines. A Turing machine has a tape that serves as unbounded external memory. A language model generating tokens on a scratchpad is functionally similar: it reads from its context (tape), performs computation (forward pass), and writes results back (generated tokens). This suggests that a transformer with a sufficiently large context window and the ability to generate intermediate tokens is computationally more powerful than a transformer forced to answer in a single step.

Scratchpads are particularly important for tasks requiring state tracking, where the model must maintain and update multiple variables across many reasoning steps. Examples include following a set of instructions ("After step 3, Alice has 5 apples, Bob has 3"), simulating algorithms ("After iteration 4, the sorted prefix is [1,3,5,7]"), or mathematical proofs ("From lemma 1 and 2, we derive..."). In each case, writing down the intermediate state prevents information loss.

### Why naive approaches fail

Relying solely on the model's internal hidden state for working memory fails because the hidden state at each position is a fixed-size vector that must represent the entire relevant state of the computation. For simple tasks, this is sufficient. But for tasks with many intermediate variables (like simulating a sorting algorithm on 20 elements), the hidden state simply does not have enough dimensions to reliably track all variables simultaneously.

Attempting to skip intermediate steps ("Just compute the final answer") collapses all computation into a single forward pass. This is like asking a student to solve a system of equations without writing anything down: possible for simple cases but increasingly error-prone as complexity grows. The scratchpad externalizes memory, reducing the cognitive load at each step.

### Mental models

- **Whiteboard**: The scratchpad is like a whiteboard during a meeting. You write intermediate results so everyone (including your future self) can reference them. Erasing the whiteboard (losing context) forces you to recompute or guess.
- **Stack frames**: Like a call stack in programming, the scratchpad stores intermediate results from sub-computations that will be needed later. Each "stack frame" is a line of reasoning that can be referenced.
- **Turing machine tape**: Generated tokens extend the "tape" that the model reads from. Each token is a cell on the tape that can be read in future computation steps.

### Visual explanations

```
  Scratchpad as external memory:

  Problem: Sort [3, 1, 4, 1, 5]

  Without scratchpad (internal state only):
  Model sees: "Sort [3, 1, 4, 1, 5] ="
  Must compute answer in ONE forward pass
  Hidden state must somehow track all comparisons and swaps
  Result: Often wrong for longer lists

  With scratchpad:
  "Sort [3, 1, 4, 1, 5]
   Step 1: Find min in [3,1,4,1,5] -> 1 at index 1
   State: [1 | 3,4,1,5]
   Step 2: Find min in [3,4,1,5] -> 1 at index 2
   State: [1,1 | 3,4,5]
   Step 3: Find min in [3,4,5] -> 3 at index 0
   State: [1,1,3 | 4,5]
   Step 4: Find min in [4,5] -> 4 at index 0
   State: [1,1,3,4 | 5]
   Result: [1,1,3,4,5]"

  Each "State:" line is readable in subsequent attention!

  Memory capacity:
  Hidden state: ~1024 floats (fixed)
  Scratchpad: grows with each generated token
  After 100 tokens: effectively 100 * 1024 floats of accessible info
```

---

## Hands-on Exploration

1. Implement a state-tracking task and compare accuracy with and without a scratchpad.
2. Show how the scratchpad prevents information loss across many steps.
3. Demonstrate that scratchpad token count is a proxy for computational complexity.
4. Implement a simple sorting algorithm using scratchpad-style step-by-step execution.

---

## Live Code

```rust
fn main() {
    println!("=== Scratchpads ===\n");

    // 1. State tracking task: follow a sequence of operations on variables
    println!("--- State Tracking with Scratchpad ---\n");

    let instructions = vec![
        "Set x = 10",
        "Set y = 5",
        "Set x = x + y",        // x=15
        "Set z = x * 2",        // z=30
        "Set y = z - x",        // y=15
        "Set x = x + y + z",    // x=60
        "Set w = x / z",        // w=2
        "Set y = w * w + 1",    // y=5
    ];

    println!("Instructions:");
    for (i, inst) in instructions.iter().enumerate() {
        println!("  {}: {}", i + 1, inst);
    }

    // Execute with scratchpad (tracking state at each step)
    println!("\nExecution with scratchpad:");
    let mut state: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut scratchpad = Vec::new();

    for (i, inst) in instructions.iter().enumerate() {
        execute_instruction(inst, &mut state);

        let state_str: Vec<String> = {
            let mut entries: Vec<(&String, &f64)> = state.iter().collect();
            entries.sort_by_key(|(k, _)| k.to_string());
            entries.iter().map(|(k, v)| format!("{}={}", k, v)).collect()
        };
        let snapshot = format!("Step {}: {} -> {{{}}}", i + 1, inst, state_str.join(", "));
        scratchpad.push(snapshot.clone());
        println!("  {}", snapshot);
    }

    // Without scratchpad: simulate model that can only access recent state
    println!("\n--- Without Scratchpad: Information Loss ---\n");
    println!("If model can only 'remember' last 3 variable states:");

    let mut limited_state: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let window_size = 3;
    let mut history: Vec<(String, f64)> = Vec::new();

    for (i, inst) in instructions.iter().enumerate() {
        // The limited model might not recall all variable values
        let needed_vars = extract_vars(inst);
        let mut all_found = true;

        for var in &needed_vars {
            if !limited_state.contains_key(var.as_str()) {
                all_found = false;
            }
        }

        if all_found || i < 2 {
            execute_instruction(inst, &mut limited_state);
            // Add to history
            if let Some(var) = extract_assigned_var(inst) {
                if let Some(val) = limited_state.get(&var) {
                    history.push((var.clone(), *val));
                }
            }
        } else {
            println!("  Step {}: CANNOT COMPUTE - missing variable from earlier!", i + 1);
            continue;
        }

        // Simulate limited memory: only keep last N entries visible
        if history.len() > window_size {
            let removed = history.remove(0);
            // "Forget" old values (simulate limited context)
            limited_state.remove(&removed.0);
        }

        let state_str: Vec<String> = {
            let mut entries: Vec<(&String, &f64)> = limited_state.iter().collect();
            entries.sort_by_key(|(k, _)| k.to_string());
            entries.iter().map(|(k, v)| format!("{}={}", k, v)).collect()
        };
        println!("  Step {}: visible state = {{{}}}", i + 1, state_str.join(", "));
    }

    // 2. Sorting with scratchpad
    println!("\n--- Selection Sort via Scratchpad ---\n");

    let mut arr = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    println!("Input: {:?}\n", arr);

    let mut scratchpad_steps = Vec::new();
    let n = arr.len();

    for i in 0..n - 1 {
        let mut min_idx = i;
        for j in (i + 1)..n {
            if arr[j] < arr[min_idx] {
                min_idx = j;
            }
        }
        if min_idx != i {
            arr.swap(i, min_idx);
        }
        let step = format!(
            "Step {}: min in {:?} is {} at idx {}, swap -> {:?}",
            i + 1,
            &arr[i..],
            arr[i],
            min_idx,
            arr
        );
        scratchpad_steps.push(step.clone());
        println!("  {}", step);
    }
    println!("\nSorted: {:?}", arr);
    println!("Scratchpad tokens used: ~{} (each step ~20 tokens)", scratchpad_steps.len() * 20);

    // 3. Computational complexity on scratchpad
    println!("\n--- Scratchpad Size as Complexity Metric ---\n");

    let task_complexities = vec![
        ("2-digit addition", 1, "1 step"),
        ("3-digit multiplication", 6, "6 steps (partial products + addition)"),
        ("Sort 5 elements", 10, "10 comparison steps"),
        ("Sort 20 elements", 190, "190 comparison steps"),
        ("2x2 matrix multiply", 12, "12 multiply-add steps"),
        ("4x4 matrix multiply", 112, "112 multiply-add steps"),
        ("Simple logical deduction", 3, "3 inference steps"),
        ("10-step logical chain", 10, "10 inference steps"),
    ];

    println!(
        "{:<30} {:>8} {:>12} {}",
        "Task", "Steps", "~Tokens", "Description"
    );
    println!("{}", "-".repeat(80));

    for (task, steps, desc) in &task_complexities {
        let tokens = steps * 15; // Rough estimate of tokens per step
        println!("{:<30} {:>8} {:>12} {}", task, steps, tokens, desc);
    }

    // 4. Accuracy vs scratchpad depth
    println!("\n--- Accuracy vs Scratchpad Usage ---\n");

    let problem_depths = [1, 3, 5, 8, 12, 20];
    let per_step_reliability = 0.97; // Each scratchpad step is 97% correct

    println!(
        "{:>12} {:>15} {:>15} {:>15}",
        "Steps", "No scratchpad", "With scratchpad", "Improvement"
    );
    println!("{}", "-".repeat(60));

    for &depth in &problem_depths {
        // Without scratchpad: single-pass, accuracy drops steeply
        let no_scratch = sigmoid((3.0 - depth as f64) * 1.5) * 100.0;

        // With scratchpad: compound per-step reliability
        let with_scratch = per_step_reliability.powi(depth as i32) * 100.0;

        let improvement = with_scratch - no_scratch;
        println!(
            "{:>12} {:>14.1}% {:>14.1}% {:>+14.1}%",
            depth, no_scratch, with_scratch, improvement
        );
    }

    println!("\n  Scratchpads provide the biggest benefit for multi-step problems");
    println!("  where the model would otherwise have to compress all reasoning");
    println!("  into a single forward pass.");
}

fn execute_instruction(inst: &str, state: &mut std::collections::HashMap<String, f64>) {
    // Parse simple instructions like "Set x = 10" or "Set x = x + y"
    let parts: Vec<&str> = inst.split_whitespace().collect();
    if parts.len() < 4 || parts[0] != "Set" || parts[2] != "=" {
        return;
    }

    let var_name = parts[1].to_string();

    if parts.len() == 4 {
        // "Set x = 10"
        if let Ok(val) = parts[3].parse::<f64>() {
            state.insert(var_name, val);
        }
    } else if parts.len() == 6 {
        // "Set x = a + b"
        let a = resolve_value(parts[3], state);
        let b = resolve_value(parts[5], state);

        let result = match parts[4] {
            "+" => a + b,
            "-" => a - b,
            "*" => a * b,
            "/" => if b != 0.0 { a / b } else { 0.0 },
            _ => 0.0,
        };
        state.insert(var_name, result);
    } else if parts.len() == 8 {
        // "Set x = a + b + c"
        let a = resolve_value(parts[3], state);
        let b = resolve_value(parts[5], state);
        let c = resolve_value(parts[7], state);
        let result = match (parts[4], parts[6]) {
            ("+", "+") => a + b + c,
            ("*", "+") => a * b + c,
            ("+", "*") => a + b * c,
            _ => a + b + c,
        };
        state.insert(var_name, result);
    }
}

fn resolve_value(token: &str, state: &std::collections::HashMap<String, f64>) -> f64 {
    if let Ok(val) = token.parse::<f64>() {
        val
    } else {
        *state.get(token).unwrap_or(&0.0)
    }
}

fn extract_vars(inst: &str) -> Vec<String> {
    let parts: Vec<&str> = inst.split_whitespace().collect();
    let mut vars = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        if i < 3 { continue; } // Skip "Set x ="
        if part.parse::<f64>().is_err()
            && *part != "+" && *part != "-" && *part != "*" && *part != "/"
            && *part != "=" && *part != "Set"
        {
            vars.push(part.to_string());
        }
    }
    vars
}

fn extract_assigned_var(inst: &str) -> Option<String> {
    let parts: Vec<&str> = inst.split_whitespace().collect();
    if parts.len() >= 2 && parts[0] == "Set" {
        Some(parts[1].to_string())
    } else {
        None
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

---

## Key Takeaways

- Scratchpads use generated tokens as external working memory, allowing the model to store and reference intermediate results that would otherwise be lost in fixed-size hidden state activations.
- The scratchpad concept connects language models to Turing machines: generated tokens extend the "tape" of accessible memory, making the model computationally more powerful.
- State-tracking tasks benefit enormously from scratchpads because each step can read the current state from previous tokens rather than relying on the model's internal memory.
- Scratchpad token count is a meaningful proxy for computational complexity: harder problems require more intermediate steps, and each step adds both computation and memory to the model's budget.
