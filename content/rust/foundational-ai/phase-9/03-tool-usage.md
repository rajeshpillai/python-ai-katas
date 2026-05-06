# Tool Usage

> Phase 9 — Reasoning Models | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

Language models are powerful general-purpose reasoners but have fundamental limitations: they cannot access real-time information, perform exact arithmetic reliably, execute code, or interact with external systems. Tool usage extends a language model's capabilities by allowing it to delegate specific tasks to specialized external tools. The model decides when to call a tool, what arguments to provide, interprets the results, and integrates them into its reasoning.

This is a paradigm shift from monolithic AI systems. Instead of training a single model to do everything, we train a model to be a skilled orchestrator that knows what it knows, what it does not know, and which tool can fill each gap. A calculator handles arithmetic perfectly. A search engine retrieves current information. A code interpreter runs programs. The model's role is to decompose problems, route sub-problems to appropriate tools, and synthesize the results into a coherent response.

Tool usage also introduces a new form of reasoning: the model must plan which tools to use, in what order, and how to handle errors or unexpected results. This is closer to how human experts work: a scientist does not compute integrals in their head; they use Mathematica. An engineer does not test hypotheses mentally; they write and run experiments. The ability to effectively use tools is itself a form of intelligence.

### Why naive approaches fail

Training a model to perform exact arithmetic internally requires enormous parameter counts for diminishing returns. A 100-billion-parameter model still makes errors on 6-digit multiplication that a simple calculator solves perfectly. The model's architecture is optimized for pattern matching and language understanding, not for exact symbolic computation.

Hard-coding tool calls (if the query contains numbers, always use a calculator) is too rigid. The model might encounter numbers in a context where calculation is not needed ("The year 1776 was important"). Conversely, it might need a calculator for a query that does not explicitly mention numbers ("How much would I save if the price drops by a third?"). The model must learn to recognize when tool usage is appropriate based on semantic understanding.

### Mental models

- **Specialist consultation**: The model is a general practitioner who knows when to refer a patient to a specialist. It recognizes the type of problem, selects the right specialist (tool), communicates the relevant information, and integrates the specialist's response into the overall treatment plan.
- **Function calling**: Tools are like library functions. The model writes the function call (with arguments), the runtime executes it, and the model uses the return value. The model is the programmer; the tools are the standard library.
- **Extended cognition**: Just as humans extend their cognitive abilities with pen and paper, calculators, and search engines, tool-using models extend their capabilities beyond what their neural networks alone can achieve.

### Visual explanations

```
  Tool usage flow:

  User: "What's the population of Tokyo times 1.5?"

  Model reasoning:
  1. I need the current population of Tokyo -> use SEARCH tool
  2. I need to multiply that number by 1.5 -> use CALCULATOR tool

  Model: [SEARCH("population of Tokyo 2024")]
  Tool:  -> "13.96 million"
  Model: [CALCULATOR("13960000 * 1.5")]
  Tool:  -> "20940000"
  Model: "The population of Tokyo is approximately 13.96 million.
          Multiplied by 1.5, that gives about 20.94 million."

  Tool routing decision tree:

  Query arrives
     |
     v
  Does it need current information? --YES--> SEARCH
     |NO
  Does it need exact computation?  --YES--> CALCULATOR
     |NO
  Does it need code execution?     --YES--> CODE_INTERPRETER
     |NO
  Does it need file access?        --YES--> FILE_READER
     |NO
  Handle with internal knowledge
```

---

## Hands-on Exploration

1. Implement a simple tool system with a calculator, a knowledge base, and a string processor.
2. Build a query router that decides which tool to use based on query content.
3. Show how tool usage improves accuracy compared to model-only answers.
4. Demonstrate multi-tool reasoning where the output of one tool feeds into another.

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    println!("=== Tool Usage ===\n");

    // Define our tool ecosystem
    let mut knowledge_base: HashMap<&str, &str> = HashMap::new();
    knowledge_base.insert("population_tokyo", "13960000");
    knowledge_base.insert("population_paris", "2161000");
    knowledge_base.insert("speed_of_light", "299792458");
    knowledge_base.insert("pi", "3.14159265358979");
    knowledge_base.insert("earth_radius_km", "6371");
    knowledge_base.insert("avogadro", "6.022e23");

    // Simulate queries that require different tool combinations
    let queries = vec![
        Query {
            text: "What is the population of Tokyo multiplied by 3?",
            steps: vec![
                Step::Search("population_tokyo"),
                Step::Calculate("* 3"),
            ],
        },
        Query {
            text: "What is the circumference of Earth in kilometers?",
            steps: vec![
                Step::Search("earth_radius_km"),
                Step::Search("pi"),
                Step::Calculate("circumference = 2 * pi * radius"),
            ],
        },
        Query {
            text: "How many times faster is light than sound (343 m/s)?",
            steps: vec![
                Step::Search("speed_of_light"),
                Step::Calculate("/ 343"),
            ],
        },
        Query {
            text: "What is sqrt(population of Paris)?",
            steps: vec![
                Step::Search("population_paris"),
                Step::Calculate("sqrt"),
            ],
        },
        Query {
            text: "Reverse the string 'reasoning models'",
            steps: vec![
                Step::StringOp("reverse", "reasoning models"),
            ],
        },
    ];

    println!("--- Executing Queries with Tool Usage ---\n");

    for query in &queries {
        println!("Query: {}", query.text);
        println!("  Plan: {} tool calls needed", query.steps.len());

        let mut context: Vec<(String, String)> = Vec::new();

        for (i, step) in query.steps.iter().enumerate() {
            match step {
                Step::Search(key) => {
                    let result = tool_search(&knowledge_base, key);
                    println!("  Step {}: SEARCH('{}') -> {}", i + 1, key, result);
                    context.push((key.to_string(), result));
                }
                Step::Calculate(operation) => {
                    let result = tool_calculate(&context, operation);
                    println!("  Step {}: CALC('{}') -> {}", i + 1, operation, result);
                    context.push(("calc_result".to_string(), result));
                }
                Step::StringOp(op, input) => {
                    let result = tool_string(op, input);
                    println!("  Step {}: STRING_{}('{}') -> {}", i + 1, op.to_uppercase(), input, result);
                    context.push(("string_result".to_string(), result));
                }
            }
        }

        let final_result = context.last().map(|(_, v)| v.as_str()).unwrap_or("N/A");
        println!("  Final answer: {}\n", final_result);
    }

    // 2. Comparison: with tools vs without tools
    println!("--- Accuracy: With Tools vs Without Tools ---\n");

    let test_cases: Vec<(&str, f64)> = vec![
        ("13960000 * 3", 41880000.0),
        ("2 * 3.14159 * 6371", 40030.17),
        ("299792458 / 343", 874028.30),
        ("sqrt(2161000)", 1470.10),
        ("23456 * 78901", 1850689456.0),
        ("1 / 7", 0.142857),
    ];

    println!(
        "{:<25} {:>15} {:>15} {:>10}",
        "Expression", "Tool result", "LLM approx*", "Error%"
    );
    println!("{}", "-".repeat(68));

    for (expr, expected) in &test_cases {
        let tool_result = basic_calculate(expr);
        // Simulate LLM approximation (introduces small errors)
        let llm_approx = simulate_llm_math(*expected);
        let tool_error = if *expected != 0.0 {
            ((tool_result - expected) / expected * 100.0).abs()
        } else {
            0.0
        };
        let llm_error = if *expected != 0.0 {
            ((llm_approx - expected) / expected * 100.0).abs()
        } else {
            0.0
        };

        println!(
            "{:<25} {:>15.2} {:>15.2} {:>4.1}% vs {:.1}%",
            expr, tool_result, llm_approx, tool_error, llm_error
        );
    }
    println!("  * Simulated LLM approximation with typical neural network errors\n");

    // 3. Tool routing decisions
    println!("--- Tool Routing Logic ---\n");

    let routing_examples = vec![
        ("What is 234 * 567?", "CALCULATOR", "Contains arithmetic"),
        ("When was the Eiffel Tower built?", "SEARCH", "Factual question"),
        ("Write a haiku about rust", "NONE (internal)", "Creative task"),
        ("What time is it in Tokyo?", "SEARCH", "Requires current data"),
        ("Sort this list: [5,2,8,1]", "CODE_INTERPRETER", "Algorithmic task"),
        ("Explain quantum entanglement", "NONE (internal)", "Knowledge explanation"),
        ("How far is Mars right now?", "SEARCH", "Dynamic real-time data"),
        ("Reverse 'hello world'", "STRING_TOOLS", "String manipulation"),
    ];

    println!("{:<40} {:<20} {}", "Query", "Route to", "Reason");
    println!("{}", "-".repeat(80));
    for (query, tool, reason) in &routing_examples {
        println!("{:<40} {:<20} {}", query, tool, reason);
    }

    // 4. Multi-tool chaining
    println!("\n--- Multi-Tool Chaining ---\n");

    println!("Complex query: 'If Tokyo's population grew by 5%% and Paris's by 10%%,");
    println!("                what would be their combined population?'\n");

    let tokyo_pop: f64 = 13960000.0;
    let paris_pop: f64 = 2161000.0;

    println!("  Step 1: SEARCH('population_tokyo') -> {}", tokyo_pop);
    println!("  Step 2: SEARCH('population_paris') -> {}", paris_pop);

    let tokyo_new = tokyo_pop * 1.05;
    println!("  Step 3: CALC('{} * 1.05') -> {:.0}", tokyo_pop, tokyo_new);

    let paris_new = paris_pop * 1.10;
    println!("  Step 4: CALC('{} * 1.10') -> {:.0}", paris_pop, paris_new);

    let combined = tokyo_new + paris_new;
    println!("  Step 5: CALC('{:.0} + {:.0}') -> {:.0}", tokyo_new, paris_new, combined);

    println!("\n  Final answer: {:.0} ({:.1} million)", combined, combined / 1_000_000.0);
    println!("  Total tool calls: 5 (2 searches + 3 calculations)");

    // 5. Error handling in tool usage
    println!("\n--- Error Handling ---\n");

    let error_scenarios = vec![
        ("Search for unknown fact", "SEARCH('population_atlantis')", "Not found -> model says 'I don't know'"),
        ("Division by zero", "CALC('100 / 0')", "Error -> model explains the issue"),
        ("Ambiguous query", "SEARCH('size of java')", "Multiple meanings -> model asks for clarification"),
        ("Tool timeout", "CODE('infinite_loop()')", "Timeout -> model reports failure gracefully"),
    ];

    for (scenario, call, handling) in &error_scenarios {
        println!("  Scenario: {}", scenario);
        println!("    Call: {}", call);
        println!("    Handling: {}\n", handling);
    }
}

enum Step<'a> {
    Search(&'a str),
    Calculate(&'a str),
    StringOp(&'a str, &'a str),
}

struct Query<'a> {
    text: &'a str,
    steps: Vec<Step<'a>>,
}

fn tool_search<'a>(kb: &'a HashMap<&str, &str>, key: &str) -> String {
    kb.get(key)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "NOT_FOUND".to_string())
}

fn tool_calculate(context: &[(String, String)], operation: &str) -> String {
    // Simple calculator that uses context values
    if operation.starts_with("* ") {
        if let Some((_, val)) = context.last() {
            if let Ok(v) = val.parse::<f64>() {
                let multiplier: f64 = operation[2..].parse().unwrap_or(1.0);
                return format!("{:.0}", v * multiplier);
            }
        }
    } else if operation.starts_with("/ ") {
        if let Some((_, val)) = context.last() {
            if let Ok(v) = val.parse::<f64>() {
                let divisor: f64 = operation[2..].parse().unwrap_or(1.0);
                return format!("{:.2}", v / divisor);
            }
        }
    } else if operation == "sqrt" {
        if let Some((_, val)) = context.last() {
            if let Ok(v) = val.parse::<f64>() {
                return format!("{:.2}", v.sqrt());
            }
        }
    } else if operation.contains("circumference") {
        // Special: 2 * pi * radius
        let pi_val: f64 = context
            .iter()
            .find(|(k, _)| k == "pi")
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(std::f64::consts::PI);
        let radius: f64 = context
            .iter()
            .find(|(k, _)| k == "earth_radius_km")
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(0.0);
        return format!("{:.2}", 2.0 * pi_val * radius);
    }
    "ERROR".to_string()
}

fn tool_string(operation: &str, input: &str) -> String {
    match operation {
        "reverse" => input.chars().rev().collect(),
        "uppercase" => input.to_uppercase(),
        "lowercase" => input.to_lowercase(),
        "length" => input.len().to_string(),
        _ => "UNKNOWN_OP".to_string(),
    }
}

fn basic_calculate(expr: &str) -> f64 {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() == 3 {
        let a: f64 = parts[0].parse().unwrap_or(0.0);
        let b: f64 = parts[2].parse().unwrap_or(0.0);
        return match parts[1] {
            "*" => a * b,
            "/" => if b != 0.0 { a / b } else { f64::NAN },
            "+" => a + b,
            "-" => a - b,
            _ => f64::NAN,
        };
    }
    if expr.starts_with("sqrt(") {
        let inner = &expr[5..expr.len() - 1];
        if let Ok(v) = inner.parse::<f64>() {
            return v.sqrt();
        }
    }
    f64::NAN
}

fn simulate_llm_math(expected: f64) -> f64 {
    // Simulate typical LLM math errors: small relative errors
    // that increase with magnitude
    let magnitude = expected.abs().max(1.0).log10();
    let error_rate = 0.001 * magnitude; // Error grows with number size
    let noise = (expected * 1.23456).sin() * error_rate;
    expected * (1.0 + noise)
}
```

---

## Key Takeaways

- Tool usage extends a language model beyond its internal capabilities by delegating specialized tasks (arithmetic, search, code execution) to purpose-built external systems.
- The model's key role shifts from being a universal solver to being an intelligent orchestrator that decomposes problems, routes sub-tasks to appropriate tools, and synthesizes results.
- Multi-tool chaining enables solving complex queries by piping the output of one tool into another, with the model managing the dataflow and reasoning at each junction.
- Effective tool usage requires the model to learn when tools are needed (routing), how to formulate tool calls (argument construction), and how to handle failures gracefully (error management).
