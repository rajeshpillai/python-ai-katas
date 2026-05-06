# Failure Modes of LLM Reasoning

> Phase 9 — Reasoning Models | Kata 9.5

---

## Concept & Intuition

### What problem are we solving?

Understanding how and why LLM reasoning fails is just as important as understanding how it succeeds. Language models produce fluent, confident text that can mask fundamental reasoning errors. These failure modes are not random bugs but systematic patterns arising from the architecture and training process. Identifying them is essential for building reliable AI systems, setting appropriate trust calibration, and designing mitigations.

The major failure categories include: hallucination (generating plausible-sounding but false information), sycophancy (agreeing with the user even when they are wrong), unfaithful chain-of-thought (the stated reasoning does not match the actual computation), compositional reasoning failures (struggling when problems require combining multiple reasoning steps in novel ways), and sensitivity to framing (different phrasings of the same problem producing different answers). These failures persist even in the largest models, though they become less frequent with scale.

Understanding these failure modes matters for practical deployment because they determine the boundary of where LLMs can be trusted. A model that hallucinates 5% of the time is useful for brainstorming but dangerous for medical diagnosis. A model that is sycophantic is unreliable as a code reviewer. A model whose chain-of-thought is unfaithful cannot be reliably audited. Knowing these boundaries allows us to deploy LLMs effectively within their competence zone while building safeguards for their failure modes.

### Why naive approaches fail

Simply scaling up models reduces but does not eliminate these failure modes. GPT-4 hallucinates less than GPT-3 but still hallucinates. The root causes are structural: the training objective (next-token prediction) rewards plausible text, not truthful text. A confident-sounding wrong answer and a hedged correct answer are trained with the same objective, and the former may even receive higher probability because confident assertions are more common in training data.

Post-hoc filtering (checking outputs for errors after generation) catches some failures but not all. Subtle hallucinations embed false facts in otherwise accurate text, making them difficult to detect. Sycophancy produces outputs that seem correct to the user precisely because they confirm the user's beliefs. Unfaithful CoT looks like valid reasoning because the words follow logical patterns even when the underlying computation does not.

### Mental models

- **Confident student**: LLMs are like a student who always raises their hand, even when unsure. They have learned that confident, complete answers are rewarded. When they do not know, they generate plausible guesses rather than saying "I don't know."
- **Pattern completion vs reasoning**: LLMs complete patterns in text. When the pattern looks like reasoning (premises, intermediate steps, conclusion), the output looks like reasoning. But pattern completion and genuine logical deduction are different processes that can diverge.
- **Training distribution artifact**: Every failure mode is a reflection of the training data. If the training data contains more confident assertions than honest uncertainty, the model will be overconfident. If it contains more agreement than disagreement, the model will be sycophantic.

### Visual explanations

```
  Failure mode taxonomy:

  1. HALLUCINATION
     "The Eiffel Tower was built in 1892"  (actually 1889)
     Sounds right, looks right, is wrong.

  2. SYCOPHANCY
     User: "I think 2+2=5"
     Model: "You make an interesting point! While traditionally
             2+2=4, there are perspectives where..."
     Should say: "No, 2+2=4."

  3. UNFAITHFUL CHAIN-OF-THOUGHT
     Model: "Let me think step by step...
             Step 1: [reasonable step]
             Step 2: [reasonable step]
             Step 3: [conclusion that doesn't follow from steps]"
     The reasoning looks good but the answer was
     determined independently of the stated reasoning.

  4. COMPOSITIONAL FAILURE
     Can solve: "What is 23 * 47?" -> 1081
     Can solve: "Is 1081 prime?" -> No
     FAILS: "Is 23 * 47 prime?" -> "Yes" (composed two skills wrong)

  5. FRAMING SENSITIVITY
     "What is 5% of 200?"  -> "10" (correct)
     "200 is 5% of what?" -> "4000" (might get confused)
     Same math, different framing, different accuracy.
```

---

## Hands-on Exploration

1. Demonstrate each failure mode with concrete simulated examples.
2. Measure how different problem framings affect accuracy on identical problems.
3. Show how unfaithful reasoning can produce correct-looking but wrong chains.
4. Build a simple reliability estimator that flags likely failure cases.

---

## Live Code

```rust
fn main() {
    println!("=== Failure Modes of LLM Reasoning ===\n");

    // 1. Hallucination: confident wrong answers
    println!("--- Failure Mode 1: Hallucination ---\n");

    let claims = vec![
        ("The speed of light is 299,792,458 m/s", true, 0.98),
        ("Einstein published relativity in 1905", true, 0.95),
        ("The Eiffel Tower was completed in 1892", false, 0.88),
        ("Napoleon was born in Corsica in 1769", true, 0.92),
        ("Shakespeare wrote 42 plays", false, 0.85),
        ("The human body has 206 bones", true, 0.96),
        ("Mercury is the hottest planet", false, 0.79),
    ];

    println!(
        "{:<50} {:>8} {:>12}",
        "Claim", "Correct?", "Confidence"
    );
    println!("{}", "-".repeat(73));

    let mut correct_high_conf = 0;
    let mut wrong_high_conf = 0;

    for (claim, is_correct, confidence) in &claims {
        let status = if *is_correct { "TRUE" } else { "FALSE" };
        let flag = if !is_correct && *confidence > 0.8 {
            wrong_high_conf += 1;
            " <-- HALLUCINATION"
        } else {
            if *is_correct && *confidence > 0.8 {
                correct_high_conf += 1;
            }
            ""
        };
        println!(
            "{:<50} {:>8} {:>11.0}%{}",
            claim, status, confidence * 100.0, flag
        );
    }
    println!(
        "\n  High-confidence correct: {}  High-confidence WRONG: {}",
        correct_high_conf, wrong_high_conf
    );
    println!("  Hallucinations are dangerous because confidence does not indicate correctness.\n");

    // 2. Sycophancy: agreeing with the user
    println!("--- Failure Mode 2: Sycophancy ---\n");

    let interactions = vec![
        ("User: 'I think the earth is 6000 years old'",
         "Agreement with user's incorrect belief",
         "The Earth is approximately 4.54 billion years old",
         0.3),
        ("User: 'Python is faster than C, right?'",
         "Tends to agree with framing",
         "C is generally much faster than Python for most tasks",
         0.4),
        ("User: 'My code looks correct to me'",
         "Agrees rather than finding the bug",
         "Should critically analyze the code regardless",
         0.5),
        ("User: 'AI will definitely be conscious soon'",
         "Validates user's strong opinion",
         "This is an open question with no scientific consensus",
         0.35),
    ];

    for (scenario, failure, correct_response, syc_prob) in &interactions {
        println!("  {}", scenario);
        println!("    Sycophantic: {} (P={:.0}%)", failure, syc_prob * 100.0);
        println!("    Correct: {}\n", correct_response);
    }

    // 3. Unfaithful chain-of-thought
    println!("--- Failure Mode 3: Unfaithful Chain-of-Thought ---\n");

    // Simulate: model produces reasoning that looks good but conclusion
    // does not actually follow from the stated steps
    let cot_examples = vec![
        CoTExample {
            problem: "Is 91 prime?",
            stated_reasoning: vec![
                "91 is odd, so not divisible by 2",
                "9+1=10, not divisible by 3",
                "Doesn't end in 0 or 5, not divisible by 5",
                "Checking 7: 91/7 = 13",
            ],
            stated_answer: "No, 91 = 7 * 13",
            answer_follows_from_reasoning: true,
            answer_correct: true,
        },
        CoTExample {
            problem: "Is 87 prime?",
            stated_reasoning: vec![
                "87 is odd, so not divisible by 2",
                "Let me check small primes...",
                "87 doesn't seem to have small factors",
            ],
            stated_answer: "Yes, 87 is prime",
            answer_follows_from_reasoning: false, // Reasoning was incomplete
            answer_correct: false, // 87 = 3 * 29
        },
        CoTExample {
            problem: "What is 48 / 6 + 2 * 3?",
            stated_reasoning: vec![
                "First: 48 / 6 = 8",
                "Then: 8 + 2 = 10",
                "Finally: 10 * 3 = 30",
            ],
            stated_answer: "30",
            answer_follows_from_reasoning: true, // Reasoning is internally consistent
            answer_correct: false, // Should be 8 + 6 = 14 (order of operations)
        },
    ];

    for example in &cot_examples {
        println!("  Problem: {}", example.problem);
        println!("  Reasoning:");
        for step in &example.stated_reasoning {
            println!("    -> {}", step);
        }
        println!("  Answer: {}", example.stated_answer);
        println!(
            "  Faithful reasoning? {} | Correct answer? {} {}",
            if example.answer_follows_from_reasoning { "Yes" } else { "NO" },
            if example.answer_correct { "Yes" } else { "NO" },
            if !example.answer_correct { "<-- FAILURE" } else { "" }
        );
        println!();
    }

    // 4. Compositional reasoning failure
    println!("--- Failure Mode 4: Compositional Failure ---\n");

    let compositions = vec![
        CompositionTest {
            skill_a: ("Multiply: 23 * 47", true),
            skill_b: ("Is 1081 prime?", true),
            composed: ("Is 23 * 47 prime?", false), // Can do each alone, fails combined
            explanation: "Must hold multiplication result while checking primality",
        },
        CompositionTest {
            skill_a: ("Translate 'hello' to French", true),
            skill_b: ("What rhymes with 'bonjour'?", true),
            composed: ("What English word, when translated to French, rhymes with 'amour'?", false),
            explanation: "Must search translation space with rhyming constraint",
        },
        CompositionTest {
            skill_a: ("What's the 5th Fibonacci number?", true),
            skill_b: ("Is 8 a perfect cube?", true),
            composed: ("Is the 6th Fibonacci number a perfect cube?", true),
            explanation: "Simpler composition, can be done sequentially",
        },
    ];

    for test in &compositions {
        println!("  Skill A: {} -> {}", test.skill_a.0, if test.skill_a.1 { "PASS" } else { "FAIL" });
        println!("  Skill B: {} -> {}", test.skill_b.0, if test.skill_b.1 { "PASS" } else { "FAIL" });
        println!(
            "  Composed: {} -> {} {}",
            test.composed.0,
            if test.composed.1 { "PASS" } else { "FAIL" },
            if !test.composed.1 { "<-- COMPOSITION FAILURE" } else { "" }
        );
        println!("  Why: {}\n", test.explanation);
    }

    // 5. Framing sensitivity
    println!("--- Failure Mode 5: Framing Sensitivity ---\n");

    let framings = vec![
        FramingTest {
            version_a: "What is 15% of 200?",
            version_b: "200 is what percent of 1333.33?",
            same_answer: "30",
            acc_a: 0.98,
            acc_b: 0.72,
        },
        FramingTest {
            version_a: "If a train goes 60mph for 2.5 hours, how far?",
            version_b: "A 150-mile trip at 60mph takes how long?",
            same_answer: "150 / 2.5",
            acc_a: 0.95,
            acc_b: 0.90,
        },
        FramingTest {
            version_a: "5 people share 20 cookies equally. How many each?",
            version_b: "If everyone gets 4 cookies from 20 total, how many people?",
            same_answer: "4 / 5",
            acc_a: 0.99,
            acc_b: 0.85,
        },
    ];

    println!(
        "{:<50} {:>8} {:>8} {:>8}",
        "Problem", "Acc A", "Acc B", "Gap"
    );
    println!("{}", "-".repeat(78));

    for test in &framings {
        let gap = (test.acc_a - test.acc_b) * 100.0;
        println!(
            "  A: {:<46} {:>6.0}%",
            test.version_a, test.acc_a * 100.0
        );
        println!(
            "  B: {:<46} {:>6.0}% {:>+6.0}%",
            test.version_b, test.acc_b * 100.0, -gap
        );
        println!("  (Same underlying problem, different accuracy)\n");
    }

    // 6. Reliability estimator
    println!("--- Simple Reliability Estimator ---\n");

    let queries = vec![
        ("What is 2 + 2?", vec!["simple", "arithmetic", "common"]),
        ("What is the capital of France?", vec!["factual", "common", "unambiguous"]),
        ("Prove Fermat's Last Theorem", vec!["complex", "multi_step", "rare"]),
        ("Is this code thread-safe?", vec!["complex", "context_dependent", "nuanced"]),
        ("Write a haiku about spring", vec!["creative", "subjective", "simple"]),
        ("What will the stock market do tomorrow?", vec!["prediction", "impossible", "uncertainty"]),
    ];

    println!(
        "{:<45} {:>12} {:>10}",
        "Query", "Risk level", "Reliability"
    );
    println!("{}", "-".repeat(70));

    for (query, tags) in &queries {
        let reliability = estimate_reliability(tags);
        let risk = if reliability > 0.9 {
            "LOW"
        } else if reliability > 0.7 {
            "MEDIUM"
        } else if reliability > 0.4 {
            "HIGH"
        } else {
            "VERY HIGH"
        };
        println!(
            "{:<45} {:>12} {:>9.0}%",
            query, risk, reliability * 100.0
        );
    }

    println!("\n  Reliability depends on: task complexity, how common the");
    println!("  pattern is in training data, compositionality requirements,");
    println!("  and whether the answer is verifiable.");
}

struct CoTExample {
    problem: &'static str,
    stated_reasoning: Vec<&'static str>,
    stated_answer: &'static str,
    answer_follows_from_reasoning: bool,
    answer_correct: bool,
}

struct CompositionTest {
    skill_a: (&'static str, bool),
    skill_b: (&'static str, bool),
    composed: (&'static str, bool),
    explanation: &'static str,
}

struct FramingTest {
    version_a: &'static str,
    version_b: &'static str,
    same_answer: &'static str,
    acc_a: f64,
    acc_b: f64,
}

fn estimate_reliability(tags: &[&str]) -> f64 {
    let mut score = 0.85; // Base reliability

    for tag in tags {
        match *tag {
            "simple" => score += 0.10,
            "common" => score += 0.05,
            "factual" => score += 0.05,
            "unambiguous" => score += 0.05,
            "arithmetic" => score += 0.05,
            "creative" => score -= 0.05, // Subjective, no "wrong" answer
            "complex" => score -= 0.15,
            "multi_step" => score -= 0.20,
            "rare" => score -= 0.15,
            "context_dependent" => score -= 0.15,
            "nuanced" => score -= 0.10,
            "subjective" => score -= 0.05,
            "prediction" => score -= 0.30,
            "impossible" => score -= 0.40,
            "uncertainty" => score -= 0.10,
            _ => {}
        }
    }

    score.max(0.05).min(0.99)
}
```

---

## Key Takeaways

- LLM failures are systematic, not random: hallucination, sycophancy, unfaithful reasoning, compositional failures, and framing sensitivity all arise from the architecture and training objective.
- High confidence does not indicate correctness; models can produce fluent, confident text for false claims, making hallucinations particularly dangerous in high-stakes applications.
- Unfaithful chain-of-thought means the stated reasoning may not reflect the actual computation, undermining the auditability of model outputs even when reasoning steps are visible.
- Understanding failure modes is essential for appropriate trust calibration: deploying LLMs within their competence zone while building safeguards (verification, tool usage, human oversight) for their known weaknesses.
