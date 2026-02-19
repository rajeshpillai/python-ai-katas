# Tool Usage

> Phase 9 — Reasoning Models | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

Language models are remarkably capable, but they have fundamental limitations. They cannot reliably perform precise arithmetic with large numbers, they cannot access real-time information, and their knowledge is frozen at their training cutoff date. Tool usage solves these problems by allowing the model to recognize when it needs external help, formulate a request to an appropriate tool (calculator, search engine, database), interpret the result, and incorporate it into its response.

This is a paradigm shift from "the model knows everything" to "the model knows what it does not know and knows how to find out." A model with tool access can answer "What is 7,847,293 * 3,291,847?" by calling a calculator rather than attempting mental arithmetic. It can answer "What was yesterday's stock price?" by querying an API rather than hallucinating a number. The model's role shifts from being a knowledge store to being an orchestrator that knows when and how to delegate.

Tool usage also dramatically extends what is possible within a single interaction. A model without tools is limited to what it can compute in its forward passes and retrieve from its weights. A model with tools can execute code, search the internet, query databases, generate images, and interact with external systems. This compositional capability means the model's effective ability is the union of its own capabilities and all its tools' capabilities.

### Why naive approaches fail

Training a model to be good at arithmetic by including more math in the training data has diminishing returns. The model might memorize common multiplications but will fail on unusual large numbers. The fundamental issue is that neural networks perform approximate, parallel computation -- they are not designed for the exact, sequential operations that arithmetic requires. No amount of training data fixes this architectural mismatch.

Trying to expand the model's knowledge by training on more recent data is expensive, slow, and always out of date. Even if you retrain monthly, the model still cannot answer questions about yesterday. And increasing model size to store more knowledge hits diminishing returns -- doubling parameters does not double the amount of retrievable knowledge.

### Mental models

- **Specialist referrals in medicine:** A general practitioner does not perform brain surgery. They diagnose the problem, identify the right specialist, write a referral, and interpret the specialist's report. The model is the GP; tools are the specialists.
- **Using a phone as an extension of your brain:** You do not memorize every phone number or every fact. You know how to look things up on your phone. This "knowing how to find out" is more powerful than "knowing the answer."
- **APIs in software engineering:** A web application does not implement its own payment processing, email sending, or map rendering. It calls specialized APIs. Similarly, a language model calls specialized tools.

### Visual explanations

```
  MODEL WITHOUT TOOLS:

  User: "What is 7847293 * 3291847?"
         │
         ▼
  ┌─────────────────┐
  │  Language Model  │──> "Approximately 25.8 trillion"
  │  (guessing)     │    (WRONG: actual is 25,830,590,697,771)
  └─────────────────┘

  MODEL WITH TOOLS:

  User: "What is 7847293 * 3291847?"
         │
         ▼
  ┌─────────────────┐     ┌──────────────┐
  │  Language Model  │────>│  Calculator  │
  │  (recognizes it │     │  7847293 *   │
  │   needs a tool) │     │  3291847     │
  └────────┬────────┘     └──────┬───────┘
           │                     │
           │  ┌──────────────────┘
           │  │ Result: 25830590697771
           ▼  ▼
  ┌─────────────────┐
  │  Language Model  │──> "7,847,293 * 3,291,847 = 25,830,590,697,771"
  │  (formats answer)│    (CORRECT)
  └─────────────────┘

  TOOL SELECTION PROCESS:

  Query ──> Model analyzes ──> Which tool?
                │
                ├──> Math question?     ──> Calculator
                ├──> Factual question?  ──> Search engine
                ├──> Code execution?    ──> Code interpreter
                └──> Within knowledge?  ──> Direct answer
```

---

## Hands-on Exploration

1. Simulate a language model attempting arithmetic with and without a calculator tool
2. Implement a tool-selection mechanism that routes queries to appropriate tools
3. Build a simple calculator tool, a lookup tool, and a string-processing tool
4. Show accuracy improvement when the model uses tools versus answering directly
5. Demonstrate tool chaining: using the output of one tool as input to another

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Simulated LLM: good at language, bad at precise computation ---
def llm_direct_answer(question_type, true_answer, difficulty):
    """Simulate LLM trying to answer without tools.
    Higher difficulty = more likely to be wrong."""
    if question_type == "math":
        # LLMs are bad at precise math
        error_rate = 1 - np.exp(-difficulty * 0.5)
        if np.random.random() < error_rate:
            noise = int(true_answer * np.random.uniform(-0.1, 0.1))
            return true_answer + noise, False
        return true_answer, True
    elif question_type == "lookup":
        # LLMs hallucinate facts with probability based on obscurity
        if np.random.random() < difficulty * 0.15:
            return "hallucinated_answer", False
        return true_answer, True
    return true_answer, True

# --- Tools ---
def calculator(expression):
    """Precise arithmetic tool."""
    a, op, b = expression
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
           "*": lambda x, y: x * y, "/": lambda x, y: x / y if y != 0 else float("inf")}
    return ops[op](a, b)

def lookup_database(key, database):
    """Knowledge retrieval tool."""
    return database.get(key, "NOT_FOUND")

def string_tool(text, operation):
    """Text processing tool."""
    if operation == "length":
        return len(text)
    elif operation == "reverse":
        return text[::-1]
    elif operation == "upper":
        return text.upper()

# --- Tool router: decides which tool to use ---
def route_to_tool(query_type, query, database=None):
    """Model decides whether to use a tool and which one."""
    if query_type == "math":
        return "calculator", calculator(query)
    elif query_type == "lookup":
        return "database", lookup_database(query, database)
    elif query_type == "string":
        return "string_tool", string_tool(query[0], query[1])
    else:
        return "direct", None

print("=" * 60)
print("TOOL USAGE: Extending Model Capabilities")
print("=" * 60)

# --- Math problems: with and without calculator ---
print("\n--- Math Problems: Direct vs Calculator ---\n")
math_problems = [
    (347, "+", 892),
    (47, "*", 23),
    (7847, "*", 3291),
    (999999, "*", 888888),
    (123456789, "+", 987654321),
]

print(f"{'Problem':<28s} {'Direct':>15s} {'Calculator':>15s} {'Correct':>12s}")
print("-" * 75)

direct_correct = 0
tool_correct = 0
for a, op, b in math_problems:
    true_answer = calculator((a, op, b))
    problem_str = f"{a} {op} {b}"
    difficulty = len(str(abs(int(true_answer)))) / 3

    direct, d_ok = llm_direct_answer("math", true_answer, difficulty)
    tool_name, tool_ans = route_to_tool("math", (a, op, b))

    d_mark = "ok" if d_ok else "WRONG"
    t_mark = "ok" if tool_ans == true_answer else "WRONG"
    direct_correct += d_ok
    tool_correct += (tool_ans == true_answer)

    print(f"{problem_str:<28s} {direct:>15,} ({d_mark:>5s}) "
          f"{tool_ans:>15,} ({t_mark:>5s}) {true_answer:>12,}")

print(f"\nDirect accuracy: {direct_correct}/{len(math_problems)}  "
      f"Tool accuracy: {tool_correct}/{len(math_problems)}")

# --- Knowledge lookup ---
print(f"\n{'=' * 60}")
print("KNOWLEDGE LOOKUP: Direct vs Database Tool")
print("=" * 60)
knowledge_db = {
    "earth_mass_kg": 5.972e24,
    "speed_of_light": 299792458,
    "python_release": "1991",
    "pi_digits_100": "3.14159265358979323846...",
    "moon_distance_km": 384400,
}
queries = [
    ("earth_mass_kg", 5.972e24, 0.3),
    ("speed_of_light", 299792458, 0.2),
    ("python_release", "1991", 0.5),
    ("moon_distance_km", 384400, 0.4),
]
print(f"\n{'Query':<20s} {'Direct':>20s} {'DB Lookup':>20s} {'Correct':>12s}")
print("-" * 76)
for key, true_val, difficulty in queries:
    direct, d_ok = llm_direct_answer("lookup", true_val, difficulty)
    _, tool_ans = route_to_tool("lookup", key, knowledge_db)
    d_sym = "ok" if d_ok else "WRONG"
    t_sym = "ok" if tool_ans == true_val else "WRONG"
    print(f"{key:<20s} {str(direct):>20s} ({d_sym:>5s}) "
          f"{str(tool_ans):>20s}")

# --- Tool chaining ---
print(f"\n{'=' * 60}")
print("TOOL CHAINING: Composing Multiple Tools")
print("=" * 60)
print("\nProblem: 'Compute (347 * 89) + (156 * 43), then report digit count'")
print("\nStep-by-step with tools:")
step1 = calculator((347, "*", 89))
print(f"  1. calculator(347 * 89)   = {step1}")
step2 = calculator((156, "*", 43))
print(f"  2. calculator(156 * 43)   = {step2}")
step3 = calculator((step1, "+", step2))
print(f"  3. calculator({step1} + {step2}) = {step3}")
step4 = string_tool(str(int(step3)), "length")
print(f"  4. string_tool('{int(step3)}', length) = {step4} digits")

# --- Large-scale accuracy comparison ---
print(f"\n{'=' * 60}")
print("1000-TRIAL ACCURACY COMPARISON")
print("=" * 60)
results = {"direct_math": 0, "tool_math": 0,
           "direct_lookup": 0, "tool_lookup": 0}

for _ in range(1000):
    # Random math problem
    a = np.random.randint(100, 100000)
    b = np.random.randint(100, 100000)
    true_ans = a * b
    difficulty = len(str(true_ans)) / 3
    _, d_ok = llm_direct_answer("math", true_ans, difficulty)
    t_ans = calculator((a, "*", b))
    results["direct_math"] += d_ok
    results["tool_math"] += (t_ans == true_ans)

    # Random lookup
    key = list(knowledge_db.keys())[np.random.randint(len(knowledge_db))]
    true_val = knowledge_db[key]
    _, d_ok = llm_direct_answer("lookup", true_val, 0.4)
    _, t_ans = route_to_tool("lookup", key, knowledge_db)
    results["direct_lookup"] += d_ok
    results["tool_lookup"] += (t_ans == true_val)

print(f"\n  {'Task':<20s} {'Direct':>10s} {'With Tool':>10s}")
print("  " + "-" * 42)
for task in ["math", "lookup"]:
    d = results[f"direct_{task}"] / 1000
    t = results[f"tool_{task}"] / 1000
    d_bar = "#" * int(d * 25)
    t_bar = "#" * int(t * 25)
    print(f"  {task:<20s} {d:>9.1%} |{d_bar}")
    print(f"  {'(with tool)':<20s} {t:>9.1%} |{t_bar}")
```

---

## Key Takeaways

- **Tools compensate for architectural limitations.** Neural networks are approximate computers; tools like calculators provide exact computation where needed.
- **Tool selection is itself a learned skill.** The model must recognize when its own capabilities are insufficient and identify the right tool for the job -- this meta-cognitive ability is critical.
- **Tool chaining enables complex workflows.** By using the output of one tool as input to another, models can solve multi-step problems that no single tool or model could handle alone.
- **Tools keep knowledge current.** Instead of retraining to update knowledge, a model with search and database tools can access real-time information.
- **The model becomes an orchestrator.** With tool access, the model's role shifts from "answering questions from memory" to "coordinating resources to solve problems" -- a much more powerful paradigm.
