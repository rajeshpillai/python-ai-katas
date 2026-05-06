# Planning vs Prediction

> Phase 9 — Reasoning Models | Kata 9.4

---

## Concept & Intuition

### What problem are we solving?

Standard language models are fundamentally predictive: they generate the most likely next token given the context. This works remarkably well for many tasks, but it has a critical limitation. Prediction is a forward-only process: the model commits to each token without considering what comes next. It cannot look ahead to see if the current choice leads to a dead end. Planning, in contrast, involves considering future states, evaluating multiple possible paths, and choosing actions that lead to desirable outcomes several steps ahead.

The distinction becomes clear in complex reasoning tasks. A predictive model solving a math problem might start a proof strategy that seems promising locally but hits a dead end five steps later. A planning model would evaluate multiple strategies before committing, potentially choosing a less obvious first step that leads to a cleaner solution. Similarly, in code generation, a predictive model might write a function signature that makes the implementation unnecessarily complex, while a planning model would consider the full implementation before choosing the signature.

This tension between prediction and planning is one of the central challenges in making language models truly capable reasoners. Techniques like tree-of-thought (exploring multiple reasoning paths), self-evaluation (the model critiques its own outputs), and iterative refinement (generating multiple drafts) all attempt to add planning capabilities on top of a fundamentally predictive architecture.

### Why naive approaches fail

Pure prediction fails on problems where the optimal path is not locally obvious. In chess, the best move is often a quiet positional move rather than an aggressive but obvious attack. Predictive models tend to choose "obvious" moves that pattern-match to common training examples rather than deeply analyzed positions.

Simple beam search adds some look-ahead but only at the token level, not at the reasoning level. A beam of width 5 explores 5 token sequences, but all 5 might represent the same flawed reasoning strategy with minor wording variations. True planning requires evaluating different reasoning strategies, not just different phrasings of the same strategy.

Generating multiple complete answers and selecting the best one (best-of-N sampling) is a brute-force approximation of planning. It works but is computationally expensive and does not provide the model with the ability to course-correct mid-generation. It is planning at the coarsest granularity.

### Mental models

- **GPS vs driving**: Prediction is like driving by following the road ahead of you. Planning is like using GPS to see the whole route, including traffic and detours, before choosing your path.
- **Chess thinking**: A beginner (prediction) plays the first good-looking move they see. A grandmaster (planning) considers multiple candidate moves, evaluates each several steps deep, and then chooses.
- **Writing a paper**: Prediction writes sentence by sentence, hoping it flows. Planning outlines the full argument first, ensures logical consistency, and then fills in the prose.

### Visual explanations

```
  Prediction (greedy, forward-only):

  Start -> A -> B -> C -> DEAD END!
  (Cannot backtrack, committed to A)

  Planning (tree search):

  Start -> A -> B -> C -> DEAD END
        -> A -> D -> E -> SUCCESS (score: 7)
        -> F -> G -> H -> SUCCESS (score: 9)  <-- BEST!

  Chooses F as first step because it leads to best outcome.

  Tree-of-thought reasoning:

  Question: "How to arrange 5 items optimally?"

  Thought 1: "Sort by size..."
     -> Evaluate: leads to partial solution, score 6/10
  Thought 2: "Group by category..."
     -> Evaluate: leads to better solution, score 8/10
  Thought 3: "Consider constraints first..."
     -> Evaluate: leads to optimal solution, score 10/10

  Planning selects Thought 3. Prediction would pick Thought 1
  (most common/obvious approach in training data).
```

---

## Hands-on Exploration

1. Implement a simple planning problem where greedy (predictive) approach fails.
2. Compare greedy path selection with look-ahead planning.
3. Implement tree-of-thought style evaluation of multiple reasoning paths.
4. Demonstrate that planning overhead is justified by improved solution quality.

---

## Live Code

```rust
fn main() {
    println!("=== Planning vs Prediction ===\n");

    // 1. Grid navigation: find path from (0,0) to (N-1,N-1)
    // Greedy goes toward the goal but hits walls
    // Planning considers multiple paths

    let grid = vec![
        vec![0, 0, 0, 0, 0, 1, 0],
        vec![0, 1, 1, 1, 0, 1, 0],
        vec![0, 0, 0, 1, 0, 0, 0],
        vec![1, 1, 0, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 0, 1, 0],
        vec![0, 1, 1, 1, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 1, 0],
    ];

    let start = (0, 0);
    let goal = (6, 6);

    println!("--- Grid Navigation ---\n");
    println!("Grid (0=open, 1=wall, S=start, G=goal):");
    print_grid_with_path(&grid, &[], start, goal);

    // Greedy approach: always move toward goal
    let greedy_path = greedy_navigate(&grid, start, goal);
    println!("Greedy path (always move toward goal):");
    if greedy_path.is_empty() {
        println!("  FAILED! Greedy got stuck.\n");
    } else {
        print_grid_with_path(&grid, &greedy_path, start, goal);
        println!("  Length: {}\n", greedy_path.len());
    }

    // Planning approach: BFS finds optimal path
    let planned_path = bfs_navigate(&grid, start, goal);
    println!("Planned path (BFS, considers all options):");
    if planned_path.is_empty() {
        println!("  No path exists.\n");
    } else {
        print_grid_with_path(&grid, &planned_path, start, goal);
        println!("  Length: {}\n", planned_path.len());
    }

    // 2. Decision-making with delayed rewards
    println!("--- Delayed Reward Problem ---\n");
    println!("Choose actions over 5 steps. Each action has immediate and future value.\n");

    let action_tree = vec![
        // Step 0: two choices
        vec![
            Action { name: "A (obvious)", immediate: 10.0, future: vec![3.0, 2.0, 1.0, 0.0] },
            Action { name: "B (subtle)",  immediate: 2.0,  future: vec![5.0, 8.0, 12.0, 15.0] },
        ],
    ];

    // Predictive: choose highest immediate reward
    let pred_choice = &action_tree[0][0]; // Action A
    let pred_total: f64 = pred_choice.immediate + pred_choice.future.iter().sum::<f64>();

    let plan_choice = &action_tree[0][1]; // Action B
    let plan_total: f64 = plan_choice.immediate + plan_choice.future.iter().sum::<f64>();

    println!("  {} -> immediate: {}, future: {:?}, total: {}",
        pred_choice.name, pred_choice.immediate, pred_choice.future, pred_total);
    println!("  {} -> immediate: {}, future: {:?}, total: {}",
        plan_choice.name, plan_choice.immediate, plan_choice.future, plan_total);
    println!("\n  Prediction picks: {} (total: {})", pred_choice.name, pred_total);
    println!("  Planning picks: {} (total: {})", plan_choice.name, plan_total);
    println!("  Planning wins by {:.0} points!\n", plan_total - pred_total);

    // 3. Tree-of-thought simulation
    println!("--- Tree-of-Thought Reasoning ---\n");

    // Problem: Find the best approach to compute sum(1..100)
    let thoughts = vec![
        Thought {
            strategy: "Brute force: add 1+2+3+...+100 one by one",
            steps: 100,
            accuracy: 0.70,
            elegance: 0.2,
        },
        Thought {
            strategy: "Pairing: (1+100)+(2+99)+...+(50+51) = 50*101",
            steps: 3,
            accuracy: 0.99,
            elegance: 0.9,
        },
        Thought {
            strategy: "Formula: n*(n+1)/2 = 100*101/2",
            steps: 2,
            accuracy: 0.99,
            elegance: 1.0,
        },
        Thought {
            strategy: "Recursion: sum(100) = 100 + sum(99) = ...",
            steps: 100,
            accuracy: 0.60,
            elegance: 0.3,
        },
    ];

    println!("  Problem: Compute 1 + 2 + 3 + ... + 100\n");

    // Evaluate each thought
    println!("{:<55} {:>6} {:>8} {:>8} {:>8}",
        "Strategy", "Steps", "Acc", "Elegant", "Score");
    println!("{}", "-".repeat(90));

    let mut best_score = 0.0;
    let mut best_idx = 0;

    for (i, thought) in thoughts.iter().enumerate() {
        let score = thought.accuracy * 0.5
            + thought.elegance * 0.3
            + (1.0 / thought.steps as f64) * 0.2;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
        println!(
            "{:<55} {:>6} {:>7.0}% {:>8.1} {:>8.3}",
            thought.strategy, thought.steps,
            thought.accuracy * 100.0, thought.elegance, score
        );
    }

    println!("\n  Predictive model likely chooses: '{}' (most common in training)",
        thoughts[0].strategy);
    println!("  Planning model evaluates all and chooses: '{}'",
        thoughts[best_idx].strategy);

    // 4. Self-evaluation and course correction
    println!("\n--- Self-Evaluation (Planning Feature) ---\n");

    let problem = "Solve: 2x + 3 = 15";

    // Attempt 1: Wrong approach
    println!("  Problem: {}\n", problem);
    println!("  Attempt 1 (prediction):");
    println!("    '2x + 3 = 15'");
    println!("    '2x = 15 + 3 = 18'   <- ERROR: should be 15 - 3");
    println!("    'x = 9'               <- Wrong answer");
    println!("    (Prediction committed to wrong step, cannot correct)\n");

    // With self-evaluation
    println!("  Attempt 2 (planning with self-evaluation):");
    println!("    '2x + 3 = 15'");
    println!("    '2x = 15 + 3 = 18'");
    println!("    [SELF-CHECK: if x=9, then 2*9+3=21 != 15. ERROR detected!]");
    println!("    [BACKTRACK]");
    println!("    '2x = 15 - 3 = 12'");
    println!("    'x = 6'");
    println!("    [VERIFY: 2*6 + 3 = 15. Correct!]");

    // 5. Cost-benefit analysis
    println!("\n--- Planning Overhead vs Quality ---\n");

    let scenarios = vec![
        ("Simple factual QA", 1, 95.0, 1, 96.0),
        ("Multi-step math", 1, 45.0, 5, 88.0),
        ("Code generation", 1, 60.0, 3, 85.0),
        ("Strategic game move", 1, 30.0, 10, 90.0),
        ("Complex proof", 1, 15.0, 8, 75.0),
    ];

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Task", "Pred cost", "Pred acc", "Plan cost", "Plan acc", "Worth it?"
    );
    println!("{}", "-".repeat(78));

    for (task, pred_cost, pred_acc, plan_cost, plan_acc) in &scenarios {
        let improvement = plan_acc - pred_acc;
        let cost_ratio = *plan_cost as f64 / *pred_cost as f64;
        let worth = if improvement / cost_ratio > 5.0 { "YES" } else { "Maybe" };
        println!(
            "{:<25} {:>10} {:>9.0}% {:>10} {:>9.0}% {:>10}",
            task, pred_cost, pred_acc, plan_cost, plan_acc, worth
        );
    }

    println!("\n  Planning is most valuable when prediction accuracy is low");
    println!("  and the cost of being wrong is high.");
}

struct Action {
    name: &'static str,
    immediate: f64,
    future: Vec<f64>,
}

struct Thought {
    strategy: &'static str,
    steps: usize,
    accuracy: f64,
    elegance: f64,
}

fn greedy_navigate(grid: &[Vec<i32>], start: (usize, usize), goal: (usize, usize)) -> Vec<(usize, usize)> {
    let mut path = vec![start];
    let mut current = start;
    let mut visited = vec![vec![false; grid[0].len()]; grid.len()];
    visited[start.0][start.1] = true;

    for _ in 0..50 { // Max steps to prevent infinite loop
        if current == goal {
            return path;
        }

        let directions: Vec<(i32, i32)> = vec![(0, 1), (1, 0), (0, -1), (-1, 0)];

        // Sort directions by distance to goal (greedy heuristic)
        let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
        for (dr, dc) in &directions {
            let nr = current.0 as i32 + dr;
            let nc = current.1 as i32 + dc;
            if nr >= 0 && nr < grid.len() as i32 && nc >= 0 && nc < grid[0].len() as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                if grid[nr][nc] == 0 && !visited[nr][nc] {
                    let dist = ((nr as f64 - goal.0 as f64).powi(2)
                        + (nc as f64 - goal.1 as f64).powi(2))
                        .sqrt();
                    candidates.push((nr, nc, dist));
                }
            }
        }

        if candidates.is_empty() {
            return vec![]; // Stuck!
        }

        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let next = (candidates[0].0, candidates[0].1);
        visited[next.0][next.1] = true;
        current = next;
        path.push(current);
    }
    vec![]
}

fn bfs_navigate(grid: &[Vec<i32>], start: (usize, usize), goal: (usize, usize)) -> Vec<(usize, usize)> {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut visited = vec![vec![false; cols]; rows];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; cols]; rows];
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(start);
    visited[start.0][start.1] = true;

    while let Some(current) = queue.pop_front() {
        if current == goal {
            // Reconstruct path
            let mut path = vec![goal];
            let mut pos = goal;
            while let Some(prev) = parent[pos.0][pos.1] {
                path.push(prev);
                pos = prev;
            }
            path.reverse();
            return path;
        }

        let directions = [(0i32, 1i32), (1, 0), (0, -1), (-1, 0)];
        for (dr, dc) in &directions {
            let nr = current.0 as i32 + dr;
            let nc = current.1 as i32 + dc;
            if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                if grid[nr][nc] == 0 && !visited[nr][nc] {
                    visited[nr][nc] = true;
                    parent[nr][nc] = Some(current);
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    vec![]
}

fn print_grid_with_path(
    grid: &[Vec<i32>],
    path: &[(usize, usize)],
    start: (usize, usize),
    goal: (usize, usize),
) {
    let path_set: std::collections::HashSet<(usize, usize)> = path.iter().cloned().collect();
    for (r, row) in grid.iter().enumerate() {
        print!("  ");
        for (c, cell) in row.iter().enumerate() {
            if (r, c) == start {
                print!("S ");
            } else if (r, c) == goal {
                print!("G ");
            } else if path_set.contains(&(r, c)) {
                print!("* ");
            } else if *cell == 1 {
                print!("# ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
    println!();
}
```

---

## Key Takeaways

- Prediction generates tokens greedily (forward-only), while planning evaluates multiple possible paths before committing, enabling recovery from dead ends and selection of globally optimal strategies.
- Problems with delayed rewards or non-obvious optimal first steps disproportionately benefit from planning, as the locally best choice is often not the globally best choice.
- Tree-of-thought reasoning, self-evaluation, and iterative refinement are practical techniques that add planning capabilities to fundamentally predictive language models.
- Planning has a cost (more computation) but is most valuable when prediction accuracy is low and the cost of errors is high, such as in multi-step reasoning, code generation, and strategic decision-making.
