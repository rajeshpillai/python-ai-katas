# Heuristics and Cost

> Phase 0 — What is AI? | Kata 0.04

---

## Concept & Intuition

### What problem are we solving?

Blind search algorithms like BFS and DFS explore the state space without any sense of direction. They do not know whether they are getting closer to or farther from the goal. Heuristic search adds domain knowledge to guide the exploration, dramatically reducing the number of states examined. The A* algorithm combines the best of both worlds: it uses a heuristic to estimate the remaining cost and tracks the actual cost so far, guaranteeing an optimal path when the heuristic is admissible (never overestimates).

The concept of cost is fundamental. Not all paths are equal: some edges in a graph are more expensive than others (longer roads, steeper terrain, higher tolls). A proper search algorithm must account for varying costs rather than simply counting steps. This is what transforms naive search into practical pathfinding.

In this kata, we implement the A* algorithm — one of the most important algorithms in all of AI — and compare it against BFS on a weighted grid. We will see how a good heuristic can reduce exploration by orders of magnitude.

### Why naive approaches fail

BFS finds the shortest path in terms of number of steps, but it ignores edge costs. On a weighted graph, the path with fewest edges might be far more expensive than a path with more edges but lower total weight. Furthermore, BFS explores states in all directions equally, wasting effort on states far from the goal. Without a heuristic, a search from New York to Los Angeles would explore states in Maine and Florida just as eagerly as states in Pennsylvania and Ohio.

### Mental models

- **f(n) = g(n) + h(n)**: A* evaluates each state by its total estimated cost: actual cost from start (g) plus estimated cost to goal (h). This balances exploitation (follow cheap paths) with exploration (move toward the goal).
- **Admissible heuristic**: A heuristic that never overestimates guarantees optimality. Manhattan distance on a grid is admissible because you can never reach the goal faster than the direct path.
- **A* as informed BFS**: A* is like BFS but instead of exploring the nearest state (by steps), it explores the most promising state (by estimated total cost).

### Visual explanations

```
  BFS explores uniformly:          A* focuses toward the goal:

  . . . . . G                     . . . . . G
  . x x x . .                     . . . x x .
  . x x x . .                     . . x x . .
  . x S x . .                     . . x . . .
  . x x x . .                     . . S . . .
  . . . . . .                     . . . . . .

  x = explored states              x = explored states
  BFS: many states explored        A*: far fewer states explored
```

---

## Hands-on Exploration

1. Create a weighted grid where each cell has a traversal cost.
2. Implement A* search with Manhattan distance as the heuristic.
3. Compare the path cost and states explored against BFS (uniform cost).
4. Experiment with different heuristics and observe their effect on optimality and speed.

---

## Live Code

```rust
fn main() {
    println!("=== A* Search: Heuristic Pathfinding ===\n");

    // Weighted grid: value = traversal cost (0 = wall)
    let grid = vec![
        vec![1, 1, 1, 0, 1, 1, 1, 1],
        vec![1, 3, 1, 0, 1, 5, 5, 1],
        vec![1, 3, 1, 1, 1, 1, 5, 1],
        vec![1, 3, 3, 3, 0, 1, 1, 1],
        vec![1, 1, 1, 3, 0, 1, 3, 1],
        vec![0, 0, 1, 1, 1, 1, 3, 1],
        vec![1, 1, 1, 0, 0, 1, 1, 1],
        vec![1, 1, 1, 1, 1, 1, 1, 1],
    ];

    let start = (0, 0);
    let goal = (7, 7);

    println!("Grid (cost values, 0 = wall):");
    for row in &grid {
        for &cell in row {
            if cell == 0 {
                print!(" ## ");
            } else {
                print!(" {:2} ", cell);
            }
        }
        println!();
    }

    // A* search
    println!("\n--- A* Search (Manhattan heuristic) ---");
    let (astar_path, astar_cost, astar_explored) =
        astar_search(&grid, start, goal, manhattan_distance);
    print_search_result(&grid, &astar_path, astar_cost, astar_explored);

    // A* with zero heuristic (equivalent to Dijkstra / uniform cost search)
    println!("\n--- Uniform Cost Search (zero heuristic) ---");
    let (ucs_path, ucs_cost, ucs_explored) =
        astar_search(&grid, start, goal, |_, _| 0.0);
    print_search_result(&grid, &ucs_path, ucs_cost, ucs_explored);

    // A* with Euclidean heuristic
    println!("\n--- A* Search (Euclidean heuristic) ---");
    let (euc_path, euc_cost, euc_explored) =
        astar_search(&grid, start, goal, euclidean_distance);
    print_search_result(&grid, &euc_path, euc_cost, euc_explored);

    // Comparison
    println!("\n=== Comparison ===");
    println!("{:<25} {:<12} {:<12}", "Algorithm", "Path Cost", "Explored");
    println!("{}", "-".repeat(49));
    println!("{:<25} {:<12.1} {:<12}", "A* (Manhattan)", astar_cost, astar_explored);
    println!("{:<25} {:<12.1} {:<12}", "Uniform Cost (Dijkstra)", ucs_cost, ucs_explored);
    println!("{:<25} {:<12.1} {:<12}", "A* (Euclidean)", euc_cost, euc_explored);

    let savings = if ucs_explored > 0 {
        (1.0 - astar_explored as f64 / ucs_explored as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "\nA* (Manhattan) explored {:.0}% fewer states than Uniform Cost",
        savings
    );

    kata_metric("astar_manhattan_cost", astar_cost);
    kata_metric("astar_manhattan_explored", astar_explored as f64);
    kata_metric("uniform_cost_explored", ucs_explored as f64);
    kata_metric("exploration_savings_pct", savings);
}

fn astar_search(
    grid: &[Vec<i32>],
    start: (usize, usize),
    goal: (usize, usize),
    heuristic: fn((usize, usize), (usize, usize)) -> f64,
) -> (Vec<(usize, usize)>, f64, usize) {
    let rows = grid.len();
    let cols = grid[0].len();

    // g_cost[r][c] = best known cost from start to (r, c)
    let mut g_cost = vec![vec![f64::INFINITY; cols]; rows];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; cols]; rows];
    let mut closed = vec![vec![false; cols]; rows];

    // Open list: (f_cost, (row, col))
    // We implement a simple priority queue using a sorted Vec
    let mut open: Vec<(f64, (usize, usize))> = Vec::new();

    g_cost[start.0][start.1] = 0.0;
    let h = heuristic(start, goal);
    open.push((h, start));

    let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut explored = 0;

    while !open.is_empty() {
        // Find the node with lowest f_cost
        let mut best_idx = 0;
        for i in 1..open.len() {
            if open[i].0 < open[best_idx].0 {
                best_idx = i;
            }
        }
        let (_f, current) = open.remove(best_idx);

        if closed[current.0][current.1] {
            continue;
        }
        closed[current.0][current.1] = true;
        explored += 1;

        if current == goal {
            let path = reconstruct_path(&parent, start, goal);
            return (path, g_cost[goal.0][goal.1], explored);
        }

        for &(dr, dc) in &directions {
            let nr = current.0 as i32 + dr;
            let nc = current.1 as i32 + dc;

            if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;

                if grid[nr][nc] == 0 || closed[nr][nc] {
                    continue;
                }

                let new_g = g_cost[current.0][current.1] + grid[nr][nc] as f64;

                if new_g < g_cost[nr][nc] {
                    g_cost[nr][nc] = new_g;
                    parent[nr][nc] = Some(current);
                    let f = new_g + heuristic((nr, nc), goal);
                    open.push((f, (nr, nc)));
                }
            }
        }
    }

    (vec![], f64::INFINITY, explored)
}

fn manhattan_distance(a: (usize, usize), b: (usize, usize)) -> f64 {
    ((a.0 as f64 - b.0 as f64).abs() + (a.1 as f64 - b.1 as f64).abs())
}

fn euclidean_distance(a: (usize, usize), b: (usize, usize)) -> f64 {
    let dr = a.0 as f64 - b.0 as f64;
    let dc = a.1 as f64 - b.1 as f64;
    (dr * dr + dc * dc).sqrt()
}

fn reconstruct_path(
    parent: &[Vec<Option<(usize, usize)>>],
    start: (usize, usize),
    goal: (usize, usize),
) -> Vec<(usize, usize)> {
    let mut path = vec![goal];
    let mut current = goal;
    while current != start {
        match parent[current.0][current.1] {
            Some(p) => {
                path.push(p);
                current = p;
            }
            None => break,
        }
    }
    path.reverse();
    path
}

fn print_search_result(
    grid: &[Vec<i32>],
    path: &[(usize, usize)],
    cost: f64,
    explored: usize,
) {
    if path.is_empty() {
        println!("No path found!");
        return;
    }
    println!("Path cost: {:.1}, States explored: {}", cost, explored);
    for (r, row) in grid.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            let on_path = path.iter().any(|&(pr, pc)| pr == r && pc == c);
            if on_path {
                print!(" ** ");
            } else if cell == 0 {
                print!(" ## ");
            } else {
                print!("  . ");
            }
        }
        println!();
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- A* combines actual cost (g) with estimated remaining cost (h) to efficiently find optimal paths.
- An admissible heuristic (one that never overestimates) guarantees A* finds the optimal solution.
- Compared to uninformed search, A* can explore dramatically fewer states while still finding the best path.
- The quality of the heuristic directly impacts efficiency: a tighter (but still admissible) heuristic means fewer states explored.
