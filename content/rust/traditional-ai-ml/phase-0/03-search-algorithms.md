# Search Algorithms

> Phase 0 — What is AI? | Kata 0.03

---

## Concept & Intuition

### What problem are we solving?

Many AI problems can be framed as search: finding a path from an initial state to a goal state through a space of possible states. A GPS finding the shortest route between two cities, a puzzle solver finding the sequence of moves to solve a Rubik's cube, or a game AI finding the best next move — all of these are search problems. The state space can be enormous, so we need systematic strategies to explore it efficiently.

Search algorithms are the workhorses of classical AI. They formalize the idea of exploring possibilities: start from where you are, consider what actions are available, take an action, and repeat until you reach your goal. The key distinction between algorithms is their strategy for deciding which state to explore next.

We will implement two foundational search algorithms: Breadth-First Search (BFS) and Depth-First Search (DFS), then compare their behavior on a maze-solving problem. Understanding these algorithms provides the foundation for more sophisticated search methods like A* that we will explore in later katas.

### Why naive approaches fail

A brute-force approach that tries every possible path quickly becomes infeasible. Consider a simple 10x10 grid maze: there could be thousands of possible paths. A 20x20 grid has millions. Without a systematic exploration strategy and a way to avoid revisiting states, you will either get stuck in loops or waste time exploring the same states repeatedly. Even with loop detection, the order in which you explore states dramatically affects how quickly you find a solution and whether that solution is optimal.

### Mental models

- **Search as tree exploration**: The initial state is the root. Each action creates a child node. Search algorithms decide which branch to explore next.
- **BFS as expanding ripple**: BFS explores all states at distance 1, then distance 2, then distance 3. Like ripples on a pond, it guarantees finding the shortest path.
- **DFS as deep dive**: DFS follows one path as deep as possible before backtracking. It uses less memory but may find a long, winding path even when a short one exists.

### Visual explanations

```
  Maze:              BFS explores            DFS explores
  S . . #            in layers:              depth-first:
  # . # .            1 2 3 #                 1 2 5 #
  . . . .            # 3 # .                 # 3 # .
  . # . G            . 4 5 6                 . 4 7 8
                     . # 6 7(G)              . # 6 9(G)

  BFS: guaranteed shortest path      DFS: finds A path, not necessarily shortest
  BFS: O(b^d) memory                 DFS: O(d) memory (b=branching, d=depth)
```

---

## Hands-on Exploration

1. Represent a maze as a 2D grid with walls, open cells, a start, and a goal.
2. Implement BFS using a queue — it explores states level by level.
3. Implement DFS using a stack — it explores states by diving deep first.
4. Compare the paths found, the number of states explored, and memory usage.

---

## Live Code

```rust
fn main() {
    println!("=== Search Algorithms: Maze Solving ===\n");

    // Define a maze: 0=open, 1=wall, 2=start, 3=goal
    let maze = vec![
        vec![2, 0, 0, 1, 0, 0, 0],
        vec![1, 0, 1, 0, 0, 1, 0],
        vec![0, 0, 0, 0, 1, 0, 0],
        vec![0, 1, 1, 0, 0, 0, 1],
        vec![0, 0, 0, 1, 0, 0, 0],
        vec![1, 1, 0, 0, 1, 1, 0],
        vec![0, 0, 0, 0, 0, 0, 3],
    ];

    let start = (0, 0);
    let goal = (6, 6);

    print_maze(&maze, &[]);

    // BFS
    println!("\n--- Breadth-First Search ---");
    let (bfs_path, bfs_explored) = bfs(&maze, start, goal);
    match &bfs_path {
        Some(path) => {
            println!("Path found! Length: {}", path.len());
            println!("States explored: {}", bfs_explored);
            print_maze(&maze, path);
        }
        None => println!("No path found!"),
    }

    // DFS
    println!("\n--- Depth-First Search ---");
    let (dfs_path, dfs_explored) = dfs(&maze, start, goal);
    match &dfs_path {
        Some(path) => {
            println!("Path found! Length: {}", path.len());
            println!("States explored: {}", dfs_explored);
            print_maze(&maze, path);
        }
        None => println!("No path found!"),
    }

    // Compare
    println!("\n--- Comparison ---");
    let bfs_len = bfs_path.as_ref().map_or(0, |p| p.len());
    let dfs_len = dfs_path.as_ref().map_or(0, |p| p.len());
    println!("BFS path length: {} (explored {} states)", bfs_len, bfs_explored);
    println!("DFS path length: {} (explored {} states)", dfs_len, dfs_explored);

    if bfs_len > 0 && dfs_len > 0 {
        println!(
            "BFS found {}path, DFS path is {}% longer",
            "shortest ",
            ((dfs_len as f64 / bfs_len as f64 - 1.0) * 100.0) as i32
        );
    }

    kata_metric("bfs_path_length", bfs_len as f64);
    kata_metric("dfs_path_length", dfs_len as f64);
    kata_metric("bfs_states_explored", bfs_explored as f64);
    kata_metric("dfs_states_explored", dfs_explored as f64);
}

fn bfs(
    maze: &[Vec<i32>],
    start: (usize, usize),
    goal: (usize, usize),
) -> (Option<Vec<(usize, usize)>>, usize) {
    let rows = maze.len();
    let cols = maze[0].len();
    let mut visited = vec![vec![false; cols]; rows];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; cols]; rows];

    // Queue for BFS (using Vec as a queue with remove(0))
    let mut queue: Vec<(usize, usize)> = Vec::new();
    queue.push(start);
    visited[start.0][start.1] = true;
    let mut explored = 0;

    let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while !queue.is_empty() {
        let current = queue.remove(0);
        explored += 1;

        if current == goal {
            return (Some(reconstruct_path(&parent, start, goal)), explored);
        }

        for &(dr, dc) in &directions {
            let nr = current.0 as i32 + dr;
            let nc = current.1 as i32 + dc;

            if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                if !visited[nr][nc] && maze[nr][nc] != 1 {
                    visited[nr][nc] = true;
                    parent[nr][nc] = Some(current);
                    queue.push((nr, nc));
                }
            }
        }
    }

    (None, explored)
}

fn dfs(
    maze: &[Vec<i32>],
    start: (usize, usize),
    goal: (usize, usize),
) -> (Option<Vec<(usize, usize)>>, usize) {
    let rows = maze.len();
    let cols = maze[0].len();
    let mut visited = vec![vec![false; cols]; rows];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; cols]; rows];

    // Stack for DFS
    let mut stack: Vec<(usize, usize)> = Vec::new();
    stack.push(start);
    let mut explored = 0;

    let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some(current) = stack.pop() {
        if visited[current.0][current.1] {
            continue;
        }
        visited[current.0][current.1] = true;
        explored += 1;

        if current == goal {
            return (Some(reconstruct_path(&parent, start, goal)), explored);
        }

        for &(dr, dc) in &directions {
            let nr = current.0 as i32 + dr;
            let nc = current.1 as i32 + dc;

            if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                let nr = nr as usize;
                let nc = nc as usize;
                if !visited[nr][nc] && maze[nr][nc] != 1 {
                    parent[nr][nc] = Some(current);
                    stack.push((nr, nc));
                }
            }
        }
    }

    (None, explored)
}

fn reconstruct_path(
    parent: &[Vec<Option<(usize, usize)>>],
    start: (usize, usize),
    goal: (usize, usize),
) -> Vec<(usize, usize)> {
    let mut path = vec![goal];
    let mut current = goal;
    while current != start {
        if let Some(p) = parent[current.0][current.1] {
            path.push(p);
            current = p;
        } else {
            break;
        }
    }
    path.reverse();
    path
}

fn print_maze(maze: &[Vec<i32>], path: &[(usize, usize)]) {
    for (r, row) in maze.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            let on_path = path.iter().any(|&(pr, pc)| pr == r && pc == c);
            if on_path && cell != 2 && cell != 3 {
                print!(" * ");
            } else {
                match cell {
                    1 => print!(" # "),
                    2 => print!(" S "),
                    3 => print!(" G "),
                    _ => print!(" . "),
                }
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

- Many AI problems can be formulated as searching through a state space from an initial state to a goal state.
- BFS explores states layer by layer and guarantees finding the shortest path, but uses more memory.
- DFS dives deep before backtracking, using less memory but potentially finding suboptimal paths.
- The choice between search strategies depends on the problem: is optimality required? How deep is the solution? How large is the branching factor?
