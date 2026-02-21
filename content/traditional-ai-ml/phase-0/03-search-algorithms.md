# Search Algorithms

> Phase 0 — What is AI? | Kata 0.3

---

## Concept & Intuition

### What problem are we solving?

Many AI problems can be framed as search: finding a path from an initial state to a goal state through a space of possibilities. How does a GPS find the shortest route? How does a chess engine evaluate millions of board positions? How does a puzzle solver find the right sequence of moves? All of these are search problems — and the algorithms we use to navigate these spaces determine whether we find a solution at all, and how efficiently we find it.

Search algorithms are foundational to AI because they formalize the idea of "exploring possibilities systematically." Before machine learning, search was the primary tool for building intelligent behavior. Even today, search remains central to game playing (Monte Carlo Tree Search in AlphaGo), planning (robot motion planning), and optimization (finding the best hyperparameters for ML models).

The key insight is that different search strategies make different tradeoffs between **completeness** (will it find a solution if one exists?), **optimality** (will it find the best solution?), and **efficiency** (how much time and memory does it need?). Understanding these tradeoffs is fundamental to choosing the right algorithm for a given problem.

### Why naive approaches fail

The most obvious approach — exhaustively trying every possible path — works only for tiny problems. The number of states in most interesting problems grows exponentially. A chess game has roughly 10^120 possible positions; the number of possible routes through a city has factorial growth. Brute force is not just slow; it is physically impossible for problems beyond trivial size.

Even "smart" exhaustive methods like breadth-first search (BFS), which guarantees finding the shortest path, can consume astronomical amounts of memory on large graphs. Depth-first search (DFS) uses less memory but can wander down infinite paths and miss nearby solutions. The challenge is to search intelligently — using domain knowledge (heuristics) to guide exploration toward promising regions of the search space.

### Mental models

- **The maze analogy**: BFS explores the maze layer by layer (all cells 1 step away, then 2 steps, etc.). DFS follows one path as far as possible before backtracking. A* uses a map to head toward the exit
- **Breadth vs. depth**: BFS is like searching for your keys by methodically checking every room on each floor before moving to the next floor. DFS is like picking a direction and following it until you hit a dead end
- **The cost of memory**: BFS remembers every node it visits (expensive). DFS only remembers the current path (cheap). A* remembers visited nodes but explores fewer of them
- **Informed vs. uninformed**: BFS and DFS are "blind" — they don't know where the goal is. A* has a "compass" (the heuristic) pointing toward the goal

### Visual explanations

```
Search Strategy Comparison
============================

BFS (Breadth-First Search)          DFS (Depth-First Search)
Explores level by level             Explores branch by branch

       S                                   S
      / \                                 / \
    [1] [2]    <- level 1               [1]  6
    / \   \                             / \
  [3] [4] [5]  <- level 2            [2]  5
  /                                   / \
 G                                  [3]  4
                                    /
Order: S,1,2,3,4,5,G              [G]
Memory: O(b^d)
Optimal: Yes (unweighted)         Order: S,1,2,3,G
                                  Memory: O(d)
                                  Optimal: No


A* Search — Best of Both Worlds
=================================

Uses: f(n) = g(n) + h(n)
       |       |       |
     total   actual  estimated
     cost    cost    cost to
             so far  goal

    S ---2--- A ---3--- G
    |                   |
    1                   1
    |                   |
    B -------5--------- C

  A* with good heuristic explores: S -> A -> G (cost 5)
  BFS explores: S -> A -> B -> G (finds path but wastes effort)
  DFS might explore: S -> B -> C -> G (cost 7, suboptimal)
```

---

## Hands-on Exploration

1. **Visualize the frontier**: Run the code below and observe how BFS and DFS expand nodes in completely different orders on the same grid. Pay attention to how many nodes each algorithm visits before finding the goal.

2. **Experiment with obstacles**: Modify the grid to add walls and observe how the algorithms route around them. Notice how BFS always finds the shortest path but A* finds it faster.

3. **Break A***: Try modifying the heuristic to overestimate distances (multiply Manhattan distance by 3). Observe that A* may no longer find the optimal path — this demonstrates why admissibility matters.

4. **Count the cost**: Compare the number of nodes expanded by BFS vs. A* on grids of increasing size. Plot the relationship — this reveals A*'s practical advantage.

---

## Live Code

```python
"""
Search Algorithms — BFS, DFS, and A* on a grid world.

This code implements three search algorithms and visualizes how they
explore a 2D grid to find a path from start to goal.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

# ============================================================
# Grid World Setup
# ============================================================

def create_grid(rows, cols, wall_fraction=0.2, seed=42):
    """Create a grid with random walls. 0=open, 1=wall."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((rows, cols), dtype=int)
    num_walls = int(rows * cols * wall_fraction)
    wall_positions = rng.choice(rows * cols, size=num_walls, replace=False)
    for pos in wall_positions:
        r, c = divmod(pos, cols)
        grid[r, c] = 1
    # Ensure start and goal are open
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid

def get_neighbors(grid, node):
    """Return valid neighbors (up, down, left, right)."""
    rows, cols = grid.shape
    r, c = node
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

# ============================================================
# Search Algorithms
# ============================================================

def bfs(grid, start, goal):
    """Breadth-First Search. Returns (path, visited_order)."""
    queue = deque([(start, [start])])
    visited = set([start])
    visited_order = [start]

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path, visited_order
        for neighbor in get_neighbors(grid, node):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, visited_order

def dfs(grid, start, goal):
    """Depth-First Search. Returns (path, visited_order)."""
    stack = [(start, [start])]
    visited = set([start])
    visited_order = [start]

    while stack:
        node, path = stack.pop()
        if node == goal:
            return path, visited_order
        for neighbor in get_neighbors(grid, node):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                stack.append((neighbor, path + [neighbor]))
    return None, visited_order

def astar(grid, start, goal):
    """A* Search with Manhattan distance heuristic. Returns (path, visited_order)."""
    def heuristic(node):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    open_set = [(heuristic(start), 0, start, [start])]
    visited = set()
    visited_order = []
    g_scores = {start: 0}

    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited:
            continue
        visited.add(node)
        visited_order.append(node)

        if node == goal:
            return path, visited_order

        for neighbor in get_neighbors(grid, node):
            new_g = g + 1
            if neighbor not in g_scores or new_g < g_scores[neighbor]:
                g_scores[neighbor] = new_g
                f = new_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor]))

    return None, visited_order

# ============================================================
# Visualization
# ============================================================

def visualize_search(grid, visited_order, path, title, ax):
    """Visualize the search process on a grid."""
    rows, cols = grid.shape
    display = np.full((rows, cols, 3), 1.0)  # White background

    # Draw walls
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                display[r, c] = [0.2, 0.2, 0.2]  # Dark gray walls

    # Draw visited nodes (gradient from light to dark blue)
    for i, (r, c) in enumerate(visited_order):
        intensity = 0.3 + 0.5 * (i / max(len(visited_order) - 1, 1))
        display[r, c] = [0.8, 0.9 - 0.4 * intensity, 1.0 - 0.6 * intensity]

    # Draw path
    if path:
        for r, c in path:
            display[r, c] = [1.0, 0.84, 0.0]  # Gold path

    # Mark start and goal
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    display[start[0], start[1]] = [0.0, 0.8, 0.0]  # Green start
    display[goal[0], goal[1]] = [1.0, 0.0, 0.0]     # Red goal

    ax.imshow(display, interpolation="nearest")
    ax.set_title(f"{title}\nVisited: {len(visited_order)} | Path: {len(path) if path else 'None'}",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

# Run the search algorithms
ROWS, COLS = 15, 20
grid = create_grid(ROWS, COLS, wall_fraction=0.25, seed=42)
start = (0, 0)
goal = (ROWS - 1, COLS - 1)

bfs_path, bfs_visited = bfs(grid, start, goal)
dfs_path, dfs_visited = dfs(grid, start, goal)
astar_path, astar_visited = astar(grid, start, goal)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

visualize_search(grid, bfs_visited, bfs_path, "BFS (Breadth-First)", axes[0])
visualize_search(grid, dfs_visited, dfs_path, "DFS (Depth-First)", axes[1])
visualize_search(grid, astar_visited, astar_path, "A* (Informed Search)", axes[2])

plt.suptitle("Search Algorithm Comparison on Grid World", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Print summary statistics
print("=" * 60)
print("SEARCH ALGORITHM COMPARISON")
print("=" * 60)
algorithms = [
    ("BFS", bfs_visited, bfs_path),
    ("DFS", dfs_visited, dfs_path),
    ("A*", astar_visited, astar_path),
]

for name, visited, path in algorithms:
    path_len = len(path) if path else "No path"
    print(f"\n{name}:")
    print(f"  Nodes visited:  {len(visited)}")
    print(f"  Path length:    {path_len}")
    if path:
        print(f"  Efficiency:     {len(path)/len(visited):.1%} (path/visited ratio)")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("- BFS finds the shortest path but explores many nodes")
print("- DFS uses less memory but may find a longer path")
print("- A* finds the shortest path while exploring fewer nodes")
print("- A*'s advantage grows with problem size")
```

---

## Key Takeaways

- **Search is foundational to AI**: Many intelligent behaviors can be framed as finding a path through a state space, from route planning to game playing to puzzle solving
- **BFS guarantees the shortest path** (in unweighted graphs) but can consume enormous memory because it stores all frontier nodes
- **DFS is memory-efficient** (only stores the current path) but provides no optimality guarantee and can explore irrelevant branches
- **A* combines the best of both worlds** by using a heuristic to guide search toward the goal, finding optimal paths while exploring fewer nodes than BFS
- **The heuristic must be admissible** (never overestimate the true cost) for A* to guarantee optimality — a bad heuristic can make A* worse than BFS
