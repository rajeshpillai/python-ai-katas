# Heuristics and Cost

> Phase 0 — What is AI? | Kata 0.4

---

## Concept & Intuition

### What problem are we solving?

In the previous kata, we saw that A* search uses a heuristic function to guide exploration toward the goal. But where do heuristics come from? How do we design them? And what happens when they are wrong? This kata dives deep into **heuristic design** — the art of creating estimates that make search efficient without sacrificing correctness.

A heuristic function h(n) estimates the cost of reaching the goal from node n. The quality of this estimate has a profound impact on search performance. A perfect heuristic would make A* go straight to the goal without exploring any unnecessary nodes. A terrible heuristic would make A* degenerate into brute-force search. The sweet spot is a heuristic that is "good enough" — close to the true cost while being fast to compute.

The formal properties that matter are **admissibility** (the heuristic never overestimates the true cost) and **consistency** (the heuristic obeys the triangle inequality). Admissibility guarantees that A* finds the optimal solution. Consistency guarantees that A* never needs to re-expand a node. Understanding these properties is critical for designing search algorithms that are both correct and efficient.

### Why naive approaches fail

The most common mistake is using a heuristic that overestimates costs. An overestimating heuristic makes A* greedy — it rushes toward the goal and may miss shorter paths that initially look worse. The resulting path can be arbitrarily bad. Conversely, a heuristic that always returns zero (trivially admissible) gives no guidance, making A* behave exactly like Dijkstra's algorithm — correct but slow.

Another pitfall is using a heuristic that is expensive to compute. If evaluating h(n) takes as long as actually searching, the heuristic provides no speedup. The best heuristics capture the essential structure of the problem in a computation that is orders of magnitude cheaper than solving the problem itself.

### Mental models

- **The GPS analogy**: A heuristic is like the straight-line distance shown on a GPS — it tells you roughly how far you are from the destination, even though the actual road distance is longer. As long as the estimate is never too high (admissible), the GPS will find the best route
- **Relaxed problems**: Many good heuristics come from solving a "relaxed" version of the problem where some constraints are removed. For a tile puzzle, allowing tiles to pass through each other gives the Manhattan distance heuristic
- **The spectrum of knowledge**: h(n) = 0 means "I know nothing" (blind search). h(n) = h*(n) means "I know everything" (perfect oracle). Real heuristics fall between these extremes
- **Tighter is better**: Among admissible heuristics, the one closer to the true cost expands fewer nodes. But tighter heuristics are usually more expensive to compute — there is always a tradeoff

### Visual explanations

```
Admissibility: Never Overestimate
===================================

True cost to goal: 10

  h(n) = 7   ADMISSIBLE (underestimates: 7 <= 10)
  h(n) = 10  ADMISSIBLE (exact: 10 <= 10)
  h(n) = 12  INADMISSIBLE (overestimates: 12 > 10) !!

              A* with admissible h  =>  Guaranteed optimal
              A* with inadmissible h  =>  May find suboptimal path


Consistency (Triangle Inequality)
===================================

  For every node n and successor n' with step cost c(n, n'):

    h(n) <= c(n, n') + h(n')

        n ---c=3--- n'
        |            |
     h(n)=5       h(n')=4

    Is 5 <= 3 + 4 = 7?  YES -> Consistent


Heuristic Quality Spectrum
============================

  h(n)=0        h(n)=Manhattan     h(n)=h*(n)
  (no info)     (moderate info)    (perfect info)
    |                |                  |
    v                v                  v
  Dijkstra         A*              Greedy direct
  (slow, optimal) (balanced)       (instant, optimal)
  Expands many    Expands some     Expands minimum
  nodes           nodes            nodes

  <── Cheaper to compute ─── Expensive to compute ──>
  <── More nodes expanded ── Fewer nodes expanded ──>
```

---

## Hands-on Exploration

1. **Compare heuristics**: Run the code below with different heuristics (Euclidean, Manhattan, Chebyshev, zero) on the same grid. Count how many nodes each heuristic causes A* to expand.

2. **Break admissibility**: Multiply the Manhattan distance by 1.5, 2.0, and 3.0. For each, check whether A* still finds the optimal path and how many nodes it expands. This shows the tradeoff between speed and optimality.

3. **Design your own heuristic**: For the grid world, invent a new admissible heuristic different from the ones provided. Verify that it never overestimates by comparing it to the true shortest path distance for several nodes.

4. **Weighted A***: Experiment with f(n) = g(n) + w*h(n) for different weights w. Plot how path quality and nodes expanded change as w increases from 1.0 (optimal A*) to higher values (faster but suboptimal).

---

## Live Code

```python
"""
Heuristics and Cost — Designing and evaluating heuristic functions for A* search.

This code compares multiple heuristics on a grid-world search problem,
measuring their impact on search efficiency and solution quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq

# ============================================================
# Grid and Search Setup
# ============================================================

def create_grid(rows, cols, wall_fraction=0.2, seed=42):
    """Create a grid with random walls."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((rows, cols), dtype=int)
    num_walls = int(rows * cols * wall_fraction)
    wall_positions = rng.choice(rows * cols, size=num_walls, replace=False)
    for pos in wall_positions:
        r, c = divmod(pos, cols)
        grid[r, c] = 1
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid

def get_neighbors(grid, node):
    """Return valid 4-connected neighbors."""
    rows, cols = grid.shape
    r, c = node
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def astar(grid, start, goal, heuristic_fn):
    """A* search with a pluggable heuristic function."""
    open_set = [(heuristic_fn(start, goal), 0, start, [start])]
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
                f = new_g + heuristic_fn(neighbor, goal)
                heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor]))

    return None, visited_order

# ============================================================
# Heuristic Functions
# ============================================================

def h_zero(node, goal):
    """Zero heuristic — equivalent to Dijkstra's algorithm."""
    return 0

def h_manhattan(node, goal):
    """Manhattan (L1) distance — admissible for 4-connected grids."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def h_euclidean(node, goal):
    """Euclidean (L2) distance — admissible, but looser than Manhattan for grids."""
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

def h_chebyshev(node, goal):
    """Chebyshev (L-inf) distance — admissible for 4-connected grids."""
    return max(abs(node[0] - goal[0]), abs(node[1] - goal[1]))

def h_overestimate(node, goal):
    """Inadmissible heuristic — 2x Manhattan distance (overestimates!)."""
    return 2 * (abs(node[0] - goal[0]) + abs(node[1] - goal[1]))

# ============================================================
# Run Comparison
# ============================================================

ROWS, COLS = 20, 25
grid = create_grid(ROWS, COLS, wall_fraction=0.25, seed=42)
start = (0, 0)
goal = (ROWS - 1, COLS - 1)

heuristics = {
    "h=0 (Dijkstra)": h_zero,
    "Chebyshev": h_chebyshev,
    "Euclidean": h_euclidean,
    "Manhattan": h_manhattan,
    "2x Manhattan\n(inadmissible!)": h_overestimate,
}

results = {}
for name, h_fn in heuristics.items():
    path, visited = astar(grid, start, goal, h_fn)
    results[name] = {"path": path, "visited": visited}

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

for ax, (name, res) in zip(axes, results.items()):
    display = np.full((ROWS, COLS, 3), 1.0)

    # Walls
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] == 1:
                display[r, c] = [0.2, 0.2, 0.2]

    # Visited nodes
    for i, (r, c) in enumerate(res["visited"]):
        t = i / max(len(res["visited"]) - 1, 1)
        display[r, c] = [0.7 + 0.3 * t, 0.85 - 0.4 * t, 1.0 - 0.6 * t]

    # Path
    if res["path"]:
        for r, c in res["path"]:
            display[r, c] = [1.0, 0.84, 0.0]

    display[start[0], start[1]] = [0.0, 0.8, 0.0]
    display[goal[0], goal[1]] = [1.0, 0.0, 0.0]

    path_len = len(res["path"]) if res["path"] else "N/A"
    ax.imshow(display, interpolation="nearest")
    ax.set_title(f"{name}\nVisited: {len(res['visited'])}\nPath: {path_len}", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Impact of Heuristic Choice on A* Search", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Quantitative Analysis
# ============================================================

print("=" * 65)
print("HEURISTIC COMPARISON — QUANTITATIVE RESULTS")
print("=" * 65)
print(f"{'Heuristic':<25} {'Visited':>10} {'Path Len':>10} {'Optimal?':>10}")
print("-" * 65)

# Find optimal path length (from Manhattan heuristic, guaranteed admissible)
optimal_len = len(results["Manhattan"]["path"]) if results["Manhattan"]["path"] else None

for name, res in results.items():
    visited = len(res["visited"])
    path_len = len(res["path"]) if res["path"] else "N/A"
    is_optimal = "Yes" if res["path"] and len(res["path"]) == optimal_len else "NO"
    clean_name = name.replace("\n", " ")
    print(f"{clean_name:<25} {visited:>10} {str(path_len):>10} {is_optimal:>10}")

# ============================================================
# Weighted A* Experiment
# ============================================================

print("\n" + "=" * 65)
print("WEIGHTED A*: f(n) = g(n) + w * h(n)")
print("=" * 65)

weights = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
w_results = []

for w in weights:
    def h_weighted(node, goal, weight=w):
        return weight * h_manhattan(node, goal)

    path, visited = astar(grid, start, goal, h_weighted)
    path_len = len(path) if path else None
    w_results.append((w, len(visited), path_len))
    ratio = path_len / optimal_len if path_len and optimal_len else float("inf")
    print(f"  w={w:<5.1f}  Visited: {len(visited):>4}  Path: {str(path_len):>4}  "
          f"Suboptimality: {ratio:.2f}x")

# Plot the tradeoff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ws = [r[0] for r in w_results]
visited_counts = [r[1] for r in w_results]
path_lens = [r[2] for r in w_results]

ax1.plot(ws, visited_counts, "bo-", linewidth=2, markersize=8)
ax1.set_xlabel("Weight w", fontsize=12)
ax1.set_ylabel("Nodes Expanded", fontsize=12)
ax1.set_title("Speed: Higher w = Fewer Nodes", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)

ax2.plot(ws, path_lens, "ro-", linewidth=2, markersize=8)
ax2.axhline(y=optimal_len, color="green", linestyle="--", label=f"Optimal ({optimal_len})")
ax2.set_xlabel("Weight w", fontsize=12)
ax2.set_ylabel("Path Length", fontsize=12)
ax2.set_title("Quality: Higher w = Longer Path", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.suptitle("Weighted A* Tradeoff: Speed vs. Optimality", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nKey insight: There is a fundamental tradeoff between search speed")
print("and solution quality. Admissible heuristics guarantee optimality;")
print("inadmissible heuristics trade optimality for speed.")
```

---

## Key Takeaways

- **A heuristic is an estimate** of the cost to reach the goal — it guides search by telling the algorithm which directions look promising
- **Admissibility (never overestimate)** is the key property that guarantees A* finds the optimal solution. Violating it trades optimality for speed
- **Consistency (triangle inequality)** is a stronger property that prevents A* from needing to re-expand nodes, improving efficiency
- **Better heuristics = fewer nodes expanded**, but there is always a tradeoff between heuristic quality and computation cost
- **Weighted A* (f = g + w*h)** provides a practical knob to tune the speed-vs-optimality tradeoff, useful when "good enough" solutions are acceptable
