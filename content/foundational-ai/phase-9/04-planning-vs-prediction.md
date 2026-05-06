# Planning vs Prediction

> Phase 9 — Reasoning Models | Kata 9.4

---

## Concept & Intuition

### What problem are we solving?

Language models are fundamentally next-token predictors: given what came before, predict what comes next. This works astonishingly well for many tasks, but it is inherently a local, myopic strategy. The model chooses each token based on immediate context without considering where the sequence will end up five, ten, or fifty tokens later. This is like navigating a maze by always choosing the corridor that looks most promising right now, without ever looking at the map.

Planning is fundamentally different. A planning agent considers future consequences before making each decision. In a chess game, a greedy player picks the move that captures the most valuable piece right now, but a planning player considers entire sequences of moves and counter-moves, choosing the move that leads to the best position many turns ahead. The difference between prediction and planning is the difference between "What is the most likely next word?" and "What next word leads to the best overall outcome?"

This distinction matters enormously for tasks where short-term and long-term objectives conflict. In writing an essay, the most likely next sentence might repeat a point already made (local coherence), while the best next sentence introduces a new argument that makes the essay stronger overall (global planning). In problem-solving, the most probable next step might follow a familiar but wrong pattern, while the optimal step requires an unintuitive detour that leads to the correct solution.

### Why naive approaches fail

Pure next-token prediction fails when the locally optimal choice is globally suboptimal. Consider generating a proof: the model might start down a promising-looking path that leads to a dead end. By the time it realizes the path is wrong, it has already committed tokens and cannot backtrack (autoregressive generation is one-directional). A planning approach would evaluate multiple potential paths before committing to any of them.

Beam search (tracking multiple candidate sequences) helps somewhat, but it is still fundamentally greedy with a wider beam. It does not perform the kind of deep lookahead that true planning requires. And increasing the beam width has diminishing returns -- you need exponentially more beams to capture exponentially many possible futures.

### Mental models

- **GPS navigation vs following your nose:** Prediction is driving toward your destination by always turning in its general direction. Planning is computing the full route first, including highways that initially go the "wrong" direction but are faster overall.
- **Chess novice vs grandmaster:** A novice captures any piece available (greedy prediction). A grandmaster sacrifices a piece now to set up a checkmate in five moves (planning with lookahead).
- **Building a house:** A prediction approach lays bricks that look good right now. A planning approach starts with blueprints, ensuring the foundation supports the roof that has not been built yet.

### Visual explanations

```
  PREDICTION (Greedy / Next-Token):

  State ──> Best immediate   ──> Best immediate   ──> ...
             move                 move
  "Always pick what seems best RIGHT NOW"

  Path:  A ──> B ──> C ──> D ──> DEAD END
               │
               └ (the right path was B -> E -> F -> GOAL,
                  but E looked worse than C at step 2)

  PLANNING (Lookahead / Search):

  State ──> Evaluate ALL paths to depth d:
            ├── A -> B -> C -> D: score = 3
            ├── A -> B -> E -> F: score = 9  ← best!
            ├── A -> G -> H -> I: score = 5
            └── A -> G -> J -> K: score = 2
         ──> Pick path to score 9
         ──> Execute first move: A -> B

  MINIMAX PLANNING (adversarial):

       MAX (us)          [choose max]
      /    \
    MIN     MIN          [opponent chooses min]
   / \     / \
  3   5   2   9          [leaf scores]

  MAX picks right branch (min is 2),
  but left branch has min=3, so MAX picks LEFT.
  Greedy would pick right (has 9), but opponent blocks it.
```

---

## Hands-on Exploration

1. Implement a simple grid world where an agent must reach a goal
2. Compare greedy prediction (always move toward goal) with planning (search for optimal path)
3. Add obstacles that make the greedy path fail but the planned path succeed
4. Implement minimax planning for a simple adversarial game
5. Measure how lookahead depth affects decision quality

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Grid world environment ---
def create_grid(size=8):
    """Create a grid with obstacles. 0=free, 1=wall, 2=goal."""
    grid = np.zeros((size, size), dtype=int)
    # Add walls that force non-greedy paths
    for i in range(1, size - 1):
        grid[i, size // 2] = 1     # vertical wall
    grid[1, size // 2] = 0         # gap at top
    grid[size - 2, size // 2] = 0  # gap at bottom
    grid[size - 1, size - 1] = 2   # goal
    return grid

def print_grid(grid, path=None, label=""):
    """Display grid with optional path."""
    print(f"\n  {label}")
    size = grid.shape[0]
    print("  " + "+" + "---" * size + "+")
    for i in range(size):
        row = "  |"
        for j in range(size):
            if path and (i, j) in path:
                if (i, j) == path[0]:
                    row += " S "
                elif (i, j) == path[-1]:
                    row += " G "
                else:
                    row += " . "
            elif grid[i, j] == 1:
                row += "###"
            elif grid[i, j] == 2:
                row += " G "
            else:
                row += "   "
        row += "|"
        print(row)
    print("  " + "+" + "---" * size + "+")

def greedy_agent(grid, start, goal):
    """Always move toward goal (Manhattan distance)."""
    path = [start]
    pos = start
    visited = {start}
    max_steps = grid.size * 2
    for _ in range(max_steps):
        if pos == goal:
            return path, True
        # Try all moves, pick the one closest to goal
        moves = [(pos[0]-1,pos[1]), (pos[0]+1,pos[1]),
                 (pos[0],pos[1]-1), (pos[0],pos[1]+1)]
        valid = [(r,c) for r,c in moves
                 if 0<=r<grid.shape[0] and 0<=c<grid.shape[1]
                 and grid[r,c] != 1 and (r,c) not in visited]
        if not valid:
            return path, False  # stuck
        # Greedy: minimize Manhattan distance
        best = min(valid, key=lambda p: abs(p[0]-goal[0])+abs(p[1]-goal[1]))
        visited.add(best)
        path.append(best)
        pos = best
    return path, False

def bfs_planner(grid, start, goal):
    """BFS: finds shortest path (optimal planning)."""
    from collections import deque
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path, True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if (0<=nr<grid.shape[0] and 0<=nc<grid.shape[1]
                    and grid[nr,nc] != 1 and (nr,nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr,nc), path + [(nr,nc)]))
    return [], False

print("=" * 60)
print("PLANNING vs PREDICTION")
print("=" * 60)

grid = create_grid(8)
start = (0, 0)
goal = (7, 7)

greedy_path, greedy_ok = greedy_agent(grid, start, goal)
planned_path, plan_ok = bfs_planner(grid, start, goal)

print_grid(grid, greedy_path,
           f"Greedy (prediction): {len(greedy_path)} steps, "
           f"{'reached goal' if greedy_ok else 'STUCK!'}")
print_grid(grid, planned_path,
           f"BFS (planning): {len(planned_path)} steps, "
           f"{'reached goal' if plan_ok else 'FAILED'}")

# --- Minimax for adversarial planning ---
print(f"\n{'=' * 60}")
print("MINIMAX: Planning with an Adversary")
print("=" * 60)

def minimax(node, depth, is_max, tree, alpha=-np.inf, beta=np.inf):
    """Minimax with alpha-beta pruning."""
    if depth == 0 or node not in tree or not tree[node]:
        return node, []
    children = tree[node]
    best_path = []
    if is_max:
        best_val = -np.inf
        for child in children:
            val, path = minimax(child, depth-1, False, tree, alpha, beta)
            if val > best_val:
                best_val = val
                best_path = [child] + path
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best_val, best_path
    else:
        best_val = np.inf
        for child in children:
            val, path = minimax(child, depth-1, True, tree, alpha, beta)
            if val < best_val:
                best_val = val
                best_path = [child] + path
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best_val, best_path

# Simple game tree (values at leaves)
tree = {
    "root": ["A", "B"],
    "A": ["A1", "A2"],
    "B": ["B1", "B2"],
    "A1": [3, 5], "A2": [6, 9],
    "B1": [1, 2], "B2": [0, 7],
    3: [], 5: [], 6: [], 9: [],
    1: [], 2: [], 0: [], 7: [],
}

print("\n  Game tree:")
print("          root")
print("         /    \\")
print("        A      B")
print("       / \\    / \\")
print("     A1  A2  B1  B2")
print("    /\\  /\\  /\\  /\\")
print("   3  5 6 9 1 2 0  7")

val, path = minimax("root", 4, True, tree)
print(f"\n  Minimax value: {val}")
print(f"  Optimal path: root -> {' -> '.join(str(p) for p in path)}")

# Greedy comparison
greedy_val = max(3, 5, 6, 9, 1, 2, 0, 7)  # greedy picks max leaf
print(f"  Greedy would aim for: {greedy_val} (but opponent blocks!)")
print(f"  Planning guarantees: {val} (best achievable against optimal opponent)")

# --- Lookahead depth matters ---
print(f"\n{'=' * 60}")
print("LOOKAHEAD DEPTH vs DECISION QUALITY")
print("=" * 60)

# Random game: sequence of choices, some lead to traps
n_decisions = 10
n_choices = 3
# Generate random reward landscape
rewards = np.random.randn(n_decisions, n_choices)
# Plant a trap: option 0 at step 3 looks great but leads to terrible step 4
rewards[3, 0] = 2.0   # tempting
rewards[4, :] = [-5.0, -5.0, 0.5]  # all bad after trap, except choice 2

def evaluate_with_lookahead(rewards, depth):
    """Greedy with varying lookahead depth."""
    total = 0
    for step in range(rewards.shape[0]):
        # Look ahead 'depth' steps from current
        best_choice = 0
        best_value = -np.inf
        for c in range(rewards.shape[1]):
            value = rewards[step, c]
            # Simple lookahead: average of best choices ahead
            for d in range(1, min(depth, rewards.shape[0] - step)):
                value += np.max(rewards[step + d]) * (0.9 ** d)
            if value > best_value:
                best_value = value
                best_choice = c
        total += rewards[step, best_choice]
    return total

print(f"\n{'Lookahead':>10s} {'Total Score':>12s} {'Quality':>10s}")
print("-" * 36)
scores = {}
for depth in [1, 2, 3, 5, 10]:
    score = evaluate_with_lookahead(rewards, depth)
    scores[depth] = score
    bar = "#" * max(0, int((score + 5) * 3))
    print(f"{depth:>10d} {score:>12.2f}   |{bar}")
print(f"\n  Depth 1 (greedy) hits the trap. Deeper lookahead avoids it.")
```

---

## Key Takeaways

- **Prediction is local; planning is global.** Next-token prediction optimizes for immediate plausibility, while planning optimizes for long-term outcome quality.
- **Greedy choices can be locally optimal but globally catastrophic.** When short-term and long-term objectives conflict, prediction fails and planning succeeds.
- **Planning requires evaluating future consequences.** Whether through tree search, simulation, or learned value functions, planning looks ahead before committing to a decision.
- **Adversarial settings demand planning.** Against an opponent who exploits greedy strategies, minimax-style planning is necessary to guarantee acceptable outcomes.
- **Modern reasoning models bridge prediction and planning.** Models like o1 generate extended reasoning traces that implicitly perform lookahead search, combining the fluency of language models with the foresight of planners.
