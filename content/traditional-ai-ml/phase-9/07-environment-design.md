# Environment Design

> Phase 9 — Reinforcement Learning | Kata 9.7

---

## Concept & Intuition

### What problem are we solving?

In reinforcement learning, the algorithm gets most of the attention, but the **environment design** — how you define states, actions, rewards, and termination conditions — often matters more. A poorly designed reward function can cause an agent to find clever shortcuts that satisfy the reward signal without solving the actual problem. A sparse reward (only given at the end of a long episode) can make learning impossibly slow. An overly complex state space can make the problem intractable.

**Reward shaping** is the art of adding intermediate rewards that guide the agent toward good behavior without changing the optimal policy. **Sparse rewards** are the most natural (win/lose at the end) but the hardest to learn from. **Dense rewards** (feedback at every step) are easier to learn from but risk introducing unintended incentives. Getting this balance right is one of the most important practical skills in applied RL.

Environment abstraction also matters. Should a robot control individual motor voltages, or high-level actions like "move forward 1 meter"? The right abstraction level makes the problem learnable. Too low-level and the action space is enormous; too high-level and the agent lacks the flexibility to find good solutions.

### Why naive approaches fail

The classic failure mode is **reward hacking**: the agent finds a way to maximize the reward signal that violates the designer's intent. A cleaning robot rewarded for not seeing dirt might learn to close its eyes. A game-playing agent rewarded for score might exploit a bug instead of playing well. Sparse rewards cause the opposite problem: the agent wanders randomly for thousands of steps without any learning signal, making progress glacially slow.

### Mental models

- **Reward shaping as breadcrumbs**: Sparse reward is like hiding treasure with no map. Shaped reward leaves breadcrumbs that lead to the treasure without revealing its exact location.
- **Potential-based shaping**: The safe way to add shaping rewards is F(s, s') = gamma * Phi(s') - Phi(s), where Phi is a potential function. This provably preserves the optimal policy.
- **Environment design as curriculum**: Start with a simple version of the problem (dense rewards, small state space) and gradually increase difficulty — like teaching a student with easy exercises before hard ones.

### Visual explanations

```
Sparse vs. Dense vs. Shaped Rewards:

Sparse reward:
  Step 1: 0  Step 2: 0  Step 3: 0  ... Step 100: 0  Goal: +1
  Problem: Agent gets no signal for 100 steps. Learning is extremely slow.

Dense reward (naive):
  Step 1: -dist_to_goal  Step 2: -dist_to_goal  ...
  Problem: Agent might learn to orbit near the goal without reaching it
           (reducing distance while avoiding the terminal state).

Potential-based shaping (safe):
  Phi(s) = -distance_to_goal(s)
  F(s, s') = gamma * Phi(s') - Phi(s)
  Original reward + F(s, s') = shaped reward
  Guarantee: same optimal policy as original!

Reward hacking examples:
  Intent: "Clean the room"     Reward: "Minimize visible dirt"
  Hack: Cover dirt with a blanket

  Intent: "Move fast"          Reward: "High velocity"
  Hack: Spin in circles (high angular velocity, zero progress)

  Intent: "Win the game"       Reward: "High score"
  Hack: Exploit a score-duplication bug
```

---

## Hands-on Exploration

1. Build a simple grid world with a goal far from start. Train an agent with sparse reward (+1 at goal only). How many episodes does it take to start learning?
2. Add dense reward (-distance_to_goal at each step). Does learning speed up? Does the agent actually reach the goal, or does it exhibit any surprising behavior?
3. Implement potential-based shaping. Compare the learned policy to the sparse-reward case — are they the same?
4. Design a reward that accidentally encourages "reward hacking." For example, reward the agent for moving closer to the goal but do not penalize circling. What happens?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Grid World for Reward Shaping Experiments ---
GRID_SIZE = 8
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT
START = (0, 0)
GOAL = (7, 7)

# @param n_episodes int 200 3000 500
n_episodes = 500
# @param gamma float 0.5 0.99 0.99
gamma = 0.99
alpha = 0.1
epsilon = 0.15

def state_idx(r, c):
    return r * GRID_SIZE + c

def idx_to_rc(s):
    return s // GRID_SIZE, s % GRID_SIZE

def distance_to_goal(r, c):
    return abs(r - GOAL[0]) + abs(c - GOAL[1])  # Manhattan distance

def step_env(r, c, action):
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    dr, dc = deltas[action]
    nr, nc = r + dr, c + dc
    nr = max(0, min(GRID_SIZE - 1, nr))
    nc = max(0, min(GRID_SIZE - 1, nc))
    done = (nr, nc) == GOAL
    return nr, nc, done

def compute_reward_sparse(r, c, nr, nc, done):
    """Sparse: reward only at goal."""
    return 1.0 if done else 0.0

def compute_reward_dense(r, c, nr, nc, done):
    """Dense: negative distance to goal."""
    return 1.0 if done else -distance_to_goal(nr, nc) / (2 * GRID_SIZE)

def compute_reward_shaped(r, c, nr, nc, done):
    """Potential-based shaping (preserves optimal policy)."""
    phi_s = -distance_to_goal(r, c)
    phi_ns = -distance_to_goal(nr, nc)
    shaping = gamma * phi_ns - phi_s
    base_reward = 1.0 if done else 0.0
    return base_reward + shaping

def compute_reward_hackable(r, c, nr, nc, done):
    """Hackable: reward for getting closer but no step penalty."""
    dist_old = distance_to_goal(r, c)
    dist_new = distance_to_goal(nr, nc)
    return 1.0 if done else (dist_old - dist_new) * 0.5

def run_q_learning(reward_fn, n_episodes, label):
    """Run Q-learning with a given reward function."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = []
    goals_reached = []

    for ep in range(n_episodes):
        r, c = START
        total_return = 0.0

        for t in range(300):
            s = state_idx(r, c)
            # Epsilon-greedy
            if np.random.random() < epsilon:
                a = np.random.randint(N_ACTIONS)
            else:
                a = np.argmax(Q[s])

            nr, nc, done = step_env(r, c, a)
            ns = state_idx(nr, nc)
            reward = reward_fn(r, c, nr, nc, done)

            # Q-learning update
            td_target = reward + gamma * np.max(Q[ns]) * (1 - done)
            Q[s, a] += alpha * (td_target - Q[s, a])

            total_return += reward
            r, c = nr, nc

            if done:
                break

        episode_returns.append(total_return)
        goals_reached.append(1 if done else 0)

    return Q, episode_returns, goals_reached

# --- Run experiments ---
reward_configs = [
    ("Sparse", compute_reward_sparse),
    ("Dense", compute_reward_dense),
    ("Shaped (potential)", compute_reward_shaped),
    ("Hackable", compute_reward_hackable),
]

print(f"=== Environment Design: Reward Shaping ===")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Start: {START}, Goal: {GOAL}")
print(f"Episodes: {n_episodes}, Gamma: {gamma}\n")

results = {}
for label, reward_fn in reward_configs:
    Q, returns, goals = run_q_learning(reward_fn, n_episodes, label)
    results[label] = (Q, returns, goals)

# --- Compare goal-reaching rates ---
print(f"{'Reward Type':>22}  {'First 100':>10}  {'Last 100':>10}  {'Total Goals':>12}")
print("-" * 60)
for label in results:
    Q, returns, goals = results[label]
    first = sum(goals[:100])
    last = sum(goals[-100:])
    total = sum(goals)
    print(f"{label:>22}  {first:>10}  {last:>10}  {total:>12}")

# --- Show learned policies ---
arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}

for label in ["Sparse", "Shaped (potential)"]:
    Q = results[label][0]
    print(f"\n=== {label} Policy ===")
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            if (r, c) == GOAL:
                row.append("G")
            else:
                s = state_idx(r, c)
                row.append(arrows[np.argmax(Q[s])])
        print(" ".join(row))

# --- Analyze reward hacking ---
print("\n=== Reward Hacking Analysis ===")
Q_hack = results["Hackable"][0]
goals_hack = results["Hackable"][2]
print(f"Hackable reward reached goal in {sum(goals_hack)}/{n_episodes} episodes")

# Check if agent oscillates near the goal
test_r, test_c = GOAL[0] - 1, GOAL[1]  # one step from goal
s = state_idx(test_r, test_c)
print(f"\nQ-values one step from goal (state ({test_r},{test_c})):")
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
for a in range(N_ACTIONS):
    print(f"  {action_names[a]:>5}: {Q_hack[s, a]:+.3f}")
best_action = action_names[np.argmax(Q_hack[s])]
print(f"  Best action: {best_action} {'(toward goal!)' if best_action == 'DOWN' else '(not toward goal?!)'}")

# --- Key insight: potential-based shaping preserves optimality ---
print("\n=== Optimality Check ===")
Q_sparse = results["Sparse"][0]
Q_shaped = results["Shaped (potential)"][0]
policy_match = 0
for s in range(N_STATES):
    if np.argmax(Q_sparse[s]) == np.argmax(Q_shaped[s]):
        policy_match += 1
print(f"Sparse vs Shaped policy agreement: {policy_match}/{N_STATES} states "
      f"({100*policy_match/N_STATES:.0f}%)")
print("(Potential-based shaping guarantees the same optimal policy)")
```

---

## Key Takeaways

- **Environment design often matters more than algorithm choice.** A well-shaped reward with a simple algorithm beats a sophisticated algorithm with a bad reward.
- **Sparse rewards are natural but hard to learn from.** The agent gets no learning signal until it stumbles onto success, which can take a very long time.
- **Potential-based reward shaping is provably safe.** It speeds up learning by adding intermediate signals without changing the optimal policy.
- **Reward hacking is a real and dangerous failure mode.** Agents optimize exactly what you measure, not what you intend. Always check for unintended shortcuts.
- **The right level of abstraction makes problems tractable.** Choosing appropriate state representations, action spaces, and episode boundaries is a design decision that profoundly affects learnability.
