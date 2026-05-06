# Markov Decision Processes

> Phase 9 — Reinforcement Learning | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

In supervised learning, we have labeled examples: input-output pairs that tell the model exactly what the right answer is. But many real-world problems do not come with labels. An agent playing a game, a robot navigating a warehouse, or a thermostat controlling temperature must make **sequences of decisions** where the consequences of each action unfold over time. The right action now depends on what happens later.

A **Markov Decision Process (MDP)** provides the mathematical framework for these sequential decision problems. It defines a set of **states** (where the agent can be), **actions** (what it can do), **transition probabilities** (how the world changes), and **rewards** (what the agent gets). The "Markov" property means the future depends only on the current state, not the full history — a simplification that makes the math tractable.

Understanding MDPs is essential because every reinforcement learning algorithm — Q-learning, policy gradient, actor-critic — assumes the problem can be modeled as an MDP (or a variant of one). Getting the MDP formulation right is half the battle.

### Why naive approaches fail

Without the MDP framework, you might try to optimize each decision greedily — always pick the action with the highest immediate reward. But this ignores delayed consequences. A chess player who only captures the nearest piece will lose to one who sacrifices a pawn for a checkmate three moves later. The MDP framework introduces the **discount factor** to balance immediate and future rewards, and transition probabilities to handle uncertainty.

### Mental models

- **Board game**: Each board position is a state. Each legal move is an action. Rolling dice determines the transition. Points scored are rewards.
- **GPS navigation**: Each intersection is a state. Turning left/right/straight are actions. Traffic conditions create stochastic transitions. Arrival time is the (negative) reward.
- **The Markov property as amnesia**: The agent only needs to look at the current snapshot, not remember how it got there — like waking up in a room and deciding what to do based solely on what you see now.

### Visual explanations

```
A simple 4-state MDP (Grid World):

    +-------+-------+
    |       |       |
    |  S0   |  S1   |
    | start |       |
    +-------+-------+
    |       |       |
    |  S2   |  S3   |
    |       | goal! |
    +-------+-------+

Actions: UP, DOWN, LEFT, RIGHT
Transitions: 80% intended direction, 10% each perpendicular
Rewards: +1 at S3 (goal), -0.04 per step (encourages efficiency)

Discount factor (gamma):
  gamma = 1.0  -->  agent values future rewards equally
  gamma = 0.5  -->  reward 3 steps away is worth 0.5^3 = 0.125
  gamma = 0.0  -->  agent is completely myopic (greedy)
```

---

## Hands-on Exploration

1. Define a small grid world on paper (3x3). Label states, choose actions, assign rewards. Write out the transition table for one state assuming deterministic moves.
2. Add stochasticity: with probability 0.8 the agent moves as intended, 0.1 it slips left, 0.1 it slips right. Rewrite the transition table.
3. Trace through a trajectory (sequence of states) from start to goal. Compute the total discounted return for gamma=0.9 and gamma=0.5. Notice how gamma changes which paths look attractive.
4. Consider a state near a penalty. With a high slip probability, is it still safe to walk past it? Think about how stochastic transitions affect strategy.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Define a simple 4x4 Grid World MDP ---
GRID_ROWS = 4
GRID_COLS = 4
N_STATES = GRID_ROWS * GRID_COLS
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_ACTIONS = len(ACTIONS)

# Special states
GOAL_STATE = 15       # bottom-right corner
TRAP_STATE = 11       # a penalty state
START_STATE = 0       # top-left corner

# Rewards
STEP_REWARD = -0.04   # small penalty per step to encourage efficiency
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0

# @param gamma float 0.0 1.0 0.9
gamma = 0.9  # discount factor
# @param slip_prob float 0.0 0.5 0.1
slip_prob = 0.1  # probability of slipping to each side

def state_to_rc(s):
    return s // GRID_COLS, s % GRID_COLS

def rc_to_state(r, c):
    return r * GRID_COLS + c

def step(state, action_idx):
    """Return list of (next_state, probability, reward) tuples."""
    if state == GOAL_STATE or state == TRAP_STATE:
        return [(state, 1.0, 0.0)]  # terminal — absorbing

    r, c = state_to_rc(state)
    deltas = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    # Intended direction + two perpendicular slips
    intended = action_idx
    if action_idx in [0, 1]:  # UP/DOWN -> slip LEFT/RIGHT
        perp = [2, 3]
    else:                      # LEFT/RIGHT -> slip UP/DOWN
        perp = [0, 1]

    outcomes = []
    for direction, prob in [(intended, 1.0 - 2 * slip_prob)] + [(p, slip_prob) for p in perp]:
        dr, dc = deltas[direction]
        nr, nc = r + dr, c + dc
        # Walls: stay in place if out of bounds
        if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
            nr, nc = r, c
        ns = rc_to_state(nr, nc)
        # Assign reward
        if ns == GOAL_STATE:
            reward = GOAL_REWARD
        elif ns == TRAP_STATE:
            reward = TRAP_REWARD
        else:
            reward = STEP_REWARD
        outcomes.append((ns, prob, reward))
    return outcomes

# --- Build full transition and reward tables ---
T = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # T[s, a, s']
R = np.zeros((N_STATES, N_ACTIONS, N_STATES))  # R[s, a, s']

for s in range(N_STATES):
    for a in range(N_ACTIONS):
        for (ns, prob, rew) in step(s, a):
            T[s, a, ns] += prob
            R[s, a, ns] = rew

# --- Simulate a random walk ---
def simulate_episode(max_steps=50):
    state = START_STATE
    trajectory = [state]
    total_return = 0.0
    discount = 1.0
    for t in range(max_steps):
        if state == GOAL_STATE or state == TRAP_STATE:
            break
        action = np.random.randint(N_ACTIONS)
        outcomes = step(state, action)
        probs = [o[1] for o in outcomes]
        idx = np.random.choice(len(outcomes), p=probs)
        next_state, _, reward = outcomes[idx]
        total_return += discount * reward
        discount *= gamma
        state = next_state
        trajectory.append(state)
    return trajectory, total_return

print("=== MDP Structure ===")
print(f"States: {N_STATES} (4x4 grid)")
print(f"Actions: {ACTIONS}")
print(f"Goal: state {GOAL_STATE}, Trap: state {TRAP_STATE}")
print(f"Discount factor (gamma): {gamma}")
print(f"Slip probability: {slip_prob}")
print()

# Run several episodes
print("=== Sample Episodes (random policy) ===")
returns = []
for ep in range(10):
    traj, ret = simulate_episode()
    returns.append(ret)
    path = ' -> '.join([f"({state_to_rc(s)[0]},{state_to_rc(s)[1]})" for s in traj])
    outcome = "GOAL" if traj[-1] == GOAL_STATE else ("TRAP" if traj[-1] == TRAP_STATE else "TIMEOUT")
    print(f"Ep {ep}: {outcome:7s} | Return={ret:+.3f} | {path}")

print(f"\nMean return over 10 episodes: {np.mean(returns):.3f}")

# Show transition probabilities for one state-action pair
s_example = 5
a_example = 3  # RIGHT
print(f"\n=== Transition probs from state {s_example} ({state_to_rc(s_example)}), action RIGHT ===")
for ns in range(N_STATES):
    if T[s_example, a_example, ns] > 0:
        print(f"  -> state {ns} ({state_to_rc(ns)}): prob={T[s_example, a_example, ns]:.2f}, reward={R[s_example, a_example, ns]:.2f}")
```

---

## Key Takeaways

- **An MDP has four components: states, actions, transitions, and rewards.** Together they fully specify a sequential decision problem.
- **The Markov property is the key assumption.** The next state depends only on the current state and action, not the full history.
- **The discount factor gamma controls how far-sighted the agent is.** Low gamma means greedy; high gamma means the agent plans ahead.
- **Stochastic transitions make planning harder.** The agent must reason about expected outcomes, not just best-case scenarios.
- **A random policy performs poorly.** This motivates the search for optimal policies, which is what the rest of reinforcement learning is about.
