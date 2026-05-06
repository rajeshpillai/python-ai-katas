# Value Functions

> Phase 9 — Reinforcement Learning | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

Given an MDP, we need a way to evaluate how good it is to be in a particular state, or to take a particular action in a state. **Value functions** provide exactly this. The **state-value function** V(s) tells us the expected total discounted reward starting from state s and following a given policy. The **action-value function** Q(s, a) tells us the expected return of taking action a in state s and then following the policy.

These functions are the backbone of reinforcement learning. If we know the true value function, we can act optimally by always choosing the action that leads to the highest-value next state. The challenge is computing these values, which is where the **Bellman equations** come in — recursive relationships that express the value of a state in terms of the values of its successor states.

The Bellman equation says: the value of a state equals the immediate reward plus the discounted value of the next state, averaged over all possible transitions. This recursive structure means we can solve for values iteratively — start with a guess, apply the Bellman update repeatedly, and converge to the true values.

### Why naive approaches fail

You might try to estimate V(s) by running many simulations from each state and averaging the returns. This Monte Carlo approach works but is slow — it requires complete episodes, and high-variance returns make convergence sluggish. The Bellman equations exploit the recursive structure to propagate value information efficiently across the entire state space in each sweep, converging much faster.

### Mental models

- **House prices by neighborhood**: The value of a house depends on the house itself plus the value of the neighborhood (neighboring states). Bellman equations compute values by looking at neighbors.
- **Dominoes falling**: Updating the value of one state triggers updates in adjacent states — information propagates through the chain.
- **V(s) is a report card for states; Q(s,a) is a report card for state-action pairs.** Both grade how good things are from here on out.

### Visual explanations

```
Bellman equation for V(s) under policy pi:

  V(s) = sum over actions a of:
           pi(a|s) * sum over next states s' of:
             T(s,a,s') * [ R(s,a,s') + gamma * V(s') ]

In plain English:
  Value of s = weighted average over actions of
               (expected immediate reward + discounted value of where we land)

Example (2-state chain):
  S0 --action--> S1 --action--> S1 (terminal, reward=+1)

  V(S1) = 1.0  (terminal reward)
  V(S0) = 0 + gamma * V(S1) = gamma * 1.0

  If gamma=0.9: V(S0) = 0.9
  If gamma=0.5: V(S0) = 0.5

Value iteration convergence:
  Sweep 0: V = [0.0, 0.0, 0.0, 0.0, ...]   (initial guess)
  Sweep 1: V = [0.0, 0.0, 0.0, 0.9, ...]   (goal neighbor updated)
  Sweep 2: V = [0.0, 0.0, 0.7, 0.9, ...]   (propagates further)
  ...
  Sweep N: V = [0.6, 0.7, 0.8, 0.9, ...]   (converged)
```

---

## Hands-on Exploration

1. Draw a 3-state MDP on paper. Assign rewards and transitions. Write the Bellman equation for each state and solve the system of equations by hand.
2. Start with V(s) = 0 for all states. Apply one Bellman update sweep. Then another. Track how values propagate from the rewarding state outward.
3. Compare V(s) values for gamma=0.99 vs gamma=0.5. Which states change the most? Why?
4. Compute Q(s, a) for a state with two actions — one risky (high reward but might lead to penalty) and one safe (low reward). See how gamma affects which action looks better.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Define a 4x4 Grid World MDP ---
GRID_ROWS = 4
GRID_COLS = 4
N_STATES = GRID_ROWS * GRID_COLS
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_ACTIONS = len(ACTIONS)

GOAL_STATE = 15
TRAP_STATE = 11
STEP_REWARD = -0.04
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
TERMINAL_STATES = {GOAL_STATE, TRAP_STATE}

# @param gamma float 0.0 1.0 0.9
gamma = 0.9
# @param n_sweeps int 1 100 30
n_sweeps = 30
slip_prob = 0.1

def state_to_rc(s):
    return s // GRID_COLS, s % GRID_COLS

def rc_to_state(r, c):
    return r * GRID_COLS + c

def get_transitions(state, action_idx):
    """Return list of (next_state, probability, reward)."""
    if state in TERMINAL_STATES:
        return [(state, 1.0, 0.0)]
    r, c = state_to_rc(state)
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    intended = action_idx
    if action_idx in [0, 1]:
        perp = [2, 3]
    else:
        perp = [0, 1]
    outcomes = []
    for direction, prob in [(intended, 1.0 - 2*slip_prob)] + [(p, slip_prob) for p in perp]:
        dr, dc = deltas[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
            nr, nc = r, c
        ns = rc_to_state(nr, nc)
        reward = GOAL_REWARD if ns == GOAL_STATE else (TRAP_REWARD if ns == TRAP_STATE else STEP_REWARD)
        outcomes.append((ns, prob, reward))
    return outcomes

# --- Value Iteration: compute optimal V(s) ---
V = np.zeros(N_STATES)

print("=== Value Iteration ===")
print(f"Gamma: {gamma}, Sweeps: {n_sweeps}\n")

for sweep in range(n_sweeps):
    V_new = np.copy(V)
    max_delta = 0.0
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        # Bellman optimality: V(s) = max_a sum_s' T(s,a,s')[R + gamma*V(s')]
        action_values = []
        for a in range(N_ACTIONS):
            q_sa = 0.0
            for (ns, prob, rew) in get_transitions(s, a):
                q_sa += prob * (rew + gamma * V[ns])
            action_values.append(q_sa)
        V_new[s] = max(action_values)
        max_delta = max(max_delta, abs(V_new[s] - V[s]))
    V = V_new
    if sweep < 5 or sweep == n_sweeps - 1:
        print(f"Sweep {sweep:2d} | max delta: {max_delta:.6f}")

# --- Display V(s) as a grid ---
print("\n=== Optimal State Values V*(s) ===")
for r in range(GRID_ROWS):
    row_vals = []
    for c in range(GRID_COLS):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_vals.append(" GOAL ")
        elif s == TRAP_STATE:
            row_vals.append(" TRAP ")
        else:
            row_vals.append(f"{V[s]:+.3f}")
    print("  ".join(row_vals))

# --- Compute Q(s,a) and extract optimal policy ---
print("\n=== Optimal Action-Values Q*(s,a) for state 0 ===")
s_example = 0
for a in range(N_ACTIONS):
    q_sa = 0.0
    for (ns, prob, rew) in get_transitions(s_example, a):
        q_sa += prob * (rew + gamma * V[ns])
    print(f"  Q({s_example}, {ACTIONS[a]:>5s}) = {q_sa:+.4f}")

# --- Extract and display the optimal policy ---
print("\n=== Optimal Policy ===")
arrows = {'UP': '^', 'DOWN': 'v', 'LEFT': '<', 'RIGHT': '>'}
for r in range(GRID_ROWS):
    row_policy = []
    for c in range(GRID_COLS):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_policy.append("G")
        elif s == TRAP_STATE:
            row_policy.append("X")
        else:
            q_values = []
            for a in range(N_ACTIONS):
                q = sum(prob * (rew + gamma * V[ns]) for ns, prob, rew in get_transitions(s, a))
                q_values.append(q)
            best_action = ACTIONS[np.argmax(q_values)]
            row_policy.append(arrows[best_action])
    print("  ".join(row_policy))

# --- Show Bellman equation convergence ---
print("\n=== Bellman Equation Check (state 0) ===")
s = 0
best_q = -np.inf
for a in range(N_ACTIONS):
    q = sum(prob * (rew + gamma * V[ns]) for ns, prob, rew in get_transitions(s, a))
    if q > best_q:
        best_q = q
print(f"V*(0) from table:    {V[0]:+.4f}")
print(f"V*(0) from Bellman:  {best_q:+.4f}")
print(f"Match: {abs(V[0] - best_q) < 1e-6}")
```

---

## Key Takeaways

- **V(s) measures how good a state is; Q(s,a) measures how good a state-action pair is.** Both compute expected discounted returns.
- **The Bellman equation is a recursive relationship.** The value of a state depends on the values of its successors, weighted by transition probabilities.
- **Value iteration applies the Bellman optimality update repeatedly until convergence.** Each sweep propagates reward information one step further.
- **From V* or Q*, we can extract the optimal policy.** Just pick the action that maximizes Q(s,a) in each state.
- **Convergence is guaranteed for finite MDPs with gamma < 1.** The contraction mapping theorem ensures the iterates converge to the unique fixed point.
