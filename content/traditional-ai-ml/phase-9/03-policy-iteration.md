# Policy Iteration

> Phase 9 — Reinforcement Learning | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

Value iteration finds the optimal value function directly, but it mixes two operations in every sweep: evaluating how good the current strategy is and improving the strategy. **Policy iteration** cleanly separates these two steps. First, **policy evaluation** computes the exact value function for the current policy. Then, **policy improvement** uses those values to find a better policy. Repeat until the policy stops changing — that is the optimal policy.

This separation has practical benefits. Policy evaluation solves a system of linear equations (or iterates until convergence), giving the exact values for the current policy. Policy improvement is then a single greedy step: for each state, pick the action with the highest expected value. The algorithm converges in surprisingly few iterations — often far fewer than value iteration — because policy space is finite and each improvement step is guaranteed to be strict (unless we are already optimal).

Policy iteration is also conceptually cleaner. It makes explicit the cycle that underlies all reinforcement learning: evaluate your current strategy, then improve it. This evaluate-improve loop appears everywhere — from tabular methods to deep RL algorithms like actor-critic.

### Why naive approaches fail

Trying all possible policies is combinatorially explosive — in an MDP with S states and A actions, there are A^S possible deterministic policies. Even a modest 100-state, 4-action problem has 4^100 policies. Policy iteration avoids brute force by making guaranteed improvements at each step, typically converging in a handful of iterations regardless of the state space size.

### Mental models

- **Exam and study cycle**: Policy evaluation is like taking an exam to see your current score. Policy improvement is like adjusting your study plan based on the results. Repeat until your score stops improving.
- **Corporate strategy review**: Evaluate current operations (compute V), then restructure departments that underperform (improve policy). Each review-restructure cycle makes the company better.
- **Newton's method for policies**: Just as Newton's method converges in few steps by using exact derivative information, policy iteration converges fast because policy evaluation gives exact values.

### Visual explanations

```
Policy Iteration Algorithm:

  Start with arbitrary policy pi_0

  Repeat:
    1. POLICY EVALUATION
       Compute V^{pi}(s) for all s
       (solve: V(s) = sum_{s'} T(s, pi(s), s')[R + gamma * V(s')])

    2. POLICY IMPROVEMENT
       For each state s:
         pi_new(s) = argmax_a sum_{s'} T(s,a,s')[R + gamma * V(s')]

    3. If pi_new == pi: STOP (optimal!)
       Else: pi = pi_new, go to step 1

Convergence example:
  Iteration 0: pi = [RIGHT, RIGHT, RIGHT, ...]  --> V = [0.1, 0.3, 0.5, ...]
  Iteration 1: pi = [DOWN,  RIGHT, RIGHT, ...]  --> V = [0.4, 0.5, 0.6, ...]
  Iteration 2: pi = [DOWN,  DOWN,  RIGHT, ...]  --> V = [0.6, 0.7, 0.8, ...]
  Iteration 3: pi = [DOWN,  DOWN,  RIGHT, ...]  --> No change! Optimal.

  Typical convergence: 3-10 iterations (even for large state spaces)
```

---

## Hands-on Exploration

1. Start with a random policy in a 3x3 grid. Compute V(s) for each state by solving the Bellman equations (treat it as 9 equations with 9 unknowns).
2. Using those V values, greedily improve the policy. Did any states change their action?
3. Evaluate the new policy. Did the values increase? (They must, by the policy improvement theorem.)
4. Compare the number of iterations policy iteration takes vs. the number of sweeps value iteration takes to reach the same answer. Which converges faster in terms of iterations?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- 4x4 Grid World MDP ---
GRID_ROWS = 4
GRID_COLS = 4
N_STATES = GRID_ROWS * GRID_COLS
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_ACTIONS = len(ACTIONS)
GOAL_STATE = 15
TRAP_STATE = 11
TERMINAL_STATES = {GOAL_STATE, TRAP_STATE}
STEP_REWARD = -0.04
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0

# @param gamma float 0.5 0.99 0.9
gamma = 0.9
# @param eval_sweeps int 1 200 50
eval_sweeps = 50  # max sweeps for policy evaluation
slip_prob = 0.1

def state_to_rc(s):
    return s // GRID_COLS, s % GRID_COLS

def rc_to_state(r, c):
    return r * GRID_COLS + c

def get_transitions(state, action_idx):
    if state in TERMINAL_STATES:
        return [(state, 1.0, 0.0)]
    r, c = state_to_rc(state)
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    intended = action_idx
    perp = [2, 3] if action_idx in [0, 1] else [0, 1]
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

# --- Policy Evaluation ---
def policy_evaluation(policy, gamma, tol=1e-6):
    V = np.zeros(N_STATES)
    for sweep in range(eval_sweeps):
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            a = policy[s]
            v_new = sum(prob * (rew + gamma * V[ns]) for ns, prob, rew in get_transitions(s, a))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < tol:
            break
    return V, sweep + 1

# --- Policy Improvement ---
def policy_improvement(V, gamma):
    new_policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        q_values = []
        for a in range(N_ACTIONS):
            q = sum(prob * (rew + gamma * V[ns]) for ns, prob, rew in get_transitions(s, a))
            q_values.append(q)
        new_policy[s] = np.argmax(q_values)
    return new_policy

# --- Policy Iteration ---
print("=== Policy Iteration ===")
print(f"Gamma: {gamma}\n")

# Start with a random policy
policy = np.random.randint(0, N_ACTIONS, size=N_STATES)

arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}

for iteration in range(20):
    # Step 1: Policy Evaluation
    V, eval_iters = policy_evaluation(policy, gamma)

    # Step 2: Policy Improvement
    new_policy = policy_improvement(V, gamma)

    # Check convergence
    policy_changed = not np.array_equal(policy, new_policy)
    n_changed = np.sum(policy != new_policy)

    print(f"Iteration {iteration}: eval took {eval_iters} sweeps, "
          f"policy changes: {n_changed}, mean V: {np.mean(V):.4f}")

    if not policy_changed:
        print(f"\nPolicy converged after {iteration} improvement steps!")
        break

    policy = new_policy

# --- Display final value function ---
print("\n=== Final Value Function V*(s) ===")
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

# --- Display final policy ---
print("\n=== Optimal Policy ===")
for r in range(GRID_ROWS):
    row_policy = []
    for c in range(GRID_COLS):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_policy.append("G")
        elif s == TRAP_STATE:
            row_policy.append("X")
        else:
            row_policy.append(arrows[policy[s]])
    print("  ".join(row_policy))

# --- Compare with Value Iteration ---
print("\n=== Comparison: Value Iteration ===")
V_vi = np.zeros(N_STATES)
for sweep in range(200):
    delta = 0.0
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        v_old = V_vi[s]
        V_vi[s] = max(
            sum(prob * (rew + gamma * V_vi[ns]) for ns, prob, rew in get_transitions(s, a))
            for a in range(N_ACTIONS)
        )
        delta = max(delta, abs(V_vi[s] - v_old))
    if delta < 1e-6:
        print(f"Value iteration converged after {sweep + 1} sweeps")
        break

print(f"Policy iteration total improvement steps: {iteration}")
print(f"Max difference in V: {np.max(np.abs(V - V_vi)):.8f}")
```

---

## Key Takeaways

- **Policy iteration separates evaluation from improvement.** This clean decomposition makes each step easier to understand and implement.
- **Policy evaluation computes exact values for a fixed policy.** It solves the Bellman equation for the current policy, not the optimality equation.
- **Policy improvement is greedy.** It picks the best action according to the current value function — no lookahead beyond one step is needed.
- **Convergence is fast.** Policy iteration typically converges in very few iterations (often under 10), even for large state spaces, because each improvement step is guaranteed to strictly increase performance.
- **Policy iteration and value iteration reach the same answer.** They differ in computational trade-offs: policy iteration does more work per iteration but needs fewer iterations.
