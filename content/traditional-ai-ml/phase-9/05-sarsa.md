# SARSA

> Phase 9 — Reinforcement Learning | Kata 9.5

---

## Concept & Intuition

### What problem are we solving?

Q-learning updates toward the **best possible** next action, regardless of what the agent actually does. This is powerful but can be dangerous: if the agent follows an exploratory policy that occasionally takes risky actions near cliffs or traps, Q-learning's values may be overly optimistic because they assume the agent will act optimally in the future (which it will not, since it is still exploring).

**SARSA** (State-Action-Reward-State-Action) is an **on-policy** TD method that fixes this. Instead of using max_a' Q(s', a'), it uses the Q-value of the action the agent **actually takes** in the next state. The update becomes: Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)], where a' is the actual next action chosen by the policy (including exploration). This means SARSA's values reflect the behavior policy's actual performance, including its mistakes from exploration.

The practical difference emerges in environments with danger zones. Q-learning learns the optimal path even if it runs along a cliff edge, because it assumes perfect future execution. SARSA learns a safer path that accounts for the fact that epsilon-greedy exploration might accidentally step off the cliff.

### Why naive approaches fail

If you use Q-learning with a high exploration rate near dangerous states, the agent may learn a theoretically optimal but practically dangerous policy. When deployed (even with a small epsilon for continued learning), the agent occasionally falls off cliffs it learned to walk beside. SARSA's on-policy learning naturally produces safer, more conservative policies that account for the exploration noise inherent in the behavior policy.

### Mental models

- **The name tells the story**: SARSA stands for (S, A, R, S', A') — the five elements needed for one update. You need to know the next action A' before updating, which is why it is on-policy.
- **Cautious vs. bold driver**: Q-learning is like a driver who plans the fastest route assuming perfect driving. SARSA is like a driver who knows they sometimes swerve, so they avoid roads next to cliffs.
- **You learn from what you actually do, not from what you could do**: On-policy means your value estimates match your actual behavior, including exploration noise.

### Visual explanations

```
SARSA vs Q-Learning Update:

  SARSA (on-policy):
    Q(S,A) += alpha * [R + gamma * Q(S', A') - Q(S,A)]
                                      ^^^^
                                      actual next action (from policy)

  Q-learning (off-policy):
    Q(S,A) += alpha * [R + gamma * max_a' Q(S', a') - Q(S,A)]
                                   ^^^^^^^^^^^^^^^^^
                                   best possible next action

The Cliff Walking Problem:

    S . . . . . . . . . . G     S = Start, G = Goal
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    X X X X X X X X X X X X     X = Cliff (reward = -100)

    Q-learning path:  S . . . . . . . . . . G   (optimal but risky)
                                              |
                      along the cliff edge ----

    SARSA path:       S . . . . . . . . . . .   (safer, avoids cliff)
                      . . . . . . . . . . . G
                      further from the cliff ---
```

---

## Hands-on Exploration

1. Draw the cliff walking environment on paper. Trace what happens when an epsilon-greedy agent walks along the cliff edge — how often does it fall?
2. Run Q-learning and SARSA on the cliff problem. Compare their learned paths. Which is safer?
3. Set epsilon=0 for both algorithms after training. Now are their policies the same? What does this tell you about on-policy vs. off-policy learning?
4. Increase epsilon from 0.1 to 0.3. Does the gap between Q-learning and SARSA paths grow? Why?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Cliff Walking Environment ---
# 4 rows x 12 columns grid
ROWS = 4
COLS = 12
N_STATES = ROWS * COLS
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_ACTIONS = len(ACTIONS)

START = (3, 0)   # bottom-left
GOAL = (3, 11)   # bottom-right
CLIFF = [(3, c) for c in range(1, 11)]  # bottom row (except start and goal)

# @param epsilon float 0.01 0.5 0.1
epsilon = 0.1
# @param alpha float 0.01 1.0 0.5
alpha = 0.5
# @param gamma float 0.5 1.0 1.0
gamma = 1.0
# @param n_episodes int 100 2000 500
n_episodes = 500

def state_idx(r, c):
    return r * COLS + c

def idx_to_rc(s):
    return s // COLS, s % COLS

def step(r, c, action_idx):
    """Take action, return (new_r, new_c, reward, done)."""
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    dr, dc = deltas[action_idx]
    nr, nc = r + dr, c + dc
    # Walls
    nr = max(0, min(ROWS - 1, nr))
    nc = max(0, min(COLS - 1, nc))
    # Check cliff
    if (nr, nc) in CLIFF:
        return START[0], START[1], -100, False  # fall, back to start
    # Check goal
    if (nr, nc) == GOAL:
        return nr, nc, -1, True
    return nr, nc, -1, False

def epsilon_greedy(Q, s, eps):
    if np.random.random() < eps:
        return np.random.randint(N_ACTIONS)
    return np.argmax(Q[s])

# --- Run both algorithms ---
def run_q_learning(n_episodes, epsilon, alpha, gamma):
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards_per_ep = []
    for ep in range(n_episodes):
        r, c = START
        total_reward = 0
        for t in range(500):
            s = state_idx(r, c)
            a = epsilon_greedy(Q, s, epsilon)
            nr, nc, reward, done = step(r, c, a)
            ns = state_idx(nr, nc)
            # Q-learning: use max over next actions
            td_target = reward + gamma * np.max(Q[ns]) * (1 - done)
            Q[s, a] += alpha * (td_target - Q[s, a])
            total_reward += reward
            r, c = nr, nc
            if done:
                break
        rewards_per_ep.append(total_reward)
    return Q, rewards_per_ep

def run_sarsa(n_episodes, epsilon, alpha, gamma):
    Q = np.zeros((N_STATES, N_ACTIONS))
    rewards_per_ep = []
    for ep in range(n_episodes):
        r, c = START
        s = state_idx(r, c)
        a = epsilon_greedy(Q, s, epsilon)
        total_reward = 0
        for t in range(500):
            nr, nc, reward, done = step(r, c, a)
            ns = state_idx(nr, nc)
            # SARSA: use the actual next action
            a_next = epsilon_greedy(Q, ns, epsilon)
            td_target = reward + gamma * Q[ns, a_next] * (1 - done)
            Q[s, a] += alpha * (td_target - Q[s, a])
            total_reward += reward
            r, c = nr, nc
            s = ns
            a = a_next
            if done:
                break
        rewards_per_ep.append(total_reward)
    return Q, rewards_per_ep

Q_ql, rewards_ql = run_q_learning(n_episodes, epsilon, alpha, gamma)
Q_sarsa, rewards_sarsa = run_sarsa(n_episodes, epsilon, alpha, gamma)

# --- Compare learning curves ---
print("=== SARSA vs Q-Learning on Cliff Walking ===")
print(f"Epsilon: {epsilon}, Alpha: {alpha}, Gamma: {gamma}")
print(f"Episodes: {n_episodes}\n")

window = 50
print(f"{'Period':>12}  {'Q-Learning':>12}  {'SARSA':>12}")
print("-" * 40)
for start in range(0, n_episodes, n_episodes // 5):
    end = min(start + window, n_episodes)
    mr_ql = np.mean(rewards_ql[start:end])
    mr_sarsa = np.mean(rewards_sarsa[start:end])
    print(f"Ep {start:>4}-{end:<4}  {mr_ql:>12.1f}  {mr_sarsa:>12.1f}")

# --- Display learned policies ---
arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}

def display_policy(Q, name):
    print(f"\n=== {name} Policy ===")
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            if (r, c) == GOAL:
                row.append("G")
            elif (r, c) in CLIFF:
                row.append("X")
            elif (r, c) == START:
                row.append(arrows[np.argmax(Q[state_idx(r, c)])])
            else:
                row.append(arrows[np.argmax(Q[state_idx(r, c)])])
        print(" ".join(row))

display_policy(Q_ql, "Q-Learning")
display_policy(Q_sarsa, "SARSA")

# --- Trace greedy paths ---
def trace_path(Q, label):
    r, c = START
    path = [(r, c)]
    for t in range(50):
        s = state_idx(r, c)
        a = np.argmax(Q[s])
        nr, nc, reward, done = step(r, c, a)
        path.append((nr, nc))
        r, c = nr, nc
        if done or (r, c) == START:
            break
    print(f"\n{label} greedy path:")
    path_str = " -> ".join([f"({r},{c})" for r, c in path])
    print(f"  {path_str}")
    # Which row does the path use?
    rows_used = set(r for r, c in path[1:-1])
    if 3 in rows_used and 2 not in rows_used:
        print(f"  --> Walks along cliff edge (row 3)")
    elif 2 in rows_used:
        print(f"  --> Takes safer path (row 2, away from cliff)")

trace_path(Q_ql, "Q-Learning")
trace_path(Q_sarsa, "SARSA")

# --- Final comparison ---
print(f"\n=== Final 50-Episode Average Reward ===")
print(f"Q-Learning: {np.mean(rewards_ql[-50:]):.1f}")
print(f"SARSA:      {np.mean(rewards_sarsa[-50:]):.1f}")
print(f"\nNote: SARSA often gets better *online* reward because its policy")
print(f"accounts for exploration noise (safer path = fewer cliff falls).")
```

---

## Key Takeaways

- **SARSA is on-policy: it learns about the policy it is actually following.** This means its Q-values reflect the exploration noise baked into epsilon-greedy behavior.
- **Q-learning is off-policy: it learns about the optimal policy regardless of the behavior policy.** This can lead to optimistic values near dangerous states.
- **In safe environments, Q-learning and SARSA converge to similar policies.** The difference emerges near cliffs, traps, or any states where exploration can be costly.
- **SARSA produces safer, more conservative policies.** When exploration might cause the agent to stumble into danger, SARSA learns to keep its distance.
- **The choice between on-policy and off-policy depends on the application.** If you will deploy with epsilon > 0 (continued learning), SARSA's caution is valuable. If you will deploy greedily (epsilon = 0), Q-learning's optimality is preferable.
