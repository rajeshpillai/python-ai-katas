# Q-Learning

> Phase 9 — Reinforcement Learning | Kata 9.4

---

## Concept & Intuition

### What problem are we solving?

Policy iteration and value iteration require a complete model of the environment — full knowledge of transition probabilities and rewards. But in most real problems, the agent does not have this model. It must learn from experience: take actions, observe outcomes, and update its understanding. **Q-learning** is the foundational model-free algorithm that learns the optimal action-value function Q*(s, a) directly from interactions with the environment.

Q-learning is an **off-policy temporal difference (TD)** method. "Temporal difference" means it updates estimates based on other estimates (bootstrapping) rather than waiting for complete episodes. "Off-policy" means the agent can learn about the optimal policy while following a different behavior policy — typically an epsilon-greedy policy that balances **exploration** (trying new actions to discover their effects) with **exploitation** (using current knowledge to maximize reward).

The core update rule is elegant: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]. The term in brackets is the **TD error** — the difference between the target (immediate reward plus the best future value) and the current estimate. The learning rate alpha controls how much each experience shifts the estimate.

### Why naive approaches fail

Without exploration, the agent gets stuck exploiting the first decent strategy it finds, never discovering better alternatives. Pure exploration (random actions) wastes time and never converges on good behavior. The epsilon-greedy strategy threads the needle: with probability epsilon, take a random action (explore); otherwise, take the best known action (exploit). Over time, epsilon can be decayed so the agent explores less as it becomes more confident.

### Mental models

- **Restaurant exploration**: You have a favorite restaurant (exploit), but occasionally try a new place (explore). Over time you build a mental "Q-table" of how good each restaurant is, and your exploration rate naturally decreases.
- **The max in the update is the key**: Q-learning always looks at the best possible next action, even if the agent did not take that action. This is why it is off-policy — it learns about the greedy policy regardless of what the behavior policy does.
- **TD error as surprise**: When the actual reward plus estimated future value differs from what Q predicted, the agent is surprised. It adjusts Q to reduce that surprise.

### Visual explanations

```
Q-Learning Update:

  Agent in state S, takes action A, gets reward R, lands in S'

  Old estimate:  Q(S, A)
  Target:        R + gamma * max_{a'} Q(S', a')
  TD error:      target - old estimate
  New estimate:  Q(S, A) + alpha * TD_error

  Q(S,A) ----[alpha]----> R + gamma * max Q(S', *)
    |                          |
    current                    bootstrapped
    estimate                   target

Epsilon-Greedy Exploration:

  With prob epsilon:    choose random action  (EXPLORE)
  With prob 1-epsilon:  choose argmax_a Q(s,a) (EXPLOIT)

  epsilon = 1.0  -->  pure random exploration
  epsilon = 0.0  -->  pure greedy exploitation
  epsilon = 0.1  -->  explore 10% of the time (typical)
```

---

## Hands-on Exploration

1. Initialize a Q-table with zeros for a 3x3 grid. Place a reward at one corner. Manually trace through 5 steps of Q-learning updates with epsilon=0.5. Watch Q values grow from the goal backward.
2. Set epsilon=0 (pure greedy). Start the agent in a corner far from the goal. Does it ever find the goal? Why not?
3. Try alpha=1.0 (fully replace old estimate with new target). How does this compare to alpha=0.1? Which is more stable?
4. After many episodes, compare the learned Q-values to the true Q-values from value iteration. How close are they?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- 5x5 Grid World ---
GRID_SIZE = 5
N_STATES = GRID_SIZE * GRID_SIZE
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_ACTIONS = len(ACTIONS)
GOAL_STATE = 24  # bottom-right
TRAP_STATE = 17  # a trap
TERMINAL_STATES = {GOAL_STATE, TRAP_STATE}
STEP_REWARD = -0.1
GOAL_REWARD = 10.0
TRAP_REWARD = -10.0

# @param epsilon float 0.01 1.0 0.1
epsilon = 0.1
# @param learning_rate float 0.01 1.0 0.1
learning_rate = 0.1
# @param gamma float 0.5 0.99 0.95
gamma = 0.95
# @param n_episodes int 100 5000 1000
n_episodes = 1000

def state_to_rc(s):
    return s // GRID_SIZE, s % GRID_SIZE

def rc_to_state(r, c):
    return r * GRID_SIZE + c

def step(state, action_idx):
    """Take action, return (next_state, reward, done)."""
    if state in TERMINAL_STATES:
        return state, 0.0, True
    r, c = state_to_rc(state)
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    # 90% intended, 10% random slip
    if np.random.random() < 0.9:
        dr, dc = deltas[action_idx]
    else:
        dr, dc = deltas[np.random.randint(N_ACTIONS)]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c  # wall: stay
    ns = rc_to_state(nr, nc)
    if ns == GOAL_STATE:
        return ns, GOAL_REWARD, True
    elif ns == TRAP_STATE:
        return ns, TRAP_REWARD, True
    return ns, STEP_REWARD, False

def choose_action(state, Q, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    return np.argmax(Q[state])

# --- Q-Learning ---
Q = np.zeros((N_STATES, N_ACTIONS))

episode_rewards = []
episode_lengths = []

print("=== Q-Learning ===")
print(f"Epsilon: {epsilon}, Alpha: {learning_rate}, Gamma: {gamma}")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Episodes: {n_episodes}\n")

for ep in range(n_episodes):
    state = 0  # start top-left
    total_reward = 0.0
    steps = 0

    for t in range(200):  # max steps per episode
        action = choose_action(state, Q, epsilon)
        next_state, reward, done = step(state, action)

        # Q-learning update (off-policy: uses max over next actions)
        td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
        td_error = td_target - Q[state, action]
        Q[state, action] += learning_rate * td_error

        total_reward += reward
        state = next_state
        steps += 1

        if done:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

# --- Report learning progress ---
print(f"{'Window':>10}  {'Mean Reward':>12}  {'Mean Length':>12}")
print("-" * 38)
windows = [(0, 100), (100, 300), (300, 600), (600, n_episodes)]
for start, end in windows:
    if end <= n_episodes:
        mr = np.mean(episode_rewards[start:end])
        ml = np.mean(episode_lengths[start:end])
        print(f"Ep {start:>4}-{end:<4}  {mr:>12.2f}  {ml:>12.1f}")

# --- Display learned Q-values for start state ---
print(f"\n=== Q-values for state 0 (start) ===")
for a in range(N_ACTIONS):
    print(f"  Q(0, {ACTIONS[a]:>5s}) = {Q[0, a]:+.3f}")

# --- Extract and display the learned policy ---
print("\n=== Learned Policy ===")
arrows = {0: '^', 1: 'v', 2: '<', 3: '>'}
for r in range(GRID_SIZE):
    row_policy = []
    for c in range(GRID_SIZE):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_policy.append("G")
        elif s == TRAP_STATE:
            row_policy.append("X")
        else:
            best_a = np.argmax(Q[s])
            row_policy.append(arrows[best_a])
    print("  ".join(row_policy))

# --- Display max Q-values as a heatmap ---
print("\n=== Max Q-values per state ===")
for r in range(GRID_SIZE):
    row_vals = []
    for c in range(GRID_SIZE):
        s = rc_to_state(r, c)
        if s == GOAL_STATE:
            row_vals.append(" GOAL ")
        elif s == TRAP_STATE:
            row_vals.append(" TRAP ")
        else:
            row_vals.append(f"{np.max(Q[s]):+.2f}")
    print("  ".join(row_vals))

# --- Test learned policy (greedy, no exploration) ---
print("\n=== Test Episode (greedy policy) ===")
state = 0
path = [state]
for t in range(50):
    action = np.argmax(Q[state])
    next_state, reward, done = step(state, action)
    state = next_state
    path.append(state)
    if done:
        break
path_str = ' -> '.join([f"({state_to_rc(s)[0]},{state_to_rc(s)[1]})" for s in path])
outcome = "GOAL" if path[-1] == GOAL_STATE else ("TRAP" if path[-1] == TRAP_STATE else "TIMEOUT")
print(f"Result: {outcome}")
print(f"Path: {path_str}")
```

---

## Key Takeaways

- **Q-learning is model-free.** It learns optimal behavior purely from experience, without knowing transition probabilities or reward functions in advance.
- **The off-policy nature is powerful.** Q-learning updates toward the best possible next action (max Q), even while the agent follows an exploratory policy. This decouples learning from behavior.
- **Epsilon-greedy balances exploration and exploitation.** Too much exploration wastes time; too little causes the agent to miss better strategies. Tuning epsilon is critical.
- **The learning rate alpha controls stability vs. speed.** High alpha learns fast but oscillates; low alpha is stable but slow. A common practice is to decay alpha over time.
- **Q-values propagate backward from rewards.** States near the goal get high Q-values first, then these propagate to earlier states over many episodes — like ripples spreading from a stone dropped in water.
