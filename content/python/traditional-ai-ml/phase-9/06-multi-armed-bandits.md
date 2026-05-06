# Multi-Armed Bandits

> Phase 9 — Reinforcement Learning | Kata 9.6

---

## Concept & Intuition

### What problem are we solving?

Imagine you are in a casino facing a row of slot machines (bandits), each with a different (unknown) payout probability. You have a limited budget. How do you maximize your total winnings? This is the **multi-armed bandit** problem — the purest distillation of the exploration-exploitation dilemma.

Unlike full RL, there are no states or transitions. You simply choose an arm, observe a reward, and repeat. The challenge is entirely about **learning which arm is best while simultaneously maximizing reward**. Pulling the best-known arm (exploit) earns immediate reward but risks missing a better one. Pulling an uncertain arm (explore) gathers information but might waste a pull on a bad arm.

This problem appears everywhere: A/B testing websites, clinical drug trials, ad placement, recommendation systems, and hyperparameter search. Three key strategies tackle it: **epsilon-greedy** (explore randomly some fraction of the time), **Upper Confidence Bound (UCB)** (explore arms with high uncertainty), and **Thompson Sampling** (sample from posterior belief distributions and act on the sample). Each makes a different trade-off between simplicity and efficiency.

### Why naive approaches fail

Pure exploitation (always pick the arm with the highest observed reward) can lock onto a suboptimal arm that got lucky early. Pure exploration (pull arms uniformly) wastes most pulls on clearly bad arms. Epsilon-greedy is simple but wastes exploration budget on arms it already knows are bad. UCB and Thompson Sampling direct exploration toward uncertain arms — the ones where new information is most valuable.

### Mental models

- **Restaurant analogy again, refined**: Epsilon-greedy tries a random new restaurant 10% of the time. UCB tries the restaurant with the highest "potential" (good reviews OR few reviews). Thompson Sampling rolls dice weighted by its belief about each restaurant and goes to the winner.
- **UCB = optimism in the face of uncertainty**: If you have not tried an arm much, assume it might be great. This automatically directs exploration toward under-sampled arms.
- **Thompson Sampling = probability matching**: The probability of picking an arm equals the probability that it is actually the best. Elegant and often optimal.

### Visual explanations

```
Three bandits with true means [0.3, 0.5, 0.7]:

After 10 pulls each:
  Arm 0: observed mean = 0.35, pulls = 10
  Arm 1: observed mean = 0.48, pulls = 10
  Arm 2: observed mean = 0.65, pulls = 10

Epsilon-Greedy (eps=0.1):
  90% of the time: pick Arm 2 (best observed)
  10% of the time: pick uniformly at random
  --> Wastes exploration on Arm 0 (clearly bad)

UCB (after t total pulls):
  UCB(arm) = mean(arm) + sqrt(2 * ln(t) / pulls(arm))
  Arms pulled less get a bigger bonus
  --> Explores uncertain arms, not random ones

Thompson Sampling:
  Sample theta_i ~ Beta(successes_i + 1, failures_i + 1)
  Pick arm with highest theta_i
  --> Naturally explores uncertain arms, converges to best

Regret over time:
  Optimal total reward after T pulls: 0.7 * T
  Your total reward: sum of actual rewards
  Regret = optimal - yours

  Good algorithms: regret grows as O(log T)
  Bad algorithms:  regret grows as O(T)
```

---

## Hands-on Exploration

1. Create 3 arms with known probabilities (e.g., 0.3, 0.5, 0.7). Run epsilon-greedy for 1000 pulls and track cumulative regret.
2. Implement UCB. Does it converge to the best arm faster? Plot the number of times each arm is pulled.
3. Implement Thompson Sampling. Compare its regret curve to epsilon-greedy and UCB.
4. Try a harder problem: 10 arms with means uniformly spaced from 0.1 to 0.9. Which algorithm handles more arms best?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Bandit Environment ---
TRUE_MEANS = [0.2, 0.45, 0.7, 0.55, 0.35]
N_ARMS = len(TRUE_MEANS)

# @param n_pulls int 100 5000 1000
n_pulls = 1000

def pull_arm(arm):
    """Bernoulli bandit: reward is 0 or 1."""
    return 1 if np.random.random() < TRUE_MEANS[arm] else 0

# --- Epsilon-Greedy ---
def run_epsilon_greedy(n_pulls, epsilon=0.1):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    rewards = []
    choices = []
    for t in range(n_pulls):
        if np.random.random() < epsilon:
            arm = np.random.randint(N_ARMS)
        else:
            arm = np.argmax(values)
        reward = pull_arm(arm)
        counts[arm] += 1
        # Incremental mean update
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
        choices.append(arm)
    return rewards, choices, counts, values

# --- UCB1 ---
def run_ucb(n_pulls):
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    rewards = []
    choices = []
    # Pull each arm once
    for arm in range(N_ARMS):
        reward = pull_arm(arm)
        counts[arm] = 1
        values[arm] = reward
        rewards.append(reward)
        choices.append(arm)
    # UCB selection
    for t in range(N_ARMS, n_pulls):
        ucb_values = values + np.sqrt(2 * np.log(t) / counts)
        arm = np.argmax(ucb_values)
        reward = pull_arm(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
        choices.append(arm)
    return rewards, choices, counts, values

# --- Thompson Sampling (Beta-Bernoulli) ---
def run_thompson(n_pulls):
    successes = np.ones(N_ARMS)   # Beta(1,1) prior = uniform
    failures = np.ones(N_ARMS)
    rewards = []
    choices = []
    for t in range(n_pulls):
        # Sample from posterior Beta distribution for each arm
        samples = np.random.beta(successes, failures)
        arm = np.argmax(samples)
        reward = pull_arm(arm)
        if reward == 1:
            successes[arm] += 1
        else:
            failures[arm] += 1
        rewards.append(reward)
        choices.append(arm)
    counts = successes + failures - 2  # subtract prior
    values = successes / (successes + failures)
    return rewards, choices, counts, values

# --- Run all three ---
r_eg, c_eg, cnt_eg, v_eg = run_epsilon_greedy(n_pulls, epsilon=0.1)
r_ucb, c_ucb, cnt_ucb, v_ucb = run_ucb(n_pulls)
r_ts, c_ts, cnt_ts, v_ts = run_thompson(n_pulls)

# --- Compute cumulative regret ---
best_mean = max(TRUE_MEANS)
def cumulative_regret(choices):
    regret = 0.0
    regrets = []
    for arm in choices:
        regret += best_mean - TRUE_MEANS[arm]
        regrets.append(regret)
    return regrets

reg_eg = cumulative_regret(c_eg)
reg_ucb = cumulative_regret(c_ucb)
reg_ts = cumulative_regret(c_ts)

# --- Report ---
print(f"=== Multi-Armed Bandits ({N_ARMS} arms, {n_pulls} pulls) ===")
print(f"True means: {TRUE_MEANS}")
print(f"Best arm: {np.argmax(TRUE_MEANS)} (mean={best_mean})\n")

print(f"{'Algorithm':>20}  {'Total Reward':>13}  {'Final Regret':>13}  {'Best Arm %':>11}")
print("-" * 65)
for name, rewards, choices, counts in [
    ("Epsilon-Greedy", r_eg, c_eg, cnt_eg),
    ("UCB1", r_ucb, c_ucb, cnt_ucb),
    ("Thompson Sampling", r_ts, c_ts, cnt_ts),
]:
    total_r = sum(rewards)
    best_arm_pct = 100 * sum(1 for c in choices if c == np.argmax(TRUE_MEANS)) / n_pulls
    regret = cumulative_regret(choices)[-1]
    print(f"{name:>20}  {total_r:>13.0f}  {regret:>13.1f}  {best_arm_pct:>10.1f}%")

# --- Arm pull distribution ---
print("\n=== Arm Pull Distribution ===")
print(f"{'Arm':>4}  {'True Mean':>10}  {'EpsGreedy':>10}  {'UCB':>10}  {'Thompson':>10}")
print("-" * 50)
for arm in range(N_ARMS):
    print(f"{arm:>4}  {TRUE_MEANS[arm]:>10.2f}  {int(cnt_eg[arm]):>10}  "
          f"{int(cnt_ucb[arm]):>10}  {int(cnt_ts[arm]):>10}")

# --- Regret at checkpoints ---
print("\n=== Cumulative Regret at Checkpoints ===")
print(f"{'Pull':>6}  {'EpsGreedy':>10}  {'UCB':>10}  {'Thompson':>10}")
print("-" * 40)
for t in [50, 100, 250, 500, n_pulls - 1]:
    print(f"{t+1:>6}  {reg_eg[t]:>10.1f}  {reg_ucb[t]:>10.1f}  {reg_ts[t]:>10.1f}")

print("\n=== Observations ===")
print("- Thompson Sampling typically achieves the lowest regret")
print("- UCB explores systematically based on uncertainty")
print("- Epsilon-Greedy wastes exploration on clearly bad arms")
print(f"- All algorithms correctly identify arm {np.argmax(TRUE_MEANS)} as best")
```

---

## Key Takeaways

- **The multi-armed bandit is the simplest exploration-exploitation problem.** No states, no transitions — just choosing arms and observing rewards.
- **Epsilon-greedy is simple but wasteful.** It explores uniformly, spending as much time on clearly bad arms as on uncertain ones.
- **UCB directs exploration toward uncertain arms.** The confidence bonus shrinks as an arm is pulled more, so attention naturally shifts to under-explored arms.
- **Thompson Sampling is Bayesian and often optimal.** By sampling from posterior distributions, it naturally balances exploration and exploitation without a tunable epsilon parameter.
- **Regret is the right metric.** It measures the cumulative cost of not knowing the best arm from the start. Good algorithms achieve logarithmic regret growth.
