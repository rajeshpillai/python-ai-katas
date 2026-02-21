# Environment Design

> Phase 9 — Reinforcement Learning | Kata 9.7

---

## Concept & Intuition

### What problem are we solving?

Before you can train an RL agent, you need an environment for it to interact with. Environment design is the art of defining states, actions, transitions, and rewards in a way that makes the learning problem well-posed. A poorly designed reward function can lead to unexpected and undesirable agent behavior ("reward hacking"), while a well-designed environment guides the agent toward the intended goal efficiently.

The key design decisions are: **state representation** (what information does the agent observe?), **action space** (what can the agent do?), **reward function** (what behavior do we want to encourage?), and **termination conditions** (when does an episode end?). Each choice profoundly affects whether the agent can learn at all, how quickly it learns, and what behavior it converges to.

Reward shaping is particularly important. Sparse rewards (only at the goal) make learning hard because the agent must stumble upon the goal by random exploration before it gets any learning signal. Dense rewards (small rewards for intermediate progress) make learning faster but risk the agent optimizing the intermediate rewards instead of the true objective.

### Why naive approaches fail

A common mistake is designing rewards that do not align with the true objective. A cleaning robot rewarded for "not seeing dirt" might learn to close its eyes. An agent rewarded for "time spent alive" might learn to never move (avoiding risky actions). These are examples of reward hacking -- the agent finds unintended loopholes in the reward function. The solution is careful reward engineering and extensive testing of agent behavior.

### Mental models

- **Reward as communication**: the reward function is how you communicate your goals to the agent. Ambiguous rewards lead to ambiguous behavior.
- **State as the agent's eyes**: the agent can only learn patterns it can observe. Missing state information leads to suboptimal policies.
- **Reward shaping as a curriculum**: dense rewards guide the agent through intermediate milestones, like a teacher giving partial credit.

### Visual explanations

```
Environment Design Checklist:
  [ ] State space: What does the agent observe?
  [ ] Action space: What can the agent do?
  [ ] Transitions: How does the world respond?
  [ ] Reward function: What behavior do we want?
  [ ] Terminal conditions: When does an episode end?
  [ ] Reset mechanism: How to start a new episode?

Reward Design Spectrum:
  Sparse:  reward = +1 at goal, 0 elsewhere
           Pro: clear objective
           Con: hard to learn (must find goal by chance)

  Dense:   reward = -distance_to_goal at each step
           Pro: guides learning
           Con: might lead to reward hacking

  Shaped:  reward = progress_toward_goal - cost_of_action
           Pro: balances guidance with objectivity
           Con: requires domain knowledge to design
```

---

## Hands-on Exploration

1. Design a simple inventory management environment: the agent decides how much stock to order each day, facing uncertain demand and holding costs.
2. Compare sparse rewards (profit at end of month) vs dense rewards (daily profit). Show that dense rewards enable faster learning.
3. Intentionally create a reward hacking scenario: design a flawed reward and show how the agent exploits it.
4. Fix the reward function and demonstrate improved agent behavior.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_int = |s: &mut u64, max: usize| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as usize) % max
    };

    // === Environment: Inventory Management ===
    // State: current stock level (0-20)
    // Action: order quantity (0-5)
    // Demand: random 0-5 per day
    // Revenue: $10 per item sold
    // Holding cost: $2 per item in stock at end of day
    // Stockout cost: $5 per unmet demand
    // Max capacity: 20

    let max_stock = 20;
    let max_order = 5;
    let max_demand = 5;
    let sell_price = 10.0;
    let hold_cost = 2.0;
    let stockout_cost = 5.0;
    let order_cost = 3.0; // per item ordered
    let episode_length = 30; // 30 days

    let n_states = max_stock + 1;
    let n_actions = max_order + 1;

    // Environment step
    let env_step = |stock: usize, order: usize, rng: &mut u64| -> (usize, f64, f64) {
        let new_stock = (stock + order).min(max_stock);
        let demand = rand_int(rng, max_demand + 1);

        let sold = demand.min(new_stock);
        let unmet = demand.saturating_sub(new_stock);
        let remaining = new_stock - sold;

        let revenue = sold as f64 * sell_price;
        let holding = remaining as f64 * hold_cost;
        let stockout = unmet as f64 * stockout_cost;
        let ordering = order as f64 * order_cost;

        let reward = revenue - holding - stockout - ordering;

        (remaining, reward, revenue)
    };

    // === Q-Learning on the inventory problem ===
    println!("=== Environment Design: Inventory Management ===\n");
    println!("State: stock level (0-{})", max_stock);
    println!("Actions: order 0-{} items", max_order);
    println!("Demand: uniform 0-{}", max_demand);
    println!("Episode: {} days\n", episode_length);

    // --- Reward Design 1: Dense (daily profit) ---
    let alpha = 0.1;
    let gamma = 0.95;
    let epsilon = 0.15;
    let n_episodes = 3000;

    let mut q_dense = vec![vec![0.0; n_actions]; n_states];
    let mut dense_rewards: Vec<f64> = Vec::new();

    for _ in 0..n_episodes {
        let mut stock = 10; // start with 10 items
        let mut total = 0.0;

        for _ in 0..episode_length {
            let action = if rand_f64(&mut rng) < epsilon {
                rand_int(&mut rng, n_actions)
            } else {
                let mut best = 0;
                for a in 1..n_actions {
                    if q_dense[stock][a] > q_dense[stock][best] { best = a; }
                }
                best
            };

            let (next_stock, reward, _) = env_step(stock, action, &mut rng);

            // Dense reward: immediate daily profit
            let max_q = q_dense[next_stock].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            q_dense[stock][action] += alpha * (reward + gamma * max_q - q_dense[stock][action]);

            total += reward;
            stock = next_stock;
        }
        dense_rewards.push(total);
    }

    // --- Reward Design 2: Sparse (end-of-episode only) ---
    let mut q_sparse = vec![vec![0.0; n_actions]; n_states];
    let mut sparse_rewards: Vec<f64> = Vec::new();

    for _ in 0..n_episodes {
        let mut stock = 10;
        let mut total = 0.0;
        let mut trajectory: Vec<(usize, usize)> = Vec::new();

        for _ in 0..episode_length {
            let action = if rand_f64(&mut rng) < epsilon {
                rand_int(&mut rng, n_actions)
            } else {
                let mut best = 0;
                for a in 1..n_actions {
                    if q_sparse[stock][a] > q_sparse[stock][best] { best = a; }
                }
                best
            };

            let (next_stock, reward, _) = env_step(stock, action, &mut rng);
            trajectory.push((stock, action));
            total += reward;
            stock = next_stock;
        }

        // Sparse: only update at end with total reward
        // Use Monte Carlo update
        let mut g = total / episode_length as f64; // average daily reward
        for &(s, a) in trajectory.iter().rev() {
            q_sparse[s][a] += alpha * (g - q_sparse[s][a]);
            g *= gamma;
        }

        sparse_rewards.push(total);
    }

    // --- Reward Design 3: Flawed (reward hacking) ---
    // Flaw: reward agent for having high stock (ignores profitability)
    let mut q_flawed = vec![vec![0.0; n_actions]; n_states];
    let mut flawed_actual_profits: Vec<f64> = Vec::new();

    for _ in 0..n_episodes {
        let mut stock = 10;
        let mut actual_profit = 0.0;

        for _ in 0..episode_length {
            let action = if rand_f64(&mut rng) < epsilon {
                rand_int(&mut rng, n_actions)
            } else {
                let mut best = 0;
                for a in 1..n_actions {
                    if q_flawed[stock][a] > q_flawed[stock][best] { best = a; }
                }
                best
            };

            let (next_stock, real_reward, _) = env_step(stock, action, &mut rng);

            // FLAWED reward: just reward high stock levels
            let flawed_reward = next_stock as f64;

            let max_q = q_flawed[next_stock].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            q_flawed[stock][action] += alpha * (flawed_reward + gamma * max_q - q_flawed[stock][action]);

            actual_profit += real_reward;
            stock = next_stock;
        }
        flawed_actual_profits.push(actual_profit);
    }

    // --- Compare results ---
    println!("=== Learning Curves (average over last 500 episodes) ===\n");
    let last = 500;
    let avg = |rewards: &[f64]| -> f64 {
        rewards[rewards.len() - last..].iter().sum::<f64>() / last as f64
    };

    println!("{:<25} {:>15}", "Reward Design", "Avg Profit");
    println!("{}", "-".repeat(42));
    println!("{:<25} {:>15.1}", "Dense (daily profit)", avg(&dense_rewards));
    println!("{:<25} {:>15.1}", "Sparse (end-of-episode)", avg(&sparse_rewards));
    println!("{:<25} {:>15.1}", "Flawed (stock level)", avg(&flawed_actual_profits));

    // --- Show learned policies ---
    println!("\n=== Learned Policies (order quantity by stock level) ===\n");
    println!("{:>6}  {:>12} {:>12} {:>12}", "Stock", "Dense", "Sparse", "Flawed");
    println!("{}", "-".repeat(48));

    for s in (0..=max_stock).step_by(2) {
        let dense_a = (0..n_actions).max_by(|&a, &b| q_dense[s][a].partial_cmp(&q_dense[s][b]).unwrap()).unwrap();
        let sparse_a = (0..n_actions).max_by(|&a, &b| q_sparse[s][a].partial_cmp(&q_sparse[s][b]).unwrap()).unwrap();
        let flawed_a = (0..n_actions).max_by(|&a, &b| q_flawed[s][a].partial_cmp(&q_flawed[s][b]).unwrap()).unwrap();
        println!("{:>6}  {:>12} {:>12} {:>12}", s,
            format!("order {}", dense_a),
            format!("order {}", sparse_a),
            format!("order {}", flawed_a));
    }

    println!("\nNote: The flawed policy always orders maximum ({}) because it optimizes",
        max_order);
    println!("for stock level rather than profit. This is reward hacking!");

    // --- Design guidelines ---
    println!("\n=== Environment Design Guidelines ===");
    println!("1. Align rewards with true objective (profit, not stock level)");
    println!("2. Use dense rewards for faster learning when possible");
    println!("3. Test for reward hacking by checking actual vs intended behavior");
    println!("4. Include costs (holding, ordering) to prevent degenerate solutions");
    println!("5. Make state observable: agent needs to see stock level to decide");

    let dense_profit = avg(&dense_rewards);
    println!();
    println!("kata_metric(\"dense_reward_profit\", {:.1})", dense_profit);
    println!("kata_metric(\"sparse_reward_profit\", {:.1})", avg(&sparse_rewards));
    println!("kata_metric(\"flawed_reward_profit\", {:.1})", avg(&flawed_actual_profits));
}
```

---

## Key Takeaways

- **Reward design is the most critical part of environment design.** A misaligned reward function leads to reward hacking -- the agent optimizes the reward without achieving the intended goal.
- **Dense rewards enable faster learning** by providing immediate feedback, but must be carefully designed to avoid incentivizing shortcuts.
- **State representation determines what the agent can learn.** Missing information in the state leads to suboptimal policies.
- **Always test agent behavior against the true objective,** not just the reward signal. The agent might maximize reward while doing something completely unintended.
