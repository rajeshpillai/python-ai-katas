# Q-Learning

> Phase 9 — Reinforcement Learning | Kata 9.4

---

## Concept & Intuition

### What problem are we solving?

Policy iteration and value iteration require a complete model of the environment -- full knowledge of transition probabilities and rewards. But in most real problems, the agent does not have this model. It must learn from experience: take actions, observe outcomes, and update its understanding. **Q-learning** is the foundational model-free algorithm that learns the optimal action-value function Q*(s, a) directly from interactions with the environment.

Q-learning is an **off-policy temporal difference (TD)** method. "Temporal difference" means it updates estimates based on other estimates (bootstrapping) rather than waiting for complete episodes. "Off-policy" means the agent can learn about the optimal policy while following a different behavior policy -- typically an epsilon-greedy policy that balances **exploration** (trying new actions) with **exploitation** (using current knowledge).

The core update rule is elegant: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]. The term in brackets is the **TD error** -- the difference between the target (immediate reward plus the best future value) and the current estimate.

### Why naive approaches fail

Without exploration, the agent gets stuck exploiting the first decent strategy it finds, never discovering better alternatives. Pure exploration (random actions) wastes time and never converges. The epsilon-greedy strategy threads the needle: with probability epsilon, take a random action (explore); otherwise, take the best known action (exploit). Over time, epsilon can be decayed to reduce exploration as confidence grows.

### Mental models

- **Restaurant exploration**: You have a favorite restaurant (exploit), but occasionally try a new place (explore). Over time you build a "Q-table" of how good each restaurant is.
- **The max in the update is the key**: Q-learning always looks at the best possible next action, even if the agent did not take that action. This is why it is off-policy.
- **TD error as surprise**: When the actual outcome differs from what Q predicted, the agent is surprised and adjusts its estimate.

### Visual explanations

```
Q-Learning Update:

  Agent in state S, takes action A, gets reward R, lands in S'

  Old estimate:  Q(S, A)
  Target:        R + gamma * max_{a'} Q(S', a')
  TD error:      target - old estimate
  New estimate:  Q(S, A) + alpha * TD_error

Epsilon-Greedy:
  With prob epsilon:    choose random action  (EXPLORE)
  With prob 1-epsilon:  choose argmax_a Q(s,a) (EXPLOIT)
```

---

## Hands-on Exploration

1. Initialize a Q-table with zeros for a 5x5 grid. Manually trace through 5 Q-learning updates with epsilon=0.5.
2. Set epsilon=0 (pure greedy). Start far from the goal. Does the agent find it?
3. Try alpha=1.0 vs alpha=0.1. Which is more stable?
4. After training, compare the learned Q-values to the optimal values from value iteration.

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

    // --- 5x5 Grid World ---
    let grid = 5;
    let n_states = grid * grid;
    let n_actions = 4;
    let goal = 24;
    let trap = 17;
    let action_names = ["UP", "DOWN", "LEFT", "RIGHT"];

    let to_rc = |s: usize| -> (usize, usize) { (s / grid, s % grid) };
    let to_s = |r: usize, c: usize| -> usize { r * grid + c };

    let step = |state: usize, action: usize, rng: &mut u64| -> (usize, f64, bool) {
        if state == goal || state == trap { return (state, 0.0, true); }
        let (r, c) = to_rc(state);

        // 90% intended action, 10% random slip
        let actual = if rand_f64(rng) < 0.9 { action } else { rand_int(rng, n_actions) };

        let (nr, nc) = match actual {
            0 => (if r > 0 { r - 1 } else { r }, c),
            1 => (if r < grid - 1 { r + 1 } else { r }, c),
            2 => (r, if c > 0 { c - 1 } else { c }),
            3 => (r, if c < grid - 1 { c + 1 } else { c }),
            _ => (r, c),
        };
        let ns = to_s(nr, nc);
        let reward = if ns == goal { 10.0 }
                     else if ns == trap { -10.0 }
                     else { -0.1 };
        let done = ns == goal || ns == trap;
        (ns, reward, done)
    };

    // --- Q-Learning ---
    let epsilon = 0.1;
    let alpha = 0.1;
    let gamma = 0.95;
    let n_episodes = 2000;

    let mut q = vec![vec![0.0; n_actions]; n_states];
    let mut episode_rewards: Vec<f64> = Vec::new();
    let mut episode_lengths: Vec<usize> = Vec::new();

    println!("=== Q-Learning ===");
    println!("Epsilon: {}, Alpha: {}, Gamma: {}", epsilon, alpha, gamma);
    println!("Grid: {}x{}, Episodes: {}\n", grid, grid, n_episodes);

    for ep in 0..n_episodes {
        let mut state = 0;
        let mut total_reward = 0.0;
        let mut steps = 0;

        for _ in 0..200 {
            // Epsilon-greedy action selection
            let action = if rand_f64(&mut rng) < epsilon {
                rand_int(&mut rng, n_actions)
            } else {
                let mut best_a = 0;
                let mut best_q = q[state][0];
                for a in 1..n_actions {
                    if q[state][a] > best_q {
                        best_q = q[state][a];
                        best_a = a;
                    }
                }
                best_a
            };

            let (next_state, reward, done) = step(state, action, &mut rng);

            // Q-learning update (off-policy: uses max over next actions)
            let max_q_next = q[next_state].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_target = reward + gamma * max_q_next * (1.0 - done as u8 as f64);
            let td_error = td_target - q[state][action];
            q[state][action] += alpha * td_error;

            total_reward += reward;
            state = next_state;
            steps += 1;

            if done { break; }
        }

        episode_rewards.push(total_reward);
        episode_lengths.push(steps);
    }

    // --- Learning progress ---
    println!("=== Learning Progress ===\n");
    println!("{:>12} {:>12} {:>12}", "Window", "Mean Reward", "Mean Steps");
    println!("{}", "-".repeat(38));
    let windows = [(0, 100), (100, 500), (500, 1000), (1000, 2000)];
    for (start, end) in &windows {
        if *end <= n_episodes {
            let mr: f64 = episode_rewards[*start..*end].iter().sum::<f64>()
                / (end - start) as f64;
            let ml: f64 = episode_lengths[*start..*end].iter().sum::<usize>() as f64
                / (end - start) as f64;
            println!("Ep {:>4}-{:<4} {:>12.2} {:>12.1}", start, end, mr, ml);
        }
    }

    // --- Learned policy ---
    let arrows = ['^', 'v', '<', '>'];
    println!("\n=== Learned Policy ===");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            if s == goal { row.push_str("  G  "); }
            else if s == trap { row.push_str("  X  "); }
            else {
                let best_a = (0..n_actions)
                    .max_by(|&a, &b| q[s][a].partial_cmp(&q[s][b]).unwrap())
                    .unwrap();
                row.push_str(&format!("  {}  ", arrows[best_a]));
            }
        }
        println!("  {}", row);
    }

    // --- Max Q-values ---
    println!("\n=== Max Q-values per state ===");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            if s == goal { row.push_str(" GOAL "); }
            else if s == trap { row.push_str(" TRAP "); }
            else {
                let max_q = q[s].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                row.push_str(&format!("{:+.2} ", max_q));
            }
        }
        println!("  {}", row);
    }

    // --- Q-values for start state ---
    println!("\n=== Q-values for state 0 (start) ===");
    for a in 0..n_actions {
        println!("  Q(0, {:>5}) = {:+.4}", action_names[a], q[0][a]);
    }

    // --- Test the learned policy (greedy, no exploration) ---
    println!("\n=== Test Episode (greedy) ===");
    let mut state = 0;
    let mut path = vec![state];
    for _ in 0..50 {
        let best_a = (0..n_actions)
            .max_by(|&a, &b| q[state][a].partial_cmp(&q[state][b]).unwrap())
            .unwrap();
        let (ns, _, done) = step(state, best_a, &mut rng);
        state = ns;
        path.push(state);
        if done { break; }
    }

    let outcome = if *path.last().unwrap() == goal { "GOAL" }
                  else if *path.last().unwrap() == trap { "TRAP" }
                  else { "TIMEOUT" };
    let path_str: String = path.iter()
        .map(|&s| { let (r, c) = to_rc(s); format!("({},{})", r, c) })
        .collect::<Vec<_>>().join(" -> ");
    println!("Result: {}", outcome);
    println!("Path: {}", path_str);

    println!();
    println!("kata_metric(\"final_mean_reward\", {:.2})",
        episode_rewards[1500..].iter().sum::<f64>() / 500.0);
    println!("kata_metric(\"final_mean_steps\", {:.1})",
        episode_lengths[1500..].iter().sum::<usize>() as f64 / 500.0);
}
```

---

## Key Takeaways

- **Q-learning is model-free.** It learns optimal behavior purely from experience, without knowing transition probabilities or reward functions.
- **The off-policy nature is powerful.** Q-learning updates toward the best possible next action (max Q), even while following an exploratory policy.
- **Epsilon-greedy balances exploration and exploitation.** Too much exploration wastes time; too little causes the agent to miss better strategies.
- **Q-values propagate backward from rewards.** States near the goal get high Q-values first, then these propagate to earlier states over many episodes.
