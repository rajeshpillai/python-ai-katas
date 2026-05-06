# SARSA

> Phase 9 — Reinforcement Learning | Kata 9.5

---

## Concept & Intuition

### What problem are we solving?

SARSA (State-Action-Reward-State-Action) is an **on-policy** TD learning algorithm. Unlike Q-learning which updates toward the *best possible* next action, SARSA updates toward the action the agent *actually takes* next. The name comes from the quintuple (S, A, R, S', A') used in each update -- the current state, current action, reward, next state, and next action.

The update rule is: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]. The critical difference from Q-learning is the absence of "max": instead of max_a' Q(S', a'), SARSA uses Q(S', A') where A' is the action actually selected by the current policy.

This makes SARSA **safer** in environments where exploration can be dangerous. Q-learning learns the optimal policy assuming perfect execution, which may involve walking near cliffs (high risk, high reward paths). SARSA learns a policy that accounts for its own exploration noise -- since it knows it will sometimes take random actions, it learns to avoid dangerous situations where a random action could be catastrophic.

### Why naive approaches fail

Q-learning can learn overly aggressive policies in dangerous environments because it assumes optimal future behavior (the max). When the agent actually follows epsilon-greedy and sometimes takes random actions near a cliff, Q-learning's policy leads to frequent falls. SARSA's on-policy nature means it learns a more conservative policy that accounts for its own imperfections.

### Mental models

- **Q-learning is an optimist**: "I'll learn the best possible behavior, assuming I always execute perfectly." This is great in safe environments.
- **SARSA is a realist**: "I'll learn what's best given how I actually behave, including my mistakes." This is safer in dangerous environments.
- **The cliff walking problem**: a classic illustration. Q-learning learns the shortest path along the cliff edge (optimal but risky with exploration). SARSA learns a longer path far from the cliff (suboptimal but safe).

### Visual explanations

```
Q-Learning vs SARSA update:

  Q-Learning (off-policy):
    Q(S,A) += alpha * [R + gamma * max_a' Q(S',a') - Q(S,A)]
                                   ^^^
                              best possible action

  SARSA (on-policy):
    Q(S,A) += alpha * [R + gamma * Q(S',A') - Q(S,A)]
                                      ^^
                              action actually taken

Cliff Walking Example:
  S . . . . . . . . . . G
  C C C C C C C C C C C C   (C = cliff, -100 reward)

  Q-learning path: S--->-->-->-->-->-->-->-->-->G  (along cliff edge)
  SARSA path:      S                             G
                   |                             |
                   +-->-->-->-->-->-->-->-->-->-->+  (safe path away from cliff)
```

---

## Hands-on Exploration

1. Implement SARSA on a grid world. Compare the learned Q-values to Q-learning's Q-values.
2. Create a "cliff walking" environment. Run both Q-learning and SARSA with epsilon=0.1. Compare their average rewards.
3. Track the number of times each algorithm falls off the cliff during training. SARSA should fall less often.
4. Vary epsilon and observe how the gap between Q-learning and SARSA changes. At epsilon=0, they should converge to the same policy.

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

    // --- Cliff Walking Environment ---
    // 4 rows x 12 columns
    // Start: bottom-left (3,0), Goal: bottom-right (3,11)
    // Cliff: bottom row between start and goal (3,1) to (3,10)
    let rows = 4;
    let cols = 12;
    let n_states = rows * cols;
    let n_actions = 4; // UP, DOWN, LEFT, RIGHT
    let start = 3 * cols + 0;  // (3, 0)
    let goal = 3 * cols + 11;  // (3, 11)

    let to_rc = |s: usize| -> (usize, usize) { (s / cols, s % cols) };
    let to_s = |r: usize, c: usize| -> usize { r * cols + c };

    let is_cliff = |s: usize| -> bool {
        let (r, c) = to_rc(s);
        r == 3 && c >= 1 && c <= 10
    };

    let step = |state: usize, action: usize| -> (usize, f64, bool) {
        if state == goal { return (state, 0.0, true); }
        let (r, c) = to_rc(state);
        let (nr, nc) = match action {
            0 => (if r > 0 { r - 1 } else { r }, c),
            1 => (if r < rows - 1 { r + 1 } else { r }, c),
            2 => (r, if c > 0 { c - 1 } else { c }),
            3 => (r, if c < cols - 1 { c + 1 } else { c }),
            _ => (r, c),
        };
        let ns = to_s(nr, nc);
        if is_cliff(ns) {
            return (start, -100.0, false); // fall off cliff, back to start
        }
        if ns == goal {
            return (ns, -1.0, true);
        }
        (ns, -1.0, false)
    };

    let epsilon_greedy = |q: &[Vec<f64>], state: usize, eps: f64, rng: &mut u64| -> usize {
        if rand_f64(rng) < eps {
            rand_int(rng, n_actions)
        } else {
            let mut best = 0;
            for a in 1..n_actions {
                if q[state][a] > q[state][best] { best = a; }
            }
            best
        }
    };

    // --- Parameters ---
    let alpha = 0.5;
    let gamma = 1.0; // undiscounted for cliff walking
    let epsilon = 0.1;
    let n_episodes = 500;

    // --- Q-Learning ---
    let mut q_ql = vec![vec![0.0; n_actions]; n_states];
    let mut ql_rewards: Vec<f64> = Vec::new();
    let mut ql_falls = 0;

    for _ in 0..n_episodes {
        let mut state = start;
        let mut total_reward = 0.0;

        for _ in 0..500 {
            let action = epsilon_greedy(&q_ql, state, epsilon, &mut rng);
            let (next_state, reward, done) = step(state, action);

            if reward <= -100.0 { ql_falls += 1; }

            let max_q_next = q_ql[next_state].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let target = reward + gamma * max_q_next * (1.0 - done as u8 as f64);
            q_ql[state][action] += alpha * (target - q_ql[state][action]);

            total_reward += reward;
            state = next_state;
            if done { break; }
        }
        ql_rewards.push(total_reward);
    }

    // --- SARSA ---
    let mut q_sarsa = vec![vec![0.0; n_actions]; n_states];
    let mut sarsa_rewards: Vec<f64> = Vec::new();
    let mut sarsa_falls = 0;

    for _ in 0..n_episodes {
        let mut state = start;
        let mut action = epsilon_greedy(&q_sarsa, state, epsilon, &mut rng);
        let mut total_reward = 0.0;

        for _ in 0..500 {
            let (next_state, reward, done) = step(state, action);

            if reward <= -100.0 { sarsa_falls += 1; }

            let next_action = epsilon_greedy(&q_sarsa, next_state, epsilon, &mut rng);

            // SARSA update: uses Q(S', A') not max Q(S', *)
            let target = reward + gamma * q_sarsa[next_state][next_action]
                * (1.0 - done as u8 as f64);
            q_sarsa[state][action] += alpha * (target - q_sarsa[state][action]);

            total_reward += reward;
            state = next_state;
            action = next_action;
            if done { break; }
        }
        sarsa_rewards.push(total_reward);
    }

    // --- Results ---
    println!("=== Cliff Walking: Q-Learning vs SARSA ===\n");
    println!("Grid: {}x{}, epsilon={}, alpha={}", rows, cols, epsilon, alpha);
    println!("Cliff penalty: -100, step cost: -1\n");

    // Display environment
    println!("Environment:");
    for r in 0..rows {
        let mut row = String::new();
        for c in 0..cols {
            let s = to_s(r, c);
            if s == start { row.push_str("S "); }
            else if s == goal { row.push_str("G "); }
            else if is_cliff(s) { row.push_str("C "); }
            else { row.push_str(". "); }
        }
        println!("  {}", row);
    }

    // Learning curves
    println!("\n=== Learning Curves (moving average) ===\n");
    println!("{:>12} {:>15} {:>15}", "Episodes", "Q-Learning", "SARSA");
    println!("{}", "-".repeat(45));
    for w in [0, 100, 200, 300, 400] {
        let end = (w + 100).min(n_episodes);
        let ql_avg: f64 = ql_rewards[w..end].iter().sum::<f64>() / (end - w) as f64;
        let sarsa_avg: f64 = sarsa_rewards[w..end].iter().sum::<f64>() / (end - w) as f64;
        println!("{:>5}-{:<5} {:>15.1} {:>15.1}", w, end, ql_avg, sarsa_avg);
    }

    println!("\n=== Safety Comparison ===");
    println!("Q-Learning cliff falls: {}", ql_falls);
    println!("SARSA cliff falls:      {}", sarsa_falls);

    // Display learned policies
    let arrows = ['^', 'v', '<', '>'];
    for (name, q_table) in [("Q-Learning", &q_ql), ("SARSA", &q_sarsa)] {
        println!("\n{} Policy:", name);
        for r in 0..rows {
            let mut row = String::new();
            for c in 0..cols {
                let s = to_s(r, c);
                if s == start { row.push_str("S "); }
                else if s == goal { row.push_str("G "); }
                else if is_cliff(s) { row.push_str("C "); }
                else {
                    let best = (0..n_actions)
                        .max_by(|&a, &b| q_table[s][a].partial_cmp(&q_table[s][b]).unwrap())
                        .unwrap();
                    row.push_str(&format!("{} ", arrows[best]));
                }
            }
            println!("  {}", row);
        }
    }

    // Final performance comparison
    let ql_final: f64 = ql_rewards[400..].iter().sum::<f64>() / 100.0;
    let sarsa_final: f64 = sarsa_rewards[400..].iter().sum::<f64>() / 100.0;

    println!("\n=== Summary ===");
    println!("Final avg reward (last 100 episodes):");
    println!("  Q-Learning: {:.1}", ql_final);
    println!("  SARSA:      {:.1}", sarsa_final);
    println!("\nSARSA is safer (fewer cliff falls) but may take a longer path.");
    println!("Q-Learning finds the optimal path but falls off the cliff more during training.");

    println!();
    println!("kata_metric(\"ql_final_reward\", {:.1})", ql_final);
    println!("kata_metric(\"sarsa_final_reward\", {:.1})", sarsa_final);
    println!("kata_metric(\"ql_cliff_falls\", {})", ql_falls);
    println!("kata_metric(\"sarsa_cliff_falls\", {})", sarsa_falls);
}
```

---

## Key Takeaways

- **SARSA is on-policy: it updates using the action actually taken,** not the best possible action. This makes it learn a policy consistent with its exploration behavior.
- **Q-learning is off-policy: it updates toward the optimal action,** learning the best policy even while exploring. This can be risky in dangerous environments.
- **SARSA learns safer policies in environments with cliffs or penalties,** because it accounts for the fact that exploration might lead to bad outcomes.
- **As epsilon approaches zero, SARSA and Q-learning converge** to the same policy, since the agent stops exploring and both algorithms target optimal behavior.
