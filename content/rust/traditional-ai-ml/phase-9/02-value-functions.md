# Value Functions

> Phase 9 — Reinforcement Learning | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

A value function answers the question: "How good is it to be in a particular state (or to take a particular action in a state)?" The **state-value function** V(s) gives the expected cumulative discounted reward starting from state s and following a specific policy. The **action-value function** Q(s,a) gives the expected return from taking action a in state s and then following the policy.

Value functions are the central concept in reinforcement learning. If you know the optimal value function V*(s) or Q*(s,a), you can derive the optimal policy directly: always move to the highest-value next state, or always take the highest Q-value action. The challenge is computing these value functions, which leads to the Bellman equations -- recursive relationships that express each state's value in terms of its successor states' values.

The Bellman equation for V under policy pi states: V_pi(s) = sum_a pi(s,a) * sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V_pi(s')]. This recursive structure is what makes dynamic programming approaches possible -- you can iteratively solve for V by repeatedly applying this equation until convergence.

### Why naive approaches fail

Computing V by simulating thousands of episodes (Monte Carlo) is slow and high-variance. You need many episodes to get reliable estimates, and each episode must run to completion. The Bellman equation provides a much more efficient path: instead of sampling, it exploits the recursive structure of the problem to compute exact values through iterative updates.

### Mental models

- **V(s) as a heat map**: high-value states are "hot" (near rewards), low-value states are "cold" (far from rewards or near traps). The agent should flow toward the heat.
- **Bellman equation as a consistency condition**: a correct value function must satisfy the Bellman equation at every state. If it does not, the values are wrong.
- **Backup operation**: computing V(s) by looking one step ahead to successor states is called a "backup" -- you back up information from future states to the current state.

### Visual explanations

```
State-Value Function V(s):
  +------+------+------+------+
  | 4.21 | 5.68 | 7.34 | 8.50 |
  +------+------+------+------+
  | 3.12 | TRAP | 8.02 | GOAL |
  +------+------+------+------+
  Values increase toward goal, decrease near trap.

Bellman Equation (state-value):
  V(s) = sum_a pi(s,a) * sum_s' P(s'|s,a) * [R + gamma * V(s')]

  For deterministic policy (one action per state):
  V(s) = sum_s' P(s'|s, pi(s)) * [R(s,pi(s),s') + gamma * V(s')]

Action-Value Function Q(s,a):
  State s5: Q(s5, UP)    = 3.8
            Q(s5, DOWN)  = 2.1
            Q(s5, LEFT)  = 3.0
            Q(s5, RIGHT) = 5.2  <-- best action
```

---

## Hands-on Exploration

1. Define a small grid world. Initialize V(s) = 0 for all states. Apply one iteration of the Bellman equation and observe how values start to propagate from the reward states.
2. Repeat the Bellman update for 50 iterations and watch V(s) converge. Plot the convergence curve.
3. Compute Q(s,a) for all state-action pairs. Derive the greedy policy from Q.
4. Compare the value function under a random policy vs the greedy policy.

---

## Live Code

```rust
fn main() {
    // --- 4x4 Grid World ---
    let grid = 4;
    let n_states = grid * grid;
    let n_actions = 4; // UP, DOWN, LEFT, RIGHT
    let goal = 15;
    let trap = 7;
    let gamma = 0.95;
    let step_r = -0.1;
    let goal_r = 10.0;
    let trap_r = -10.0;

    let to_rc = |s: usize| -> (usize, usize) { (s / grid, s % grid) };
    let to_s = |r: usize, c: usize| -> usize { r * grid + c };

    // Deterministic transitions (no slip for clarity)
    let transition = |s: usize, a: usize| -> (usize, f64) {
        if s == goal || s == trap { return (s, 0.0); }
        let (r, c) = to_rc(s);
        let (nr, nc) = match a {
            0 => (if r > 0 { r - 1 } else { r }, c),
            1 => (if r < grid - 1 { r + 1 } else { r }, c),
            2 => (r, if c > 0 { c - 1 } else { c }),
            3 => (r, if c < grid - 1 { c + 1 } else { c }),
            _ => (r, c),
        };
        let ns = to_s(nr, nc);
        let reward = if ns == goal { goal_r }
                     else if ns == trap { trap_r }
                     else { step_r };
        (ns, reward)
    };

    // --- Policy Evaluation: Iterative Bellman updates ---
    println!("=== Value Functions ===\n");

    // Random policy: uniform over all actions
    let random_policy: Vec<Vec<f64>> = (0..n_states)
        .map(|_| vec![0.25; n_actions])
        .collect();

    // Iterative policy evaluation for V(s)
    let mut v = vec![0.0; n_states];
    let mut convergence = Vec::new();

    println!("=== Iterative Policy Evaluation (random policy) ===\n");
    println!("{:>5} {:>12} {:>12}", "Iter", "Max Change", "V(state 0)");
    println!("{}", "-".repeat(32));

    for iter in 0..100 {
        let mut v_new = vec![0.0; n_states];
        let mut max_change = 0.0_f64;

        for s in 0..n_states {
            if s == goal || s == trap {
                v_new[s] = 0.0;
                continue;
            }

            let mut value = 0.0;
            for a in 0..n_actions {
                let (ns, r) = transition(s, a);
                value += random_policy[s][a] * (r + gamma * v[ns]);
            }
            v_new[s] = value;
            max_change = max_change.max((v[s] - v_new[s]).abs());
        }

        v = v_new;
        convergence.push(max_change);

        if iter < 5 || iter == 9 || iter == 19 || iter == 49 || iter == 99 {
            println!("{:>5} {:>12.6} {:>12.4}", iter + 1, max_change, v[0]);
        }

        if max_change < 1e-8 { break; }
    }

    // Display V(s) as grid
    println!("\nV(s) under random policy:");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            if s == goal { row.push_str("  GOAL  "); }
            else if s == trap { row.push_str("  TRAP  "); }
            else { row.push_str(&format!(" {:+.3} ", v[s])); }
        }
        println!("  {}", row);
    }

    // --- Compute Q(s,a) from V(s) ---
    println!("\n=== Action-Value Function Q(s,a) ===\n");
    let mut q = vec![vec![0.0; n_actions]; n_states];

    for s in 0..n_states {
        for a in 0..n_actions {
            let (ns, r) = transition(s, a);
            q[s][a] = r + gamma * v[ns];
        }
    }

    // Show Q values for a few states
    let action_names = ["UP", "DOWN", "LEFT", "RIGHT"];
    for &s in &[0, 5, 10, 14] {
        let (r, c) = to_rc(s);
        println!("  State {} (r={}, c={}):", s, r, c);
        for a in 0..n_actions {
            let best = q[s].iter().cloned().fold(f64::MIN, f64::max);
            let marker = if (q[s][a] - best).abs() < 1e-6 { " <-- best" } else { "" };
            println!("    Q(s{}, {:>5}) = {:+.4}{}", s, action_names[a], q[s][a], marker);
        }
    }

    // --- Greedy policy from V ---
    println!("\n=== Greedy Policy (derived from V) ===\n");
    let arrows = ['^', 'v', '<', '>'];
    let mut greedy_policy = vec![0; n_states];

    for s in 0..n_states {
        if s == goal || s == trap { continue; }
        let best_a = (0..n_actions)
            .max_by(|&a, &b| q[s][a].partial_cmp(&q[s][b]).unwrap())
            .unwrap();
        greedy_policy[s] = best_a;
    }

    println!("Greedy policy arrows:");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            let ch = if s == goal { 'G' }
                     else if s == trap { 'X' }
                     else { arrows[greedy_policy[s]] };
            row.push_str(&format!("  {}  ", ch));
        }
        println!("  {}", row);
    }

    // --- Evaluate greedy policy ---
    let det_greedy: Vec<Vec<f64>> = (0..n_states).map(|s| {
        let mut p = vec![0.0; n_actions];
        p[greedy_policy[s]] = 1.0;
        p
    }).collect();

    let mut v_greedy = vec![0.0; n_states];
    for _ in 0..100 {
        let mut v_new = vec![0.0; n_states];
        for s in 0..n_states {
            if s == goal || s == trap { continue; }
            for a in 0..n_actions {
                let (ns, r) = transition(s, a);
                v_new[s] += det_greedy[s][a] * (r + gamma * v_greedy[ns]);
            }
        }
        v_greedy = v_new;
    }

    println!("\nV(s) under greedy policy:");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            if s == goal { row.push_str("  GOAL  "); }
            else if s == trap { row.push_str("  TRAP  "); }
            else { row.push_str(&format!(" {:+.3} ", v_greedy[s])); }
        }
        println!("  {}", row);
    }

    // --- Compare ---
    println!("\n=== Comparison ===");
    println!("  V(start) random policy: {:+.4}", v[0]);
    println!("  V(start) greedy policy: {:+.4}", v_greedy[0]);
    println!("  Improvement: {:.1}x", v_greedy[0] / v[0]);

    println!();
    println!("kata_metric(\"v_random_start\", {:.4})", v[0]);
    println!("kata_metric(\"v_greedy_start\", {:.4})", v_greedy[0]);
    println!("kata_metric(\"convergence_iters\", {})", convergence.len());
}
```

---

## Key Takeaways

- **V(s) measures expected cumulative reward from a state under a given policy.** It is the fundamental measure of "how good is this state?"
- **Q(s,a) measures expected return from taking action a in state s.** The optimal policy is simply: pick the action with the highest Q value.
- **The Bellman equation expresses value recursively.** Each state's value depends on its successor states' values, enabling iterative computation.
- **Policy evaluation converges to the true value function.** Repeated Bellman updates provably converge, revealing the quality of any fixed policy.
