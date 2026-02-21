# Policy Iteration

> Phase 9 — Reinforcement Learning | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

Policy iteration is a dynamic programming algorithm that finds the **optimal policy** for an MDP when the transition probabilities and rewards are fully known. It alternates between two steps: **policy evaluation** (compute V(s) for the current policy) and **policy improvement** (update the policy to be greedy with respect to the current V). This alternation provably converges to the optimal policy in a finite number of iterations.

The algorithm starts with an arbitrary policy (e.g., random). It evaluates that policy to get V(s), then improves the policy by choosing the action that maximizes the expected value at each state. The improved policy is at least as good as the old one. If no improvement is possible, the policy is already optimal.

**Value iteration** is a closely related algorithm that combines evaluation and improvement into a single update: V(s) = max_a [R(s,a) + gamma * sum P(s'|s,a) V(s')]. Instead of fully evaluating a policy before improving, it takes one evaluation step and immediately improves. Value iteration is simpler to implement but may take more iterations to converge.

### Why naive approaches fail

Trying to find the optimal policy by enumerating all possible policies is intractable -- for n states and m actions, there are m^n possible deterministic policies. Policy iteration avoids this exponential search by making local improvements that are guaranteed to be globally optimal at convergence. Each iteration reduces the number of possible policies remaining, and convergence is guaranteed in at most m^n iterations (usually far fewer).

### Mental models

- **Policy iteration as alternating refinement**: "given my current strategy, how good is each state?" then "given the values, what is a better strategy?" Repeat until stable.
- **Value iteration as a shortcut**: instead of fully evaluating before improving, take one step of evaluation and improve immediately. Faster per iteration, but more iterations needed.
- **Convergence as equilibrium**: the algorithm converges when the value function and policy are mutually consistent -- neither wants to change given the other.

### Visual explanations

```
Policy Iteration:

  Start: random policy pi_0

  Repeat:
    1. Policy Evaluation: compute V_pi(s) for all s
       (solve Bellman equations for current policy)

    2. Policy Improvement: for each state s,
       pi_new(s) = argmax_a sum P(s'|s,a)[R + gamma*V_pi(s')]

    3. If pi_new == pi_old: STOP (optimal!)
       Else: pi = pi_new, go to step 1.

  Typically converges in 3-10 iterations!

Value Iteration:
  V(s) = max_a [ R(s,a) + gamma * sum P(s'|s,a) * V(s') ]
  Repeat until V converges.
  Extract policy: pi(s) = argmax_a [same expression]
```

---

## Hands-on Exploration

1. Implement policy evaluation (iterative Bellman updates until convergence). Verify it produces the correct V for a known policy.
2. Implement policy improvement: derive the greedy policy from V. Show that the new policy is at least as good.
3. Combine into full policy iteration. Track how many outer iterations are needed to converge.
4. Implement value iteration and compare: same final result, different number of iterations.

---

## Live Code

```rust
fn main() {
    // --- 5x5 Grid World ---
    let grid = 5;
    let n_states = grid * grid;
    let n_actions = 4;
    let goal = 24;      // bottom-right
    let trap = 12;      // center
    let gamma = 0.95;
    let action_names = ["UP", "DOWN", "LEFT", "RIGHT"];

    let to_rc = |s: usize| -> (usize, usize) { (s / grid, s % grid) };
    let to_s = |r: usize, c: usize| -> usize { r * grid + c };

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
        let reward = if ns == goal { 10.0 }
                     else if ns == trap { -10.0 }
                     else { -0.1 };
        (ns, reward)
    };

    // === Policy Evaluation ===
    fn policy_eval(
        policy: &[usize], n_states: usize, n_actions: usize,
        gamma: f64, goal: usize, trap: usize,
        transition: &dyn Fn(usize, usize) -> (usize, f64),
    ) -> Vec<f64> {
        let mut v = vec![0.0; n_states];
        for _ in 0..500 {
            let mut max_change = 0.0_f64;
            for s in 0..n_states {
                if s == goal || s == trap { continue; }
                let a = policy[s];
                let (ns, r) = transition(s, a);
                let new_v = r + gamma * v[ns];
                max_change = max_change.max((v[s] - new_v).abs());
                v[s] = new_v;
            }
            if max_change < 1e-8 { break; }
        }
        v
    }

    // === Policy Improvement ===
    fn policy_improve(
        v: &[f64], n_states: usize, n_actions: usize,
        gamma: f64, goal: usize, trap: usize,
        transition: &dyn Fn(usize, usize) -> (usize, f64),
    ) -> Vec<usize> {
        let mut policy = vec![0; n_states];
        for s in 0..n_states {
            if s == goal || s == trap { continue; }
            let mut best_val = f64::NEG_INFINITY;
            for a in 0..n_actions {
                let (ns, r) = transition(s, a);
                let val = r + gamma * v[ns];
                if val > best_val {
                    best_val = val;
                    policy[s] = a;
                }
            }
        }
        policy
    }

    // === Policy Iteration ===
    println!("=== Policy Iteration ===\n");

    let mut policy: Vec<usize> = vec![0; n_states]; // start: all UP
    let mut pi_iter = 0;

    loop {
        pi_iter += 1;

        // Policy evaluation
        let v = policy_eval(&policy, n_states, n_actions, gamma, goal, trap, &transition);

        // Policy improvement
        let new_policy = policy_improve(&v, n_states, n_actions, gamma, goal, trap, &transition);

        // Check convergence
        let changed = policy.iter().zip(&new_policy)
            .filter(|(a, b)| a != b).count();

        println!("Iteration {}: {} states changed policy, V(start)={:.4}",
            pi_iter, changed, v[0]);

        if changed == 0 {
            println!("\nPolicy iteration converged in {} iterations!", pi_iter);
            break;
        }
        policy = new_policy;

        if pi_iter > 50 { break; } // safety
    }

    // Final evaluation
    let v_pi = policy_eval(&policy, n_states, n_actions, gamma, goal, trap, &transition);

    // Display policy
    let arrows = ['^', 'v', '<', '>'];
    println!("\nOptimal Policy (Policy Iteration):");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            let ch = if s == goal { 'G' }
                     else if s == trap { 'X' }
                     else { arrows[policy[s]] };
            row.push_str(&format!("  {}  ", ch));
        }
        println!("  {}", row);
    }

    println!("\nV*(s):");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            if s == goal { row.push_str("  GOAL "); }
            else if s == trap { row.push_str("  TRAP "); }
            else { row.push_str(&format!("{:+.2} ", v_pi[s])); }
        }
        println!("  {}", row);
    }

    // === Value Iteration ===
    println!("\n=== Value Iteration ===\n");

    let mut v_vi = vec![0.0; n_states];
    let mut vi_iter = 0;

    loop {
        vi_iter += 1;
        let mut max_change = 0.0_f64;

        for s in 0..n_states {
            if s == goal || s == trap { continue; }
            let mut best_val = f64::NEG_INFINITY;
            for a in 0..n_actions {
                let (ns, r) = transition(s, a);
                let val = r + gamma * v_vi[ns];
                if val > best_val { best_val = val; }
            }
            max_change = max_change.max((v_vi[s] - best_val).abs());
            v_vi[s] = best_val;
        }

        if vi_iter <= 5 || vi_iter % 20 == 0 || max_change < 1e-8 {
            println!("Iteration {:>3}: max_change = {:.8}, V(start) = {:.4}",
                vi_iter, max_change, v_vi[0]);
        }

        if max_change < 1e-8 {
            println!("\nValue iteration converged in {} iterations!", vi_iter);
            break;
        }
        if vi_iter > 500 { break; }
    }

    // Extract policy from V
    let vi_policy = policy_improve(&v_vi, n_states, n_actions, gamma, goal, trap, &transition);

    println!("\nOptimal Policy (Value Iteration):");
    for r in 0..grid {
        let mut row = String::new();
        for c in 0..grid {
            let s = to_s(r, c);
            let ch = if s == goal { 'G' }
                     else if s == trap { 'X' }
                     else { arrows[vi_policy[s]] };
            row.push_str(&format!("  {}  ", ch));
        }
        println!("  {}", row);
    }

    // --- Comparison ---
    println!("\n=== Comparison ===");
    println!("Policy Iteration: {} outer iterations", pi_iter);
    println!("Value Iteration:  {} iterations", vi_iter);

    let policies_match = policy.iter().zip(&vi_policy)
        .all(|(a, b)| a == b || *a == goal || *a == trap);
    println!("Policies match: {}", policies_match);

    let v_diff: f64 = v_pi.iter().zip(&v_vi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Max V difference: {:.8}", v_diff);
    println!("V*(start): PI={:.4}, VI={:.4}", v_pi[0], v_vi[0]);

    println!();
    println!("kata_metric(\"policy_iteration_iters\", {})", pi_iter);
    println!("kata_metric(\"value_iteration_iters\", {})", vi_iter);
    println!("kata_metric(\"v_optimal_start\", {:.4})", v_pi[0]);
}
```

---

## Key Takeaways

- **Policy iteration alternates between evaluation and improvement,** converging to the optimal policy in surprisingly few iterations (often 3-10).
- **Value iteration combines evaluation and improvement into one step,** requiring more iterations but simpler implementation.
- **Both methods find the same optimal policy.** They are different paths to the same answer, trading off iteration count vs computation per iteration.
- **These methods require a complete model (transition probabilities and rewards).** When the model is unknown, model-free methods like Q-learning are needed.
