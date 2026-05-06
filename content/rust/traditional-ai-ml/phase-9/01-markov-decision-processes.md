# Markov Decision Processes

> Phase 9 — Reinforcement Learning | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

A Markov Decision Process (MDP) is the mathematical framework for sequential decision-making under uncertainty. Unlike supervised learning where you have labeled examples, in an MDP an agent must learn to make a sequence of decisions (actions) that maximize cumulative reward over time. The key challenge: actions have delayed consequences -- a decision now affects not just the immediate reward but all future rewards.

An MDP is defined by five components: **States** (S) -- where the agent can be; **Actions** (A) -- what the agent can do; **Transition function** P(s'|s,a) -- the probability of reaching state s' after taking action a in state s; **Reward function** R(s,a,s') -- the immediate payoff; and **Discount factor** gamma -- how much future rewards are worth compared to immediate ones.

The **Markov property** is the key assumption: the future depends only on the current state, not on the history of how we got there. This memoryless property is what makes MDPs tractable -- the agent only needs to know where it is, not where it has been.

### Why naive approaches fail

Greedy strategies that maximize immediate reward often fail because they ignore long-term consequences. A chess player who only captures the most valuable piece available might walk into a trap. The discount factor gamma balances short-term and long-term thinking: gamma near 0 makes the agent myopic; gamma near 1 makes it far-sighted.

### Mental models

- **Chess game**: each board position is a state, each move is an action, capturing a piece is a reward. The goal is not to maximize the next capture but to win the game.
- **Discount as impatience**: $100 today is worth more than $100 next year. Gamma encodes this preference -- lower gamma means more impatient.
- **Markov property as amnesia**: the agent has amnesia about the past. All the information it needs is encoded in the current state.

### Visual explanations

```
MDP Components:

  States:  S = {s0, s1, s2, s3_goal, s4_trap}
  Actions: A = {up, down, left, right}
  Transitions: P(s'|s,a) - stochastic (may slip)
  Rewards: R(goal) = +10, R(trap) = -10, R(step) = -0.1
  Gamma:   0.95

  Grid World:
  +-----+-----+-----+
  | s0  | s1  | s2  |
  +-----+-----+-----+
  | s3  | TRAP| GOAL|
  +-----+-----+-----+

  Policy: pi(s) = action to take in state s
  Goal: find pi* that maximizes expected cumulative discounted reward
    E[R_0 + gamma*R_1 + gamma^2*R_2 + ...]
```

---

## Hands-on Exploration

1. Define a simple 3x3 grid world MDP with states, actions, transitions, and rewards. Enumerate all components explicitly.
2. Compute the expected return for a specific policy by simulating many episodes and averaging the cumulative discounted rewards.
3. Try two different policies (random vs always-go-right) and compare their expected returns.
4. Vary gamma from 0.1 to 0.99 and observe how it changes which policy is preferred.

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

    // --- 4x4 Grid World MDP ---
    let grid_size = 4;
    let n_states = grid_size * grid_size;
    let n_actions = 4; // 0=up, 1=down, 2=left, 3=right
    let goal_state = 15; // bottom-right
    let trap_state = 11; // a trap
    let action_names = ["UP", "DOWN", "LEFT", "RIGHT"];

    let step_reward = -0.1;
    let goal_reward = 10.0;
    let trap_reward = -10.0;
    let slip_prob = 0.1; // 10% chance of random action

    let to_rc = |s: usize| -> (usize, usize) { (s / grid_size, s % grid_size) };
    let to_state = |r: usize, c: usize| -> usize { r * grid_size + c };

    // Transition function
    let step = |state: usize, action: usize, rng: &mut u64| -> (usize, f64, bool) {
        if state == goal_state || state == trap_state {
            return (state, 0.0, true);
        }

        // Slip: with probability slip_prob, take random action
        let actual_action = if rand_f64(rng) < slip_prob {
            rand_int(rng, n_actions)
        } else {
            action
        };

        let (r, c) = to_rc(state);
        let (nr, nc) = match actual_action {
            0 => (if r > 0 { r - 1 } else { r }, c),
            1 => (if r < grid_size - 1 { r + 1 } else { r }, c),
            2 => (r, if c > 0 { c - 1 } else { c }),
            3 => (r, if c < grid_size - 1 { c + 1 } else { c }),
            _ => (r, c),
        };
        let next_state = to_state(nr, nc);

        let reward = if next_state == goal_state { goal_reward }
                     else if next_state == trap_state { trap_reward }
                     else { step_reward };
        let done = next_state == goal_state || next_state == trap_state;

        (next_state, reward, done)
    };

    // --- Display the grid ---
    println!("=== Markov Decision Process: 4x4 Grid World ===\n");
    println!("Grid:");
    for r in 0..grid_size {
        let mut row_str = String::new();
        for c in 0..grid_size {
            let s = to_state(r, c);
            let cell = if s == goal_state { " GOAL " }
                       else if s == trap_state { " TRAP " }
                       else { "  .   " };
            row_str.push_str(&format!("|{}", cell));
        }
        row_str.push('|');
        println!("  {}", row_str);
    }
    println!("\nMDP components:");
    println!("  States:      {} (4x4 grid)", n_states);
    println!("  Actions:     {} ({:?})", n_actions, action_names);
    println!("  Goal:        state {} (reward = {:+.1})", goal_state, goal_reward);
    println!("  Trap:        state {} (reward = {:+.1})", trap_state, trap_reward);
    println!("  Step reward: {:+.1}", step_reward);
    println!("  Slip prob:   {:.0}%", slip_prob * 100.0);

    // --- Evaluate a policy by simulation ---
    fn evaluate_policy(
        policy: &[usize], gamma: f64, n_episodes: usize,
        grid_size: usize, goal: usize, trap: usize,
        step_r: f64, goal_r: f64, trap_r: f64, slip: f64,
        rng: &mut u64,
    ) -> f64 {
        let n_actions = 4;
        let mut total_return = 0.0;

        for _ in 0..n_episodes {
            let mut state = 0; // start top-left
            let mut episode_return = 0.0;
            let mut discount = 1.0;

            for _ in 0..100 { // max steps
                if state == goal || state == trap { break; }

                let action = policy[state];

                // Take step with possible slip
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((*rng >> 33) as f64) / 2147483648.0;
                let actual = if u < slip {
                    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((*rng >> 33) as usize) % n_actions
                } else { action };

                let (r, c) = (state / grid_size, state % grid_size);
                let (nr, nc) = match actual {
                    0 => (if r > 0 { r - 1 } else { r }, c),
                    1 => (if r < grid_size - 1 { r + 1 } else { r }, c),
                    2 => (r, if c > 0 { c - 1 } else { c }),
                    3 => (r, if c < grid_size - 1 { c + 1 } else { c }),
                    _ => (r, c),
                };
                let next = nr * grid_size + nc;

                let reward = if next == goal { goal_r }
                             else if next == trap { trap_r }
                             else { step_r };

                episode_return += discount * reward;
                discount *= gamma;
                state = next;
            }
            total_return += episode_return;
        }
        total_return / n_episodes as f64
    }

    // --- Compare policies ---
    println!("\n=== Policy Evaluation (1000 episodes each) ===\n");

    // Policy 1: Random
    let random_policy: Vec<usize> = (0..n_states).map(|_| rand_int(&mut rng, n_actions)).collect();

    // Policy 2: Always go right then down
    let right_down_policy: Vec<usize> = (0..n_states).map(|s| {
        let (r, c) = to_rc(s);
        if c < grid_size - 1 { 3 } // right
        else { 1 } // down
    }).collect();

    // Policy 3: Navigate toward goal avoiding trap
    let smart_policy: Vec<usize> = (0..n_states).map(|s| {
        let (r, c) = to_rc(s);
        let (gr, gc) = to_rc(goal_state);
        let (tr, _tc) = to_rc(trap_state);
        // Avoid trap row if possible
        if r == tr && c < grid_size - 1 && r > 0 { 0 } // go up to avoid trap
        else if c < gc { 3 } // go right
        else if r < gr { 1 } // go down
        else if r > gr { 0 } // go up
        else { 3 } // default right
    }).collect();

    let gammas = [0.5, 0.9, 0.95, 0.99];

    println!("{:<25} {:>8} {:>8} {:>8} {:>8}", "Policy", "g=0.50", "g=0.90", "g=0.95", "g=0.99");
    println!("{}", "-".repeat(60));

    let policies: Vec<(&str, &[usize])> = vec![
        ("Random", &random_policy),
        ("Right-then-Down", &right_down_policy),
        ("Smart (avoid trap)", &smart_policy),
    ];

    for (name, policy) in &policies {
        let mut vals = Vec::new();
        for &gamma in &gammas {
            let value = evaluate_policy(
                policy, gamma, 1000, grid_size, goal_state, trap_state,
                step_reward, goal_reward, trap_reward, slip_prob, &mut rng,
            );
            vals.push(value);
        }
        println!("{:<25} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            name, vals[0], vals[1], vals[2], vals[3]);
    }

    // --- Display policies as arrow maps ---
    let arrows = ['^', 'v', '<', '>'];
    println!("\n=== Policy Visualization ===");
    for (name, policy) in &policies {
        println!("\n{}:", name);
        for r in 0..grid_size {
            let mut row = String::new();
            for c in 0..grid_size {
                let s = to_state(r, c);
                let ch = if s == goal_state { 'G' }
                         else if s == trap_state { 'X' }
                         else { arrows[policy[s]] };
                row.push_str(&format!("  {}  ", ch));
            }
            println!("  {}", row);
        }
    }

    // --- Demonstrate the Markov property ---
    println!("\n=== Markov Property Demonstration ===");
    println!("Transition probabilities from state 5 (row 1, col 1), action=RIGHT:");
    let test_state = 5;
    let test_action = 3; // RIGHT
    let mut outcome_counts = vec![0; n_states];
    let n_trials = 10000;

    for _ in 0..n_trials {
        let (next, _, _) = step(test_state, test_action, &mut rng);
        outcome_counts[next] += 1;
    }

    for s in 0..n_states {
        if outcome_counts[s] > 0 {
            let (r, c) = to_rc(s);
            println!("  P(s'={} [r={},c={}] | s={}, a=RIGHT) = {:.3}",
                s, r, c, test_state, outcome_counts[s] as f64 / n_trials as f64);
        }
    }
    println!("  (Depends only on current state, not history -- Markov property)");

    let smart_value = evaluate_policy(
        &smart_policy, 0.95, 1000, grid_size, goal_state, trap_state,
        step_reward, goal_reward, trap_reward, slip_prob, &mut rng,
    );
    println!();
    println!("kata_metric(\"smart_policy_value\", {:.3})", smart_value);
    println!("kata_metric(\"n_states\", {})", n_states);
    println!("kata_metric(\"n_actions\", {})", n_actions);
}
```

---

## Key Takeaways

- **MDPs formalize sequential decision-making** with states, actions, transitions, rewards, and a discount factor.
- **The Markov property means the future depends only on the present state,** not the history. This makes the problem tractable.
- **The discount factor gamma controls the planning horizon.** Low gamma makes the agent myopic; high gamma makes it consider long-term consequences.
- **Policy evaluation reveals how good a fixed strategy is.** The expected cumulative discounted reward is the fundamental measure of policy quality.
