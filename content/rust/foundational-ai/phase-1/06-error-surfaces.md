# Error Surfaces

> Phase 1 — What Does It Mean to Learn? | Kata 1.6

---

## Concept & Intuition

### What problem are we solving?

An error surface (or loss landscape) is the surface formed by plotting the loss as a function of the model parameters. For a model with two parameters (weight and bias), the error surface is a 3D landscape where the x and y axes are the parameter values and the z axis is the loss. Training a model means navigating this landscape to find the lowest point — the parameter values that minimize the loss.

For linear regression with MSE loss, the error surface is a convex bowl (a paraboloid). This is wonderful because convex surfaces have a single minimum — there are no local minima to get trapped in. No matter where you start, if you keep moving downhill, you will reach the global minimum. This is why linear regression always finds the optimal solution.

For more complex models (neural networks), the error surface is non-convex: it has multiple valleys, saddle points, and plateaus. Understanding the shape of the error surface is crucial for choosing optimization strategies and understanding why training sometimes fails.

### Why naive approaches fail

Without understanding the error surface, optimization behavior seems mysterious. Why does the model converge sometimes but diverge other times? Why does a small change in learning rate cause catastrophic failure? The answer is always in the shape of the error surface — steep regions cause instability, flat regions cause stagnation, and saddle points cause the illusion of convergence.

### Mental models

- **Loss as terrain**: Imagine the error surface as a physical landscape. Training is like rolling a ball downhill. The ball settles in valleys (minima). Steeper terrain means faster but potentially unstable movement.
- **Contour lines**: Viewed from above, the error surface looks like a topographic map. Circular contours mean uniform curvature. Elongated contours mean the surface is steep in one direction and flat in another — this makes optimization harder.
- **Convex = easy, non-convex = hard**: A convex surface guarantees you can find the global minimum. A non-convex surface means you might find a local minimum instead.

### Visual explanations

```
  Error surface for y = wx + b:

  loss  (viewed from above as contour lines)
    │
    │    ╭────────╮
    │   ╭┤        ├╮
    │  ╭┤│   *    │├╮    * = minimum (optimal w, b)
    │  │││        │││
    │  ╰┤│        │├╯
    │   ╰┤        ├╯
    │    ╰────────╯
    └──────────────── w
                 b (into page)

  The contours are ellipses centered on the optimal point.
```

---

## Hands-on Exploration

1. Compute the loss for a grid of parameter values to map out the error surface.
2. Visualize the error surface as an ASCII contour map.
3. Identify the minimum and observe how the loss changes as you move away from it.

---

## Live Code

```rust
fn main() {
    // === Error Surfaces ===
    // The loss landscape shows how loss varies with model parameters.

    // Simple dataset: y = 2x + 3 + noise
    let data: Vec<(f64, f64)> = vec![
        (1.0, 5.2), (2.0, 7.1), (3.0, 8.8), (4.0, 11.3),
        (5.0, 13.0), (6.0, 14.9), (7.0, 17.2), (8.0, 18.8),
        (9.0, 21.1), (10.0, 23.0),
    ];

    // Loss function: MSE for y = wx + b
    let compute_loss = |w: f64, b: f64| -> f64 {
        let n = data.len() as f64;
        data.iter()
            .map(|(x, y)| {
                let pred = w * x + b;
                (y - pred) * (y - pred)
            })
            .sum::<f64>() / n
    };

    // === Find the optimal parameters by grid search ===
    println!("=== Error Surface Exploration ===\n");
    println!("  Model: y = w*x + b");
    println!("  True parameters: w=2.0, b=3.0\n");

    let w_range = (0.0, 4.0);
    let b_range = (0.0, 6.0);
    let grid_size = 50;

    let mut min_loss = f64::INFINITY;
    let mut best_w = 0.0;
    let mut best_b = 0.0;

    // Compute loss on grid
    let mut loss_grid: Vec<Vec<f64>> = vec![vec![0.0; grid_size]; grid_size];
    for i in 0..grid_size {
        for j in 0..grid_size {
            let w = w_range.0 + (w_range.1 - w_range.0) * i as f64 / (grid_size - 1) as f64;
            let b = b_range.0 + (b_range.1 - b_range.0) * j as f64 / (grid_size - 1) as f64;
            let loss = compute_loss(w, b);
            loss_grid[i][j] = loss;

            if loss < min_loss {
                min_loss = loss;
                best_w = w;
                best_b = b;
            }
        }
    }

    println!("  Grid search result:");
    println!("    Best w = {:.3}, Best b = {:.3}", best_w, best_b);
    println!("    Min loss = {:.4}\n", min_loss);

    // === ASCII contour plot ===
    println!("=== Error Surface Contour Map ===\n");

    let display_h = 25;
    let display_w = 50;

    // Compute display grid
    let mut display_losses: Vec<Vec<f64>> = vec![vec![0.0; display_w]; display_h];
    let mut max_loss = 0.0_f64;
    for i in 0..display_h {
        for j in 0..display_w {
            let w = w_range.0 + (w_range.1 - w_range.0) * j as f64 / (display_w - 1) as f64;
            let b = b_range.0 + (b_range.1 - b_range.0) * i as f64 / (display_h - 1) as f64;
            let loss = compute_loss(w, b);
            display_losses[i][j] = loss;
            if loss > max_loss { max_loss = loss; }
        }
    }

    // Define contour levels
    let contour_chars = [' ', '.', ':', '-', '=', '+', '#', '%', '@'];
    let n_levels = contour_chars.len();

    println!("  b");
    for i in (0..display_h).rev() {
        let b_val = b_range.0 + (b_range.1 - b_range.0) * i as f64 / (display_h - 1) as f64;
        let mut line = String::new();
        for j in 0..display_w {
            let loss = display_losses[i][j];
            // Use log scale for better visualization
            let normalized = (loss.ln() - min_loss.max(0.01).ln())
                / (max_loss.ln() - min_loss.max(0.01).ln());
            let level = (normalized * (n_levels - 1) as f64) as usize;
            let level = level.min(n_levels - 1);

            // Mark the minimum
            let w_val = w_range.0 + (w_range.1 - w_range.0) * j as f64 / (display_w - 1) as f64;
            if (w_val - best_w).abs() < 0.1 && (b_val - best_b).abs() < 0.15 {
                line.push('*');
            } else {
                line.push(contour_chars[level]);
            }
        }
        if i == display_h - 1 || i == display_h / 2 || i == 0 {
            println!("  {:>4.1}|{}", b_val, line);
        } else {
            println!("      |{}", line);
        }
    }
    println!("      +{}", "-".repeat(display_w));
    println!("       {:.1}{}w{}{:.1}",
        w_range.0,
        " ".repeat(display_w / 2 - 4),
        " ".repeat(display_w / 2 - 3),
        w_range.1);
    println!();
    println!("  Legend: ' '=low loss, '@'=high loss, '*'=minimum");
    println!("  The contours form an elliptical bowl (convex surface).\n");

    // === Cross-sections through the error surface ===
    println!("=== Cross-Sections Through the Minimum ===\n");

    // Fix b at optimal, vary w
    println!("  Fixing b={:.2}, varying w:", best_b);
    for i in 0..20 {
        let w = w_range.0 + (w_range.1 - w_range.0) * i as f64 / 19.0;
        let loss = compute_loss(w, best_b);
        let bar_len = (loss / max_loss * 40.0) as usize;
        let bar_len = bar_len.min(40);
        let marker = if (w - best_w).abs() < 0.15 { " ← min" } else { "" };
        println!("    w={:.2}: loss={:>8.2}  |{}|{}",
            w, loss, "█".repeat(bar_len), marker);
    }

    println!();

    // Fix w at optimal, vary b
    println!("  Fixing w={:.2}, varying b:", best_w);
    for i in 0..15 {
        let b = b_range.0 + (b_range.1 - b_range.0) * i as f64 / 14.0;
        let loss = compute_loss(best_w, b);
        let bar_len = (loss / max_loss * 40.0) as usize;
        let bar_len = bar_len.min(40);
        let marker = if (b - best_b).abs() < 0.25 { " ← min" } else { "" };
        println!("    b={:.2}: loss={:>8.2}  |{}|{}",
            b, loss, "█".repeat(bar_len), marker);
    }

    println!();
    println!("  Both cross-sections are parabolas (quadratic) — confirming the");
    println!("  error surface is a convex bowl with a single global minimum.\n");

    // === Gradient at various points ===
    println!("=== Gradients Point Toward the Minimum ===\n");

    let epsilon = 0.001;
    let points = vec![
        (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (3.5, 1.5),
    ];

    println!("  {:>6} {:>6} {:>10} {:>10} {:>10}",
        "w", "b", "∂L/∂w", "∂L/∂b", "loss");
    println!("  {:->6} {:->6} {:->10} {:->10} {:->10}", "", "", "", "", "");

    for (w, b) in &points {
        let loss = compute_loss(*w, *b);
        let grad_w = (compute_loss(w + epsilon, *b) - compute_loss(w - epsilon, *b)) / (2.0 * epsilon);
        let grad_b = (compute_loss(*w, b + epsilon) - compute_loss(*w, b - epsilon)) / (2.0 * epsilon);

        println!("  {:>6.1} {:>6.1} {:>+10.2} {:>+10.2} {:>10.2}", w, b, grad_w, grad_b, loss);
    }

    println!();
    println!("  Gradients are large far from the minimum and near zero at it.");
    println!("  The gradient always points in the direction of steepest ascent.");

    println!();
    println!("Key insight: The error surface shows HOW loss depends on parameters.");
    println!("Convex surfaces guarantee a single minimum. Gradient descent follows the slope down.");
}
```

---

## Key Takeaways

- The error surface maps every possible parameter setting to a loss value — training means finding the lowest point on this surface.
- For linear regression with MSE, the error surface is convex (a bowl shape), guaranteeing a single global minimum with no local traps.
- Contour plots reveal the geometry: elongated contours indicate that one parameter direction is steeper than another, which affects optimization speed.
- Gradients at any point on the surface point toward steepest ascent — negating them gives the direction of steepest descent toward the minimum.
