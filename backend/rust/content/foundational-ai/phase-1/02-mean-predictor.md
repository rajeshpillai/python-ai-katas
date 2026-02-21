# Mean Predictor

> Phase 1 — What Does It Mean to Learn? | Kata 1.2

---

## Concept & Intuition

### What problem are we solving?

The mean predictor is the constant predictor from the previous kata — but now we extend it by computing *conditional* means. Instead of one global average, we compute the average for different subgroups of data. "What is the average price for 3-bedroom houses?" is more useful than "What is the average price?" because it uses information from the features.

This is the first step toward real learning: partitioning the data by feature values and computing group-specific predictions. A conditional mean predictor based on categorical features is essentially a lookup table. For continuous features, we can bin the values and compute the mean within each bin. These simple strategies are surprisingly effective baselines and illustrate a core principle: the more specific your conditioning, the better your predictions — up to a point.

The tension between global and conditional means illustrates the bias-variance tradeoff. A global mean has high bias (it ignores useful information) but low variance (it is stable). A highly conditional mean (e.g., one per unique data point) has low bias but high variance — it overfits. Finding the right level of conditioning is the art of machine learning.

### Why naive approaches fail

Computing a mean for every unique combination of features quickly leads to tiny groups with unreliable averages. If you condition on too many features, some groups will have only one observation, and the "average" is just that single value — pure memorization. This is overfitting in its simplest form.

### Mental models

- **Lookup table**: A conditional mean predictor is just a table: "if bedrooms=3, predict $275K." Simple, interpretable, and often surprisingly effective.
- **From global to local**: As you condition on more features, predictions become more local (specific) but less stable. This is the bias-variance tradeoff in action.
- **Mean as center of gravity**: The mean is the point that minimizes total squared distance from all data points. The conditional mean does this within each subgroup.

### Visual explanations

```
  Global mean:                  Conditional means:
  (one prediction for all)      (one per group)

  price │                       price │
        │  . .   . .                  │  . .      group B mean
        │──────────────────           │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        │  . .   . .                  │  . .
        │     . .                     │     . .    group A mean
        │                             │─ ─ ─ ─ ─ ─
        └───────────── x              └───────────── x
```

---

## Hands-on Exploration

1. Compute the global mean of a target variable as the baseline.
2. Compute conditional means by grouping on a categorical feature and observe the improvement.
3. Bin a continuous feature and compute bin-level means to see how conditioning improves predictions.

---

## Live Code

```rust
fn main() {
    // === Mean Predictor ===
    // From global mean to conditional means: the first step toward learning.

    // Dataset: [sq_feet, bedrooms, neighborhood, price]
    let data: Vec<(f64, i32, &str, f64)> = vec![
        (1400.0, 3, "A", 250000.0),
        (1800.0, 4, "B", 340000.0),
        (1100.0, 2, "A", 195000.0),
        (2200.0, 5, "B", 430000.0),
        (1600.0, 3, "A", 275000.0),
        (900.0,  1, "A", 150000.0),
        (2000.0, 4, "B", 395000.0),
        (1300.0, 2, "A", 220000.0),
        (1700.0, 3, "B", 310000.0),
        (1500.0, 3, "A", 265000.0),
        (2100.0, 4, "B", 410000.0),
        (1000.0, 2, "A", 180000.0),
        (1900.0, 4, "A", 350000.0),
        (1200.0, 2, "B", 230000.0),
        (1650.0, 3, "B", 295000.0),
    ];

    let prices: Vec<f64> = data.iter().map(|d| d.3).collect();
    let n = prices.len() as f64;

    // === Global mean ===
    let global_mean: f64 = prices.iter().sum::<f64>() / n;
    let global_mse: f64 = prices.iter()
        .map(|p| (p - global_mean) * (p - global_mean))
        .sum::<f64>() / n;

    println!("=== Mean Predictor: From Global to Conditional ===\n");
    println!("  1. Global Mean Predictor");
    println!("     Prediction: ${:.0} for everything", global_mean);
    println!("     MSE: {:.0}\n", global_mse);

    // === Conditional mean by neighborhood ===
    println!("  2. Conditional Mean by Neighborhood");
    let neighborhoods = ["A", "B"];
    let mut cond_mse_neighborhood = 0.0;
    let mut cond_preds: Vec<f64> = Vec::new();

    for &hood in &neighborhoods {
        let group: Vec<f64> = data.iter()
            .filter(|d| d.2 == hood)
            .map(|d| d.3)
            .collect();
        let group_mean = group.iter().sum::<f64>() / group.len() as f64;
        let group_mse: f64 = group.iter()
            .map(|p| (p - group_mean) * (p - group_mean))
            .sum::<f64>() / group.len() as f64;
        println!("     Neighborhood {}: n={}, mean=${:.0}, within-group MSE={:.0}",
            hood, group.len(), group_mean, group_mse);

        for d in &data {
            if d.2 == hood {
                cond_preds.push(group_mean);
            }
        }
        cond_mse_neighborhood += group_mse * group.len() as f64;
    }
    cond_mse_neighborhood /= n;
    let improvement_hood = (1.0 - cond_mse_neighborhood / global_mse) * 100.0;
    println!("     Overall MSE: {:.0} ({:.1}% improvement over global)\n",
        cond_mse_neighborhood, improvement_hood);

    // === Conditional mean by bedrooms ===
    println!("  3. Conditional Mean by Bedrooms");
    let mut bedroom_counts: Vec<i32> = data.iter().map(|d| d.1).collect();
    bedroom_counts.sort();
    bedroom_counts.dedup();

    let mut cond_mse_bedrooms = 0.0;

    for &beds in &bedroom_counts {
        let group: Vec<f64> = data.iter()
            .filter(|d| d.1 == beds)
            .map(|d| d.3)
            .collect();
        let group_mean = group.iter().sum::<f64>() / group.len() as f64;
        let group_mse: f64 = group.iter()
            .map(|p| (p - group_mean) * (p - group_mean))
            .sum::<f64>() / group.len() as f64;
        cond_mse_bedrooms += group_mse * group.len() as f64;
        println!("     {} bedrooms: n={}, mean=${:.0}, within-group MSE={:.0}",
            beds, group.len(), group_mean, group_mse);
    }
    cond_mse_bedrooms /= n;
    let improvement_beds = (1.0 - cond_mse_bedrooms / global_mse) * 100.0;
    println!("     Overall MSE: {:.0} ({:.1}% improvement over global)\n",
        cond_mse_bedrooms, improvement_beds);

    // === Conditional mean by sq_feet bins ===
    println!("  4. Conditional Mean by Square Footage Bins");
    let bin_edges = [0.0, 1200.0, 1600.0, 2000.0, 3000.0];
    let bin_names = ["<1200", "1200-1600", "1600-2000", "2000+"];

    let mut cond_mse_sqft = 0.0;

    for i in 0..bin_names.len() {
        let lo = bin_edges[i];
        let hi = bin_edges[i + 1];
        let group: Vec<f64> = data.iter()
            .filter(|d| d.0 >= lo && d.0 < hi)
            .map(|d| d.3)
            .collect();
        if group.is_empty() { continue; }
        let group_mean = group.iter().sum::<f64>() / group.len() as f64;
        let group_mse: f64 = group.iter()
            .map(|p| (p - group_mean) * (p - group_mean))
            .sum::<f64>() / group.len() as f64;
        cond_mse_sqft += group_mse * group.len() as f64;
        println!("     {} sqft: n={}, mean=${:.0}, within-group MSE={:.0}",
            bin_names[i], group.len(), group_mean, group_mse);
    }
    cond_mse_sqft /= n;
    let improvement_sqft = (1.0 - cond_mse_sqft / global_mse) * 100.0;
    println!("     Overall MSE: {:.0} ({:.1}% improvement over global)\n",
        cond_mse_sqft, improvement_sqft);

    // === Summary comparison ===
    println!("=== Summary ===\n");
    println!("  {:>25} {:>14} {:>14}",
        "Strategy", "MSE", "Improvement");
    println!("  {:->25} {:->14} {:->14}", "", "", "");
    println!("  {:>25} {:>14.0} {:>13.1}%", "Global mean", global_mse, 0.0);
    println!("  {:>25} {:>14.0} {:>13.1}%", "By neighborhood", cond_mse_neighborhood, improvement_hood);
    println!("  {:>25} {:>14.0} {:>13.1}%", "By bedrooms", cond_mse_bedrooms, improvement_beds);
    println!("  {:>25} {:>14.0} {:>13.1}%", "By sq_feet bins", cond_mse_sqft, improvement_sqft);

    println!();
    println!("Key insight: Conditioning on features improves predictions.");
    println!("The mean predictor is the seed from which all learning grows.");
}
```

---

## Key Takeaways

- The global mean predictor ignores all features — it is the simplest baseline for regression problems.
- Conditional means partition data by feature values and predict the group average, which is the first step toward real learning.
- Each conditioning feature reduces MSE by explaining some variance — the amount of improvement tells you how informative that feature is.
- Over-conditioning (too many groups, too few observations per group) leads to unstable predictions — this is the bias-variance tradeoff in its simplest form.
