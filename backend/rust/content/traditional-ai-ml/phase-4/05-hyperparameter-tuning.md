# Hyperparameter Tuning

> Phase 4 — Model Evaluation & Selection | Kata 4.05

---

## Concept & Intuition

### What problem are we solving?

Every ML algorithm has hyperparameters: settings that are not learned from data but set by the practitioner before training. KNN has K (number of neighbors). Decision trees have max depth and min samples per leaf. Ridge regression has lambda. These hyperparameters control the model's complexity and behavior, and choosing them well can mean the difference between a mediocre model and an excellent one.

Hyperparameter tuning is the systematic search for the best combination of hyperparameter values. The key constraint is that you must evaluate each setting on validation data, not training data, to avoid overfitting to a specific configuration. Grid search (trying all combinations) is exhaustive but expensive. Random search is surprisingly effective because it explores the hyperparameter space more efficiently when only a few parameters actually matter.

In this kata, we implement grid search and random search with cross-validation, applying them to tune a KNN classifier's hyperparameters.

### Why naive approaches fail

Setting hyperparameters by intuition or default values leaves performance on the table. Trying a few values manually and picking the best is informal grid search with selection bias. Tuning on the test set overfits to that specific test set, producing optimistically biased estimates. The proper approach uses a separate validation set or cross-validation within the training set, with the test set reserved for final, unbiased evaluation.

### Mental models

- **Grid search as exhaustive enumeration**: Try every combination on a predefined grid. Guaranteed to find the best combination on the grid, but expensive with many hyperparameters.
- **Random search as efficient sampling**: Randomly sample combinations. When only 1-2 hyperparameters matter (which is common), random search is much more efficient because it explores more unique values per parameter.
- **Nested cross-validation**: The outer loop evaluates the model. The inner loop tunes hyperparameters. This prevents information leakage from tuning.

### Visual explanations

```
  Grid Search (3x3=9 trials):    Random Search (9 trials):

  param2                          param2
    |  x  x  x                     |   x    x
    |  x  x  x                     | x     x
    |  x  x  x                     |    x  x
    +---------- param1              |  x   x   x
                                    +---------- param1
  Covers only 3 unique             Covers 9 unique values
  values per parameter             per parameter
```

---

## Hands-on Exploration

1. Define a hyperparameter search space for KNN (K, distance metric weighting).
2. Implement grid search with cross-validation.
3. Implement random search with cross-validation.
4. Compare the two approaches in terms of best score and computational cost.

---

## Live Code

```rust
fn main() {
    println!("=== Hyperparameter Tuning ===\n");

    // Generate dataset
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for _ in 0..40 {
        let x1 = 1.0 + randf(&mut rng) * 5.0;
        let x2 = 1.0 + randf(&mut rng) * 5.0;
        features.push(vec![x1, x2]);
        labels.push(0);
    }
    for _ in 0..40 {
        let x1 = 4.0 + randf(&mut rng) * 5.0;
        let x2 = 4.0 + randf(&mut rng) * 5.0;
        features.push(vec![x1, x2]);
        labels.push(1);
    }

    let (scaled, _, _) = standardize(&features);
    let n = scaled.len();

    // Hold out final test set
    let mut idx: Vec<usize> = (0..n).collect();
    shuffle(&mut idx, 99);
    let test_n = 16;
    let train_idx: Vec<usize> = idx[..n-test_n].to_vec();
    let test_idx: Vec<usize> = idx[n-test_n..].to_vec();

    let x_pool: Vec<Vec<f64>> = train_idx.iter().map(|&i| scaled[i].clone()).collect();
    let y_pool: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| scaled[i].clone()).collect();
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    println!("Training pool: {} samples, Test: {} samples\n", x_pool.len(), test_n);

    // Define hyperparameter space
    let k_values = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21];
    let weight_options = vec![0, 1]; // 0=uniform, 1=distance-weighted
    let p_norms = vec![1, 2]; // 1=manhattan, 2=euclidean

    let total_grid = k_values.len() * weight_options.len() * p_norms.len();
    println!("Hyperparameter space: {} combinations\n", total_grid);

    // Grid Search
    println!("--- Grid Search (5-fold CV) ---");
    let cv_folds = stratified_k_fold(&y_pool, 5, 42);
    let mut grid_results: Vec<(usize, usize, usize, f64, f64)> = Vec::new();
    let mut grid_evals = 0;

    for &k in &k_values {
        for &weighted in &weight_options {
            for &p in &p_norms {
                let mut fold_scores = Vec::new();
                for test_fold in &cv_folds {
                    let train_fold: Vec<usize> = (0..x_pool.len()).filter(|i| !test_fold.contains(i)).collect();
                    let xtr: Vec<Vec<f64>> = train_fold.iter().map(|&i| x_pool[i].clone()).collect();
                    let ytr: Vec<usize> = train_fold.iter().map(|&i| y_pool[i]).collect();
                    let xte: Vec<Vec<f64>> = test_fold.iter().map(|&i| x_pool[i].clone()).collect();
                    let yte: Vec<usize> = test_fold.iter().map(|&i| y_pool[i]).collect();
                    fold_scores.push(knn_eval(&xtr, &ytr, &xte, &yte, k, weighted == 1, p));
                    grid_evals += 1;
                }
                let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                let std = (fold_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                    / fold_scores.len() as f64).sqrt();
                grid_results.push((k, weighted, p, mean, std));
            }
        }
    }

    grid_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("Top 5 configurations:");
    println!("{:<6} {:>8} {:>8} {:>10} {:>10}", "K", "Weight", "P-norm", "CV Mean", "CV Std");
    println!("{}", "-".repeat(44));
    for i in 0..5.min(grid_results.len()) {
        let (k, w, p, mean, std) = grid_results[i];
        let wt = if w == 0 { "uniform" } else { "distance" };
        println!("{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}%", k, wt, p, mean * 100.0, std * 100.0);
    }
    println!("  Total evaluations: {} (folds * configs)", grid_evals);

    // Random Search
    println!("\n--- Random Search (5-fold CV, 20 random trials) ---");
    let n_random = 20;
    let mut random_results: Vec<(usize, usize, usize, f64, f64)> = Vec::new();
    let mut random_evals = 0;

    for trial in 0..n_random {
        rng = lcg(rng);
        let k = k_values[(rng as usize) % k_values.len()];
        rng = lcg(rng);
        let weighted = weight_options[(rng as usize) % weight_options.len()];
        rng = lcg(rng);
        let p = p_norms[(rng as usize) % p_norms.len()];

        let mut fold_scores = Vec::new();
        for test_fold in &cv_folds {
            let train_fold: Vec<usize> = (0..x_pool.len()).filter(|i| !test_fold.contains(i)).collect();
            let xtr: Vec<Vec<f64>> = train_fold.iter().map(|&i| x_pool[i].clone()).collect();
            let ytr: Vec<usize> = train_fold.iter().map(|&i| y_pool[i]).collect();
            let xte: Vec<Vec<f64>> = test_fold.iter().map(|&i| x_pool[i].clone()).collect();
            let yte: Vec<usize> = test_fold.iter().map(|&i| y_pool[i]).collect();
            fold_scores.push(knn_eval(&xtr, &ytr, &xte, &yte, k, weighted == 1, p));
            random_evals += 1;
        }
        let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std = (fold_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / fold_scores.len() as f64).sqrt();
        random_results.push((k, weighted, p, mean, std));
    }

    random_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    let (rk, rw, rp, rm, rs) = random_results[0];
    let rwt = if rw == 0 { "uniform" } else { "distance" };
    println!("  Best: K={}, weight={}, p={}, CV={:.1}% +/- {:.1}%", rk, rwt, rp, rm*100.0, rs*100.0);
    println!("  Total evaluations: {} ({:.0}% of grid)", random_evals, random_evals as f64 / grid_evals as f64 * 100.0);

    // Compare on test set
    println!("\n--- Final Test Set Evaluation ---");
    let (gk, gw, gp, gm, _) = grid_results[0];
    let grid_test = knn_eval(&x_pool, &y_pool, &x_test, &y_test, gk, gw == 1, gp);
    let random_test = knn_eval(&x_pool, &y_pool, &x_test, &y_test, rk, rw == 1, rp);

    println!("  Grid search best:   K={}, test acc={:.1}% (CV was {:.1}%)", gk, grid_test*100.0, gm*100.0);
    println!("  Random search best: K={}, test acc={:.1}% (CV was {:.1}%)", rk, random_test*100.0, rm*100.0);

    kata_metric("grid_best_cv", gm);
    kata_metric("random_best_cv", rm);
    kata_metric("grid_test_acc", grid_test);
    kata_metric("random_test_acc", random_test);
    kata_metric("grid_total_evals", grid_evals as f64);
    kata_metric("random_total_evals", random_evals as f64);
}

fn knn_eval(xtr: &[Vec<f64>], ytr: &[usize], xte: &[Vec<f64>], yte: &[usize], k: usize, weighted: bool, p: usize) -> f64 {
    let preds: Vec<usize> = xte.iter().map(|x| {
        let mut dists: Vec<(usize, f64)> = xtr.iter().enumerate().map(|(i, xi)| {
            let d = if p == 1 { xi.iter().zip(x.iter()).map(|(a,b)| (a-b).abs()).sum() }
                    else { xi.iter().zip(x.iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt() };
            (i, d)
        }).collect();
        dists.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());

        if weighted {
            let mut wv = [0.0f64; 2];
            for i in 0..k.min(dists.len()) { wv[ytr[dists[i].0]] += 1.0/(dists[i].1+1e-8); }
            if wv[1] > wv[0] { 1 } else { 0 }
        } else {
            let mut votes = [0usize; 2];
            for i in 0..k.min(dists.len()) { votes[ytr[dists[i].0]] += 1; }
            if votes[1] > votes[0] { 1 } else { 0 }
        }
    }).collect();
    preds.iter().zip(yte.iter()).filter(|(p,a)| p==a).count() as f64 / yte.len() as f64
}

fn stratified_k_fold(labels: &[usize], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for class in 0..=1 {
        let mut ci: Vec<usize> = labels.iter().enumerate().filter(|(_,&l)| l==class).map(|(i,_)| i).collect();
        shuffle(&mut ci, seed + class as u64);
        for (i, idx) in ci.into_iter().enumerate() { folds[i % k].push(idx); }
    }
    folds
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64; let p = data[0].len();
    let mut m = vec![0.0; p]; let mut s = vec![0.0; p];
    for j in 0..p { let c: Vec<f64> = data.iter().map(|r| r[j]).collect();
        m[j] = c.iter().sum::<f64>()/n; s[j] = (c.iter().map(|x| (x-m[j]).powi(2)).sum::<f64>()/n).sqrt(); }
    let sc = data.iter().map(|r| r.iter().enumerate().map(|(j,&v)| if s[j]<1e-10{0.0}else{(v-m[j])/s[j]}).collect()).collect();
    (sc, m, s)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut r = seed; for i in (1..arr.len()).rev() { r = lcg(r); arr.swap(i, (r%(i as u64+1)) as usize); }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Hyperparameter tuning systematically searches for the best model configuration using cross-validation to avoid overfitting to a specific data split.
- Grid search is exhaustive and guaranteed to find the best combination on the grid, but scales exponentially with the number of hyperparameters.
- Random search explores more unique values per parameter and is often as effective as grid search with fewer evaluations, especially when only a few hyperparameters matter.
- Always reserve a final test set that is never used during tuning. Cross-validation within the training set handles model selection; the test set provides the unbiased final estimate.
