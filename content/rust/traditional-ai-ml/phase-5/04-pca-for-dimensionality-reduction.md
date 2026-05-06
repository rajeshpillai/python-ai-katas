# PCA for Dimensionality Reduction

> Phase 5 — Unsupervised Learning | Kata 5.04

---

## Concept & Intuition

### What problem are we solving?

High-dimensional data is hard to visualize, slow to process, and often redundant. Principal Component Analysis (PCA) reduces dimensionality by finding the directions (principal components) along which the data varies most. It projects the data onto these directions, preserving as much variance (information) as possible while reducing the number of features.

PCA is the most widely used dimensionality reduction technique. It compresses 100 features into 10 while retaining 95% of the information. It reveals the intrinsic dimensionality of the data (maybe only 3 of your 50 features carry independent information). And it decorrelates features, which can improve the performance of algorithms that assume feature independence.

In this kata, we implement PCA from scratch using the covariance matrix and the power iteration method for finding eigenvectors. We demonstrate how PCA compresses data and what information is retained versus lost.

### Why naive approaches fail

Simply dropping features (e.g., keeping the first 10 and discarding the rest) is arbitrary and throws away potentially important information. Feature selection based on individual variance ignores correlations between features. PCA is superior because it finds the optimal linear combinations of features that capture the most variance, accounting for all correlations.

### Mental models

- **PCA as finding the best viewing angle**: Imagine a 3D scatter plot of data. PCA finds the camera angle from which the data looks most spread out. The first principal component is the direction of maximum spread; the second is the direction of maximum spread perpendicular to the first.
- **Eigenvectors as axes, eigenvalues as importance**: Each principal component is an eigenvector of the covariance matrix. Its eigenvalue tells you how much variance it captures. Large eigenvalue = important direction.
- **Explained variance ratio**: If the first 3 components explain 95% of the variance, you can safely reduce to 3 dimensions with minimal information loss.

### Visual explanations

```
  Original 2D data:        After PCA:              Reduced to 1D:

  x2                       PC2                      PC1
  |  . .  . .              |  . .                   . . .. . .. . .
  | . .  .  .              | . .  .                 (projected onto PC1)
  |. .  .  .    ------>    |. .  .     ------>
  | .  .  .                +--------- PC1           95% of variance
  |.  .  .                  (rotated)               retained
  +---------- x1

  PC1 captures most variance    PC2 has little variance
```

---

## Hands-on Exploration

1. Standardize the data (zero mean, unit variance).
2. Compute the covariance matrix.
3. Find eigenvalues and eigenvectors using power iteration.
4. Project data onto the top K principal components and measure explained variance.

---

## Live Code

```rust
fn main() {
    println!("=== PCA for Dimensionality Reduction ===\n");

    // Generate 5D data where only 2 dimensions carry real information
    let mut rng = 42u64;
    let n = 50;
    let mut data: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n {
        let z1 = randf(&mut rng) * 5.0; // main signal 1
        let z2 = randf(&mut rng) * 3.0; // main signal 2
        let noise1 = randf(&mut rng) * 0.3;
        let noise2 = randf(&mut rng) * 0.3;
        let noise3 = randf(&mut rng) * 0.3;

        // 5 features are functions of 2 latent variables + noise
        data.push(vec![
            z1 + noise1,
            z2 + noise2,
            z1 * 0.8 + z2 * 0.3 + noise3,
            z1 * 0.5 - z2 * 0.5 + randf(&mut rng) * 0.3,
            z2 * 0.9 + randf(&mut rng) * 0.3,
        ]);
    }

    let feature_names = vec!["f1", "f2", "f3", "f4", "f5"];
    let p = 5;
    println!("Dataset: {} samples, {} features\n", n, p);

    // Step 1: Standardize
    let (centered, means, stds) = standardize(&data);

    // Step 2: Covariance matrix
    let cov = covariance_matrix(&centered);
    println!("--- Covariance Matrix ---");
    print!("{:>8}", "");
    for name in &feature_names { print!("{:>8}", name); }
    println!();
    for (i, row) in cov.iter().enumerate() {
        print!("{:>8}", feature_names[i]);
        for val in row { print!("{:>8.3}", val); }
        println!();
    }

    // Step 3: Eigendecomposition via power iteration
    println!("\n--- Eigenvalues and Eigenvectors ---");
    let (eigenvalues, eigenvectors) = eigen_decomposition(&cov, p);

    let total_var: f64 = eigenvalues.iter().sum();
    let mut cumulative = 0.0;

    println!("{:<6} {:>12} {:>12} {:>12}", "PC", "Eigenvalue", "Var %", "Cumul %");
    println!("{}", "-".repeat(44));
    for i in 0..p {
        cumulative += eigenvalues[i];
        let var_pct = eigenvalues[i] / total_var * 100.0;
        let cum_pct = cumulative / total_var * 100.0;
        let bar = "#".repeat((var_pct / 3.0) as usize);
        println!("{:<6} {:>12.4} {:>11.1}% {:>11.1}% |{}", i + 1, eigenvalues[i], var_pct, cum_pct, bar);
    }

    // Step 4: Project onto top 2 components
    let n_components = 2;
    println!("\n--- Projection to {} dimensions ---", n_components);
    let projection_matrix: Vec<Vec<f64>> = (0..n_components)
        .map(|i| eigenvectors[i].clone())
        .collect();

    let projected: Vec<Vec<f64>> = centered.iter().map(|row| {
        projection_matrix.iter().map(|pc| {
            row.iter().zip(pc.iter()).map(|(x, w)| x * w).sum()
        }).collect()
    }).collect();

    // Explained variance
    let explained: f64 = eigenvalues[..n_components].iter().sum();
    let explained_pct = explained / total_var * 100.0;
    println!("  Explained variance: {:.1}%", explained_pct);
    println!("  Dimensions reduced: {} -> {}", p, n_components);

    // Component loadings
    println!("\n--- Principal Component Loadings ---");
    println!("(How much each original feature contributes to each PC)\n");
    print!("{:>8}", "Feature");
    for i in 0..n_components { print!("{:>8}", format!("PC{}", i + 1)); }
    println!();
    println!("{}", "-".repeat(8 + 8 * n_components));
    for j in 0..p {
        print!("{:>8}", feature_names[j]);
        for i in 0..n_components {
            print!("{:>8.3}", eigenvectors[i][j]);
        }
        println!();
    }

    // Reconstruction error
    println!("\n--- Reconstruction Error ---");
    let reconstructed: Vec<Vec<f64>> = projected.iter().map(|proj| {
        let mut recon = vec![0.0; p];
        for (i, pc) in projection_matrix.iter().enumerate() {
            for j in 0..p { recon[j] += proj[i] * pc[j]; }
        }
        recon
    }).collect();

    let recon_error: f64 = centered.iter().zip(reconstructed.iter()).map(|(orig, recon)| {
        orig.iter().zip(recon.iter()).map(|(o, r)| (o - r).powi(2)).sum::<f64>()
    }).sum::<f64>() / n as f64;

    println!("  Mean reconstruction error: {:.4}", recon_error);
    println!("  This represents the {:.1}% of variance NOT captured", 100.0 - explained_pct);

    // Scree plot
    println!("\n--- Scree Plot ---");
    for (i, &ev) in eigenvalues.iter().enumerate() {
        let bar = "#".repeat((ev / eigenvalues[0] * 25.0) as usize);
        println!("  PC{}: {:.4} |{}", i + 1, ev, bar);
    }

    kata_metric("explained_variance_2pc", explained_pct);
    kata_metric("reconstruction_error", recon_error);
    kata_metric("eigenvalue_1", eigenvalues[0]);
    kata_metric("eigenvalue_2", eigenvalues[1]);
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64; let p = data[0].len();
    let mut means = vec![0.0; p]; let mut stds = vec![0.0; p];
    for j in 0..p { let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        means[j] = col.iter().sum::<f64>()/n;
        stds[j] = (col.iter().map(|x| (x-means[j]).powi(2)).sum::<f64>()/n).sqrt(); }
    let centered = data.iter().map(|row| row.iter().enumerate().map(|(j,&v)|
        if stds[j]<1e-10{0.0}else{(v-means[j])/stds[j]}).collect()).collect();
    (centered, means, stds)
}

fn covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len() as f64;
    let p = data[0].len();
    let mut cov = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            let sum: f64 = data.iter().map(|row| row[i] * row[j]).sum();
            cov[i][j] = sum / n;
        }
    }
    cov
}

fn eigen_decomposition(matrix: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut deflated = matrix.to_vec();

    for _ in 0..n {
        let (eigenvalue, eigenvector) = power_iteration(&deflated, 200);
        eigenvalues.push(eigenvalue);
        eigenvectors.push(eigenvector.clone());

        // Deflate: remove this component from the matrix
        let p = deflated.len();
        for i in 0..p {
            for j in 0..p {
                deflated[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    eigenvalues
}

fn power_iteration(matrix: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = matrix.len();
    let mut v: Vec<f64> = vec![1.0; n];
    let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    for x in &mut v { *x /= norm; }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Multiply: w = M * v
        let w: Vec<f64> = matrix.iter().map(|row| {
            row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
        }).collect();

        // Eigenvalue estimate
        eigenvalue = w.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

        // Normalize
        let norm = (w.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm < 1e-12 { break; }
        v = w.iter().map(|x| x / norm).collect();
    }

    (eigenvalue, v)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- PCA finds the orthogonal directions of maximum variance in the data, enabling dimensionality reduction with minimal information loss.
- The eigenvalues of the covariance matrix tell you how much variance each principal component captures. The explained variance ratio guides how many components to keep.
- Component loadings reveal which original features contribute most to each principal component, aiding interpretation.
- PCA is a linear technique. For nonlinear dimensionality reduction, methods like t-SNE (next kata) are needed.
