# PCA Intuition

> Phase 4 — Representation Learning | Kata 4.2

---

## Concept & Intuition

### What problem are we solving?

Principal Component Analysis (PCA) is the foundational technique for dimensionality reduction. It finds the directions in which the data varies most and projects the data onto those directions. If your data has 100 features but most of the variation can be captured by just 5 directions, PCA compresses the data from 100 dimensions to 5 with minimal information loss.

PCA works by computing the eigenvectors of the data's covariance matrix. The eigenvector with the largest eigenvalue points in the direction of maximum variance — this is the first principal component. The second component is the direction of maximum variance orthogonal to the first, and so on. Each eigenvalue tells you how much variance that component explains.

The deep learning connection is direct: PCA is a linear version of what autoencoders do. An autoencoder learns to compress data into a lower-dimensional representation and reconstruct it — if the autoencoder has no activation functions, it learns exactly the PCA solution. PCA gives us intuition for what representation learning means: finding a compact description that preserves the important information.

### Why naive approaches fail

Simply removing features (e.g., keeping only the first 5 columns) throws away information arbitrarily. PCA instead finds the optimal linear projection that preserves the most variance. Two features that are highly correlated contain mostly the same information — PCA combines them into a single component that captures what they share. This is far more intelligent than naive feature selection.

### Mental models

- **PCA as finding the best camera angle**: Imagine a 3D object. From some angles it looks like a blob (low information). From the optimal angle, you can see all the important structure. PCA finds that optimal angle.
- **Variance = information**: Directions with high variance are where data points differ from each other. Directions with zero variance are where all points are identical. PCA keeps the high-variance directions and discards the rest.
- **Rotate, then drop**: PCA rotates the coordinate system to align with the data's natural axes, then drops the axes that carry little information.

### Visual explanations

```
  2D data:                     After PCA:
    y │    . .  .              PC2 │
      │   . . . . .                │    . . . . . . .
      │  . . . . . .               │  . . . . . . . .
      │ . . . . .                  │    . . . . . .
      │. . . .                     └──────────────────── PC1
      └──────────── x              (most variance along PC1)

  PCA rotates the axes to align with the direction of maximum spread.
  Projecting onto just PC1 captures most of the information.
```

---

## Hands-on Exploration

1. Implement PCA from scratch: center data, compute covariance, find eigenvectors.
2. Project data onto principal components and measure explained variance.
3. Observe how many components are needed to capture most of the information.

---

## Live Code

```rust
fn main() {
    // === PCA from Scratch ===
    // Finding the directions of maximum variance.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // Generate correlated 2D data
    let n = 100;
    let mut data: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n {
        let t = rand_f64() * 3.0;
        let noise = rand_f64() * 0.3;
        // x1 and x2 are correlated (both depend on t)
        let x1 = 2.0 * t + 1.0 + noise;
        let x2 = 1.5 * t + 0.5 + rand_f64() * 0.3;
        data.push(vec![x1, x2]);
    }

    let dim = data[0].len();

    println!("=== PCA Intuition ===\n");
    println!("  Dataset: {} points in {}D (correlated features)\n", n, dim);

    // === Step 1: Center the data ===
    let means: Vec<f64> = (0..dim).map(|d| {
        data.iter().map(|row| row[d]).sum::<f64>() / n as f64
    }).collect();

    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(means.iter()).map(|(x, m)| x - m).collect()
    }).collect();

    println!("  Step 1: Center the data (subtract mean)");
    println!("  Means: [{:.3}, {:.3}]\n", means[0], means[1]);

    // === Step 2: Compute covariance matrix ===
    let mut cov = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            cov[i][j] = centered.iter()
                .map(|row| row[i] * row[j])
                .sum::<f64>() / (n - 1) as f64;
        }
    }

    println!("  Step 2: Compute covariance matrix");
    println!("  Cov = [{:.4}, {:.4}]", cov[0][0], cov[0][1]);
    println!("        [{:.4}, {:.4}]\n", cov[1][0], cov[1][1]);
    println!("  High off-diagonal values ({:.4}) indicate strong correlation.\n", cov[0][1]);

    // === Step 3: Compute eigenvalues and eigenvectors (2x2 case) ===
    // For 2x2 matrix [[a,b],[c,d]]:
    // eigenvalues: λ = (a+d)/2 ± sqrt(((a-d)/2)^2 + b*c)
    let a = cov[0][0];
    let b_val = cov[0][1];
    let c = cov[1][0];
    let d = cov[1][1];

    let trace = a + d;
    let det = a * d - b_val * c;
    let discriminant = (trace * trace / 4.0 - det).max(0.0).sqrt();

    let lambda1 = trace / 2.0 + discriminant;
    let lambda2 = trace / 2.0 - discriminant;

    // Eigenvectors
    let compute_eigenvector = |lambda: f64| -> Vec<f64> {
        let v = if (a - lambda).abs() > 1e-10 {
            vec![b_val, lambda - a]
        } else {
            vec![lambda - d, c]
        };
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        vec![v[0] / norm, v[1] / norm]
    };

    let pc1 = compute_eigenvector(lambda1);
    let pc2 = compute_eigenvector(lambda2);

    let total_var = lambda1 + lambda2;
    let explained1 = lambda1 / total_var * 100.0;
    let explained2 = lambda2 / total_var * 100.0;

    println!("  Step 3: Eigendecomposition");
    println!("  Eigenvalue 1: {:.4} ({:.1}% of variance)", lambda1, explained1);
    println!("  Eigenvalue 2: {:.4} ({:.1}% of variance)", lambda2, explained2);
    println!("  PC1 direction: [{:.4}, {:.4}]", pc1[0], pc1[1]);
    println!("  PC2 direction: [{:.4}, {:.4}]", pc2[0], pc2[1]);
    println!();

    // === Step 4: Project data ===
    let projected: Vec<Vec<f64>> = centered.iter().map(|row| {
        let proj1 = row[0] * pc1[0] + row[1] * pc1[1];
        let proj2 = row[0] * pc2[0] + row[1] * pc2[1];
        vec![proj1, proj2]
    }).collect();

    println!("  Step 4: Project data onto principal components\n");

    // === Variance explained bar chart ===
    println!("  === Variance Explained ===\n");
    println!("    PC1: {:.1}%  |{}|",
        explained1, "█".repeat((explained1 / 2.0) as usize));
    println!("    PC2: {:.1}%  |{}|",
        explained2, "█".repeat((explained2 / 2.0) as usize));
    println!("    Total:  {:.1}%\n", explained1 + explained2);

    // === Reconstruction error with 1 component ===
    let mut recon_error_1pc = 0.0;
    let mut recon_error_2pc = 0.0;

    for (i, row) in centered.iter().enumerate() {
        // Reconstruct from 1 PC
        let proj1 = projected[i][0];
        let recon_1 = vec![proj1 * pc1[0], proj1 * pc1[1]];
        let err_1: f64 = row.iter().zip(recon_1.iter())
            .map(|(orig, rec)| (orig - rec) * (orig - rec))
            .sum();
        recon_error_1pc += err_1;

        // Reconstruct from 2 PCs (should be ~0)
        let proj2 = projected[i][1];
        let recon_2 = vec![
            proj1 * pc1[0] + proj2 * pc2[0],
            proj1 * pc1[1] + proj2 * pc2[1],
        ];
        let err_2: f64 = row.iter().zip(recon_2.iter())
            .map(|(orig, rec)| (orig - rec) * (orig - rec))
            .sum();
        recon_error_2pc += err_2;
    }

    recon_error_1pc /= n as f64;
    recon_error_2pc /= n as f64;

    println!("  === Reconstruction Error ===\n");
    println!("    Using 1 PC:  MSE = {:.6}", recon_error_1pc);
    println!("    Using 2 PCs: MSE = {:.6} (perfect: used all components)", recon_error_2pc);
    println!();

    // === Higher-dimensional example ===
    println!("  === Higher-Dimensional Example (5D → PCA) ===\n");

    // Generate 5D data where only 2 dimensions carry real information
    let n_high = 80;
    let mut data_5d: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n_high {
        let t1 = rand_f64() * 3.0;
        let t2 = rand_f64() * 2.0;
        // 5 features, but only 2 underlying factors
        data_5d.push(vec![
            2.0 * t1 + 0.5 * t2 + rand_f64() * 0.1,
            1.5 * t1 - 0.3 * t2 + rand_f64() * 0.1,
            0.8 * t1 + 1.2 * t2 + rand_f64() * 0.1,
            -0.5 * t1 + 2.0 * t2 + rand_f64() * 0.1,
            1.0 * t1 + 0.7 * t2 + rand_f64() * 0.1,
        ]);
    }

    // Center
    let dim5 = 5;
    let means5: Vec<f64> = (0..dim5).map(|d| {
        data_5d.iter().map(|row| row[d]).sum::<f64>() / n_high as f64
    }).collect();
    let centered5: Vec<Vec<f64>> = data_5d.iter().map(|row| {
        row.iter().zip(means5.iter()).map(|(x, m)| x - m).collect()
    }).collect();

    // Compute 5x5 covariance
    let mut cov5 = vec![vec![0.0; dim5]; dim5];
    for i in 0..dim5 {
        for j in 0..dim5 {
            cov5[i][j] = centered5.iter()
                .map(|row| row[i] * row[j])
                .sum::<f64>() / (n_high - 1) as f64;
        }
    }

    // Power iteration to find eigenvalues (approximate)
    let power_iteration = |mat: &Vec<Vec<f64>>, n_iter: usize| -> (f64, Vec<f64>) {
        let dim = mat.len();
        let mut v: Vec<f64> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.5 }).collect();

        for _ in 0..n_iter {
            // Multiply
            let mut new_v: Vec<f64> = (0..dim).map(|i|
                (0..dim).map(|j| mat[i][j] * v[j]).sum::<f64>()
            ).collect();
            // Normalize
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut new_v { *x /= norm; }
            v = new_v;
        }

        // Eigenvalue = v^T A v
        let av: Vec<f64> = (0..dim).map(|i|
            (0..dim).map(|j| mat[i][j] * v[j]).sum::<f64>()
        ).collect();
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(vi, avi)| vi * avi).sum();

        (eigenvalue, v)
    };

    // Deflation to find multiple eigenvalues
    let mut remaining_cov = cov5.clone();
    let mut eigenvalues = Vec::new();

    for _ in 0..dim5 {
        let (eigenval, eigenvec) = power_iteration(&remaining_cov, 100);
        eigenvalues.push(eigenval);

        // Deflate: remove this component
        for i in 0..dim5 {
            for j in 0..dim5 {
                remaining_cov[i][j] -= eigenval * eigenvec[i] * eigenvec[j];
            }
        }
    }

    let total_var5: f64 = eigenvalues.iter().sum();
    let mut cumulative = 0.0;

    println!("  5D data with 2 true underlying factors:\n");
    println!("  {:>4} {:>12} {:>12} {:>12}",
        "PC", "Eigenvalue", "% Variance", "Cumulative");
    println!("  {:->4} {:->12} {:->12} {:->12}", "", "", "", "");

    for (i, &ev) in eigenvalues.iter().enumerate() {
        let pct = ev.max(0.0) / total_var5 * 100.0;
        cumulative += pct;
        let bar_len = (pct / 2.0) as usize;
        println!("  {:>4} {:>12.4} {:>11.1}% {:>11.1}%  |{}|",
            i + 1, ev.max(0.0), pct, cumulative, "█".repeat(bar_len));
    }

    println!();
    println!("  The first 2 PCs capture ~{:.0}% of variance — matching the 2 true factors!",
        eigenvalues[0..2].iter().sum::<f64>() / total_var5 * 100.0);
    println!("  We can reduce 5D to 2D with minimal information loss.\n");

    println!("Key insight: PCA finds the directions of maximum variance.");
    println!("It is dimensionality reduction by rotation and truncation —");
    println!("the linear ancestor of deep representation learning.");
}
```

---

## Key Takeaways

- PCA finds the directions of maximum variance in data and projects onto those directions, achieving optimal linear dimensionality reduction.
- The eigenvalues of the covariance matrix tell you how much variance each principal component explains — use this to decide how many dimensions to keep.
- Data often lives on a lower-dimensional manifold: 5D data with 2 true factors can be compressed to 2D with minimal loss.
- PCA is the linear version of autoencoder representation learning — understanding PCA gives deep intuition for what neural networks learn internally.
