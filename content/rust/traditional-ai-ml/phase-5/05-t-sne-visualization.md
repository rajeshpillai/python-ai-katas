# t-SNE Visualization

> Phase 5 — Unsupervised Learning | Kata 5.05

---

## Concept & Intuition

### What problem are we solving?

PCA finds the best linear projection for preserving variance, but many real-world datasets have nonlinear structure that PCA cannot capture. t-SNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality reduction technique specifically designed for visualization. It preserves local neighborhood structure: points that are close in high-dimensional space remain close in the 2D embedding.

t-SNE works by converting pairwise distances in high-dimensional space into probabilities (using Gaussian kernels), then finding a low-dimensional arrangement where similar pairwise probabilities hold (using the heavier-tailed Student-t distribution). The mismatch between these distributions is minimized using gradient descent on the KL divergence.

In this kata, we implement a simplified t-SNE from scratch. While production t-SNE uses Barnes-Hut approximations for speed, our implementation demonstrates the core algorithm on small datasets.

### Why naive approaches fail

PCA preserves global structure (overall variance) but often fails to reveal cluster separation in high dimensions. Points from different clusters that are far apart in the original space may overlap when projected linearly onto 2D. t-SNE focuses on preserving local neighborhoods — if two points were near each other in the original space, t-SNE keeps them near each other in 2D. This makes it excellent for visualizing clusters, even when they are interleaved in complex ways.

### Mental models

- **High-D similarities as probabilities**: For each pair of points, compute a probability proportional to their closeness. Close points get high probability, far points get low probability.
- **Student-t in low-D**: The t-distribution has heavier tails than the Gaussian. This means moderate distances in high-D can map to larger distances in low-D, preventing the "crowding problem" where everything collapses to the center.
- **Perplexity as neighborhood size**: Perplexity controls how many effective neighbors each point considers. Low perplexity focuses on very local structure; high perplexity captures more global structure.

### Visual explanations

```
  PCA projection:             t-SNE embedding:

    A A B B                   A A     B B
    A B B A                   A A     B B
    B A A B                   A A     B B
    B B A A
                              C C     D D
  Clusters overlap!           C C     D D

  PCA: linear, preserves      t-SNE: nonlinear, preserves
  global variance              local neighborhoods
```

---

## Hands-on Exploration

1. Compute pairwise affinities in high-dimensional space using Gaussian kernels.
2. Initialize low-dimensional embeddings randomly.
3. Minimize KL divergence using gradient descent.
4. Compare t-SNE embeddings with PCA projections.

---

## Live Code

```rust
fn main() {
    println!("=== t-SNE Visualization ===\n");

    // Generate 5D data with 3 clear clusters
    let mut rng = 42u64;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Cluster 0
    for _ in 0..15 {
        data.push(vec![randf(&mut rng)+1.0, randf(&mut rng)+1.0, randf(&mut rng)+1.0,
                        randf(&mut rng)*0.3, randf(&mut rng)*0.3]);
        labels.push(0);
    }
    // Cluster 1
    for _ in 0..15 {
        data.push(vec![randf(&mut rng)+5.0, randf(&mut rng)+5.0, randf(&mut rng)+1.0,
                        randf(&mut rng)*0.3+3.0, randf(&mut rng)*0.3]);
        labels.push(1);
    }
    // Cluster 2
    for _ in 0..15 {
        data.push(vec![randf(&mut rng)+1.0, randf(&mut rng)+5.0, randf(&mut rng)+5.0,
                        randf(&mut rng)*0.3, randf(&mut rng)*0.3+3.0]);
        labels.push(2);
    }

    let n = data.len();
    println!("Dataset: {} points in 5D, 3 clusters\n", n);

    // Step 1: Compute pairwise distances
    let distances = pairwise_distances(&data);

    // Step 2: Compute high-dimensional affinities
    let perplexity = 10.0;
    let p_matrix = compute_pairwise_affinities(&distances, perplexity);
    println!("Perplexity: {}", perplexity);

    // Step 3: Run t-SNE
    println!("\n--- Running t-SNE (200 iterations) ---");
    let mut embedding = tsne(&distances, &p_matrix, 2, 200, 0.5, &mut rng);

    // Display embedding
    println!("\n--- t-SNE Embedding ---");
    ascii_scatter_labeled(&embedding, &labels);

    // Compare with PCA
    println!("\n--- PCA Projection (2D) ---");
    let (centered, _, _) = standardize(&data);
    let cov = covariance_matrix(&centered);
    let (eigenvalues, eigenvectors) = eigen_decomp(&cov, 2);

    let pca_embedding: Vec<Vec<f64>> = centered.iter().map(|row| {
        (0..2).map(|i| row.iter().zip(eigenvectors[i].iter()).map(|(x, w)| x * w).sum()).collect()
    }).collect();

    ascii_scatter_labeled(&pca_embedding, &labels);

    // Cluster separation comparison
    println!("\n--- Cluster Separation Score ---");
    let tsne_sep = cluster_separation(&embedding, &labels, 3);
    let pca_sep = cluster_separation(&pca_embedding, &labels, 3);
    println!("  t-SNE separation score: {:.4}", tsne_sep);
    println!("  PCA separation score:   {:.4}", pca_sep);
    println!("  (Higher = better cluster separation)");

    // Perplexity effect
    println!("\n--- Perplexity Effect ---");
    for &perp in &[5.0, 10.0, 20.0, 30.0] {
        let p_mat = compute_pairwise_affinities(&distances, perp);
        let emb = tsne(&distances, &p_mat, 2, 150, 0.5, &mut rng);
        let sep = cluster_separation(&emb, &labels, 3);
        println!("  Perplexity {:>5.0}: separation = {:.4}", perp, sep);
    }

    kata_metric("tsne_separation", tsne_sep);
    kata_metric("pca_separation", pca_sep);
    kata_metric("perplexity", perplexity);
    kata_metric("n_samples", n as f64);
}

fn tsne(
    distances: &[Vec<f64>],
    p_matrix: &[Vec<f64>],
    out_dim: usize,
    n_iter: usize,
    learning_rate: f64,
    rng: &mut u64,
) -> Vec<Vec<f64>> {
    let n = distances.len();

    // Initialize randomly
    let mut y: Vec<Vec<f64>> = (0..n).map(|_| {
        (0..out_dim).map(|_| randf(rng) * 0.01).collect()
    }).collect();

    let mut momentum: Vec<Vec<f64>> = vec![vec![0.0; out_dim]; n];

    for iter in 0..n_iter {
        // Compute low-dimensional affinities (Student-t with 1 df)
        let mut q_matrix = vec![vec![0.0; n]; n];
        let mut q_sum = 0.0;

        for i in 0..n {
            for j in (i+1)..n {
                let dist_sq: f64 = y[i].iter().zip(y[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum();
                let q = 1.0 / (1.0 + dist_sq);
                q_matrix[i][j] = q;
                q_matrix[j][i] = q;
                q_sum += 2.0 * q;
            }
        }

        // Normalize
        if q_sum > 0.0 {
            for i in 0..n { for j in 0..n { q_matrix[i][j] /= q_sum; } }
        }

        // Compute gradients
        let mut grad = vec![vec![0.0; out_dim]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let dist_sq: f64 = y[i].iter().zip(y[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum();
                let q = 1.0 / (1.0 + dist_sq);
                let mult = 4.0 * (p_matrix[i][j] - q_matrix[i][j]) * q;
                for d in 0..out_dim {
                    grad[i][d] += mult * (y[i][d] - y[j][d]);
                }
            }
        }

        // Update with momentum
        let mom = if iter < 50 { 0.5 } else { 0.8 };
        for i in 0..n {
            for d in 0..out_dim {
                momentum[i][d] = mom * momentum[i][d] - learning_rate * grad[i][d];
                y[i][d] += momentum[i][d];
            }
        }

        if iter % 50 == 0 || iter == n_iter - 1 {
            let kl = kl_divergence(p_matrix, &q_matrix, n);
            println!("  Iteration {}: KL divergence = {:.6}", iter, kl);
        }
    }

    y
}

fn compute_pairwise_affinities(distances: &[Vec<f64>], perplexity: f64) -> Vec<Vec<f64>> {
    let n = distances.len();
    let target_entropy = perplexity.ln();
    let mut p = vec![vec![0.0; n]; n];

    for i in 0..n {
        // Binary search for sigma
        let mut sigma = 1.0;
        let mut lo = 0.001;
        let mut hi = 100.0;

        for _ in 0..50 {
            let mut sum = 0.0;
            for j in 0..n {
                if i == j { continue; }
                sum += (-distances[i][j].powi(2) / (2.0 * sigma * sigma)).exp();
            }

            let mut entropy = 0.0;
            for j in 0..n {
                if i == j { continue; }
                let pij = (-distances[i][j].powi(2) / (2.0 * sigma * sigma)).exp() / sum.max(1e-10);
                if pij > 1e-10 { entropy -= pij * pij.ln(); }
            }

            if (entropy - target_entropy).abs() < 1e-5 { break; }
            if entropy > target_entropy { hi = sigma; sigma = (lo + hi) / 2.0; }
            else { lo = sigma; sigma = (lo + hi) / 2.0; }
        }

        let mut sum = 0.0;
        for j in 0..n {
            if i == j { continue; }
            sum += (-distances[i][j].powi(2) / (2.0 * sigma * sigma)).exp();
        }
        for j in 0..n {
            if i == j { continue; }
            p[i][j] = (-distances[i][j].powi(2) / (2.0 * sigma * sigma)).exp() / sum.max(1e-10);
        }
    }

    // Symmetrize
    for i in 0..n { for j in (i+1)..n {
        let sym = (p[i][j] + p[j][i]) / (2.0 * n as f64);
        p[i][j] = sym; p[j][i] = sym;
    }}

    p
}

fn kl_divergence(p: &[Vec<f64>], q: &[Vec<f64>], n: usize) -> f64 {
    let mut kl = 0.0;
    for i in 0..n { for j in 0..n {
        if i != j && p[i][j] > 1e-12 && q[i][j] > 1e-12 {
            kl += p[i][j] * (p[i][j] / q[i][j]).ln();
        }
    }}
    kl
}

fn pairwise_distances(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n { for j in (i+1)..n {
        let dist = data[i].iter().zip(data[j].iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
        d[i][j] = dist; d[j][i] = dist;
    }}
    d
}

fn cluster_separation(embedding: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let mut intra = 0.0; let mut intra_n = 0;
    let mut inter = 0.0; let mut inter_n = 0;
    for i in 0..embedding.len() { for j in (i+1)..embedding.len() {
        let d = embedding[i].iter().zip(embedding[j].iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
        if labels[i] == labels[j] { intra += d; intra_n += 1; }
        else { inter += d; inter_n += 1; }
    }}
    let avg_intra = if intra_n > 0 { intra / intra_n as f64 } else { 1.0 };
    let avg_inter = if inter_n > 0 { inter / inter_n as f64 } else { 1.0 };
    avg_inter / avg_intra
}

fn ascii_scatter_labeled(data: &[Vec<f64>], labels: &[usize]) {
    let symbols = ['A', 'B', 'C', 'D'];
    let h = 12; let w = 40;
    let (xmin,xmax,ymin,ymax) = bounds(data);
    let mut grid = vec![vec![' '; w]; h];
    for (i, p) in data.iter().enumerate() {
        let c = ((p[0]-xmin)/(xmax-xmin+1e-10)*(w-1) as f64).round().max(0.0).min((w-1) as f64) as usize;
        let r = ((ymax-p[1])/(ymax-ymin+1e-10)*(h-1) as f64).round().max(0.0).min((h-1) as f64) as usize;
        grid[r][c] = symbols[labels[i] % symbols.len()];
    }
    for row in &grid { println!("  |{}", row.iter().collect::<String>()); }
    println!("  +{}", "-".repeat(w));
}

fn bounds(d: &[Vec<f64>]) -> (f64,f64,f64,f64) {
    (d.iter().map(|p|p[0]).fold(f64::INFINITY,f64::min), d.iter().map(|p|p[0]).fold(f64::NEG_INFINITY,f64::max),
     d.iter().map(|p|p[1]).fold(f64::INFINITY,f64::min), d.iter().map(|p|p[1]).fold(f64::NEG_INFINITY,f64::max))
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n=data.len() as f64; let p=data[0].len();
    let mut m=vec![0.0;p]; let mut s=vec![0.0;p];
    for j in 0..p{let c:Vec<f64>=data.iter().map(|r|r[j]).collect(); m[j]=c.iter().sum::<f64>()/n; s[j]=(c.iter().map(|x|(x-m[j]).powi(2)).sum::<f64>()/n).sqrt();}
    let sc=data.iter().map(|r|r.iter().enumerate().map(|(j,&v)|if s[j]<1e-10{0.0}else{(v-m[j])/s[j]}).collect()).collect();
    (sc,m,s)
}

fn covariance_matrix(d: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n=d.len() as f64; let p=d[0].len(); let mut c=vec![vec![0.0;p];p];
    for i in 0..p{for j in 0..p{c[i][j]=d.iter().map(|r|r[i]*r[j]).sum::<f64>()/n;}} c
}

fn eigen_decomp(matrix: &[Vec<f64>], k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut evals=Vec::new(); let mut evecs=Vec::new(); let mut def=matrix.to_vec();
    for _ in 0..k { let(ev,vec)=power_iter(&def,200); evals.push(ev); evecs.push(vec.clone());
        let p=def.len(); for i in 0..p{for j in 0..p{def[i][j]-=ev*vec[i]*vec[j];}} }
    (evals,evecs)
}

fn power_iter(m: &[Vec<f64>], iters: usize) -> (f64, Vec<f64>) {
    let n=m.len(); let mut v:Vec<f64>=vec![1.0;n]; let norm=(v.iter().map(|x|x*x).sum::<f64>()).sqrt();
    for x in &mut v{*x/=norm;} let mut ev=0.0;
    for _ in 0..iters { let w:Vec<f64>=m.iter().map(|r|r.iter().zip(v.iter()).map(|(a,b)|a*b).sum()).collect();
        ev=w.iter().zip(v.iter()).map(|(a,b)|a*b).sum(); let norm=(w.iter().map(|x|x*x).sum::<f64>()).sqrt();
        if norm<1e-12{break;} v=w.iter().map(|x|x/norm).collect(); }
    (ev,v)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- t-SNE creates nonlinear 2D embeddings optimized for visualization by preserving local neighborhood structure from high-dimensional data.
- The perplexity parameter controls the balance between local and global structure. Typical values range from 5 to 50.
- t-SNE is for visualization only — distances in the embedding are not directly interpretable, and it should not be used for downstream ML tasks.
- PCA is fast and deterministic but limited to linear projections. t-SNE is slower and stochastic but captures nonlinear structure that PCA misses.
