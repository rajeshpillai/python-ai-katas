# K-Means Clustering

> Phase 5 — Unsupervised Learning | Kata 5.01

---

## Concept & Intuition

### What problem are we solving?

K-Means clustering discovers natural groups in unlabeled data. Given N data points and a desired number of clusters K, the algorithm partitions the data so that each point belongs to the cluster whose center (centroid) is nearest. The centroids are iteratively adjusted until they stabilize. No labels are needed — the algorithm finds structure purely from the geometry of the data.

K-Means is the most widely used clustering algorithm because it is simple, fast, and works well when clusters are roughly spherical and similar in size. Applications range from customer segmentation to image compression to gene expression analysis. Understanding K-Means is essential because it introduces core unsupervised learning concepts: objective functions, iterative optimization, and the challenge of evaluating results without ground truth labels.

In this kata, we implement K-Means from scratch using Lloyd's algorithm (the standard iterative approach) and explore initialization strategies, convergence behavior, and how to choose K.

### Why naive approaches fail

Random cluster assignment without iterative refinement is unlikely to find meaningful groups. Initializing all centroids at the same point leads to degenerate solutions where only one cluster is used. Poor initialization (random centroids that happen to be close together) can cause the algorithm to converge to a suboptimal local minimum. K-Means++ initialization addresses this by spreading initial centroids far apart, dramatically improving convergence.

### Mental models

- **K-Means as expectation-maximization**: Alternate between (1) assigning points to nearest centroid (E-step) and (2) updating centroids to the mean of assigned points (M-step). Each step reduces the total within-cluster variance.
- **Objective function**: K-Means minimizes the sum of squared distances from each point to its assigned centroid (inertia). This is a non-convex objective, which is why different initializations can lead to different solutions.
- **The elbow method**: Plot inertia vs. K. As K increases, inertia always decreases. The "elbow" — where the rate of decrease sharply changes — suggests a good K.

### Visual explanations

```
  Iteration 1:           Iteration 3:           Converged:
  (random centroids)     (centroids moving)     (stable)

  * .  .    x            *  .  .   x            *  .  .   x
  .  .   . .             .  .   . .             .  .   . .
  . .  . .               .  .  . .              .  .  . .
     .  .                   .  .                   .  .
  x   . .  .   *         x   . .  .   *         x   . .  .   *
    .  .  .  .              .  .  .  .              .  .  .  .

  * x = centroids         Centroids move toward    Centroids stable
  . = data points         cluster centers          = converged
```

---

## Hands-on Exploration

1. Implement Lloyd's K-Means algorithm: initialize, assign, update, repeat.
2. Implement K-Means++ initialization for better starting centroids.
3. Run the elbow method to find the optimal K.
4. Visualize cluster assignments and centroids.

---

## Live Code

```rust
fn main() {
    println!("=== K-Means Clustering ===\n");

    // Generate 3 clusters
    let mut rng = 42u64;
    let mut data: Vec<Vec<f64>> = Vec::new();

    // Cluster 0: centered at (2, 2)
    for _ in 0..30 { data.push(vec![2.0 + randf(&mut rng)*2.0, 2.0 + randf(&mut rng)*2.0]); }
    // Cluster 1: centered at (8, 3)
    for _ in 0..25 { data.push(vec![8.0 + randf(&mut rng)*2.0, 3.0 + randf(&mut rng)*2.0]); }
    // Cluster 2: centered at (5, 8)
    for _ in 0..25 { data.push(vec![5.0 + randf(&mut rng)*2.0, 8.0 + randf(&mut rng)*2.0]); }

    let n = data.len();
    println!("Dataset: {} points in 2D\n", n);

    // K-Means with K=3
    println!("--- K-Means (K=3) ---");
    let (assignments, centroids, inertia, iters) = kmeans(&data, 3, 100, &mut rng);

    println!("Converged in {} iterations", iters);
    println!("Inertia (total within-cluster variance): {:.4}", inertia);
    for (i, c) in centroids.iter().enumerate() {
        let count = assignments.iter().filter(|&&a| a == i).count();
        println!("  Cluster {}: centroid=({:.2}, {:.2}), size={}", i, c[0], c[1], count);
    }

    // ASCII visualization
    println!("\n--- Cluster Visualization ---");
    ascii_clusters(&data, &assignments, &centroids);

    // Elbow method
    println!("\n--- Elbow Method (K=1 to 8) ---");
    println!("{:<6} {:>12} {:>12}", "K", "Inertia", "Delta");
    println!("{}", "-".repeat(32));

    let mut prev_inertia = f64::INFINITY;
    let mut elbow_k = 1;
    let mut max_delta = 0.0;

    for k in 1..=8 {
        let (_, _, inertia_k, _) = kmeans(&data, k, 100, &mut rng);
        let delta = prev_inertia - inertia_k;
        let bar = "#".repeat((inertia_k / 50.0).min(30.0) as usize);
        println!("{:<6} {:>12.2} {:>12.2} |{}", k, inertia_k, delta, bar);

        if k > 1 && prev_inertia < f64::INFINITY {
            let improvement = delta / prev_inertia;
            if delta > max_delta && k <= 5 {
                max_delta = delta;
            }
        }

        prev_inertia = inertia_k;
    }

    // Silhouette score
    println!("\n--- Silhouette Analysis ---");
    for k in 2..=5 {
        let (assign, _, _, _) = kmeans(&data, k, 100, &mut rng);
        let sil = silhouette_score(&data, &assign, k);
        let bar = "#".repeat((sil * 30.0).max(0.0) as usize);
        println!("  K={}: silhouette={:.4} |{}", k, sil, bar);
    }

    // K-Means++ vs Random initialization
    println!("\n--- K-Means++ vs Random Init (10 runs each) ---");
    let mut random_inertias = Vec::new();
    let mut pp_inertias = Vec::new();

    for trial in 0..10 {
        rng = lcg(rng);
        let (_, _, ri, _) = kmeans_random_init(&data, 3, 100, &mut rng);
        random_inertias.push(ri);
        let (_, _, pi, _) = kmeans(&data, 3, 100, &mut rng);
        pp_inertias.push(pi);
    }

    let r_mean = random_inertias.iter().sum::<f64>() / 10.0;
    let r_std = (random_inertias.iter().map(|x| (x-r_mean).powi(2)).sum::<f64>()/10.0).sqrt();
    let p_mean = pp_inertias.iter().sum::<f64>() / 10.0;
    let p_std = (pp_inertias.iter().map(|x| (x-p_mean).powi(2)).sum::<f64>()/10.0).sqrt();

    println!("  Random init: inertia = {:.2} +/- {:.2}", r_mean, r_std);
    println!("  K-Means++:   inertia = {:.2} +/- {:.2}", p_mean, p_std);

    kata_metric("kmeans_inertia_k3", inertia);
    kata_metric("kmeans_iterations", iters as f64);
    kata_metric("n_clusters", 3.0);
}

fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize, rng: &mut u64) -> (Vec<usize>, Vec<Vec<f64>>, f64, usize) {
    let dim = data[0].len();
    // K-Means++ initialization
    let mut centroids: Vec<Vec<f64>> = Vec::new();
    *rng = lcg(*rng);
    centroids.push(data[(*rng as usize) % data.len()].clone());

    for _ in 1..k {
        let dists: Vec<f64> = data.iter().map(|p| {
            centroids.iter().map(|c| dist(p, c)).fold(f64::INFINITY, f64::min)
        }).collect();
        let total: f64 = dists.iter().sum();
        *rng = lcg(*rng);
        let mut target = (*rng as f64 / u64::MAX as f64) * total;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            target -= d;
            if target <= 0.0 { chosen = i; break; }
        }
        centroids.push(data[chosen].clone());
    }

    let mut assignments = vec![0usize; data.len()];

    for iter in 0..max_iter {
        // Assign
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let nearest = centroids.iter().enumerate()
                .min_by(|a, b| dist(point, a.1).partial_cmp(&dist(point, b.1)).unwrap())
                .unwrap().0;
            if assignments[i] != nearest { changed = true; }
            assignments[i] = nearest;
        }

        if !changed { return (assignments, centroids, compute_inertia(data, &assignments, &centroids), iter + 1); }

        // Update centroids
        for c in 0..k {
            let members: Vec<&Vec<f64>> = data.iter().enumerate()
                .filter(|(i, _)| assignments[*i] == c).map(|(_, p)| p).collect();
            if members.is_empty() { continue; }
            centroids[c] = vec![0.0; dim];
            for p in &members {
                for d in 0..dim { centroids[c][d] += p[d]; }
            }
            for d in 0..dim { centroids[c][d] /= members.len() as f64; }
        }
    }

    let inertia = compute_inertia(data, &assignments, &centroids);
    (assignments, centroids, inertia, max_iter)
}

fn kmeans_random_init(data: &[Vec<f64>], k: usize, max_iter: usize, rng: &mut u64) -> (Vec<usize>, Vec<Vec<f64>>, f64, usize) {
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = Vec::new();
    for _ in 0..k {
        *rng = lcg(*rng);
        centroids.push(data[(*rng as usize) % data.len()].clone());
    }
    let mut assignments = vec![0usize; data.len()];
    for iter in 0..max_iter {
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let nearest = centroids.iter().enumerate()
                .min_by(|a, b| dist(point, a.1).partial_cmp(&dist(point, b.1)).unwrap()).unwrap().0;
            if assignments[i] != nearest { changed = true; }
            assignments[i] = nearest;
        }
        if !changed { return (assignments, centroids, compute_inertia(data, &assignments, &centroids), iter+1); }
        for c in 0..k {
            let members: Vec<&Vec<f64>> = data.iter().enumerate().filter(|(i,_)| assignments[*i]==c).map(|(_,p)| p).collect();
            if members.is_empty() { continue; }
            centroids[c] = vec![0.0; dim];
            for p in &members { for d in 0..dim { centroids[c][d] += p[d]; } }
            for d in 0..dim { centroids[c][d] /= members.len() as f64; }
        }
    }
    (assignments, centroids, compute_inertia(data, &assignments, &centroids), max_iter)
}

fn compute_inertia(data: &[Vec<f64>], assign: &[usize], centroids: &[Vec<Vec<f64>>]) -> f64 {
    data.iter().enumerate().map(|(i, p)| dist(p, &centroids[assign[i]]).powi(2)).sum()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn silhouette_score(data: &[Vec<f64>], assign: &[usize], k: usize) -> f64 {
    let n = data.len();
    let mut total = 0.0;
    for i in 0..n {
        let a_i = {
            let same: Vec<f64> = data.iter().enumerate()
                .filter(|(j,_)| *j != i && assign[*j] == assign[i])
                .map(|(_,p)| dist(&data[i], p)).collect();
            if same.is_empty() { 0.0 } else { same.iter().sum::<f64>() / same.len() as f64 }
        };
        let b_i = (0..k).filter(|&c| c != assign[i]).map(|c| {
            let other: Vec<f64> = data.iter().enumerate()
                .filter(|(j,_)| assign[*j] == c).map(|(_,p)| dist(&data[i], p)).collect();
            if other.is_empty() { f64::INFINITY } else { other.iter().sum::<f64>() / other.len() as f64 }
        }).fold(f64::INFINITY, f64::min);
        total += (b_i - a_i) / a_i.max(b_i).max(1e-10);
    }
    total / n as f64
}

fn ascii_clusters(data: &[Vec<f64>], assign: &[usize], centroids: &[Vec<Vec<f64>>]) {
    let symbols = ['0', '1', '2', '3', '4'];
    let h = 14; let w = 40;
    let (xmin, xmax, ymin, ymax) = bounds(data);
    let mut grid = vec![vec![' '; w]; h];
    for (i, p) in data.iter().enumerate() {
        let c = ((p[0]-xmin)/(xmax-xmin)*(w-1) as f64).round() as usize;
        let r = ((ymax-p[1])/(ymax-ymin)*(h-1) as f64).round() as usize;
        if c < w && r < h { grid[r][c] = symbols[assign[i] % symbols.len()]; }
    }
    for (i, c) in centroids.iter().enumerate() {
        let col = ((c[0]-xmin)/(xmax-xmin)*(w-1) as f64).round() as usize;
        let row = ((ymax-c[1])/(ymax-ymin)*(h-1) as f64).round() as usize;
        if col < w && row < h { grid[row][col] = '*'; }
    }
    for row in &grid { println!("  |{}", row.iter().collect::<String>()); }
    println!("  +{}", "-".repeat(w));
    println!("  * = centroids, 0/1/2 = cluster assignments");
}

fn bounds(data: &[Vec<f64>]) -> (f64,f64,f64,f64) {
    let xmin = data.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let xmax = data.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let ymin = data.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let ymax = data.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    (xmin, xmax, ymin, ymax)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- K-Means partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to the cluster mean.
- K-Means++ initialization spreads initial centroids apart, producing better and more consistent results than random initialization.
- The elbow method and silhouette score help determine the optimal number of clusters K.
- K-Means assumes spherical, equally-sized clusters and is sensitive to outliers. It is a starting point for clustering, not the final word.
