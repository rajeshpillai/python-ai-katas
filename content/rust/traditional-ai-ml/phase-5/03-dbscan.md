# DBSCAN

> Phase 5 — Unsupervised Learning | Kata 5.03

---

## Concept & Intuition

### What problem are we solving?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) discovers clusters as dense regions of points separated by regions of low density. Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance, can find clusters of arbitrary shape, and explicitly identifies noise points that do not belong to any cluster.

The algorithm has two parameters: epsilon (the radius of the neighborhood) and min_points (the minimum number of points required to form a dense region). A point is a core point if it has at least min_points neighbors within epsilon distance. Non-core points that are within epsilon of a core point are border points. Points that are neither core nor border are noise.

In this kata, we implement DBSCAN from scratch and demonstrate its ability to find non-spherical clusters and handle noise — two tasks where K-Means struggles.

### Why naive approaches fail

K-Means assumes clusters are convex (spherical) and equal in size. When data contains crescent-shaped clusters, elongated clusters, or clusters of varying density, K-Means fails dramatically. It also assigns every point to a cluster, even outliers that clearly do not belong. DBSCAN handles all these cases by defining clusters through local density rather than distance to a centroid.

### Mental models

- **Dense regions as clusters**: A cluster is a maximal set of density-connected points. Think of it as placing a circle of radius epsilon around each point and connecting points whose circles overlap, as long as the circles contain enough points.
- **Core, border, noise**: Core points are in the heart of a cluster (many neighbors). Border points are on the edge (near a core point but not dense enough themselves). Noise points are isolated.
- **Epsilon and min_points**: Epsilon controls the scale of density. Min_points controls the minimum cluster size. Together they define what "dense" means for your data.

### Visual explanations

```
  K-Means fails:              DBSCAN succeeds:

      111111                      AAAAAA
     1 111  222                  A AAAA  BBBBB
    11  1  2222                 AA  A  BBBB
     111  2222                   AAA  BBBB
           22                         BB
   *   *                       *   *           <- noise points

  K-Means: wrong clusters     DBSCAN: correct clusters + noise
```

---

## Hands-on Exploration

1. Implement neighborhood queries (find all points within epsilon distance).
2. Implement DBSCAN: classify points as core, border, or noise, then expand clusters.
3. Test on data with non-spherical clusters and noise.
4. Explore the effect of epsilon and min_points parameters.

---

## Live Code

```rust
fn main() {
    println!("=== DBSCAN Clustering ===\n");

    let mut rng = 42u64;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut true_labels: Vec<i32> = Vec::new(); // -1 for noise

    // Cluster 0: ring/arc shape
    for i in 0..25 {
        let angle = std::f64::consts::PI * i as f64 / 24.0;
        let r = 3.0 + randf(&mut rng) * 0.5;
        data.push(vec![r * angle.cos(), r * angle.sin()]);
        true_labels.push(0);
    }

    // Cluster 1: compact blob
    for _ in 0..20 {
        data.push(vec![5.0 + randf(&mut rng) * 1.0, 5.0 + randf(&mut rng) * 1.0]);
        true_labels.push(1);
    }

    // Noise points
    for _ in 0..8 {
        data.push(vec![randf(&mut rng) * 8.0, randf(&mut rng) * 8.0]);
        true_labels.push(-1);
    }

    let n = data.len();
    println!("Dataset: {} points ({} in clusters, {} noise)\n", n,
        true_labels.iter().filter(|&&l| l >= 0).count(),
        true_labels.iter().filter(|&&l| l < 0).count());

    // DBSCAN with different parameters
    println!("--- Parameter Sweep ---");
    println!("{:<8} {:>10} {:>10} {:>8} {:>8}", "Eps", "MinPts", "Clusters", "Noise", "Points");
    println!("{}", "-".repeat(46));

    let eps_values = vec![0.5, 0.8, 1.0, 1.5, 2.0, 3.0];
    let min_pts_values = vec![3, 5];

    for &eps in &eps_values {
        for &min_pts in &min_pts_values {
            let (labels, n_clusters) = dbscan(&data, eps, min_pts);
            let n_noise = labels.iter().filter(|&&l| l == -1).count();
            let n_assigned = n - n_noise;
            println!("{:<8.1} {:>10} {:>10} {:>8} {:>8}",
                eps, min_pts, n_clusters, n_noise, n_assigned);
        }
    }

    // Best parameters
    let eps = 1.0;
    let min_pts = 3;
    println!("\n--- DBSCAN (eps={}, min_pts={}) ---", eps, min_pts);
    let (labels, n_clusters) = dbscan(&data, eps, min_pts);

    // Point classification
    let n_core = count_core_points(&data, eps, min_pts);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let n_border = n - n_core - n_noise;

    println!("  Clusters found: {}", n_clusters);
    println!("  Core points:    {}", n_core);
    println!("  Border points:  {}", n_border);
    println!("  Noise points:   {}", n_noise);

    // Cluster sizes
    for c in 0..n_clusters as i32 {
        let size = labels.iter().filter(|&&l| l == c).count();
        println!("  Cluster {}: {} points", c, size);
    }

    // Visualization
    println!("\n--- Cluster Visualization ---");
    ascii_dbscan(&data, &labels);

    // Compare with K-Means
    println!("\n--- DBSCAN vs K-Means ---");
    let (km_labels, _, _, _) = kmeans(&data, 2, 100, &mut rng);
    let km_noise = 0; // K-Means assigns everything

    println!("  DBSCAN: {} clusters, {} noise points", n_clusters, n_noise);
    println!("  K-Means (K=2): 2 clusters, 0 noise points");
    println!("  K-Means forces noise points into clusters!");

    // Epsilon neighborhood analysis
    println!("\n--- Neighborhood Size Distribution ---");
    let mut neighbor_counts: Vec<usize> = data.iter().map(|p| {
        data.iter().filter(|q| euclidean(p, q) <= eps && euclidean(p, q) > 0.0).count()
    }).collect();
    neighbor_counts.sort();

    println!("  Min neighbors: {}", neighbor_counts[0]);
    println!("  Max neighbors: {}", neighbor_counts[n-1]);
    println!("  Median neighbors: {}", neighbor_counts[n/2]);
    println!("  Points with < {} neighbors (noise candidates): {}",
        min_pts, neighbor_counts.iter().filter(|&&c| c < min_pts).count());

    kata_metric("n_clusters", n_clusters as f64);
    kata_metric("n_noise", n_noise as f64);
    kata_metric("n_core", n_core as f64);
    kata_metric("epsilon", eps);
    kata_metric("min_points", min_pts as f64);
}

fn dbscan(data: &[Vec<f64>], eps: f64, min_pts: usize) -> (Vec<i32>, usize) {
    let n = data.len();
    let mut labels = vec![-1i32; n]; // -1 = unvisited/noise
    let mut cluster_id: i32 = 0;

    for i in 0..n {
        if labels[i] != -1 { continue; } // Already processed

        let neighbors = region_query(data, i, eps);

        if neighbors.len() < min_pts {
            // Noise (might become border later)
            continue;
        }

        // Start new cluster
        labels[i] = cluster_id;
        let mut seed_set: Vec<usize> = neighbors.clone();
        let mut j = 0;

        while j < seed_set.len() {
            let q = seed_set[j];

            if labels[q] == -1 {
                labels[q] = cluster_id; // Was noise, now border
            }

            if labels[q] != -1 && labels[q] != cluster_id {
                j += 1;
                continue; // Already in another cluster? Skip. (simplified)
            }

            labels[q] = cluster_id;

            let q_neighbors = region_query(data, q, eps);
            if q_neighbors.len() >= min_pts {
                // q is a core point, add its neighbors to seed set
                for &nn in &q_neighbors {
                    if labels[nn] == -1 || labels[nn] == -1 {
                        if !seed_set.contains(&nn) {
                            seed_set.push(nn);
                        }
                    }
                }
            }

            j += 1;
        }

        cluster_id += 1;
    }

    (labels, cluster_id as usize)
}

fn region_query(data: &[Vec<f64>], point_idx: usize, eps: f64) -> Vec<usize> {
    data.iter().enumerate()
        .filter(|(j, p)| *j != point_idx && euclidean(&data[point_idx], p) <= eps)
        .map(|(j, _)| j)
        .collect()
}

fn count_core_points(data: &[Vec<f64>], eps: f64, min_pts: usize) -> usize {
    data.iter().enumerate().filter(|(i, _)| {
        region_query(data, *i, eps).len() >= min_pts
    }).count()
}

fn ascii_dbscan(data: &[Vec<f64>], labels: &[i32]) {
    let symbols = ['0', '1', '2', '3', '4', '5'];
    let h = 14; let w = 40;
    let (xmin, xmax, ymin, ymax) = bounds(data);
    let mut grid = vec![vec![' '; w]; h];

    for (i, p) in data.iter().enumerate() {
        let c = ((p[0]-xmin)/(xmax-xmin+0.01)*(w-1) as f64).round().max(0.0) as usize;
        let r = ((ymax-p[1])/(ymax-ymin+0.01)*(h-1) as f64).round().max(0.0) as usize;
        if c < w && r < h {
            grid[r][c] = if labels[i] == -1 { '*' }
                else { symbols[labels[i] as usize % symbols.len()] };
        }
    }

    for row in &grid { println!("  |{}", row.iter().collect::<String>()); }
    println!("  +{}", "-".repeat(w));
    println!("  0/1/2 = cluster labels, * = noise");
}

fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize, rng: &mut u64) -> (Vec<usize>, Vec<Vec<f64>>, f64, usize) {
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = (0..k).map(|_| { *rng = lcg(*rng); data[(*rng as usize)%data.len()].clone() }).collect();
    let mut assign = vec![0usize; data.len()];
    for iter in 0..max_iter {
        let mut changed = false;
        for (i, p) in data.iter().enumerate() {
            let near = centroids.iter().enumerate().min_by(|a,b| euclidean(p,a.1).partial_cmp(&euclidean(p,b.1)).unwrap()).unwrap().0;
            if assign[i]!=near{changed=true;} assign[i]=near;
        }
        if !changed { let inertia = data.iter().enumerate().map(|(i,p)| euclidean(p,&centroids[assign[i]]).powi(2)).sum(); return (assign, centroids, inertia, iter+1); }
        for c in 0..k {
            let members: Vec<&Vec<f64>> = data.iter().enumerate().filter(|(i,_)| assign[*i]==c).map(|(_,p)| p).collect();
            if members.is_empty(){continue;} centroids[c]=vec![0.0;dim];
            for p in &members{for d in 0..dim{centroids[c][d]+=p[d];}} for d in 0..dim{centroids[c][d]/=members.len() as f64;}
        }
    }
    let inertia = data.iter().enumerate().map(|(i,p)| euclidean(p,&centroids[assign[i]]).powi(2)).sum();
    (assign, centroids, inertia, max_iter)
}

fn bounds(data: &[Vec<f64>]) -> (f64,f64,f64,f64) {
    (data.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
     data.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
     data.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
     data.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max))
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)|(x-y).powi(2)).sum::<f64>().sqrt() }

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- DBSCAN finds clusters as dense regions separated by sparse regions, without requiring K to be specified in advance.
- It naturally handles noise (points in sparse regions are labeled as outliers) and discovers clusters of arbitrary shape.
- The epsilon and min_points parameters define the density threshold. Choosing them well requires understanding the data's distance distribution.
- DBSCAN struggles with clusters of varying density. Points in sparser clusters may be classified as noise if epsilon is too small for them.
