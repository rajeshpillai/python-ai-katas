# Hierarchical Clustering

> Phase 5 — Unsupervised Learning | Kata 5.02

---

## Concept & Intuition

### What problem are we solving?

Hierarchical clustering builds a tree of clusters that shows how data points group together at different levels of granularity. Unlike K-Means, which requires you to specify K upfront, hierarchical clustering produces a complete hierarchy (dendrogram) from which you can extract any number of clusters by cutting at the desired level. This is valuable when you want to explore clustering structure at multiple scales.

Agglomerative (bottom-up) clustering starts with each point as its own cluster and repeatedly merges the two closest clusters until only one remains. The choice of how to measure distance between clusters — single linkage (minimum), complete linkage (maximum), or average linkage — produces different cluster shapes and behaviors.

In this kata, we implement agglomerative hierarchical clustering from scratch, building the merge history (dendrogram) and exploring different linkage strategies.

### Why naive approaches fail

K-Means forces a flat partition and requires K in advance. If the data has a hierarchical structure (e.g., customers group into segments, which group into markets, which group into regions), K-Means at any single K misses the multi-scale structure. Hierarchical clustering preserves all levels, letting you explore coarse and fine groupings from a single run.

### Mental models

- **Bottom-up tree building**: Start with N singleton clusters. At each step, merge the two closest clusters. After N-1 merges, everything is in one cluster. The merge history is the dendrogram.
- **Linkage as merge criterion**: Single linkage (minimum distance) creates long, chain-like clusters. Complete linkage (maximum distance) creates compact, spherical clusters. Average linkage balances both.
- **Cutting the dendrogram**: Choose a height to cut the dendrogram, and the branches below the cut define your clusters. Lower cuts give more, smaller clusters; higher cuts give fewer, larger clusters.

### Visual explanations

```
  Dendrogram:

  Height
  5.0  |         ______|______
       |        |              |
  3.0  |    ____|____      ____|
       |   |         |    |    |
  1.5  |  _|_       _|_   |    |
       | |   |     |   |  |    |
  0.0  | A   B     C   D  E    F

  Cut at height 3.0 -> 2 clusters: {A,B,C,D} and {E,F}
  Cut at height 1.5 -> 3 clusters: {A,B}, {C,D}, {E,F}
```

---

## Hands-on Exploration

1. Implement agglomerative clustering with single, complete, and average linkage.
2. Build the merge history (dendrogram) recording which clusters merged at what distance.
3. Extract flat clusters by cutting the dendrogram at a specified height.
4. Compare the three linkage methods on the same data.

---

## Live Code

```rust
fn main() {
    println!("=== Hierarchical Clustering ===\n");

    let mut rng = 42u64;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut true_labels: Vec<usize> = Vec::new();

    // 3 clusters of different shapes
    for _ in 0..12 { data.push(vec![1.0+randf(&mut rng)*1.5, 1.0+randf(&mut rng)*1.5]); true_labels.push(0); }
    for _ in 0..10 { data.push(vec![5.0+randf(&mut rng)*1.5, 5.0+randf(&mut rng)*1.5]); true_labels.push(1); }
    for _ in 0..10 { data.push(vec![8.0+randf(&mut rng)*1.5, 1.5+randf(&mut rng)*1.5]); true_labels.push(2); }

    let n = data.len();
    println!("Dataset: {} points\n", n);

    // Run hierarchical clustering with different linkages
    for linkage in &["single", "complete", "average"] {
        println!("--- {} Linkage ---", linkage);
        let merges = agglomerative(&data, linkage);

        // Print merge history (last 10 merges)
        println!("  Last merges:");
        let start = if merges.len() > 8 { merges.len() - 8 } else { 0 };
        for i in start..merges.len() {
            let (c1, c2, dist, size) = &merges[i];
            println!("    Merge clusters {} and {} at distance {:.3} (new size: {})",
                c1, c2, dist, size);
        }

        // Cut at different numbers of clusters
        println!("\n  Clustering results:");
        for n_clusters in &[2, 3, 4] {
            let assignments = cut_dendrogram(&merges, n, *n_clusters);
            let sizes: Vec<usize> = (0..*n_clusters)
                .map(|c| assignments.iter().filter(|&&a| a == c).count()).collect();
            println!("    K={}: sizes={:?}", n_clusters, sizes);
        }

        // K=3 comparison
        let assign3 = cut_dendrogram(&merges, n, 3);
        let ari = adjusted_rand_index(&true_labels, &assign3);
        println!("    K=3 Adjusted Rand Index: {:.4}", ari);

        println!();
    }

    // ASCII dendrogram (simplified)
    println!("--- Simplified Dendrogram (average linkage) ---");
    let merges = agglomerative(&data, "average");
    print_simple_dendrogram(&merges, n);

    // Comparison with K-Means
    println!("\n--- Comparison with K-Means ---");
    let hc_assign = cut_dendrogram(&merges, n, 3);
    let (km_assign, _, _, _) = kmeans(&data, 3, 100, &mut rng);

    let hc_ari = adjusted_rand_index(&true_labels, &hc_assign);
    let km_ari = adjusted_rand_index(&true_labels, &km_assign);
    println!("  Hierarchical ARI: {:.4}", hc_ari);
    println!("  K-Means ARI:      {:.4}", km_ari);

    kata_metric("hc_ari_average", hc_ari);
    kata_metric("kmeans_ari", km_ari);
    kata_metric("n_points", n as f64);
}

fn agglomerative(data: &[Vec<f64>], linkage: &str) -> Vec<(usize, usize, f64, usize)> {
    let n = data.len();
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<bool> = vec![true; n + n]; // enough space for merged clusters
    let mut merges: Vec<(usize, usize, f64, usize)> = Vec::new();
    let mut next_id = n;

    // Distance cache
    let mut dist_matrix = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n { for j in (i+1)..n {
        let d = euclidean(&data[i], &data[j]);
        dist_matrix[i][j] = d; dist_matrix[j][i] = d;
    }}

    for _ in 0..(n - 1) {
        // Find closest pair of active clusters
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        let active_ids: Vec<usize> = (0..cluster_members.len()).filter(|&i| active[i]).collect();

        for ii in 0..active_ids.len() {
            for jj in (ii+1)..active_ids.len() {
                let ci = active_ids[ii];
                let cj = active_ids[jj];
                let d = cluster_distance(&cluster_members[ci], &cluster_members[cj], data, linkage);
                if d < best_dist { best_dist = d; best_i = ci; best_j = cj; }
            }
        }

        // Merge
        let new_members: Vec<usize> = cluster_members[best_i].iter()
            .chain(cluster_members[best_j].iter()).cloned().collect();
        let new_size = new_members.len();

        active[best_i] = false;
        active[best_j] = false;
        active.push(true);
        cluster_members.push(new_members);

        merges.push((best_i, best_j, best_dist, new_size));
        next_id += 1;
    }

    merges
}

fn cluster_distance(c1: &[usize], c2: &[usize], data: &[Vec<f64>], linkage: &str) -> f64 {
    match linkage {
        "single" => {
            let mut min_d = f64::INFINITY;
            for &i in c1 { for &j in c2 { min_d = min_d.min(euclidean(&data[i], &data[j])); } }
            min_d
        }
        "complete" => {
            let mut max_d = 0.0;
            for &i in c1 { for &j in c2 { max_d = max_d.max(euclidean(&data[i], &data[j])); } }
            max_d
        }
        _ => { // average
            let mut sum = 0.0; let mut count = 0;
            for &i in c1 { for &j in c2 { sum += euclidean(&data[i], &data[j]); count += 1; } }
            sum / count as f64
        }
    }
}

fn cut_dendrogram(merges: &[(usize, usize, f64, usize)], n: usize, k: usize) -> Vec<usize> {
    // Apply first n-k merges
    let mut labels: Vec<usize> = (0..n).collect();
    let merges_to_apply = n - k;

    for i in 0..merges_to_apply.min(merges.len()) {
        let (c1, c2, _, _) = merges[i];
        let label1 = labels.iter().copied().min().unwrap();
        // Merge: relabel all members of c2 cluster to c1's label
        // This is simplified - we track through original assignments
    }

    // Re-implement with union-find approach
    let mut parent: Vec<usize> = (0..n + merges.len()).collect();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i { parent[i] = find(parent, parent[i]); }
        parent[i]
    }

    let merges_to_apply = if merges.len() > k - 1 { merges.len() - (k - 1) } else { 0 };
    for i in 0..merges_to_apply {
        let (c1, c2, _, _) = merges[i];
        let r1 = find(&mut parent, c1);
        let r2 = find(&mut parent, c2);
        parent[r2] = r1;
    }

    // Get cluster labels for original points
    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let unique_roots: Vec<usize> = {
        let mut u: Vec<usize> = roots.clone(); u.sort(); u.dedup(); u
    };
    roots.iter().map(|r| unique_roots.iter().position(|u| u == r).unwrap()).collect()
}

fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    // Contingency table approach (simplified)
    let mut same_both = 0;
    let mut same_a = 0;
    let mut same_b = 0;
    for i in 0..n {
        for j in (i+1)..n {
            let sa = a[i] == a[j];
            let sb = b[i] == b[j];
            if sa && sb { same_both += 1; }
            if sa { same_a += 1; }
            if sb { same_b += 1; }
        }
    }
    let total_pairs = n * (n - 1) / 2;
    let expected = same_a as f64 * same_b as f64 / total_pairs as f64;
    let max_index = (same_a + same_b) as f64 / 2.0;
    if (max_index - expected).abs() < 1e-10 { return 0.0; }
    (same_both as f64 - expected) / (max_index - expected)
}

fn print_simple_dendrogram(merges: &[(usize, usize, f64, usize)], n: usize) {
    println!("  Merge distances (last 10):");
    let start = if merges.len() > 10 { merges.len() - 10 } else { 0 };
    let max_dist = merges.last().map(|m| m.2).unwrap_or(1.0);
    for i in start..merges.len() {
        let bar_len = (merges[i].2 / max_dist * 30.0) as usize;
        println!("    Step {:>3} (size {:>3}): {:.3} |{}",
            i + 1, merges[i].3, merges[i].2, "#".repeat(bar_len));
    }
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

fn euclidean(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt() }

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Hierarchical clustering builds a tree (dendrogram) showing how clusters merge, allowing you to explore groupings at any level of granularity.
- The linkage criterion determines cluster shape: single linkage creates chains, complete linkage creates compact groups, average linkage balances both.
- Unlike K-Means, hierarchical clustering does not require specifying K in advance — you can cut the dendrogram at any height to get any number of clusters.
- The Adjusted Rand Index provides a quantitative measure of clustering quality when ground truth labels are available.
