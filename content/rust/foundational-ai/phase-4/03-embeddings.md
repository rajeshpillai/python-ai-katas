# Embeddings

> Phase 4 — Representation Learning | Kata 4.3

---

## Concept & Intuition

### What problem are we solving?

Embeddings are dense, low-dimensional vector representations of discrete objects (words, users, products, categories). Instead of representing a word as a one-hot vector with 50,000 dimensions (one for each word in the vocabulary), we learn a dense vector of, say, 128 dimensions that captures the word's meaning. Similar words get similar vectors — "king" and "queen" are close in embedding space, while "king" and "banana" are far apart.

Embeddings solve the curse of dimensionality for categorical data. A one-hot encoding of 10,000 products creates 10,000 sparse features that carry no information about similarity. An embedding maps those 10,000 products to 50-dimensional vectors where similar products are near each other. The model can then generalize: if it learns something about product A, it automatically applies that knowledge to similar products nearby in embedding space.

The magic of embeddings is that they are learned, not designed. During training, the embedding vectors are adjusted so that items that appear in similar contexts get similar representations. This is how Word2Vec works: words that appear in similar sentences develop similar embeddings. The structure of the embedding space (distances, directions) becomes meaningful.

### Why naive approaches fail

One-hot encoding treats every category as equally different from every other category. "Dog" is as different from "cat" as it is from "skyscraper." This wastes model capacity and prevents generalization. Embeddings learn a geometry where distances are meaningful — similar items cluster together, and even directions can be meaningful (e.g., the "gender" direction in word embedding space).

### Mental models

- **Embedding as a compressed address**: A one-hot vector is like using a full street address for every word. An embedding is like using GPS coordinates — much more compact, and proximity is meaningful.
- **Lookup table that learns**: An embedding layer is literally a matrix where row i is the vector for item i. Training adjusts these vectors so the geometry captures similarity.
- **Embeddings capture relationships**: The famous example: king - man + woman ≈ queen. Directions in embedding space correspond to semantic relationships.

### Visual explanations

```
  One-hot encoding:                Learned embedding:
  "cat"  → [1, 0, 0, 0, 0]       "cat"  → [0.2, 0.8, -0.1]
  "dog"  → [0, 1, 0, 0, 0]       "dog"  → [0.3, 0.7, -0.2]
  "fish" → [0, 0, 1, 0, 0]       "fish" → [0.1, 0.3, 0.9]
  "tree" → [0, 0, 0, 1, 0]       "tree" → [-0.8, 0.1, 0.4]
  "rock" → [0, 0, 0, 0, 1]       "rock" → [-0.7, -0.3, 0.5]

  5D, all distances equal          3D, similar items cluster!
```

---

## Hands-on Exploration

1. Implement an embedding lookup table and learn embeddings via gradient descent.
2. Train embeddings so that items in similar contexts develop similar vectors.
3. Visualize the learned embedding space and observe clustering.

---

## Live Code

```rust
fn main() {
    // === Embeddings from Scratch ===
    // Learning dense vector representations for discrete items.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    println!("=== Embeddings: Learning Dense Representations ===\n");

    // === Problem Setup ===
    // We have items that appear in contexts. Items in similar contexts should
    // get similar embeddings. This is like a simplified Word2Vec.

    let items = vec![
        "cat", "dog", "fish", "bird",       // animals
        "car", "truck", "bike", "bus",       // vehicles
        "apple", "banana", "grape", "mango", // fruits
    ];
    let n_items = items.len();
    let embed_dim = 4;

    // Define co-occurrence contexts (items that appear together)
    // Animals with animals, vehicles with vehicles, fruits with fruits
    let contexts: Vec<(usize, usize)> = vec![
        // Animals co-occur
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        // Vehicles co-occur
        (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        // Fruits co-occur
        (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11),
        // Cross-group (rare, negative examples)
    ];

    // Negative examples: items from different groups
    let negatives: Vec<(usize, usize)> = vec![
        (0, 4), (0, 8), (1, 5), (1, 9), (2, 6), (2, 10),
        (3, 7), (3, 11), (4, 8), (5, 9), (6, 10), (7, 11),
    ];

    // === Initialize embeddings ===
    let mut embeddings: Vec<Vec<f64>> = (0..n_items)
        .map(|_| (0..embed_dim).map(|_| rand_f64() * 0.5).collect())
        .collect();

    println!("  {} items, {} dimensions per embedding\n", n_items, embed_dim);
    println!("  Items: {:?}\n", items);

    // === Training: make similar items close, dissimilar items far ===
    // Using a simple contrastive loss:
    // For similar pairs: loss = ||a - b||^2 (minimize distance)
    // For dissimilar pairs: loss = max(0, margin - ||a - b||)^2 (push apart)

    let lr = 0.05;
    let margin = 2.0;
    let n_epochs = 300;

    println!("  Training embeddings...\n");
    println!("  {:>6} {:>12} {:>12} {:>12}",
        "epoch", "pos_loss", "neg_loss", "total_loss");
    println!("  {:->6} {:->12} {:->12} {:->12}", "", "", "", "");

    for epoch in 0..n_epochs {
        let mut pos_loss = 0.0;
        let mut neg_loss = 0.0;

        // Positive pairs: pull together
        for &(i, j) in &contexts {
            let dist_sq: f64 = (0..embed_dim)
                .map(|d| (embeddings[i][d] - embeddings[j][d]).powi(2))
                .sum();
            pos_loss += dist_sq;

            // Gradient: d(||a-b||^2)/da = 2(a-b)
            for d in 0..embed_dim {
                let grad = 2.0 * (embeddings[i][d] - embeddings[j][d]);
                embeddings[i][d] -= lr * grad;
                embeddings[j][d] += lr * grad; // opposite direction
            }
        }

        // Negative pairs: push apart
        for &(i, j) in &negatives {
            let dist: f64 = (0..embed_dim)
                .map(|d| (embeddings[i][d] - embeddings[j][d]).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < margin {
                let loss = (margin - dist).powi(2);
                neg_loss += loss;

                // Gradient: push apart
                for d in 0..embed_dim {
                    let diff = embeddings[i][d] - embeddings[j][d];
                    let grad = -2.0 * (margin - dist) * diff / (dist + 1e-8);
                    embeddings[i][d] -= lr * grad;
                    embeddings[j][d] += lr * grad;
                }
            }
        }

        let total = pos_loss + neg_loss;
        if epoch < 10 || epoch % 30 == 0 || epoch == n_epochs - 1 {
            println!("  {:>6} {:>12.4} {:>12.4} {:>12.4}",
                epoch, pos_loss, neg_loss, total);
        }
    }

    // === Show learned embeddings ===
    println!("\n=== Learned Embeddings ===\n");
    println!("  {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Item", "dim0", "dim1", "dim2", "dim3");
    println!("  {:->8} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "");

    for (i, name) in items.iter().enumerate() {
        println!("  {:>8} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            name, embeddings[i][0], embeddings[i][1],
            embeddings[i][2], embeddings[i][3]);
    }

    // === Distance matrix ===
    println!("\n=== Pairwise Distances ===\n");

    let cosine_sim = |a: &[f64], b: &[f64]| -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
    };

    let euclidean = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    };

    // Show distances within and across groups
    println!("  Within-group distances (should be SMALL):");
    let groups = vec![
        ("Animals",  vec![0, 1, 2, 3]),
        ("Vehicles", vec![4, 5, 6, 7]),
        ("Fruits",   vec![8, 9, 10, 11]),
    ];

    for (name, indices) in &groups {
        let mut avg_dist = 0.0;
        let mut count = 0;
        for i in 0..indices.len() {
            for j in (i+1)..indices.len() {
                avg_dist += euclidean(&embeddings[indices[i]], &embeddings[indices[j]]);
                count += 1;
            }
        }
        avg_dist /= count as f64;
        println!("    {}: avg distance = {:.3}", name, avg_dist);
    }

    println!("\n  Cross-group distances (should be LARGE):");
    for i in 0..groups.len() {
        for j in (i+1)..groups.len() {
            let mut avg_dist = 0.0;
            let mut count = 0;
            for &a in &groups[i].1 {
                for &b in &groups[j].1 {
                    avg_dist += euclidean(&embeddings[a], &embeddings[b]);
                    count += 1;
                }
            }
            avg_dist /= count as f64;
            println!("    {} ↔ {}: avg distance = {:.3}", groups[i].0, groups[j].0, avg_dist);
        }
    }

    // === Nearest neighbors ===
    println!("\n=== Nearest Neighbors (Most Similar Items) ===\n");

    for query_idx in [0, 4, 8] { // cat, car, apple
        let mut distances: Vec<(usize, f64)> = (0..n_items)
            .filter(|&i| i != query_idx)
            .map(|i| (i, euclidean(&embeddings[query_idx], &embeddings[i])))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        println!("  Nearest to '{}': {} ({:.3}), {} ({:.3}), {} ({:.3})",
            items[query_idx],
            items[distances[0].0], distances[0].1,
            items[distances[1].0], distances[1].1,
            items[distances[2].0], distances[2].1);
    }

    // === Embedding arithmetic ===
    println!("\n=== Embedding Arithmetic ===\n");

    // "animal" direction: average(animals) - average(vehicles)
    let avg_vec = |indices: &[usize]| -> Vec<f64> {
        let n = indices.len() as f64;
        (0..embed_dim).map(|d| {
            indices.iter().map(|&i| embeddings[i][d]).sum::<f64>() / n
        }).collect()
    };

    let animal_center = avg_vec(&[0, 1, 2, 3]);
    let vehicle_center = avg_vec(&[4, 5, 6, 7]);
    let fruit_center = avg_vec(&[8, 9, 10, 11]);

    println!("  Group centroids:");
    println!("    Animals:  [{:.3}, {:.3}, {:.3}, {:.3}]",
        animal_center[0], animal_center[1], animal_center[2], animal_center[3]);
    println!("    Vehicles: [{:.3}, {:.3}, {:.3}, {:.3}]",
        vehicle_center[0], vehicle_center[1], vehicle_center[2], vehicle_center[3]);
    println!("    Fruits:   [{:.3}, {:.3}, {:.3}, {:.3}]",
        fruit_center[0], fruit_center[1], fruit_center[2], fruit_center[3]);

    println!();
    println!("  The embedding space has learned a geometry where");
    println!("  similar items cluster together and groups are separated.");

    println!();
    println!("Key insight: Embeddings transform discrete items into continuous vectors");
    println!("where distance reflects similarity. This enables generalization:");
    println!("knowledge about one item transfers to nearby items in embedding space.");
}
```

---

## Key Takeaways

- Embeddings map discrete items (words, categories, products) to dense, low-dimensional vectors where distance reflects similarity.
- One-hot encoding treats all items as equally different; embeddings learn a geometry where similar items cluster together.
- Embeddings are learned during training — items that appear in similar contexts develop similar vector representations.
- The structure of the embedding space becomes meaningful: distances, directions, and arithmetic operations on vectors carry semantic information.
