# Positional Encoding

> Phase 7 — Attention & Transformers | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

Self-attention is permutation equivariant: if you shuffle the input positions, the output at each position is the same (just shuffled correspondingly). This means that a transformer without positional encoding treats "dog bites man" and "man bites dog" as equivalent because it has no concept of order. Since word order is fundamental to language meaning, we must explicitly inject positional information into the model.

The original transformer paper uses sinusoidal positional encodings: each position is represented by a vector of sines and cosines at different frequencies. Position p gets the encoding PE(p, 2i) = sin(p / 10000^(2i/d)) and PE(p, 2i+1) = cos(p / 10000^(2i/d)), where d is the embedding dimension and i is the dimension index. These encodings are added directly to the input embeddings before the first attention layer.

The sinusoidal design has two elegant properties. First, each position has a unique encoding, so the model can distinguish any two positions. Second, the encoding of relative positions is representable as a linear transformation of absolute positions: PE(p+k) can be expressed as a linear function of PE(p) for any fixed offset k. This means the model can learn to attend to relative positions (like "the word 3 positions back") rather than only absolute positions. Modern transformers often use learned positional embeddings or relative position encodings like RoPE, but the sinusoidal encoding illustrates the core principles beautifully.

### Why naive approaches fail

The simplest approach is to add a scalar position index (0, 1, 2, ...) to the embedding. This fails because the magnitude of the position number quickly dominates the embedding values, distorting the semantic content. Position 1000 would have a drastically different scale than position 1.

Normalizing positions to [0, 1] (dividing by sequence length) makes the encoding dependent on sequence length, which varies across examples. A position at 0.5 means "middle" in a length-100 sequence but also "middle" in a length-10 sequence, conflating positions 50 and 5.

One-hot position vectors solve uniqueness but are high-dimensional (one dimension per possible position) and cannot generalize to unseen lengths. They also provide no notion of distance: positions 1 and 2 are as different from each other as positions 1 and 1000.

### Mental models

- **Musical tuning forks**: Each dimension of the positional encoding vibrates at a different frequency, like a set of tuning forks. Low-frequency dimensions vary slowly (capturing coarse position) while high-frequency dimensions vary rapidly (capturing fine position). Together, they uniquely identify each position.
- **Clock hands**: Think of a clock with many hands rotating at different speeds. The hour hand (low frequency) tells you the rough position; the second hand (high frequency) tells you the precise position. The combination is unique for every moment.
- **GPS coordinates**: Just as latitude and longitude uniquely identify a location on Earth, the combination of sines and cosines at different frequencies uniquely identifies each position in the sequence.

### Visual explanations

```
  Sinusoidal positional encoding (dim=8, positions 0-7):

  pos  dim0   dim1   dim2   dim3   dim4   dim5   dim6   dim7
       sin    cos    sin    cos    sin    cos    sin    cos
       f=1    f=1    f=.1   f=.1   f=.01  f=.01  f=.001 f=.001

   0:  0.00   1.00   0.00   1.00   0.00   1.00   0.00   1.00
   1:  0.84   0.54   0.10   0.99   0.01   1.00   0.00   1.00
   2:  0.91  -0.42   0.20   0.98   0.02   1.00   0.00   1.00
   3:  0.14  -0.99   0.29   0.96   0.03   1.00   0.00   1.00
   4: -0.76  -0.65   0.39   0.92   0.04   1.00   0.00   1.00

  High-freq dims (left):  change rapidly -> fine position
  Low-freq dims (right):  change slowly  -> coarse position

  Relative position as linear transform:
  PE(p+k) = R_k * PE(p)   where R_k is a rotation matrix
  This means "3 positions ahead" is the same transformation
  regardless of absolute position!
```

---

## Hands-on Exploration

1. Implement sinusoidal positional encoding and visualize the pattern.
2. Show that each position has a unique encoding using distance metrics.
3. Demonstrate the relative position property by computing PE(p+k) - PE(p) for different p.
4. Compare positional encoding approaches and their trade-offs.

---

## Live Code

```rust
fn main() {
    println!("=== Positional Encoding ===\n");

    let d_model = 16; // Embedding dimension
    let max_len = 20; // Sequence length

    // Generate sinusoidal positional encodings
    let encodings: Vec<Vec<f64>> = (0..max_len)
        .map(|pos| positional_encoding(pos, d_model))
        .collect();

    // Visualize the encoding pattern
    println!("--- Sinusoidal Encoding Pattern (first 10 positions, 8 dims) ---\n");
    print!("{:>5}", "pos");
    for d in 0..8 {
        print!("{:>8}", format!("d{}", d));
    }
    println!();

    for pos in 0..10 {
        print!("{:>5}", pos);
        for d in 0..8 {
            print!("{:>8.3}", encodings[pos][d]);
        }
        println!();
    }

    // ASCII heatmap of the full encoding
    println!("\n--- Encoding Heatmap (20 positions x 16 dims) ---\n");
    print!("{:>5}", "");
    for d in 0..d_model {
        print!("{}", if d % 4 == 0 { "|" } else { " " });
    }
    println!();

    for pos in 0..max_len {
        print!("{:>4} ", pos);
        for d in 0..d_model {
            let v = encodings[pos][d];
            let c = if v > 0.5 {
                '#'
            } else if v > 0.0 {
                '+'
            } else if v > -0.5 {
                '-'
            } else {
                ' '
            };
            print!("{}", c);
        }
        println!();
    }
    println!("\n  High-freq dims (left) change rapidly; low-freq dims (right) change slowly.\n");

    // Uniqueness: pairwise distances
    println!("--- Position Uniqueness (pairwise L2 distances) ---\n");
    print!("{:>5}", "");
    for j in 0..8 {
        print!("{:>6}", j);
    }
    println!();

    for i in 0..8 {
        print!("{:>5}", i);
        for j in 0..8 {
            let dist = l2_distance(&encodings[i], &encodings[j]);
            print!("{:>6.2}", dist);
        }
        println!();
    }
    println!("\n  Diagonal is 0 (same position). Adjacent positions have similar distances.");
    println!("  Distant positions have larger distances.\n");

    // Relative position property
    println!("--- Relative Position Property ---\n");
    println!("Difference vectors PE(p+k) - PE(p) for fixed k should be similar:");

    for k in &[1, 2, 3] {
        let mut diffs = Vec::new();
        for p in 0..(max_len - *k) {
            let diff: Vec<f64> = encodings[p + k]
                .iter()
                .zip(encodings[p].iter())
                .map(|(a, b)| a - b)
                .collect();
            diffs.push(diff);
        }

        // Measure consistency: how similar are the difference vectors?
        let mut similarities = Vec::new();
        for i in 0..diffs.len() - 1 {
            similarities.push(cosine_similarity(&diffs[i], &diffs[i + 1]));
        }
        let avg_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let min_sim = similarities.iter().cloned().fold(f64::INFINITY, f64::min);

        println!(
            "  k={}: avg cosine similarity of PE(p+{}) - PE(p) = {:.4} (min={:.4})",
            k, k, avg_sim, min_sim
        );
    }
    println!("\n  High similarity means the 'direction' for relative offset k");
    println!("  is consistent regardless of absolute position.\n");

    // Demonstrate effect on attention
    println!("--- Effect on Attention ---\n");

    // Two identical words at different positions
    let word_embedding = vec![0.5; d_model];

    let pos3: Vec<f64> = word_embedding
        .iter()
        .zip(encodings[3].iter())
        .map(|(w, p)| w + p)
        .collect();
    let pos7: Vec<f64> = word_embedding
        .iter()
        .zip(encodings[7].iter())
        .map(|(w, p)| w + p)
        .collect();

    let sim_with_pe = cosine_similarity(&pos3, &pos7);
    let sim_without_pe = cosine_similarity(&word_embedding, &word_embedding);

    println!("  Same word at pos 3 and pos 7:");
    println!("    Without PE: similarity = {:.4} (identical, no position info)", sim_without_pe);
    println!("    With PE:    similarity = {:.4} (different, positions distinguished)", sim_with_pe);

    // Different words at same position vs same words at different positions
    let word_a = vec![0.8, 0.1, 0.3, 0.9, 0.2, 0.5, 0.1, 0.7, 0.3, 0.4, 0.6, 0.2, 0.8, 0.1, 0.5, 0.3];
    let word_b = vec![0.1, 0.7, 0.2, 0.4, 0.8, 0.3, 0.6, 0.1, 0.5, 0.2, 0.9, 0.4, 0.1, 0.7, 0.3, 0.6];

    let a_pos5: Vec<f64> = word_a.iter().zip(encodings[5].iter()).map(|(w, p)| w + p).collect();
    let b_pos5: Vec<f64> = word_b.iter().zip(encodings[5].iter()).map(|(w, p)| w + p).collect();
    let a_pos15: Vec<f64> = word_a.iter().zip(encodings[15].iter()).map(|(w, p)| w + p).collect();

    println!("\n  Different words at same position (5): sim = {:.4}", cosine_similarity(&a_pos5, &b_pos5));
    println!("  Same word at positions 5 and 15:     sim = {:.4}", cosine_similarity(&a_pos5, &a_pos15));
    println!("\n  The encoding balances word identity and position information.");
}

fn positional_encoding(pos: usize, d_model: usize) -> Vec<f64> {
    let mut encoding = vec![0.0; d_model];
    for i in 0..d_model {
        let dim_pair = i / 2;
        let freq = 1.0 / 10000.0_f64.powf(2.0 * dim_pair as f64 / d_model as f64);
        let angle = pos as f64 * freq;
        encoding[i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
    }
    encoding
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}
```

---

## Key Takeaways

- Transformers need explicit positional encoding because self-attention is inherently position-agnostic and would treat any permutation of the input identically.
- Sinusoidal encodings use sines and cosines at different frequencies to create unique, continuous position representations that generalize to unseen sequence lengths.
- The relative position property means that the transformation from PE(p) to PE(p+k) is consistent for any absolute position p, enabling the model to learn relative position patterns.
- Positional encodings are added to word embeddings, creating representations that balance semantic content with positional information, allowing the model to use both.
