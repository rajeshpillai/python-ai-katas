# Visualizing Distributions

> Phase 0 — Foundations | Kata 0.5

---

## Concept & Intuition

### What problem are we solving?

A distribution describes how values in a dataset are spread out. Are most values clustered in the center, or spread evenly? Are there outliers? Is the distribution symmetric or skewed? Before building any model, you need to understand the shape of your data, because many algorithms make assumptions about distributions (e.g., linear regression assumes normally distributed errors).

Visualizing distributions means building histograms, computing percentiles, and characterizing shape with statistics like skewness and kurtosis. In a world without plotting libraries, we build ASCII histograms — which forces us to understand exactly what a histogram is: counting how many values fall into each bin.

The most important distribution in statistics and machine learning is the normal (Gaussian) distribution, characterized by its mean and standard deviation. But real data is often not normal — it may be skewed, multimodal, or heavy-tailed. Recognizing these shapes tells you which tools are appropriate and which assumptions you can rely on.

### Why naive approaches fail

Treating all data as normally distributed when it is not can lead to systematically wrong predictions. For example, income data is heavily right-skewed — the mean income is much higher than the median because a few very high earners pull the average up. Using the mean as a "typical" value would be misleading. Always visualize your distributions before making assumptions.

### Mental models

- **Histogram as a fingerprint**: The shape of a distribution is like a fingerprint for your data. Two datasets can have the same mean but look completely different.
- **Mean vs. median**: The mean is pulled by outliers; the median splits the data in half. When they differ substantially, your distribution is skewed.
- **Standard deviation as spread**: One standard deviation from the mean captures about 68% of normally distributed data. Two standard deviations capture about 95%.

### Visual explanations

```
  Normal distribution:        Skewed distribution:
  (symmetric, bell-shaped)    (long tail to the right)

  count                       count
    │     ███                   │ ████
    │   ███████                 │ ██████
    │  █████████                │ █████████
    │ ███████████               │ ████████████
    │█████████████              │ ██████████████████
    └─────────── value          └─────────────────── value
       mean=median                 median < mean
```

---

## Hands-on Exploration

1. Generate data from different distributions (uniform, normal-like, skewed).
2. Build ASCII histograms to visualize the shape of each distribution.
3. Compute summary statistics and observe how they relate to the distribution shape.

---

## Live Code

```rust
fn main() {
    // === Visualizing Distributions ===
    // Understanding the shape of your data through histograms and statistics.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_uniform = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64  // uniform [0, 1)
    };

    // Approximate normal distribution using Box-Muller-like approach
    // Sum of 12 uniform [0,1) values minus 6 ≈ N(0,1)
    let mut rand_normal = |mean: f64, std: f64| -> f64 {
        let sum: f64 = (0..12).map(|_| rand_uniform()).sum::<f64>();
        mean + std * (sum - 6.0)
    };

    // === Helper: build and print ASCII histogram ===
    let print_histogram = |data: &[f64], title: &str, n_bins: usize| {
        println!("  {}", title);

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let bin_width = range / n_bins as f64;

        // Count values in each bin
        let mut bins = vec![0usize; n_bins];
        for &val in data {
            let bin = ((val - min) / bin_width) as usize;
            let bin = bin.min(n_bins - 1);
            bins[bin] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_scale = 40.0 / max_count as f64;

        for i in 0..n_bins {
            let lo = min + i as f64 * bin_width;
            let hi = lo + bin_width;
            let bar_len = (bins[i] as f64 * bar_scale) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("    [{:>6.1}, {:>6.1}) {:>4} │{}",
                lo, hi, bins[i], bar);
        }
        println!();
    };

    // === Distribution 1: Uniform ===
    let n = 500;
    let uniform_data: Vec<f64> = (0..n).map(|_| rand_uniform() * 10.0).collect();

    println!("=== Distribution Shapes ===\n");
    print_histogram(&uniform_data, "Uniform Distribution [0, 10):", 10);

    // === Distribution 2: Normal (bell curve) ===
    let normal_data: Vec<f64> = (0..n).map(|_| rand_normal(50.0, 10.0)).collect();
    print_histogram(&normal_data, "Normal Distribution (mean=50, std=10):", 12);

    // === Distribution 3: Right-skewed (exponential-like) ===
    let skewed_data: Vec<f64> = (0..n).map(|_| {
        // -ln(U) gives exponential distribution
        let u = rand_uniform();
        let u = if u < 1e-10 { 1e-10 } else { u };
        -u.ln() * 10.0
    }).collect();
    print_histogram(&skewed_data, "Right-Skewed Distribution (exponential-like):", 12);

    // === Distribution 4: Bimodal ===
    let bimodal_data: Vec<f64> = (0..n).map(|_| {
        if rand_uniform() < 0.5 {
            rand_normal(30.0, 5.0)
        } else {
            rand_normal(70.0, 5.0)
        }
    }).collect();
    print_histogram(&bimodal_data, "Bimodal Distribution (two peaks):", 15);

    // === Summary Statistics Comparison ===
    println!("=== Summary Statistics ===\n");

    let datasets: Vec<(&str, &Vec<f64>)> = vec![
        ("Uniform", &uniform_data),
        ("Normal", &normal_data),
        ("Right-skewed", &skewed_data),
        ("Bimodal", &bimodal_data),
    ];

    println!("  {:>14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Distribution", "Mean", "Median", "Std", "Min", "Max", "Skewness");
    println!("  {:->14} {:->8} {:->8} {:->8} {:->8} {:->8} {:->10}",
        "", "", "", "", "", "", "");

    for (name, data) in &datasets {
        let n_f = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n_f;

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance: f64 = data.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>() / n_f;
        let std = variance.sqrt();

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        // Skewness: E[(X-μ)³] / σ³
        let skewness: f64 = data.iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>() / n_f;

        println!("  {:>14} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>+10.3}",
            name, mean, median, std, min, max, skewness);
    }

    println!();

    // === Percentiles ===
    println!("=== Percentiles (Normal Distribution) ===\n");

    let mut sorted_normal = normal_data.clone();
    sorted_normal.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentiles = [5, 10, 25, 50, 75, 90, 95];
    for &p in &percentiles {
        let idx = (p as f64 / 100.0 * (sorted_normal.len() - 1) as f64) as usize;
        let val = sorted_normal[idx];
        let bar_pos = ((val - 10.0) / 80.0 * 50.0) as usize;
        let bar_pos = bar_pos.min(50);
        let marker = format!("{}|", " ".repeat(bar_pos));
        println!("    P{:>2} = {:>6.1}  {}", p, val, marker);
    }

    println!();
    println!("Key insight: The shape of a distribution (symmetric, skewed, bimodal)");
    println!("determines which statistics are meaningful and which models are appropriate.");
}
```

---

## Key Takeaways

- A distribution describes how values are spread out — visualizing it reveals patterns that summary statistics alone can miss.
- Histograms are the fundamental tool for visualizing distributions: they count how many values fall into each range (bin).
- Different distribution shapes (uniform, normal, skewed, bimodal) call for different modeling approaches and different summary statistics.
- Skewness indicates asymmetry — when data is skewed, the median is more representative than the mean.
