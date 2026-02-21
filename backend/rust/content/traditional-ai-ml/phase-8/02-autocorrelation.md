# Autocorrelation

> Phase 8 — Time Series & Sequential Data | Kata 8.2

---

## Concept & Intuition

### What problem are we solving?

Autocorrelation measures how much a time series is correlated with lagged versions of itself. If today's stock price is similar to yesterday's, the series has high autocorrelation at lag 1. If this week's sales resemble last week's, autocorrelation at lag 7 is high. Understanding autocorrelation tells you **which past values are most informative for predicting the future** and reveals the memory structure of the data.

The **autocorrelation function (ACF)** computes the correlation between y[t] and y[t-k] for each lag k. The **partial autocorrelation function (PACF)** measures the correlation at lag k after removing the effects of all intermediate lags. This distinction is critical: high ACF at lag 2 might just be because lag 2 is correlated with lag 1 (which is correlated with the present). PACF at lag 2 tells you whether lag 2 has *direct* predictive value beyond what lag 1 already provides.

ACF and PACF plots are the primary diagnostic tools for selecting time series models. An AR(p) model has PACF that cuts off after lag p. An MA(q) model has ACF that cuts off after lag q. These patterns guide model selection before any fitting occurs.

### Why naive approaches fail

Including too many lags wastes parameters on uninformative features. Including too few lags misses important structure. Without autocorrelation analysis, you are guessing how much "memory" the model needs. ACF and PACF give you a principled answer: use lags where autocorrelation is statistically significant, and stop where it drops to near zero.

### Mental models

- **ACF as an echo**: autocorrelation measures how loudly the past echoes into the present. Strong echoes (high autocorrelation) mean the past is predictive.
- **PACF as direct influence**: ACF includes indirect effects (lag 3 through lag 2 through lag 1). PACF isolates the direct link between lag k and the present.
- **Significance bands**: values within the 95% confidence band (~1.96/sqrt(n)) are probably just noise. Only act on correlations outside the bands.

### Visual explanations

```
ACF Plot (Autocorrelation):
  Lag 0: ||||||||||||||||||||  1.00  (always 1)
  Lag 1: ||||||||||||||||||    0.90  (high: yesterday predicts today)
  Lag 2: ||||||||||||||||      0.81  (still high: 2 days ago matters)
  Lag 3: ||||||||||||||        0.72
  ...
  Lag 12: ||||||||||||         0.60  (seasonal peak!)
  ...
  Lag 20: ||                   0.10

PACF Plot (Partial Autocorrelation):
  Lag 1: ||||||||||||||||||    0.90  (strong direct effect)
  Lag 2: ||                   0.05  (no direct effect after accounting for lag 1)
  Lag 3: |                    0.02
  --> Suggests AR(1) model: only lag 1 has direct predictive value

ACF/PACF signatures:
  AR(p):  ACF decays gradually,     PACF cuts off after lag p
  MA(q):  ACF cuts off after lag q, PACF decays gradually
  ARMA:   Both decay gradually
```

---

## Hands-on Exploration

1. Generate an AR(2) process and compute the ACF. Verify that autocorrelation decays exponentially.
2. Compute the PACF and verify it cuts off sharply after lag 2, confirming the AR(2) structure.
3. Generate an MA(1) process and compare its ACF (cuts off after lag 1) vs the AR process.
4. Apply ACF/PACF to real-world-like data with seasonality and identify the seasonal lag.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    let mut rand_normal = |s: &mut u64| -> f64 {
        // Box-Muller approximation
        let u1 = rand_f64(s).max(1e-10);
        let u2 = rand_f64(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // --- Generate AR(2) process: y[t] = 0.6*y[t-1] - 0.3*y[t-2] + noise ---
    let n = 500;
    let mut ar2 = vec![0.0; n];
    for t in 2..n {
        ar2[t] = 0.6 * ar2[t - 1] - 0.3 * ar2[t - 2] + rand_normal(&mut rng);
    }

    // --- Generate MA(1) process: y[t] = noise[t] + 0.8*noise[t-1] ---
    let mut ma1 = vec![0.0; n];
    let mut prev_noise = rand_normal(&mut rng);
    for t in 0..n {
        let current_noise = rand_normal(&mut rng);
        ma1[t] = current_noise + 0.8 * prev_noise;
        prev_noise = current_noise;
    }

    // --- Autocorrelation Function (ACF) ---
    fn acf(series: &[f64], max_lag: usize) -> Vec<f64> {
        let n = series.len() as f64;
        let mean: f64 = series.iter().sum::<f64>() / n;
        let var: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        (0..=max_lag).map(|k| {
            if var < 1e-10 { return 0.0; }
            let cov: f64 = (k..series.len()).map(|t| {
                (series[t] - mean) * (series[t - k] - mean)
            }).sum::<f64>() / n;
            cov / var
        }).collect()
    }

    // --- Partial Autocorrelation Function (PACF) via Durbin-Levinson ---
    fn pacf(series: &[f64], max_lag: usize) -> Vec<f64> {
        let acf_vals = acf(series, max_lag);
        let mut pacf_vals = vec![0.0; max_lag + 1];
        pacf_vals[0] = 1.0;

        if max_lag == 0 { return pacf_vals; }
        pacf_vals[1] = acf_vals[1];

        // Durbin-Levinson recursion
        let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
        phi[1][1] = acf_vals[1];

        for k in 2..=max_lag {
            let mut num = acf_vals[k];
            for j in 1..k {
                num -= phi[k - 1][j] * acf_vals[k - j];
            }
            let mut den = 1.0;
            for j in 1..k {
                den -= phi[k - 1][j] * acf_vals[j];
            }
            phi[k][k] = if den.abs() > 1e-10 { num / den } else { 0.0 };
            pacf_vals[k] = phi[k][k];

            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
        }

        pacf_vals
    }

    let max_lag = 15;
    let significance = 1.96 / (n as f64).sqrt(); // 95% confidence band

    // --- AR(2) analysis ---
    let ar2_acf = acf(&ar2, max_lag);
    let ar2_pacf = pacf(&ar2, max_lag);

    println!("=== AR(2) Process: y[t] = 0.6*y[t-1] - 0.3*y[t-2] + noise ===\n");
    println!("95% significance band: +/- {:.3}\n", significance);

    println!("ACF (decays gradually for AR processes):");
    println!("{:>4} {:>8} {:>30} {:>5}", "Lag", "ACF", "Plot", "Sig?");
    for k in 0..=max_lag {
        let bar_len = (ar2_acf[k].abs() * 25.0) as usize;
        let bar: String = if ar2_acf[k] >= 0.0 {
            format!("{:>25}", std::iter::repeat('|').take(bar_len).collect::<String>())
        } else {
            format!("{:<25}", std::iter::repeat('|').take(bar_len).collect::<String>())
        };
        let sig = if k > 0 && ar2_acf[k].abs() > significance { "*" } else { "" };
        println!("{:>4} {:>8.3} {} {:>5}", k, ar2_acf[k], bar, sig);
    }

    println!("\nPACF (cuts off after lag 2 for AR(2)):");
    println!("{:>4} {:>8} {:>30} {:>5}", "Lag", "PACF", "Plot", "Sig?");
    for k in 0..=max_lag.min(8) {
        let bar_len = (ar2_pacf[k].abs() * 25.0) as usize;
        let sign = if ar2_pacf[k] >= 0.0 { "+" } else { "-" };
        let bar: String = std::iter::repeat('|').take(bar_len).collect();
        let sig = if k > 0 && ar2_pacf[k].abs() > significance { "***" } else { "" };
        println!("{:>4} {:>8.3}  {} {} {}", k, ar2_pacf[k], sign, bar, sig);
    }
    println!("  --> PACF significant at lags 1 and 2, then drops. Suggests AR(2).");

    // --- MA(1) analysis ---
    let ma1_acf = acf(&ma1, max_lag);
    let ma1_pacf = pacf(&ma1, max_lag);

    println!("\n=== MA(1) Process: y[t] = noise[t] + 0.8*noise[t-1] ===\n");

    println!("ACF (cuts off after lag 1 for MA(1)):");
    println!("{:>4} {:>8} {:>5}", "Lag", "ACF", "Sig?");
    for k in 0..=max_lag.min(8) {
        let sig = if k > 0 && ma1_acf[k].abs() > significance { "***" } else { "" };
        let bar_len = (ma1_acf[k].abs() * 25.0) as usize;
        let bar: String = std::iter::repeat('|').take(bar_len).collect();
        println!("{:>4} {:>8.3}  {} {}", k, ma1_acf[k], bar, sig);
    }
    println!("  --> ACF significant at lag 1, then drops. Suggests MA(1).");

    println!("\nPACF (decays gradually for MA processes):");
    println!("{:>4} {:>8} {:>5}", "Lag", "PACF", "Sig?");
    for k in 0..=max_lag.min(8) {
        let sig = if k > 0 && ma1_pacf[k].abs() > significance { "***" } else { "" };
        println!("{:>4} {:>8.3}  {}", k, ma1_pacf[k], sig);
    }

    // --- Seasonal series ---
    println!("\n=== Seasonal Series (period=12) ===\n");
    let period = 12;
    let mut seasonal_series = vec![0.0; n];
    for t in 0..n {
        let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
        seasonal_series[t] = seasonal + rand_normal(&mut rng) * 2.0;
    }

    let seasonal_acf = acf(&seasonal_series, 30);
    println!("ACF at key lags (expect peaks at multiples of 12):");
    for k in [1, 6, 11, 12, 13, 24, 25] {
        if k < seasonal_acf.len() {
            let peak = if k % period == 0 { "<-- seasonal peak" } else { "" };
            println!("  Lag {:>2}: ACF = {:>7.3}  {}", k, seasonal_acf[k], peak);
        }
    }

    // --- Model selection guide ---
    println!("\n=== ACF/PACF Model Selection Guide ===");
    println!("{:<15} {:<25} {:<25}", "Model", "ACF Pattern", "PACF Pattern");
    println!("{}", "-".repeat(65));
    println!("{:<15} {:<25} {:<25}", "AR(p)", "Gradual decay", "Cuts off after lag p");
    println!("{:<15} {:<25} {:<25}", "MA(q)", "Cuts off after lag q", "Gradual decay");
    println!("{:<15} {:<25} {:<25}", "ARMA(p,q)", "Gradual decay", "Gradual decay");
    println!("{:<15} {:<25} {:<25}", "Seasonal", "Peaks at multiples of s", "Peak at lag s");

    println!();
    println!("kata_metric(\"ar2_acf_lag1\", {:.3})", ar2_acf[1]);
    println!("kata_metric(\"ar2_pacf_lag2\", {:.3})", ar2_pacf[2]);
    println!("kata_metric(\"ar2_pacf_lag3\", {:.3})", ar2_pacf[3]);
    println!("kata_metric(\"ma1_acf_lag1\", {:.3})", ma1_acf[1]);
    println!("kata_metric(\"ma1_acf_lag2\", {:.3})", ma1_acf[2]);
}
```

---

## Key Takeaways

- **ACF measures total correlation between a series and its lags.** It reveals the overall memory structure and seasonal patterns.
- **PACF isolates the direct effect of each lag** after removing intermediate influences. It is essential for AR model order selection.
- **ACF/PACF patterns are diagnostic signatures.** AR models show gradual ACF decay with sharp PACF cutoff; MA models show the opposite.
- **Significance bands prevent over-reading noise.** Only lags with autocorrelation outside the confidence bands should influence model design.
