# Time Series Basics

> Phase 8 — Time Series & Sequential Data | Kata 8.1

---

## Concept & Intuition

### What problem are we solving?

A time series is a sequence of data points ordered by time: stock prices, temperature readings, website traffic, heart rate measurements. Unlike tabular data where rows are independent, time series data has **temporal structure** -- what happened yesterday affects what happens today. This autocorrelation is both the challenge and the opportunity: we can exploit past patterns to predict the future.

The fundamental components of a time series are: **trend** (long-term direction -- rising, falling, or flat), **seasonality** (repeating patterns at fixed intervals -- daily, weekly, yearly), and **residuals** (random noise left after removing trend and seasonality). Decomposing a time series into these components is the first step in understanding and forecasting it.

Stationarity is a critical concept: a stationary time series has constant mean, constant variance, and constant autocorrelation structure over time. Most forecasting methods assume stationarity. Non-stationary series (with trends or changing variance) must be transformed (differencing, log transforms) before modeling.

### Why naive approaches fail

Treating time series as regular tabular data ignores temporal dependencies. Shuffling the data (standard in cross-validation) destroys the time ordering. Using future data to predict the past (lookahead bias) inflates accuracy. Time series requires specialized techniques: rolling windows instead of random splits, lagged features instead of independent features, and stationarity checks before modeling.

### Mental models

- **Decomposition as X-ray**: just as an X-ray reveals the skeleton beneath the skin, decomposition reveals the structural components beneath the noisy surface.
- **Stationarity as a flat playing field**: forecasting a stationary series is like predicting scores in a balanced game. Non-stationarity is like predicting a game where the rules keep changing.
- **Lag features as memory**: including yesterday's value as a feature gives the model "memory" of the recent past.

### Visual explanations

```
Time Series Decomposition:

  Original: ~~~~~~~~~~~~~/\/\/\/\~~~~~~~~~~~~~  (noisy, trending up)

  Trend:    ________________________________/   (smooth upward)
  Season:   /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/   (repeating pattern)
  Residual: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  (random noise)

  Original = Trend + Seasonality + Residual

Stationarity check:
  Stationary:      mean ------- constant -------
  Non-stationary:  mean _____ / _____ / _____ /   (drifting up)

  Differencing:  diff[t] = y[t] - y[t-1]
  Removes trend, makes series more stationary
```

---

## Hands-on Exploration

1. Generate a time series with a known trend, seasonal pattern, and noise. Plot the raw series and its components.
2. Implement a moving average to estimate the trend. Subtract it to reveal the seasonal pattern.
3. Compute first-order differences and observe how differencing removes the trend.
4. Implement a simple lag-based predictor: predict y[t] using y[t-1], y[t-2], etc. Compare to predicting the mean.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Generate time series: trend + seasonality + noise ---
    let n = 200;
    let season_period = 12; // monthly seasonality

    let mut trend = Vec::new();
    let mut season = Vec::new();
    let mut noise = Vec::new();
    let mut series = Vec::new();

    for t in 0..n {
        let tr = 50.0 + 0.3 * t as f64; // linear trend
        let se = 10.0 * (2.0 * pi * t as f64 / season_period as f64).sin()
               + 5.0 * (4.0 * pi * t as f64 / season_period as f64).cos();
        let no = (rand_f64(&mut rng) - 0.5) * 8.0;

        trend.push(tr);
        season.push(se);
        noise.push(no);
        series.push(tr + se + no);
    }

    println!("=== Time Series Basics ===");
    println!("Generated {} data points with trend, seasonality (period={}), and noise\n",
        n, season_period);

    // --- Display sample values ---
    println!("Sample values (first 12 = one full season):");
    println!("{:>4} {:>10} {:>10} {:>10} {:>10}", "t", "Series", "Trend", "Season", "Noise");
    println!("{}", "-".repeat(48));
    for t in 0..12 {
        println!("{:>4} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            t, series[t], trend[t], season[t], noise[t]);
    }

    // --- Moving average to estimate trend ---
    println!("\n=== Trend Estimation (Moving Average) ===");
    let window = season_period; // window = season period removes seasonality
    let mut moving_avg = vec![0.0; n];
    for t in 0..n {
        let start = if t >= window / 2 { t - window / 2 } else { 0 };
        let end = (t + window / 2 + 1).min(n);
        let sum: f64 = series[start..end].iter().sum();
        moving_avg[t] = sum / (end - start) as f64;
    }

    println!("Moving average (window={}):", window);
    println!("{:>4} {:>10} {:>10} {:>10}", "t", "Series", "MA_Trend", "True_Trend");
    for t in (0..n).step_by(20) {
        println!("{:>4} {:>10.2} {:>10.2} {:>10.2}", t, series[t], moving_avg[t], trend[t]);
    }

    // Trend estimation error
    let trend_mae: f64 = (window / 2..n - window / 2).map(|t| {
        (moving_avg[t] - trend[t]).abs()
    }).sum::<f64>() / (n - window) as f64;
    println!("Trend estimation MAE: {:.3}", trend_mae);

    // --- Detrended series = series - moving_avg ---
    let detrended: Vec<f64> = series.iter().zip(&moving_avg)
        .map(|(s, m)| s - m).collect();

    // --- Differencing ---
    println!("\n=== Differencing (removes trend) ===");
    let diff1: Vec<f64> = (1..n).map(|t| series[t] - series[t - 1]).collect();

    // Check stationarity via simple mean comparison (first half vs second half)
    let mean_first_half: f64 = series[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
    let mean_second_half: f64 = series[n / 2..].iter().sum::<f64>() / (n / 2) as f64;
    let diff_mean_first: f64 = diff1[..diff1.len() / 2].iter().sum::<f64>() / (diff1.len() / 2) as f64;
    let diff_mean_second: f64 = diff1[diff1.len() / 2..].iter().sum::<f64>() / (diff1.len() / 2) as f64;

    println!("Original series:");
    println!("  First half mean:  {:.2}", mean_first_half);
    println!("  Second half mean: {:.2}", mean_second_half);
    println!("  Difference:       {:.2} (non-stationary!)", (mean_second_half - mean_first_half).abs());

    println!("Differenced series:");
    println!("  First half mean:  {:.2}", diff_mean_first);
    println!("  Second half mean: {:.2}", diff_mean_second);
    println!("  Difference:       {:.2} (more stationary!)", (diff_mean_second - diff_mean_first).abs());

    // --- Lag-based prediction ---
    println!("\n=== Lag-Based Prediction ===");

    let test_start = 150;

    // Method 1: Predict the mean
    let train_mean: f64 = series[..test_start].iter().sum::<f64>() / test_start as f64;
    let mse_mean: f64 = series[test_start..].iter()
        .map(|&y| (y - train_mean).powi(2)).sum::<f64>() / (n - test_start) as f64;

    // Method 2: Naive (predict last value)
    let mse_naive: f64 = (test_start..n).map(|t| {
        (series[t] - series[t - 1]).powi(2)
    }).sum::<f64>() / (n - test_start) as f64;

    // Method 3: Seasonal naive (predict value from one season ago)
    let mse_seasonal: f64 = (test_start..n).map(|t| {
        if t >= season_period {
            (series[t] - series[t - season_period]).powi(2)
        } else {
            (series[t] - train_mean).powi(2)
        }
    }).sum::<f64>() / (n - test_start) as f64;

    // Method 4: Linear regression on lag features
    let n_lags = 3;
    let mut lag_mse = 0.0;
    let mut lag_count = 0;

    // Fit simple linear model on lag features
    // y[t] = w0 + w1*y[t-1] + w2*y[t-2] + w3*y[t-3]
    let train_start = n_lags;

    // Build lag features for training
    let mut lag_x: Vec<Vec<f64>> = Vec::new();
    let mut lag_y: Vec<f64> = Vec::new();
    for t in train_start..test_start {
        let row: Vec<f64> = (1..=n_lags).map(|l| series[t - l]).collect();
        lag_x.push(row);
        lag_y.push(series[t]);
    }

    // Simple linear regression via gradient descent
    let mut weights = vec![0.0; n_lags];
    let y_mean_lag: f64 = lag_y.iter().sum::<f64>() / lag_y.len() as f64;
    let lr = 0.00001;

    for _ in 0..5000 {
        let mut grad = vec![0.0; n_lags];
        for i in 0..lag_x.len() {
            let pred: f64 = weights.iter().zip(&lag_x[i]).map(|(w, x)| w * x).sum::<f64>() + y_mean_lag;
            let err = pred - lag_y[i];
            for j in 0..n_lags {
                grad[j] += err * lag_x[i][j];
            }
        }
        for j in 0..n_lags {
            weights[j] -= lr * 2.0 * grad[j] / lag_x.len() as f64;
        }
    }

    // Predict test set
    for t in test_start..n {
        let features: Vec<f64> = (1..=n_lags).map(|l| series[t - l]).collect();
        let pred: f64 = weights.iter().zip(&features).map(|(w, x)| w * x).sum::<f64>() + y_mean_lag;
        lag_mse += (series[t] - pred).powi(2);
        lag_count += 1;
    }
    lag_mse /= lag_count as f64;

    println!("{:<25} {:>10}", "Method", "Test MSE");
    println!("{}", "-".repeat(37));
    println!("{:<25} {:>10.2}", "Predict mean", mse_mean);
    println!("{:<25} {:>10.2}", "Naive (last value)", mse_naive);
    println!("{:<25} {:>10.2}", "Seasonal naive", mse_seasonal);
    println!("{:<25} {:>10.2}", "Lag regression (3 lags)", lag_mse);

    println!("\nLag regression weights: y[t] = {:.3} + {:.3}*y[t-1] + {:.3}*y[t-2] + {:.3}*y[t-3]",
        y_mean_lag, weights[0], weights[1], weights[2]);

    // --- Basic statistics ---
    println!("\n=== Series Statistics ===");
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let var: f64 = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let min = series.iter().cloned().fold(f64::MAX, f64::min);
    let max = series.iter().cloned().fold(f64::MIN, f64::max);

    println!("Mean: {:.2}", mean);
    println!("Std:  {:.2}", var.sqrt());
    println!("Min:  {:.2}", min);
    println!("Max:  {:.2}", max);

    println!();
    println!("kata_metric(\"mse_mean\", {:.2})", mse_mean);
    println!("kata_metric(\"mse_naive\", {:.2})", mse_naive);
    println!("kata_metric(\"mse_seasonal\", {:.2})", mse_seasonal);
    println!("kata_metric(\"mse_lag_regression\", {:.2})", lag_mse);
}
```

---

## Key Takeaways

- **Time series have temporal structure that standard ML ignores.** Observations are ordered and dependent; shuffling them destroys the signal.
- **Decomposition reveals trend, seasonality, and noise.** Understanding these components is the foundation for choosing the right model.
- **Stationarity is a prerequisite for most forecasting methods.** Differencing is the simplest way to remove trends and make a series stationary.
- **Lag features give models "memory" of the past.** Even a simple linear model on lag features can outperform naive baselines significantly.
