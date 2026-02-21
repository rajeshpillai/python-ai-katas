# Exponential Smoothing

> Phase 8 — Time Series & Sequential Data | Kata 8.4

---

## Concept & Intuition

### What problem are we solving?

Exponential smoothing is a family of forecasting methods that produce predictions as weighted averages of past observations, where the weights decrease exponentially as observations get older. The most recent data gets the most weight, the next most recent gets less, and so on -- giving the model a "fading memory" that naturally emphasizes recent patterns.

**Simple Exponential Smoothing (SES)** handles series with no trend or seasonality. It has one parameter alpha (smoothing factor) that controls how quickly the model "forgets" the past. **Double Exponential Smoothing (Holt's method)** adds a trend component with its own smoothing parameter beta. **Triple Exponential Smoothing (Holt-Winters)** adds a seasonal component with parameter gamma. Each extension adds one component and one parameter.

The beauty of exponential smoothing is its simplicity and interpretability. The forecast is an explicit, closed-form weighted average of all past observations. No matrix inversions, no iterative optimization -- just a simple recursive update rule. Despite this simplicity, exponential smoothing often competes with much more complex methods.

### Why naive approaches fail

Simple moving averages give equal weight to all observations in the window and zero weight to observations outside it. This is unnatural -- yesterday should matter more than last month. The hard cutoff also causes forecast jumps when old observations leave the window. Exponential smoothing solves both problems with smooth, gradually decaying weights that have no hard boundary.

### Mental models

- **Fading memory**: like how your memory of events fades over time -- yesterday is vivid, last month is hazy, last year is dim.
- **Alpha as attention span**: alpha near 1.0 means "focus almost entirely on the latest observation" (short memory). Alpha near 0.0 means "give significant weight to old observations" (long memory).
- **Recursive update**: each forecast is a blend of the new observation and the previous forecast: F[t+1] = alpha * y[t] + (1-alpha) * F[t].

### Visual explanations

```
Exponential weights (alpha = 0.3):
  y[t]   weight = 0.300  ||||||||||||||||
  y[t-1] weight = 0.210  |||||||||||
  y[t-2] weight = 0.147  ||||||||
  y[t-3] weight = 0.103  |||||
  y[t-4] weight = 0.072  ||||
  ...weights decay as (1-alpha)^k

Simple Exponential Smoothing:
  F[t+1] = alpha * y[t] + (1-alpha) * F[t]
         = alpha * y[t] + alpha*(1-alpha)*y[t-1] + alpha*(1-alpha)^2*y[t-2] + ...

Holt's (Double) -- adds trend:
  Level:  L[t] = alpha * y[t] + (1-alpha) * (L[t-1] + T[t-1])
  Trend:  T[t] = beta * (L[t] - L[t-1]) + (1-beta) * T[t-1]
  Forecast h steps ahead: F[t+h] = L[t] + h * T[t]

Holt-Winters (Triple) -- adds seasonality:
  Level:  L[t] = alpha * (y[t] - S[t-s]) + (1-alpha) * (L[t-1] + T[t-1])
  Trend:  T[t] = beta * (L[t] - L[t-1]) + (1-beta) * T[t-1]
  Season: S[t] = gamma * (y[t] - L[t]) + (1-gamma) * S[t-s]
  Forecast: F[t+h] = L[t] + h*T[t] + S[t+h-s]
```

---

## Hands-on Exploration

1. Implement Simple Exponential Smoothing. Vary alpha from 0.1 to 0.9 and observe how the smoothed line becomes more or less responsive.
2. Generate a series with an upward trend. Show that SES fails (lags behind) but Holt's method tracks the trend.
3. Add seasonality and implement Holt-Winters. Show that the seasonal component captures repeating patterns.
4. Implement a simple grid search over alpha, beta, gamma to find the best parameters by minimizing MSE on a validation set.

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

    // --- Generate time series with trend + seasonality + noise ---
    let n = 120; // 10 "years" of monthly data
    let season_period = 12;
    let mut series = Vec::new();

    for t in 0..n {
        let trend = 100.0 + 0.5 * t as f64;
        let seasonal = 15.0 * (2.0 * pi * t as f64 / season_period as f64).sin()
                     + 8.0 * (4.0 * pi * t as f64 / season_period as f64).cos();
        let noise = (rand_f64(&mut rng) - 0.5) * 10.0;
        series.push(trend + seasonal + noise);
    }

    let test_start = 96; // last 24 months for testing

    println!("=== Exponential Smoothing ===");
    println!("Series: {} observations, test from t={}\n", n, test_start);

    // === Simple Exponential Smoothing (SES) ===
    fn ses(series: &[f64], alpha: f64) -> Vec<f64> {
        let mut forecast = vec![series[0]]; // initialize with first observation
        for t in 1..series.len() {
            let f = alpha * series[t - 1] + (1.0 - alpha) * forecast[t - 1];
            forecast.push(f);
        }
        forecast
    }

    println!("=== Simple Exponential Smoothing ===\n");
    let alphas = [0.1, 0.3, 0.5, 0.8, 0.95];
    println!("{:>8} {:>12} {:>12}", "Alpha", "Train MSE", "Test MSE");
    println!("{}", "-".repeat(34));

    for &alpha in &alphas {
        let forecasts = ses(&series, alpha);
        let train_mse: f64 = (1..test_start).map(|t| {
            (series[t] - forecasts[t]).powi(2)
        }).sum::<f64>() / (test_start - 1) as f64;
        let test_mse: f64 = (test_start..n).map(|t| {
            (series[t] - forecasts[t]).powi(2)
        }).sum::<f64>() / (n - test_start) as f64;
        println!("{:>8.2} {:>12.2} {:>12.2}", alpha, train_mse, test_mse);
    }

    // === Holt's Double Exponential Smoothing ===
    fn holt(series: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
        let mut level = series[0];
        let mut trend = series[1] - series[0];
        let mut forecasts = vec![level];

        for t in 1..series.len() {
            let new_level = alpha * series[t] + (1.0 - alpha) * (level + trend);
            let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
            level = new_level;
            trend = new_trend;
            forecasts.push(level + trend); // one-step-ahead forecast
        }
        forecasts
    }

    println!("\n=== Holt's Double Exponential Smoothing ===\n");
    let params = [(0.3, 0.1), (0.5, 0.1), (0.3, 0.3), (0.5, 0.3)];
    println!("{:>8} {:>8} {:>12} {:>12}", "Alpha", "Beta", "Train MSE", "Test MSE");
    println!("{}", "-".repeat(44));

    for &(alpha, beta) in &params {
        let forecasts = holt(&series, alpha, beta);
        let train_mse: f64 = (2..test_start).map(|t| {
            (series[t] - forecasts[t]).powi(2)
        }).sum::<f64>() / (test_start - 2) as f64;
        let test_mse: f64 = (test_start..n).map(|t| {
            (series[t] - forecasts[t]).powi(2)
        }).sum::<f64>() / (n - test_start) as f64;
        println!("{:>8.2} {:>8.2} {:>12.2} {:>12.2}", alpha, beta, train_mse, test_mse);
    }

    // === Holt-Winters Triple Exponential Smoothing ===
    fn holt_winters(
        series: &[f64], alpha: f64, beta: f64, gamma: f64, period: usize,
    ) -> Vec<f64> {
        let n = series.len();
        if n < 2 * period { return vec![series[0]; n]; }

        // Initialize level and trend from first two seasons
        let mut level: f64 = series[..period].iter().sum::<f64>() / period as f64;
        let level2: f64 = series[period..2 * period].iter().sum::<f64>() / period as f64;
        let mut trend = (level2 - level) / period as f64;

        // Initialize seasonal component
        let mut seasonal = vec![0.0; n + period];
        for i in 0..period {
            seasonal[i] = series[i] - level;
        }

        let mut forecasts = vec![0.0; n];
        for t in 0..period {
            forecasts[t] = level + trend + seasonal[t];
        }

        for t in period..n {
            let new_level = alpha * (series[t] - seasonal[t - period])
                          + (1.0 - alpha) * (level + trend);
            let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
            seasonal[t] = gamma * (series[t] - new_level) + (1.0 - gamma) * seasonal[t - period];

            level = new_level;
            trend = new_trend;
            forecasts[t] = level + trend + seasonal[t]; // one-step-ahead
        }
        forecasts
    }

    println!("\n=== Holt-Winters Triple Exponential Smoothing ===\n");

    // Grid search for best parameters
    let mut best_mse = f64::MAX;
    let mut best_params = (0.0, 0.0, 0.0);
    let val_start = 72;

    for alpha_i in 0..5 {
        let alpha = 0.1 + alpha_i as f64 * 0.2;
        for beta_i in 0..3 {
            let beta = 0.05 + beta_i as f64 * 0.1;
            for gamma_i in 0..3 {
                let gamma = 0.05 + gamma_i as f64 * 0.1;
                let forecasts = holt_winters(&series[..test_start], alpha, beta, gamma, season_period);
                let val_mse: f64 = (val_start..test_start).map(|t| {
                    (series[t] - forecasts[t]).powi(2)
                }).sum::<f64>() / (test_start - val_start) as f64;
                if val_mse < best_mse {
                    best_mse = val_mse;
                    best_params = (alpha, beta, gamma);
                }
            }
        }
    }

    let (ba, bb, bg) = best_params;
    println!("Best params (grid search): alpha={:.2}, beta={:.2}, gamma={:.2}", ba, bb, bg);
    println!("Validation MSE: {:.2}\n", best_mse);

    // Final evaluation with best params
    let hw_forecasts = holt_winters(&series, ba, bb, bg, season_period);
    let ses_forecasts = ses(&series, 0.3);
    let holt_forecasts = holt(&series, 0.3, 0.1);

    let ses_test_mse: f64 = (test_start..n).map(|t| {
        (series[t] - ses_forecasts[t]).powi(2)
    }).sum::<f64>() / (n - test_start) as f64;
    let holt_test_mse: f64 = (test_start..n).map(|t| {
        (series[t] - holt_forecasts[t]).powi(2)
    }).sum::<f64>() / (n - test_start) as f64;
    let hw_test_mse: f64 = (test_start..n).map(|t| {
        (series[t] - hw_forecasts[t]).powi(2)
    }).sum::<f64>() / (n - test_start) as f64;

    println!("=== Final Comparison (Test Set) ===\n");
    println!("{:<30} {:>12}", "Method", "Test MSE");
    println!("{}", "-".repeat(44));
    println!("{:<30} {:>12.2}", "SES (alpha=0.3)", ses_test_mse);
    println!("{:<30} {:>12.2}", "Holt (alpha=0.3, beta=0.1)", holt_test_mse);
    println!("{:<30} {:>12.2}",
        format!("Holt-Winters (a={:.2},b={:.2},g={:.2})", ba, bb, bg), hw_test_mse);

    // --- Show forecast vs actual for last few points ---
    println!("\n=== Sample Forecasts (last 10 test points) ===");
    println!("{:>4} {:>10} {:>10} {:>10} {:>10}", "t", "Actual", "SES", "Holt", "H-W");
    for t in (n - 10)..n {
        println!("{:>4} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            t, series[t], ses_forecasts[t], holt_forecasts[t], hw_forecasts[t]);
    }

    println!();
    println!("kata_metric(\"ses_mse\", {:.2})", ses_test_mse);
    println!("kata_metric(\"holt_mse\", {:.2})", holt_test_mse);
    println!("kata_metric(\"holt_winters_mse\", {:.2})", hw_test_mse);
}
```

---

## Key Takeaways

- **Exponential smoothing gives exponentially decaying weights to past observations.** More recent data matters more, with the rate of decay controlled by alpha.
- **SES handles level only; Holt adds trend; Holt-Winters adds seasonality.** Each extension adds one component and one smoothing parameter.
- **Alpha controls the trade-off between responsiveness and smoothness.** High alpha reacts quickly to changes but is noisy; low alpha is smooth but slow to adapt.
- **Holt-Winters is remarkably competitive** despite its simplicity. For many seasonal time series, it matches or beats more complex methods.
