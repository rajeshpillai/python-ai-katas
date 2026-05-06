# Forecasting Evaluation

> Phase 8 — Time Series & Sequential Data | Kata 8.5

---

## Concept & Intuition

### What problem are we solving?

Evaluating time series forecasts requires different techniques than evaluating standard ML models. You cannot randomly split the data because the temporal order matters. You cannot use a single train/test split because performance might depend on which time period you chose. And standard metrics like MSE do not tell you whether your model beats trivial baselines.

Time series evaluation requires: (1) **proper splitting** that respects temporal order, (2) **specialized metrics** like MAPE, MASE, and relative error, (3) **rolling evaluation** (walk-forward validation) that tests the model across many different time periods, and (4) **baseline comparisons** to ensure your model actually adds value over simple heuristics like "predict yesterday's value."

Walk-forward validation is the time series equivalent of cross-validation. Instead of random folds, you expand the training window forward through time: train on months 1-12 and test on month 13, then train on months 1-13 and test on month 14, and so on. This gives you multiple test points while always maintaining temporal order.

### Why naive approaches fail

Random train/test splits create lookahead bias -- the model has access to future data during training. A single fixed split is fragile -- the model might perform well on one period but poorly on another. Using only MSE makes it impossible to compare across series with different scales. Without baseline comparison, you might celebrate a model that actually performs worse than the trivial "repeat last value" predictor.

### Mental models

- **Walk-forward as expanding window**: the training set grows with each step, always predicting the next unseen point. This mimics real deployment.
- **MASE as the fairness judge**: Mean Absolute Scaled Error compares your model to the naive baseline, with MASE < 1 meaning your model is better than naive.
- **Multiple horizons**: a 1-step forecast is easier than a 10-step forecast. Evaluating at multiple horizons reveals how quickly forecast quality degrades.

### Visual explanations

```
Walk-Forward Validation:
  Time:  |--1--2--3--4--5--6--7--8--9--10--|

  Step 1: [TRAIN: 1-5] --> test: 6
  Step 2: [TRAIN: 1-6] --> test: 7
  Step 3: [TRAIN: 1-7] --> test: 8
  Step 4: [TRAIN: 1-8] --> test: 9
  Step 5: [TRAIN: 1-9] --> test: 10

  Average all test errors for final score.

Metrics:
  MAE  = mean(|actual - forecast|)         Scale-dependent
  MAPE = mean(|actual - forecast| / |actual|)  Percentage, but fails near 0
  MASE = MAE / MAE_naive                   Scale-free, robust
  RMSE = sqrt(mean((actual - forecast)^2)) Penalizes large errors

  MASE < 1.0 --> better than naive baseline
  MASE = 1.0 --> same as naive
  MASE > 1.0 --> worse than naive (bad!)
```

---

## Hands-on Exploration

1. Implement walk-forward validation. At each step, expand the training window and predict the next point. Collect all test errors.
2. Compute MAE, RMSE, MAPE, and MASE for a simple model. Show how MASE relates to the naive baseline.
3. Compare two forecasting methods using walk-forward validation. Show that the method with lower MASE is genuinely better.
4. Evaluate forecasts at multiple horizons (1-step, 3-step, 7-step) and show how accuracy degrades with longer horizons.

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

    // --- Generate time series ---
    let n = 150;
    let period = 12;
    let mut series = Vec::new();
    for t in 0..n {
        let trend = 50.0 + 0.3 * t as f64;
        let seasonal = 12.0 * (2.0 * pi * t as f64 / period as f64).sin();
        let noise = (rand_f64(&mut rng) - 0.5) * 8.0;
        series.push(trend + seasonal + noise);
    }

    println!("=== Forecasting Evaluation ===");
    println!("Series length: {}, evaluation starts at t={}\n", n, 100);

    // --- Forecasting methods ---
    // 1. Naive: predict last value
    // 2. Seasonal naive: predict value from one season ago
    // 3. Simple moving average (SMA)
    // 4. Exponential smoothing

    fn ses_forecast(series: &[f64], alpha: f64) -> f64 {
        let mut level = series[0];
        for &y in &series[1..] {
            level = alpha * y + (1.0 - alpha) * level;
        }
        level
    }

    fn sma_forecast(series: &[f64], window: usize) -> f64 {
        let start = if series.len() > window { series.len() - window } else { 0 };
        let vals = &series[start..];
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    // --- Evaluation metrics ---
    fn mae(errors: &[f64]) -> f64 {
        errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64
    }

    fn rmse(errors: &[f64]) -> f64 {
        (errors.iter().map(|e| e.powi(2)).sum::<f64>() / errors.len() as f64).sqrt()
    }

    fn mape(actuals: &[f64], forecasts: &[f64]) -> f64 {
        actuals.iter().zip(forecasts).map(|(a, f)| {
            if a.abs() > 1e-10 { (a - f).abs() / a.abs() } else { 0.0 }
        }).sum::<f64>() / actuals.len() as f64 * 100.0
    }

    fn mase(errors: &[f64], series: &[f64]) -> f64 {
        // MASE = MAE / MAE_naive_1step
        let mae_model = mae(errors);
        let naive_errors: Vec<f64> = (1..series.len()).map(|t| {
            series[t] - series[t - 1]
        }).collect();
        let mae_naive = mae(&naive_errors);
        if mae_naive < 1e-10 { return f64::MAX; }
        mae_model / mae_naive
    }

    // --- Walk-Forward Validation ---
    println!("=== Walk-Forward Validation ===\n");

    let eval_start = 100;
    let min_train = 50;

    struct ForecastResult {
        name: String,
        errors: Vec<f64>,
        actuals: Vec<f64>,
        forecasts: Vec<f64>,
    }

    let mut results: Vec<ForecastResult> = Vec::new();

    // Method names and forecasters
    let method_names = ["Naive", "Seasonal Naive", "SMA(12)", "SES(0.3)"];

    for (m, name) in method_names.iter().enumerate() {
        let mut errors = Vec::new();
        let mut actuals = Vec::new();
        let mut forecasts = Vec::new();

        for t in eval_start..n {
            let train = &series[..t];
            let actual = series[t];

            let forecast = match m {
                0 => *train.last().unwrap(),                          // Naive
                1 => if t >= period { series[t - period] } else { *train.last().unwrap() }, // Seasonal naive
                2 => sma_forecast(train, 12),                         // SMA
                3 => ses_forecast(train, 0.3),                        // SES
                _ => 0.0,
            };

            errors.push(actual - forecast);
            actuals.push(actual);
            forecasts.push(forecast);
        }

        results.push(ForecastResult {
            name: name.to_string(), errors, actuals, forecasts,
        });
    }

    // --- Display metrics ---
    println!("{:<20} {:>8} {:>8} {:>8} {:>8}", "Method", "MAE", "RMSE", "MAPE%", "MASE");
    println!("{}", "-".repeat(56));

    for result in &results {
        let m = mae(&result.errors);
        let r = rmse(&result.errors);
        let mp = mape(&result.actuals, &result.forecasts);
        let ms = mase(&result.errors, &series[..eval_start]);

        println!("{:<20} {:>8.2} {:>8.2} {:>8.1} {:>8.3}", result.name, m, r, mp, ms);
    }

    println!("\nMASE interpretation:");
    println!("  MASE < 1.0: better than naive baseline");
    println!("  MASE = 1.0: same as naive");
    println!("  MASE > 1.0: worse than naive\n");

    // --- Multi-horizon evaluation ---
    println!("=== Multi-Horizon Evaluation ===\n");
    let horizons = [1, 3, 6, 12];

    println!("{:<20} {:>6} {:>8}", "Method", "h", "MAE");
    println!("{}", "-".repeat(36));

    for &h in &horizons {
        for (m, name) in method_names.iter().enumerate() {
            let mut errors = Vec::new();

            for t in eval_start..(n - h) {
                let train = &series[..t];
                let actual = series[t + h - 1]; // h-step ahead actual

                let forecast = match m {
                    0 => *train.last().unwrap(),
                    1 => if t + h - 1 >= period { series[t + h - 1 - period] } else { *train.last().unwrap() },
                    2 => sma_forecast(train, 12),
                    3 => ses_forecast(train, 0.3),
                    _ => 0.0,
                };

                errors.push(actual - forecast);
            }

            if !errors.is_empty() {
                println!("{:<20} {:>6} {:>8.2}", name, h, mae(&errors));
            }
        }
        println!();
    }

    // --- Rolling window vs expanding window ---
    println!("=== Rolling Window vs Expanding Window ===\n");

    let window_size = 60;
    let mut rolling_errors = Vec::new();
    let mut expanding_errors = Vec::new();

    for t in eval_start..n {
        let actual = series[t];

        // Rolling: fixed window
        let rolling_start = if t > window_size { t - window_size } else { 0 };
        let rolling_pred = ses_forecast(&series[rolling_start..t], 0.3);
        rolling_errors.push(actual - rolling_pred);

        // Expanding: all history
        let expanding_pred = ses_forecast(&series[..t], 0.3);
        expanding_errors.push(actual - expanding_pred);
    }

    println!("{:<25} {:>8} {:>8}", "Validation Strategy", "MAE", "RMSE");
    println!("{}", "-".repeat(43));
    println!("{:<25} {:>8.2} {:>8.2}", "Rolling window (60)",
        mae(&rolling_errors), rmse(&rolling_errors));
    println!("{:<25} {:>8.2} {:>8.2}", "Expanding window",
        mae(&expanding_errors), rmse(&expanding_errors));

    // --- Error distribution analysis ---
    println!("\n=== Error Distribution (SES) ===");
    let ses_errors = &results[3].errors;
    let err_mean: f64 = ses_errors.iter().sum::<f64>() / ses_errors.len() as f64;
    let err_std: f64 = (ses_errors.iter().map(|e| (e - err_mean).powi(2)).sum::<f64>()
        / ses_errors.len() as f64).sqrt();
    let pos_errors = ses_errors.iter().filter(|&&e| e > 0.0).count();
    let neg_errors = ses_errors.len() - pos_errors;

    println!("  Mean error:     {:+.3} (should be ~0 for unbiased)", err_mean);
    println!("  Std error:      {:.3}", err_std);
    println!("  Positive/Negative: {}/{} ({:.0}%/{:.0}%)",
        pos_errors, neg_errors,
        pos_errors as f64 / ses_errors.len() as f64 * 100.0,
        neg_errors as f64 / ses_errors.len() as f64 * 100.0);

    let best_mase = results.iter().map(|r| {
        mase(&r.errors, &series[..eval_start])
    }).fold(f64::MAX, f64::min);

    println!();
    println!("kata_metric(\"best_mase\", {:.3})", best_mase);
    println!("kata_metric(\"n_eval_points\", {})", n - eval_start);
}
```

---

## Key Takeaways

- **Walk-forward validation respects temporal order.** Never randomly shuffle time series data; always train on the past and test on the future.
- **MASE is the gold standard time series metric.** It is scale-free, handles zeros, and directly compares your model to the naive baseline.
- **Multiple horizons reveal forecast decay.** A model excellent at 1-step forecasting may be terrible at 7-step forecasting.
- **Always compare to baselines.** The naive forecast (predict last value) and seasonal naive are surprisingly hard to beat on many real-world series.
