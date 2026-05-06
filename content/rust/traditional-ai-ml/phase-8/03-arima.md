# ARIMA

> Phase 8 — Time Series & Sequential Data | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

ARIMA (AutoRegressive Integrated Moving Average) is the workhorse model for univariate time series forecasting. It combines three components: **AR(p)** uses the past p values as predictors, **I(d)** uses differencing d times to achieve stationarity, and **MA(q)** uses the past q forecast errors as predictors. Together, ARIMA(p,d,q) can model a wide range of time series patterns.

The AR component captures momentum: if yesterday was high, today tends to be high. The MA component captures shock effects: if yesterday's forecast was too low (positive error), today's forecast adjusts upward. The I component handles non-stationarity: by differencing, a trending series becomes stationary, and the AR+MA components can model the stationary residuals.

Model selection (choosing p, d, q) is guided by ACF/PACF analysis and information criteria (AIC/BIC). The process is: (1) determine d by testing for stationarity, (2) examine PACF to choose p, (3) examine ACF to choose q, (4) fit the model and check that residuals resemble white noise.

### Why naive approaches fail

Using AR alone cannot model series where past shocks have persistent effects (MA behavior). Using MA alone cannot model series with strong momentum (AR behavior). Using either without differencing fails on trending data. ARIMA integrates all three aspects, providing a flexible framework. However, ARIMA cannot capture nonlinear patterns or complex seasonality -- for those, you need extensions like SARIMA or machine learning approaches.

### Mental models

- **AR as inertia**: the series tends to continue in its current direction, like a ball rolling.
- **MA as correction**: the model adjusts based on recent prediction mistakes, like a thermostat reacting to temperature errors.
- **I (differencing) as detrending**: by looking at changes rather than levels, we remove the trend and focus on the dynamics.
- **AIC/BIC as Occam's razor**: these criteria balance model fit against complexity, penalizing unnecessary parameters.

### Visual explanations

```
ARIMA(p, d, q):

  Step 1: Difference d times
    z[t] = diff^d(y[t])

  Step 2: Fit AR(p) + MA(q) on z[t]:
    z[t] = c + phi_1*z[t-1] + ... + phi_p*z[t-p]
             + theta_1*e[t-1] + ... + theta_q*e[t-q]
             + e[t]

  where e[t] = z[t] - predicted_z[t]  (forecast errors)

Example ARIMA(2,1,1):
  d=1: z[t] = y[t] - y[t-1]           (first difference)
  AR(2): phi_1*z[t-1] + phi_2*z[t-2]  (momentum from 2 lags)
  MA(1): theta_1*e[t-1]               (correction from last error)

Model Selection:
  Stationarity test --> choose d
  PACF cutoff at lag p --> choose p
  ACF cutoff at lag q --> choose q
```

---

## Hands-on Exploration

1. Generate a non-stationary time series with trend. Apply differencing and verify stationarity improves.
2. Implement an AR(p) model by fitting linear regression on lag features. Evaluate on held-out test data.
3. Add MA terms by tracking and using past forecast errors as additional features.
4. Combine AR and MA into a simple ARIMA implementation. Compare ARIMA(1,1,0), ARIMA(0,1,1), and ARIMA(1,1,1).

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // --- Generate non-stationary ARIMA(1,1,1) process ---
    let n = 300;
    let true_phi = 0.6;   // AR coefficient
    let true_theta = 0.4; // MA coefficient

    // Generate: diff(y)[t] = phi*diff(y)[t-1] + theta*e[t-1] + e[t]
    let mut y = vec![100.0]; // starting value
    let mut errors = vec![0.0];
    let mut diffs = vec![0.0];

    for t in 1..n {
        let e = rand_normal(&mut rng) * 2.0;
        let d = true_phi * diffs[t - 1] + true_theta * errors[t - 1] + e;
        diffs.push(d);
        errors.push(e);
        y.push(y[t - 1] + d);
    }

    println!("=== ARIMA Implementation ===");
    println!("Generated ARIMA(1,1,1) process: phi={}, theta={}, n={}\n", true_phi, true_theta, n);

    // --- Differencing ---
    let diff1: Vec<f64> = (1..n).map(|t| y[t] - y[t - 1]).collect();

    let mean_orig: f64 = y.iter().sum::<f64>() / n as f64;
    let std_orig: f64 = (y.iter().map(|v| (v - mean_orig).powi(2)).sum::<f64>() / n as f64).sqrt();
    let mean_diff: f64 = diff1.iter().sum::<f64>() / diff1.len() as f64;
    let std_diff: f64 = (diff1.iter().map(|v| (v - mean_diff).powi(2)).sum::<f64>() / diff1.len() as f64).sqrt();

    println!("Stationarity check:");
    println!("  Original: mean={:.2}, std={:.2}", mean_orig, std_orig);
    println!("  Differenced (d=1): mean={:.2}, std={:.2}", mean_diff, std_diff);
    println!("  --> Differencing reduces the non-stationarity\n");

    let test_start = 200;

    // --- AR(p) model ---
    fn fit_ar(series: &[f64], p: usize, train_end: usize) -> Vec<f64> {
        // Returns coefficients [phi_1, ..., phi_p, intercept]
        // Using normal equations via gradient descent
        let n = train_end - p;
        let y_mean: f64 = series[p..train_end].iter().sum::<f64>() / n as f64;
        let mut w = vec![0.0; p];
        let lr = 0.001;

        for _ in 0..3000 {
            let mut grad = vec![0.0; p];
            for t in p..train_end {
                let mut pred = y_mean;
                for j in 0..p {
                    pred += w[j] * (series[t - 1 - j] - y_mean);
                }
                let err = pred - series[t];
                for j in 0..p {
                    grad[j] += err * (series[t - 1 - j] - y_mean);
                }
            }
            for j in 0..p {
                w[j] -= lr * 2.0 * grad[j] / n as f64;
            }
        }

        let mut result = w;
        result.push(y_mean);
        result
    }

    fn predict_ar(series: &[f64], coeffs: &[f64], start: usize, end: usize) -> (Vec<f64>, f64) {
        let p = coeffs.len() - 1;
        let y_mean = coeffs[p];
        let mut preds = Vec::new();
        let mut mse = 0.0;

        for t in start..end {
            let mut pred = y_mean;
            for j in 0..p {
                if t > j {
                    pred += coeffs[j] * (series[t - 1 - j] - y_mean);
                }
            }
            preds.push(pred);
            mse += (series[t] - pred).powi(2);
        }
        mse /= (end - start) as f64;
        (preds, mse)
    }

    // --- ARIMA implementation ---
    fn fit_arima(y: &[f64], p: usize, d: usize, q: usize, train_end: usize)
        -> (Vec<f64>, Vec<f64>, f64, Vec<f64>) // (ar_coeffs, ma_coeffs, intercept, residuals)
    {
        // Step 1: Difference d times
        let mut series = y.to_vec();
        for _ in 0..d {
            let new: Vec<f64> = (1..series.len()).map(|t| series[t] - series[t - 1]).collect();
            series = new;
        }

        let effective_train = train_end - d;
        let start = p.max(q);

        // Iterative fitting: alternate between AR and MA estimation
        let n_train = effective_train - start;
        let mean: f64 = series[start..effective_train].iter().sum::<f64>() / n_train as f64;
        let mut phi = vec![0.0; p];
        let mut theta = vec![0.0; q];
        let mut residuals = vec![0.0; series.len()];
        let lr = 0.0005;

        for _iter in 0..50 {
            // Compute residuals with current parameters
            for t in start..effective_train {
                let mut pred = mean;
                for j in 0..p {
                    if t > j { pred += phi[j] * (series[t - 1 - j] - mean); }
                }
                for j in 0..q {
                    if t > j { pred += theta[j] * residuals[t - 1 - j]; }
                }
                residuals[t] = series[t] - pred;
            }

            // Update AR coefficients
            for j in 0..p {
                let mut grad = 0.0;
                for t in start..effective_train {
                    let mut pred = mean;
                    for k in 0..p { if t > k { pred += phi[k] * (series[t-1-k] - mean); } }
                    for k in 0..q { if t > k { pred += theta[k] * residuals[t-1-k]; } }
                    let err = pred - series[t];
                    if t > j { grad += err * (series[t-1-j] - mean); }
                }
                phi[j] -= lr * 2.0 * grad / n_train as f64;
            }

            // Update MA coefficients
            for j in 0..q {
                let mut grad = 0.0;
                for t in start..effective_train {
                    let mut pred = mean;
                    for k in 0..p { if t > k { pred += phi[k] * (series[t-1-k] - mean); } }
                    for k in 0..q { if t > k { pred += theta[k] * residuals[t-1-k]; } }
                    let err = pred - series[t];
                    if t > j { grad += err * residuals[t-1-j]; }
                }
                theta[j] -= lr * 2.0 * grad / n_train as f64;
            }
        }

        (phi, theta, mean, residuals)
    }

    fn predict_arima(y: &[f64], phi: &[f64], theta: &[f64], mean: f64,
                      residuals: &[f64], d: usize, start: usize, end: usize) -> (Vec<f64>, f64) {
        let p = phi.len();
        let q = theta.len();

        // Difference the series
        let mut series = y.to_vec();
        for _ in 0..d {
            let new: Vec<f64> = (1..series.len()).map(|t| series[t] - series[t - 1]).collect();
            series = new;
        }

        let mut preds = Vec::new();
        let mut mse = 0.0;
        let adj_start = start - d;
        let adj_end = end - d;
        let mut resid = residuals.to_vec();

        for t in adj_start..adj_end {
            let mut pred = mean;
            for j in 0..p {
                if t > j { pred += phi[j] * (series[t-1-j] - mean); }
            }
            for j in 0..q {
                if t > j { pred += theta[j] * resid[t-1-j]; }
            }

            // Convert back from differenced to original scale
            let actual_diff = series[t];
            resid.push(actual_diff - pred);

            // For MSE, compare differenced predictions
            mse += (actual_diff - pred).powi(2);
            preds.push(pred);
        }
        mse /= (adj_end - adj_start) as f64;
        (preds, mse)
    }

    // --- Compare different ARIMA configurations ---
    println!("=== Model Comparison ===\n");

    let configs = vec![
        ("AR(1) on diff", 1, 1, 0),
        ("AR(2) on diff", 2, 1, 0),
        ("MA(1) on diff", 0, 1, 1),
        ("ARIMA(1,1,1)", 1, 1, 1),
        ("ARIMA(2,1,1)", 2, 1, 1),
    ];

    println!("{:<20} {:>12} {:>10} {:>10}", "Model", "Test MSE", "AR coeffs", "MA coeffs");
    println!("{}", "-".repeat(55));

    for (name, p, d, q) in &configs {
        let (phi, theta, mean, resid) = fit_arima(&y, *p, *d, *q, test_start);
        let (_, mse) = predict_arima(&y, &phi, &theta, mean, &resid, *d, test_start, n);

        let ar_str: String = phi.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(",");
        let ma_str: String = theta.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(",");
        println!("{:<20} {:>12.4} {:>10} {:>10}", name, mse, ar_str, ma_str);
    }

    // --- Detailed ARIMA(1,1,1) results ---
    println!("\n=== Detailed ARIMA(1,1,1) ===");
    let (phi, theta, mean, resid) = fit_arima(&y, 1, 1, 1, test_start);
    println!("Estimated phi (AR):   {:.4} (true: {})", phi[0], true_phi);
    println!("Estimated theta (MA): {:.4} (true: {})", theta[0], true_theta);
    println!("Intercept (mean):     {:.4}", mean);

    // Residual diagnostics
    let train_resid: Vec<f64> = resid[2..test_start - 1].to_vec();
    let resid_mean: f64 = train_resid.iter().sum::<f64>() / train_resid.len() as f64;
    let resid_std: f64 = (train_resid.iter().map(|r| (r - resid_mean).powi(2)).sum::<f64>()
        / train_resid.len() as f64).sqrt();

    println!("\nResidual diagnostics (should look like white noise):");
    println!("  Mean:  {:.4} (should be ~0)", resid_mean);
    println!("  Std:   {:.4}", resid_std);

    // Simple AIC approximation: AIC = n*ln(MSE) + 2*k
    println!("\n=== Information Criteria (simplified AIC) ===");
    for (name, p, d, q) in &configs {
        let (phi, theta, mean, resid) = fit_arima(&y, *p, *d, *q, test_start);
        let train_mse: f64 = resid[(*p).max(*q)..test_start - *d].iter()
            .map(|r| r.powi(2)).sum::<f64>()
            / (test_start - *d - (*p).max(*q)) as f64;
        let k = *p + *q + 1; // number of parameters
        let aic = (test_start as f64) * train_mse.ln() + 2.0 * k as f64;
        let _ = (phi, theta, mean); // suppress warnings
        println!("  {:<20} params={}, AIC={:.1}", name, k, aic);
    }

    println!();
    let (_, final_mse) = predict_arima(&y, &phi, &theta, mean, &resid, 1, test_start, n);
    println!("kata_metric(\"arima_test_mse\", {:.4})", final_mse);
    println!("kata_metric(\"estimated_phi\", {:.4})", phi[0]);
    println!("kata_metric(\"estimated_theta\", {:.4})", theta[0]);
}
```

---

## Key Takeaways

- **ARIMA combines autoregression, differencing, and moving average** into a flexible framework for univariate forecasting.
- **Differencing (the I in ARIMA) handles non-stationarity.** Most real-world series need d=1 (first difference); rarely d=2.
- **ACF/PACF guide parameter selection.** PACF cutoff suggests p (AR order); ACF cutoff suggests q (MA order).
- **Residuals should resemble white noise.** If residuals show structure (significant autocorrelation), the model is missing something and needs more parameters or a different approach.
