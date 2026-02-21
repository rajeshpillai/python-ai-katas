# DateTime Features

> Phase 7 — Feature Engineering & Pipelines | Kata 7.4

---

## Concept & Intuition

### What problem are we solving?

Timestamps are everywhere -- transaction times, sensor readings, log entries, user activity. But a raw timestamp like "2024-03-15 14:30:00" is meaningless to a model. It is a single number (seconds since epoch) that encodes many overlapping patterns: hour-of-day effects (lunch rush), day-of-week effects (weekend vs weekday), monthly seasonality (holiday shopping), and long-term trends. DateTime feature engineering extracts these distinct components so the model can learn each pattern separately.

The key insight is that time is **cyclical**: hour 23 is close to hour 0, December is close to January. Encoding cyclical features requires trigonometric transforms (sine/cosine encoding) rather than simple integers, because the integer encoding creates an artificial cliff (23 -> 0 looks like a huge jump when it is actually a small one).

Beyond extraction, derived features capture domain-specific temporal patterns: "is it a business hour?", "days since last purchase", "rolling average over the past 7 days." These features encode knowledge about how time affects the target variable in your specific domain.

### Why naive approaches fail

Using a raw Unix timestamp as a feature forces the model to learn all temporal patterns from a single monotonically increasing number. Using hour-of-day as an integer (0-23) creates a false discontinuity between 23 and 0. Using month as an integer creates a similar problem between December (12) and January (1). Cyclical encoding with sine and cosine preserves the circular nature of time, giving the model a smooth, continuous representation.

### Mental models

- **Clock face**: hours wrap around a circle. Sine and cosine map positions on this circle to two coordinates, preserving the distance between any two times.
- **Multiple rhythms**: time contains weekly, daily, monthly, and yearly rhythms superimposed. Feature extraction teases them apart.
- **Context over raw value**: "Tuesday at 2 PM" is more informative than "timestamp 1710510600."

### Visual explanations

```
Cyclical Encoding of Hour (0-23):

  hour_sin = sin(2 * pi * hour / 24)
  hour_cos = cos(2 * pi * hour / 24)

  Hour 0:  sin=0.00, cos=1.00   (midnight)
  Hour 6:  sin=1.00, cos=0.00   (6 AM)
  Hour 12: sin=0.00, cos=-1.00  (noon)
  Hour 18: sin=-1.00, cos=0.00  (6 PM)
  Hour 23: sin=-0.26, cos=0.97  (close to midnight!)

  On a unit circle:
       12
        |
  18 ---+--- 6
        |
        0

Integer encoding:  dist(23, 0) = 23  (WRONG! they're 1 hour apart)
Cyclical encoding: dist(23, 0) ~ 0.26 (CORRECT!)
```

---

## Hands-on Exploration

1. Generate hourly data with a clear daily pattern (e.g., energy usage peaks during business hours). Extract hour, day-of-week, and month features.
2. Compare integer encoding vs cyclical (sin/cos) encoding for hour-of-day. Show that cyclical encoding gives better predictions near midnight.
3. Create "is_business_hour", "is_weekend", and "days_since_start" features. Evaluate their impact on prediction accuracy.
4. Build a rolling-average feature and show how it captures short-term trends that static features miss.

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

    // --- Simulate 30 days of hourly energy usage data ---
    let n_days = 30;
    let hours_per_day = 24;
    let n = n_days * hours_per_day; // 720 data points

    // Each data point: (day, hour, day_of_week, energy_usage)
    let mut timestamps: Vec<(usize, usize, usize)> = Vec::new(); // (day, hour, dow)
    let mut usage: Vec<f64> = Vec::new();

    for day in 0..n_days {
        let dow = day % 7; // 0=Monday, 6=Sunday
        for hour in 0..hours_per_day {
            timestamps.push((day, hour, dow));

            // True pattern: daily cycle + weekend effect + trend + noise
            let daily_effect = 50.0 * (2.0 * pi * (hour as f64 - 8.0) / 24.0).cos() * (-1.0);
            let business_boost = if dow < 5 && hour >= 8 && hour <= 18 { 30.0 } else { 0.0 };
            let weekend_dip = if dow >= 5 { -20.0 } else { 0.0 };
            let trend = 0.5 * day as f64; // slight upward trend
            let noise = (rand_f64(&mut rng) - 0.5) * 15.0;

            let value = 200.0 + daily_effect + business_boost + weekend_dip + trend + noise;
            usage.push(value.max(50.0));
        }
    }

    println!("=== DateTime Feature Engineering ===");
    println!("Dataset: {} days x {} hours = {} observations\n", n_days, hours_per_day, n);

    // --- Feature Extraction ---

    // 1. Raw integer features
    let hours: Vec<f64> = timestamps.iter().map(|&(_, h, _)| h as f64).collect();
    let dows: Vec<f64> = timestamps.iter().map(|&(_, _, d)| d as f64).collect();
    let days: Vec<f64> = timestamps.iter().map(|&(d, _, _)| d as f64).collect();

    // 2. Cyclical encoding
    let hour_sin: Vec<f64> = hours.iter().map(|&h| (2.0 * pi * h / 24.0).sin()).collect();
    let hour_cos: Vec<f64> = hours.iter().map(|&h| (2.0 * pi * h / 24.0).cos()).collect();
    let dow_sin: Vec<f64> = dows.iter().map(|&d| (2.0 * pi * d / 7.0).sin()).collect();
    let dow_cos: Vec<f64> = dows.iter().map(|&d| (2.0 * pi * d / 7.0).cos()).collect();

    // 3. Boolean/derived features
    let is_weekend: Vec<f64> = timestamps.iter()
        .map(|&(_, _, d)| if d >= 5 { 1.0 } else { 0.0 }).collect();
    let is_business: Vec<f64> = timestamps.iter()
        .map(|&(_, h, d)| if d < 5 && h >= 8 && h <= 18 { 1.0 } else { 0.0 }).collect();

    // 4. Rolling average (window=24 hours)
    let window = 24;
    let mut rolling_avg: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let start = if i >= window { i - window } else { 0 };
        let sum: f64 = usage[start..i + 1].iter().sum();
        rolling_avg[i] = sum / (i + 1 - start) as f64;
    }

    // --- Demonstrate cyclical encoding ---
    println!("=== Cyclical Encoding Demo ===\n");
    println!("{:>6} {:>8} {:>8} {:>10}", "Hour", "sin", "cos", "note");
    println!("{}", "-".repeat(36));
    for &h in &[0, 3, 6, 9, 12, 15, 18, 21, 23] {
        let s = (2.0 * pi * h as f64 / 24.0).sin();
        let c = (2.0 * pi * h as f64 / 24.0).cos();
        let note = match h {
            0 => "midnight",
            6 => "morning",
            12 => "noon",
            18 => "evening",
            23 => "near midnight",
            _ => "",
        };
        println!("{:>6} {:>8.3} {:>8.3} {:>10}", h, s, c, note);
    }

    // Distance between hours using cyclical encoding
    println!("\nDistance between hours (Euclidean on sin/cos):");
    let dist = |h1: f64, h2: f64| -> f64 {
        let s1 = (2.0 * pi * h1 / 24.0).sin();
        let c1 = (2.0 * pi * h1 / 24.0).cos();
        let s2 = (2.0 * pi * h2 / 24.0).sin();
        let c2 = (2.0 * pi * h2 / 24.0).cos();
        ((s1 - s2).powi(2) + (c1 - c2).powi(2)).sqrt()
    };
    println!("  dist(23, 0)  = {:.3}  (should be small - 1 hour apart)", dist(23.0, 0.0));
    println!("  dist(23, 1)  = {:.3}  (should be small - 2 hours apart)", dist(23.0, 1.0));
    println!("  dist(0, 12)  = {:.3}  (should be large - 12 hours apart)", dist(0.0, 12.0));
    println!("  dist(6, 18)  = {:.3}  (should be large - 12 hours apart)", dist(6.0, 18.0));

    // --- Compare feature sets for prediction ---
    println!("\n=== Prediction with Different Feature Sets ===\n");

    let split = (n as f64 * 0.8) as usize;

    // Simple linear regression helper
    fn fit_predict(
        features: &[Vec<f64>], target: &[f64], split: usize,
    ) -> (Vec<f64>, f64) {
        let n_feat = features.len();
        let n_total = target.len();

        // Standardize
        let mut means = vec![0.0; n_feat];
        let mut stds = vec![1.0; n_feat];
        for f in 0..n_feat {
            means[f] = features[f][..split].iter().sum::<f64>() / split as f64;
            let var: f64 = features[f][..split].iter()
                .map(|&x| (x - means[f]).powi(2)).sum::<f64>() / split as f64;
            stds[f] = var.sqrt().max(1e-8);
        }
        let y_mean: f64 = target[..split].iter().sum::<f64>() / split as f64;

        // Gradient descent
        let mut w = vec![0.0; n_feat];
        let lr = 0.01;
        for _ in 0..2000 {
            let mut grad = vec![0.0; n_feat];
            for i in 0..split {
                let mut pred = 0.0;
                for f in 0..n_feat {
                    pred += w[f] * (features[f][i] - means[f]) / stds[f];
                }
                let err = pred - (target[i] - y_mean);
                for f in 0..n_feat {
                    grad[f] += err * (features[f][i] - means[f]) / stds[f];
                }
            }
            for f in 0..n_feat {
                w[f] -= lr * 2.0 * grad[f] / split as f64;
            }
        }

        // Predict on test set
        let mut preds = Vec::new();
        let mut mse_sum = 0.0;
        for i in split..n_total {
            let mut pred = y_mean;
            for f in 0..n_feat {
                pred += w[f] * (features[f][i] - means[f]) / stds[f];
            }
            preds.push(pred);
            mse_sum += (target[i] - pred).powi(2);
        }
        let mse = mse_sum / (n_total - split) as f64;
        (preds, mse)
    }

    // Feature set 1: raw integer hour + dow
    let (_, mse1) = fit_predict(
        &[hours.clone(), dows.clone(), days.clone()],
        &usage, split,
    );

    // Feature set 2: cyclical encoding
    let (_, mse2) = fit_predict(
        &[hour_sin.clone(), hour_cos.clone(), dow_sin.clone(), dow_cos.clone(), days.clone()],
        &usage, split,
    );

    // Feature set 3: cyclical + boolean features
    let (_, mse3) = fit_predict(
        &[hour_sin.clone(), hour_cos.clone(), dow_sin.clone(), dow_cos.clone(),
          is_weekend.clone(), is_business.clone(), days.clone()],
        &usage, split,
    );

    // Feature set 4: all features + rolling average
    let (_, mse4) = fit_predict(
        &[hour_sin.clone(), hour_cos.clone(), dow_sin.clone(), dow_cos.clone(),
          is_weekend.clone(), is_business.clone(), days.clone(), rolling_avg.clone()],
        &usage, split,
    );

    println!("{:<45} {:>10}", "Feature Set", "Test MSE");
    println!("{}", "-".repeat(57));
    println!("{:<45} {:>10.2}", "Raw integers (hour, dow, day)", mse1);
    println!("{:<45} {:>10.2}", "Cyclical (sin/cos hour, sin/cos dow, day)", mse2);
    println!("{:<45} {:>10.2}", "+ is_weekend, is_business_hour", mse3);
    println!("{:<45} {:>10.2}", "+ rolling_average_24h", mse4);

    // --- Show extracted features for sample timestamps ---
    println!("\n=== Sample Feature Values ===\n");
    println!("{:>4} {:>4} {:>4} {:>7} {:>7} {:>7} {:>7} {:>5} {:>5}",
        "Day", "Hour", "DoW", "h_sin", "h_cos", "d_sin", "d_cos", "wknd", "biz");
    for &idx in &[0, 10, 24, 120, 168, 500] {
        if idx < n {
            let (d, h, dow) = timestamps[idx];
            println!("{:>4} {:>4} {:>4} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>5} {:>5}",
                d, h, dow, hour_sin[idx], hour_cos[idx], dow_sin[idx], dow_cos[idx],
                is_weekend[idx] as i32, is_business[idx] as i32);
        }
    }

    println!();
    println!("kata_metric(\"mse_raw_integers\", {:.2})", mse1);
    println!("kata_metric(\"mse_cyclical\", {:.2})", mse2);
    println!("kata_metric(\"mse_full_features\", {:.2})", mse4);
}
```

---

## Key Takeaways

- **Raw timestamps must be decomposed into meaningful components.** Hour, day-of-week, month, and derived features each capture distinct patterns.
- **Cyclical encoding (sin/cos) preserves temporal continuity.** Without it, midnight (hour 0) appears maximally distant from 11 PM (hour 23), which is wrong.
- **Domain-specific boolean features (is_weekend, is_business_hour) encode expert knowledge** that helps the model without requiring it to learn these patterns from scratch.
- **Rolling averages capture recent trends** that static time features cannot, adding temporal context to each observation.
