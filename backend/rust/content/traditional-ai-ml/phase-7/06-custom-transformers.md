# Custom Transformers

> Phase 7 — Feature Engineering & Pipelines | Kata 7.6

---

## Concept & Intuition

### What problem are we solving?

Built-in transformers (scalers, imputers) cover common operations, but real-world feature engineering often requires custom logic: computing domain-specific ratios, applying business rules, extracting features from structured data, or performing conditional transformations. Custom transformers let you wrap any arbitrary logic into the pipeline ecosystem, gaining all the benefits of pipelines (no data leakage, composability, reproducibility) for your custom code.

In Rust, creating a custom transformer means implementing the `Transformer` trait with `fit()` and `transform()` methods. The `fit()` method learns any necessary parameters from training data (e.g., percentile thresholds, vocabulary mappings), and `transform()` applies the learned transformation. For stateless transformations (where `fit()` learns nothing), you can implement a simple function transformer wrapper.

The key design decision is **stateful vs stateless**. If your transformation depends on learned parameters (normalization constants, bin edges, frequency tables), it must learn them in `fit()` from training data only. If it is a pure mathematical operation (log transform, squaring), it is stateless and the `fit()` method does nothing.

### Why naive approaches fail

Writing custom preprocessing as standalone functions outside the pipeline recreates data leakage risks. If your custom function computes a normalization constant from the full dataset, it leaks test information into training. By implementing the logic as a proper transformer with separate `fit()` and `transform()` methods, the pipeline ensures the function is fit only on training data and applied consistently to all data.

### Mental models

- **Fit/transform contract**: `fit()` learns from data (like a student studying), `transform()` applies what was learned (like taking the test). The separation prevents leakage.
- **Adapter pattern**: your custom logic is the core algorithm; the Transformer trait wraps it so it can plug into the Pipeline system.
- **Stateful vs stateless**: if your transformation depends on data-derived parameters, you need `fit()`. If it is pure computation, implement a no-op `fit()`.

### Visual explanations

```
Custom Transformer Structure in Rust:

  struct MyTransformer {
      param: f64,                    // hyperparameter (set at creation)
      learned_values: Vec<f64>,      // learned in fit() from training data
  }

  impl Transformer for MyTransformer {
      fn fit(&mut self, data: &[Vec<f64>]) {
          self.learned_values = ...;  // learn from data
      }

      fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
          // use self.learned_values to transform
      }
  }

Stateless Function Transformer:
  struct FnTransformer { f: fn(f64) -> f64 }
  // fit() does nothing; transform() applies f to each element

Pipeline Usage:
  let pipeline = Pipeline::new(vec![
      Box::new(MyTransformer::new(5.0)),
      Box::new(StandardScaler::new()),
      Box::new(LinearModel::new()),
  ]);
```

---

## Hands-on Exploration

1. Create a stateless `LogTransformer` that applies ln(1+x) to all features. Verify it works in a pipeline with cross-validation.
2. Build a stateful `PercentileClipper` that learns the 5th and 95th percentiles from training data and clips values accordingly.
3. Create a `FeatureInteractor` that computes ratios and products of specified column pairs, adding them as new features.
4. Chain all custom transformers into a pipeline and show that the full pipeline produces better results than raw features.

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

    // === Transformer Trait ===
    trait Transformer {
        fn fit(&mut self, data: &[Vec<f64>]);
        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>>;
        fn name(&self) -> &str;
    }

    // === 1. Stateless LogTransformer ===
    struct LogTransformer;

    impl Transformer for LogTransformer {
        fn fit(&mut self, _data: &[Vec<f64>]) {
            // Stateless: nothing to learn
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().map(|&v| (1.0 + v.abs()).ln() * v.signum()).collect()
            }).collect()
        }

        fn name(&self) -> &str { "LogTransformer" }
    }

    // === 2. Stateful PercentileClipper ===
    struct PercentileClipper {
        lower_pct: f64,
        upper_pct: f64,
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
    }

    impl PercentileClipper {
        fn new(lower_pct: f64, upper_pct: f64) -> Self {
            PercentileClipper {
                lower_pct, upper_pct,
                lower_bounds: Vec::new(), upper_bounds: Vec::new(),
            }
        }
    }

    impl Transformer for PercentileClipper {
        fn fit(&mut self, data: &[Vec<f64>]) {
            if data.is_empty() { return; }
            let n_feat = data[0].len();
            self.lower_bounds = vec![0.0; n_feat];
            self.upper_bounds = vec![0.0; n_feat];

            for f in 0..n_feat {
                let mut col: Vec<f64> = data.iter().map(|r| r[f]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let lo_idx = (col.len() as f64 * self.lower_pct / 100.0) as usize;
                let hi_idx = ((col.len() as f64 * self.upper_pct / 100.0) as usize)
                    .min(col.len() - 1);
                self.lower_bounds[f] = col[lo_idx];
                self.upper_bounds[f] = col[hi_idx];
            }
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().enumerate().map(|(f, &v)| {
                    v.max(self.lower_bounds[f]).min(self.upper_bounds[f])
                }).collect()
            }).collect()
        }

        fn name(&self) -> &str { "PercentileClipper" }
    }

    // === 3. FeatureInteractor ===
    struct FeatureInteractor {
        ratio_pairs: Vec<(usize, usize)>,
        product_pairs: Vec<(usize, usize)>,
    }

    impl FeatureInteractor {
        fn new(ratio_pairs: Vec<(usize, usize)>, product_pairs: Vec<(usize, usize)>) -> Self {
            FeatureInteractor { ratio_pairs, product_pairs }
        }
    }

    impl Transformer for FeatureInteractor {
        fn fit(&mut self, _data: &[Vec<f64>]) {
            // Stateless: nothing to learn
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                let mut new_row = row.clone();
                for &(i, j) in &self.ratio_pairs {
                    new_row.push(row[i] / (row[j] + 1e-8));
                }
                for &(i, j) in &self.product_pairs {
                    new_row.push(row[i] * row[j]);
                }
                new_row
            }).collect()
        }

        fn name(&self) -> &str { "FeatureInteractor" }
    }

    // === 4. StandardScaler ===
    struct StandardScaler {
        means: Vec<f64>,
        stds: Vec<f64>,
    }

    impl StandardScaler {
        fn new() -> Self {
            StandardScaler { means: Vec::new(), stds: Vec::new() }
        }
    }

    impl Transformer for StandardScaler {
        fn fit(&mut self, data: &[Vec<f64>]) {
            if data.is_empty() { return; }
            let n = data.len() as f64;
            let n_feat = data[0].len();
            self.means = vec![0.0; n_feat];
            self.stds = vec![1.0; n_feat];
            for f in 0..n_feat {
                self.means[f] = data.iter().map(|r| r[f]).sum::<f64>() / n;
                let var: f64 = data.iter()
                    .map(|r| (r[f] - self.means[f]).powi(2)).sum::<f64>() / n;
                self.stds[f] = var.sqrt().max(1e-8);
            }
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().enumerate()
                    .map(|(f, &v)| (v - self.means[f]) / self.stds[f])
                    .collect()
            }).collect()
        }

        fn name(&self) -> &str { "StandardScaler" }
    }

    // === Pipeline ===
    struct Pipeline {
        steps: Vec<Box<dyn Transformer>>,
    }

    impl Pipeline {
        fn new(steps: Vec<Box<dyn Transformer>>) -> Self {
            Pipeline { steps }
        }

        fn fit_transform(&mut self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let mut current = data.to_vec();
            for step in self.steps.iter_mut() {
                step.fit(&current);
                current = step.transform(&current);
            }
            current
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let mut current = data.to_vec();
            for step in self.steps.iter() {
                current = step.transform(&current);
            }
            current
        }
    }

    // --- Generate housing dataset ---
    let n = 300;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut target: Vec<f64> = Vec::new();

    for _ in 0..n {
        let sqft = 500.0 + rand_f64(&mut rng) * 3000.0;
        let bedrooms = 1.0 + (rand_f64(&mut rng) * 5.0).floor();
        let bathrooms = 1.0 + (rand_f64(&mut rng) * 3.0).floor();
        let age = rand_f64(&mut rng) * 50.0;

        // Add outliers (5%)
        let sqft_val = if rand_f64(&mut rng) < 0.05 { sqft * 5.0 } else { sqft };

        // True target depends on sqft/bedrooms ratio
        let price = 100.0 * (sqft / bedrooms) - 1500.0 * age + 50000.0
            + (rand_f64(&mut rng) - 0.5) * 30000.0;
        data.push(vec![sqft_val, bedrooms, bathrooms, age]);
        target.push(price.max(10000.0));
    }

    let split = 200;
    let train_data = &data[..split];
    let test_data = &data[split..];
    let train_y = &target[..split];
    let test_y = &target[split..];

    // --- Linear regression helper ---
    fn linear_regression(
        train: &[Vec<f64>], train_y: &[f64],
        test: &[Vec<f64>], test_y: &[f64],
    ) -> f64 {
        let n_feat = train[0].len();
        let n = train.len() as f64;
        let y_mean: f64 = train_y.iter().sum::<f64>() / n;
        let mut w = vec![0.0; n_feat];
        let lr = 0.001;

        for _ in 0..2000 {
            let mut grad = vec![0.0; n_feat];
            for i in 0..train.len() {
                let pred: f64 = w.iter().zip(&train[i]).map(|(wi, xi)| wi * xi).sum::<f64>() + y_mean;
                let err = pred - train_y[i];
                for f in 0..n_feat {
                    grad[f] += err * train[i][f];
                }
            }
            for f in 0..n_feat {
                w[f] -= lr * 2.0 * grad[f] / n;
            }
        }

        let mut mse = 0.0;
        for i in 0..test.len() {
            let pred: f64 = w.iter().zip(&test[i]).map(|(wi, xi)| wi * xi).sum::<f64>() + y_mean;
            mse += (test_y[i] - pred).powi(2);
        }
        mse / test.len() as f64
    }

    // === Evaluate different transformer combinations ===
    println!("=== Custom Transformers in Pipelines ===\n");

    // 1. Raw features + scaler only
    let mut pipe1 = Pipeline::new(vec![
        Box::new(StandardScaler::new()),
    ]);
    let train1 = pipe1.fit_transform(train_data);
    let test1 = pipe1.transform(test_data);
    let mse1 = linear_regression(&train1, train_y, &test1, test_y);

    // 2. Clipper + scaler
    let mut pipe2 = Pipeline::new(vec![
        Box::new(PercentileClipper::new(5.0, 95.0)),
        Box::new(StandardScaler::new()),
    ]);
    let train2 = pipe2.fit_transform(train_data);
    let test2 = pipe2.transform(test_data);
    let mse2 = linear_regression(&train2, train_y, &test2, test_y);

    // 3. Clipper + interactions + scaler
    let mut pipe3 = Pipeline::new(vec![
        Box::new(PercentileClipper::new(5.0, 95.0)),
        Box::new(FeatureInteractor::new(
            vec![(0, 1), (0, 2)],  // sqft/beds, sqft/baths
            vec![(0, 3)],           // sqft*age
        )),
        Box::new(StandardScaler::new()),
    ]);
    let train3 = pipe3.fit_transform(train_data);
    let test3 = pipe3.transform(test_data);
    let mse3 = linear_regression(&train3, train_y, &test3, test_y);

    // 4. Full pipeline: log + clip + interactions + scale
    let mut pipe4 = Pipeline::new(vec![
        Box::new(LogTransformer),
        Box::new(PercentileClipper::new(5.0, 95.0)),
        Box::new(FeatureInteractor::new(
            vec![(0, 1), (0, 2)],
            vec![(0, 3)],
        )),
        Box::new(StandardScaler::new()),
    ]);
    let train4 = pipe4.fit_transform(train_data);
    let test4 = pipe4.transform(test_data);
    let mse4 = linear_regression(&train4, train_y, &test4, test_y);

    println!("{:<50} {:>8} {:>12}", "Pipeline Configuration", "Feats", "Test MSE");
    println!("{}", "-".repeat(72));
    println!("{:<50} {:>8} {:>12.0}", "Scaler only", train1[0].len(), mse1);
    println!("{:<50} {:>8} {:>12.0}", "PercentileClipper -> Scaler", train2[0].len(), mse2);
    println!("{:<50} {:>8} {:>12.0}",
        "Clipper -> Interactor -> Scaler", train3[0].len(), mse3);
    println!("{:<50} {:>8} {:>12.0}",
        "Log -> Clipper -> Interactor -> Scaler", train4[0].len(), mse4);

    // --- Show learned parameters ---
    println!("\n=== Stateful Transformer: Learned Parameters ===");
    let mut demo_clipper = PercentileClipper::new(5.0, 95.0);
    demo_clipper.fit(train_data);
    let feat_names = ["sqft", "bedrooms", "bathrooms", "age"];
    println!("\nPercentileClipper learned bounds (from training data):");
    for (f, name) in feat_names.iter().enumerate() {
        println!("  {:<12}: [{:.1}, {:.1}]", name, demo_clipper.lower_bounds[f], demo_clipper.upper_bounds[f]);
    }

    // --- Show feature expansion ---
    println!("\n=== Feature Expansion through Pipeline ===");
    println!("Input:  {} features [sqft, beds, baths, age]", data[0].len());
    println!("After FeatureInteractor: {} features", train3[0].len());
    println!("  [sqft, beds, baths, age, sqft/beds, sqft/baths, sqft*age]");

    println!("\nSample transformation (first row):");
    println!("  Raw:         {:?}", &data[0].iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>());
    println!("  Transformed: {:?}", &train4[0].iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());

    let best_mse = mse1.min(mse2).min(mse3).min(mse4);
    println!("\n=== Summary ===");
    println!("Best pipeline MSE: {:.0}", best_mse);
    println!("Improvement over raw: {:.1}%", (1.0 - best_mse / mse1) * 100.0);

    println!();
    println!("kata_metric(\"mse_raw\", {:.0})", mse1);
    println!("kata_metric(\"mse_best_pipeline\", {:.0})", best_mse);
    println!("kata_metric(\"improvement_pct\", {:.1})", (1.0 - best_mse / mse1) * 100.0);
}
```

---

## Key Takeaways

- **Custom transformers wrap any logic into the pipeline ecosystem.** Implement the Transformer trait (fit/transform), and your code becomes composable, leak-proof, and reproducible.
- **The fit/transform separation prevents data leakage.** Parameters learned in fit() come only from training data, even for custom logic.
- **Stateless transformers (log, polynomials) need only a trivial fit() method.** The transform() carries all the logic.
- **Stateful transformers (clippers, encoders) learn from training data in fit()** and apply those learned parameters consistently to any new data in transform().
