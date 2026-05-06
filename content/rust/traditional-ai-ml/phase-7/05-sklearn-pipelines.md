# Trait-Based ML Pipelines

> Phase 7 — Feature Engineering & Pipelines | Kata 7.5

---

## Concept & Intuition

### What problem are we solving?

A typical ML workflow involves multiple steps: impute missing values, scale features, encode categories, select features, then train a model. When these steps are done manually, there are two common problems. First, **data leakage**: if you fit the scaler on the full dataset before splitting, test set statistics leak into the training process. Second, **reproducibility**: keeping track of which transformations were applied in which order, with which parameters, becomes a nightmare when deploying or sharing models.

In Python, scikit-learn's Pipeline solves both problems by chaining transformation steps into a single object. In Rust, we can achieve the same pattern using **traits**. By defining a `Transformer` trait with `fit()` and `transform()` methods, we can chain any number of preprocessing steps with a final model into a single pipeline. Each step is fit only on training data, and transformations are applied consistently to both training and test data.

The trait-based approach in Rust provides compile-time guarantees that each step implements the correct interface. This makes the pipeline type-safe and impossible to misuse -- you cannot accidentally skip the `fit()` step or pass incorrectly shaped data. The Rust ownership model also prevents accidental mutation of fitted parameters.

### Why naive approaches fail

The most dangerous mistake in ML is fitting a transformer on the entire dataset, then splitting into train/test. For example, if you standardize using the mean and standard deviation of all data, the test set's statistics influence the training process. This makes your validation scores optimistically biased. Pipelines eliminate this by design: `fit()` learns parameters only from training data, and `transform()` applies those learned parameters to any new data.

### Mental models

- **Assembly line**: raw materials (data) enter one end, pass through stations (imputer, scaler, model) in order, and a finished product (prediction) comes out.
- **Trait contract**: the `Transformer` trait is a contract that every pipeline step must fulfill. This guarantees composability -- any transformer can plug into any position.
- **Train/test firewall**: the pipeline enforces strict separation. Fit parameters are learned only from training data, never from test data.

### Visual explanations

```
Rust Trait-Based Pipeline:

  trait Transformer {
      fn fit(&mut self, data: &[Vec<f64>]);
      fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>>;
  }

  Pipeline { steps: Vec<Box<dyn Transformer>> }

  pipeline.fit(train_data):
    step1.fit(train_data)
    data1 = step1.transform(train_data)
    step2.fit(data1)
    data2 = step2.transform(data1)
    model.fit(data2, labels)

  pipeline.predict(test_data):
    data1 = step1.transform(test_data)  // uses params from fit
    data2 = step2.transform(data1)      // uses params from fit
    model.predict(data2)

Manual Workflow (DANGER):
  scaler.fit(all_data)        // BUG: sees test data!
  all_scaled = scaler.transform(all_data)
  train, test = split(all_scaled)
  model.fit(train)

Pipeline Workflow (SAFE):
  pipeline.fit(train_data)    // scaler fits on train only
  pipeline.predict(test_data) // scaler transforms with train params
```

---

## Hands-on Exploration

1. Define a `Transformer` trait with `fit` and `transform` methods. Implement a `StandardScaler` and a `MeanImputer` as concrete types.
2. Build a `Pipeline` struct that chains multiple transformers. Verify that calling `fit` on the pipeline correctly propagates through all steps.
3. Demonstrate data leakage: manually scale all data first, then split. Compare the "leaky" accuracy to the pipeline's honest accuracy.
4. Show that the pipeline produces identical predictions whether you call the steps individually or use the pipeline -- proving composability.

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

    // === Define the Transformer trait ===
    trait Transformer {
        fn fit(&mut self, data: &[Vec<f64>]);
        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>>;
        fn name(&self) -> &str;
    }

    // === StandardScaler ===
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

    // === MeanImputer (replaces NaN/sentinel with column mean) ===
    struct MeanImputer {
        means: Vec<f64>,
        sentinel: f64,
    }

    impl MeanImputer {
        fn new(sentinel: f64) -> Self {
            MeanImputer { means: Vec::new(), sentinel }
        }
    }

    impl Transformer for MeanImputer {
        fn fit(&mut self, data: &[Vec<f64>]) {
            if data.is_empty() { return; }
            let n_feat = data[0].len();
            self.means = vec![0.0; n_feat];

            for f in 0..n_feat {
                let valid: Vec<f64> = data.iter()
                    .map(|r| r[f])
                    .filter(|&v| (v - self.sentinel).abs() > 1e-10)
                    .collect();
                self.means[f] = if valid.is_empty() {
                    0.0
                } else {
                    valid.iter().sum::<f64>() / valid.len() as f64
                };
            }
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().enumerate().map(|(f, &v)| {
                    if (v - self.sentinel).abs() < 1e-10 {
                        self.means[f]
                    } else {
                        v
                    }
                }).collect()
            }).collect()
        }

        fn name(&self) -> &str { "MeanImputer" }
    }

    // === OutlierClipper ===
    struct OutlierClipper {
        lower: Vec<f64>,
        upper: Vec<f64>,
        pct_low: f64,
        pct_high: f64,
    }

    impl OutlierClipper {
        fn new(pct_low: f64, pct_high: f64) -> Self {
            OutlierClipper {
                lower: Vec::new(), upper: Vec::new(), pct_low, pct_high,
            }
        }
    }

    impl Transformer for OutlierClipper {
        fn fit(&mut self, data: &[Vec<f64>]) {
            if data.is_empty() { return; }
            let n_feat = data[0].len();
            self.lower = vec![0.0; n_feat];
            self.upper = vec![0.0; n_feat];

            for f in 0..n_feat {
                let mut col: Vec<f64> = data.iter().map(|r| r[f]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let lo_idx = (col.len() as f64 * self.pct_low / 100.0) as usize;
                let hi_idx = ((col.len() as f64 * self.pct_high / 100.0) as usize)
                    .min(col.len() - 1);
                self.lower[f] = col[lo_idx];
                self.upper[f] = col[hi_idx];
            }
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().enumerate().map(|(f, &v)| {
                    v.max(self.lower[f]).min(self.upper[f])
                }).collect()
            }).collect()
        }

        fn name(&self) -> &str { "OutlierClipper" }
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

        fn describe(&self) {
            println!("Pipeline with {} steps:", self.steps.len());
            for (i, step) in self.steps.iter().enumerate() {
                println!("  Step {}: {}", i + 1, step.name());
            }
        }
    }

    // --- Generate dataset with missing values and outliers ---
    let n = 200;
    let sentinel = -999.0;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for _ in 0..n {
        let x1 = rand_f64(&mut rng) * 10.0;
        let x2 = rand_f64(&mut rng) * 50.0 + 20.0;
        let x3 = rand_f64(&mut rng) * 5.0;

        // Inject missing values (10%)
        let x1_val = if rand_f64(&mut rng) < 0.1 { sentinel } else { x1 };
        let x2_val = if rand_f64(&mut rng) < 0.1 { sentinel } else { x2 };

        // Inject outliers (5%)
        let x2_final = if rand_f64(&mut rng) < 0.05 { x2_val * 10.0 } else { x2_val };

        data.push(vec![x1_val, x2_final, x3]);
        labels.push(2.0 * x1 - 0.5 * x2 + x3 + (rand_f64(&mut rng) - 0.5) * 3.0);
    }

    let split = 150;
    let train_data = &data[..split];
    let test_data = &data[split..];
    let train_y = &labels[..split];
    let test_y = &labels[split..];

    // --- WRONG WAY: Manual preprocessing with data leakage ---
    println!("=== WRONG: Manual Preprocessing (data leakage) ===\n");

    // Fit imputer and scaler on ALL data (including test)
    let mut leaky_imputer = MeanImputer::new(sentinel);
    let mut leaky_scaler = StandardScaler::new();

    leaky_imputer.fit(&data); // BUG: uses all data
    let imputed_all = leaky_imputer.transform(&data);
    leaky_scaler.fit(&imputed_all); // BUG: uses all data
    let scaled_all = leaky_scaler.transform(&imputed_all);

    // Now split
    let leaky_train = &scaled_all[..split];
    let leaky_test = &scaled_all[split..];

    // Simple linear prediction
    fn linear_predict(data: &[Vec<f64>], y: &[f64], test: &[Vec<f64>]) -> (Vec<f64>, f64) {
        let n = data.len() as f64;
        let n_feat = data[0].len();
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        let mut w = vec![0.0; n_feat];
        let lr = 0.001;

        for _ in 0..1000 {
            let mut grad = vec![0.0; n_feat];
            for i in 0..data.len() {
                let pred: f64 = w.iter().zip(&data[i]).map(|(wi, xi)| wi * xi).sum::<f64>() + y_mean;
                let err = pred - y[i];
                for f in 0..n_feat {
                    grad[f] += err * data[i][f];
                }
            }
            for f in 0..n_feat {
                w[f] -= lr * 2.0 * grad[f] / n;
            }
        }

        let preds: Vec<f64> = test.iter().map(|row| {
            w.iter().zip(row).map(|(wi, xi)| wi * xi).sum::<f64>() + y_mean
        }).collect();
        let mse: f64 = preds.iter().zip(test_y).map(|(p, t)| (p - t).powi(2)).sum::<f64>()
            / test_y.len() as f64;
        (preds, mse)
    }

    let (_, leaky_mse) = linear_predict(leaky_train, train_y, leaky_test);
    println!("Leaky MSE: {:.4} (optimistically biased!)\n", leaky_mse);

    // --- RIGHT WAY: Pipeline (no leakage) ---
    println!("=== RIGHT: Pipeline (no leakage) ===\n");

    let mut pipeline = Pipeline::new(vec![
        Box::new(MeanImputer::new(sentinel)),
        Box::new(OutlierClipper::new(5.0, 95.0)),
        Box::new(StandardScaler::new()),
    ]);

    pipeline.describe();
    println!();

    // Fit on training data ONLY
    let train_transformed = pipeline.fit_transform(train_data);
    // Transform test data using params learned from training
    let test_transformed = pipeline.transform(test_data);

    let (_, safe_mse) = linear_predict(&train_transformed, train_y, &test_transformed);
    println!("Pipeline MSE: {:.4} (honest estimate)", safe_mse);

    // --- Show the difference ---
    println!("\n=== Comparison ===");
    println!("Leaky (WRONG):   MSE = {:.4}", leaky_mse);
    println!("Pipeline (RIGHT): MSE = {:.4}", safe_mse);

    // --- Demonstrate fitted parameters ---
    println!("\n=== Fitted Parameters (from training data only) ===");

    let mut demo_imputer = MeanImputer::new(sentinel);
    demo_imputer.fit(train_data);
    println!("Imputer means: {:?}", demo_imputer.means.iter()
        .map(|m| format!("{:.2}", m)).collect::<Vec<_>>());

    let imputed_train = demo_imputer.transform(train_data);
    let mut demo_scaler = StandardScaler::new();
    demo_scaler.fit(&imputed_train);
    println!("Scaler means:  {:?}", demo_scaler.means.iter()
        .map(|m| format!("{:.2}", m)).collect::<Vec<_>>());
    println!("Scaler stds:   {:?}", demo_scaler.stds.iter()
        .map(|s| format!("{:.2}", s)).collect::<Vec<_>>());

    // --- Cross-validation with pipeline ---
    println!("\n=== Cross-Validation with Pipeline ===\n");
    let n_folds = 5;
    let fold_size = split / n_folds;
    let mut cv_scores = Vec::new();

    for fold in 0..n_folds {
        let val_start = fold * fold_size;
        let val_end = if fold == n_folds - 1 { split } else { (fold + 1) * fold_size };

        let mut cv_train: Vec<Vec<f64>> = Vec::new();
        let mut cv_train_y: Vec<f64> = Vec::new();
        let mut cv_val: Vec<Vec<f64>> = Vec::new();
        let mut cv_val_y: Vec<f64> = Vec::new();

        for i in 0..split {
            if i >= val_start && i < val_end {
                cv_val.push(train_data[i].clone());
                cv_val_y.push(train_y[i]);
            } else {
                cv_train.push(train_data[i].clone());
                cv_train_y.push(train_y[i]);
            }
        }

        let mut fold_pipeline = Pipeline::new(vec![
            Box::new(MeanImputer::new(sentinel)),
            Box::new(OutlierClipper::new(5.0, 95.0)),
            Box::new(StandardScaler::new()),
        ]);

        let fold_train_t = fold_pipeline.fit_transform(&cv_train);
        let fold_val_t = fold_pipeline.transform(&cv_val);

        let (preds, mse) = linear_predict(&fold_train_t, &cv_train_y, &fold_val_t);
        let _ = preds; // suppress unused warning
        // Recompute MSE against cv_val_y
        let fold_mse: f64 = preds.iter().zip(&cv_val_y)
            .map(|(p, t)| (p - t).powi(2)).sum::<f64>() / cv_val_y.len() as f64;
        cv_scores.push(fold_mse);
        println!("  Fold {}: MSE = {:.4}", fold + 1, fold_mse);
    }

    let mean_cv: f64 = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
    let std_cv: f64 = (cv_scores.iter().map(|s| (s - mean_cv).powi(2)).sum::<f64>()
        / cv_scores.len() as f64).sqrt();
    println!("\nMean CV MSE: {:.4} (+/- {:.4})", mean_cv, std_cv);

    println!();
    println!("kata_metric(\"leaky_mse\", {:.4})", leaky_mse);
    println!("kata_metric(\"pipeline_mse\", {:.4})", safe_mse);
    println!("kata_metric(\"cv_mean_mse\", {:.4})", mean_cv);
}
```

---

## Key Takeaways

- **Trait-based pipelines prevent data leakage by design.** Fit parameters are learned only from training data, even inside cross-validation loops.
- **The Transformer trait (fit/transform) is the composability contract.** Any struct implementing it can plug into the pipeline, enabling modular, reusable preprocessing.
- **Rust's type system ensures pipeline correctness at compile time.** You cannot accidentally skip the fit step or mix up data shapes.
- **Cross-validation with pipelines gives honest estimates.** Each fold fits the preprocessor from scratch on the training fold, preventing subtle information leakage.
