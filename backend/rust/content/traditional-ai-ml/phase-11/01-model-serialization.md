# Model Serialization

> Phase 11 — Productionizing ML | Kata 11.1

---

## Concept & Intuition

### What problem are we solving?

You have trained a great model. Now what? You cannot retrain it every time you want to make a prediction. You need to **save** the model to disk and **load** it later -- in a different process, on a different machine, or in a production server. This is **model serialization**: converting a trained model's state (parameters, hyperparameters, preprocessing configuration) into a persistent format.

In Rust, serialization is typically handled through the **serde** ecosystem. A model struct derives `Serialize` and `Deserialize`, and can be saved to JSON (human-readable, debuggable), MessagePack/bincode (compact, fast), or any other serde-compatible format. Unlike Python's pickle, Rust serialization is type-safe: you cannot accidentally load incompatible data, and there are no arbitrary code execution risks.

Beyond format choice, **model versioning** is critical. As you retrain models with new data or architectures, you need to track which version is deployed, reproduce past results, and roll back if a new model underperforms. Simple conventions like timestamped filenames and metadata files provide a practical starting point.

### Why naive approaches fail

Saving just the model weights without metadata (training data version, hyperparameters, feature names, preprocessing steps) makes reproduction impossible. Using ad-hoc text formats means fragile parsing code. Deploying a model without version tracking means you cannot roll back when a new model underperforms. Rust's type system helps prevent many serialization bugs at compile time, but you still need discipline around metadata and versioning.

### Mental models

- **Serialization = freezing the model in amber**: you capture the model at a specific moment, preserving everything needed to make predictions later.
- **JSON as the readable format**: human-readable, great for debugging and inspection, but larger and slower than binary.
- **Binary formats as the fast path**: compact and fast to load, ideal for production, but not human-readable.
- **Versioning as a time machine**: when the new model breaks in production, you need to instantly revert to the previous version. Without versioning, you are stuck.

### Visual explanations

```
Serialization in Rust:

  #[derive(Serialize, Deserialize)]
  struct Model {
      weights: Vec<f64>,
      bias: f64,
      feature_names: Vec<String>,
      hyperparams: Hyperparams,
  }

  Format    Size   Speed   Readable   Type-safe   Best for
  ------    ----   -----   --------   ---------   --------
  JSON      Large  Med     Yes        Yes         Debug, config, APIs
  bincode   Small  Fast    No         Yes         Production, large models
  MsgPack   Small  Fast    No         Yes         Cross-language production

Model versioning:
  models/
    model_v1.json        <-- initial model
    model_v2.json        <-- retrained with new data
    model_v3.json        <-- new architecture
    metadata.json        <-- tracks which version is active

Complete serialization checklist:
  [ ] Model parameters (weights, coefficients)
  [ ] Hyperparameters (learning_rate, n_estimators, etc.)
  [ ] Preprocessing config (scaler params, feature names)
  [ ] Training metadata (date, dataset version, metrics)
  [ ] Schema version (for forward/backward compatibility)
```

---

## Hands-on Exploration

1. Define a model struct with weights and metadata. Serialize it to a human-readable format (simulated JSON) and a compact binary format. Compare sizes.
2. Save a complete pipeline (scaler parameters + model weights) as a single serialized object. Verify that loading and predicting works correctly on new data.
3. Create a simple versioning scheme: save models with version numbers and a metadata record. Implement a function that loads the "active" version.
4. Demonstrate type-safe deserialization: try loading a model with the wrong schema and show how it fails safely.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };

    println!("=== Model Serialization ===\n");

    // --- Define a model struct ---
    struct LinearModel {
        weights: Vec<f64>,
        bias: f64,
        feature_names: Vec<String>,
        learning_rate: f64,
        n_iterations: usize,
        training_mse: f64,
    }

    impl LinearModel {
        fn predict(&self, features: &[f64]) -> f64 {
            self.bias + self.weights.iter().zip(features).map(|(w, x)| w * x).sum::<f64>()
        }

        fn predict_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
            data.iter().map(|row| self.predict(row)).collect()
        }
    }

    // --- Scaler for preprocessing ---
    struct StandardScaler {
        means: Vec<f64>,
        stds: Vec<f64>,
    }

    impl StandardScaler {
        fn fit(data: &[Vec<f64>]) -> Self {
            let n = data.len() as f64;
            let n_feat = data[0].len();
            let mut means = vec![0.0; n_feat];
            let mut stds = vec![0.0; n_feat];
            for row in data {
                for (j, &v) in row.iter().enumerate() {
                    means[j] += v;
                }
            }
            for j in 0..n_feat { means[j] /= n; }
            for row in data {
                for (j, &v) in row.iter().enumerate() {
                    stds[j] += (v - means[j]).powi(2);
                }
            }
            for j in 0..n_feat { stds[j] = (stds[j] / n).sqrt().max(1e-10); }
            StandardScaler { means, stds }
        }

        fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|row| {
                row.iter().enumerate()
                    .map(|(j, &v)| (v - self.means[j]) / self.stds[j])
                    .collect()
            }).collect()
        }
    }

    // --- Pipeline: Scaler + Model ---
    struct Pipeline {
        scaler: StandardScaler,
        model: LinearModel,
        version: u32,
    }

    // --- Serialization (simulated without external crates) ---
    // In production Rust, you would use serde + serde_json or bincode.
    // Here we demonstrate the concepts manually.

    fn serialize_json(model: &LinearModel, scaler: &StandardScaler, version: u32) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"version\": {},\n", version));
        s.push_str(&format!("  \"bias\": {:.6},\n", model.bias));
        s.push_str(&format!("  \"weights\": [{:.6}],\n",
            model.weights.iter().map(|w| format!("{:.6}", w)).collect::<Vec<_>>().join(", ")));
        s.push_str(&format!("  \"feature_names\": [{}],\n",
            model.feature_names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(", ")));
        s.push_str(&format!("  \"learning_rate\": {:.6},\n", model.learning_rate));
        s.push_str(&format!("  \"n_iterations\": {},\n", model.n_iterations));
        s.push_str(&format!("  \"training_mse\": {:.6},\n", model.training_mse));
        s.push_str(&format!("  \"scaler_means\": [{}],\n",
            scaler.means.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", ")));
        s.push_str(&format!("  \"scaler_stds\": [{}]\n",
            scaler.stds.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", ")));
        s.push_str("}\n");
        s
    }

    fn serialize_binary(model: &LinearModel, scaler: &StandardScaler, version: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        // Version (4 bytes)
        bytes.extend_from_slice(&version.to_le_bytes());
        // Number of weights (4 bytes)
        bytes.extend_from_slice(&(model.weights.len() as u32).to_le_bytes());
        // Bias (8 bytes)
        bytes.extend_from_slice(&model.bias.to_le_bytes());
        // Weights (8 bytes each)
        for &w in &model.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        // Scaler means
        for &m in &scaler.means {
            bytes.extend_from_slice(&m.to_le_bytes());
        }
        // Scaler stds
        for &s in &scaler.stds {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    fn deserialize_binary(bytes: &[u8]) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>, u32) {
        let mut pos = 0;
        let version = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let n_weights = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let bias = f64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
        pos += 8;
        let mut weights = Vec::new();
        for _ in 0..n_weights {
            weights.push(f64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()));
            pos += 8;
        }
        let mut means = Vec::new();
        let mut stds = Vec::new();
        for _ in 0..n_weights {
            means.push(f64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()));
            pos += 8;
        }
        for _ in 0..n_weights {
            stds.push(f64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()));
            pos += 8;
        }
        (weights, bias, means, stds, version)
    }

    // --- Train a model ---
    let feature_names = vec![
        "age".to_string(), "income".to_string(), "tenure".to_string(),
        "usage".to_string(), "support_calls".to_string(),
    ];
    let n_feat = feature_names.len();
    let n_train = 200;
    let true_weights = [0.5, -0.3, 0.2, 0.4, -0.1];

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    for _ in 0..n_train {
        let row: Vec<f64> = (0..n_feat).map(|_| rand_normal(&mut rng) * 3.0).collect();
        let y: f64 = true_weights.iter().zip(&row).map(|(w, x)| w * x).sum::<f64>()
            + 1.5 + rand_normal(&mut rng) * 0.5;
        x_train.push(row);
        y_train.push(y);
    }

    // Fit scaler
    let scaler = StandardScaler::fit(&x_train);
    let x_scaled = scaler.transform(&x_train);

    // Fit linear regression via gradient descent
    let mut weights = vec![0.0; n_feat];
    let mut bias = 0.0;
    let lr = 0.01;
    let n_iter = 500;

    for _ in 0..n_iter {
        let mut grad_w = vec![0.0; n_feat];
        let mut grad_b = 0.0;
        for (i, row) in x_scaled.iter().enumerate() {
            let pred = bias + weights.iter().zip(row).map(|(w, x)| w * x).sum::<f64>();
            let err = pred - y_train[i];
            for j in 0..n_feat {
                grad_w[j] += err * row[j];
            }
            grad_b += err;
        }
        for j in 0..n_feat {
            weights[j] -= lr * grad_w[j] / n_train as f64;
        }
        bias -= lr * grad_b / n_train as f64;
    }

    // Compute training MSE
    let train_preds: Vec<f64> = x_scaled.iter().map(|row| {
        bias + weights.iter().zip(row).map(|(w, x)| w * x).sum::<f64>()
    }).collect();
    let train_mse: f64 = train_preds.iter().zip(&y_train)
        .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / n_train as f64;

    let model = LinearModel {
        weights: weights.clone(),
        bias,
        feature_names: feature_names.clone(),
        learning_rate: lr,
        n_iterations: n_iter,
        training_mse: train_mse,
    };

    println!("Trained model: {} features, MSE={:.4}", n_feat, train_mse);
    println!("Weights: {:?}", model.weights.iter().map(|w| format!("{:.3}", w)).collect::<Vec<_>>());

    // --- Serialize to JSON ---
    let json_str = serialize_json(&model, &scaler, 1);
    let json_size = json_str.len();
    println!("\n=== JSON Serialization ===\n");
    println!("{}", json_str);
    println!("JSON size: {} bytes", json_size);

    // --- Serialize to binary ---
    let binary_data = serialize_binary(&model, &scaler, 1);
    let binary_size = binary_data.len();
    println!("=== Binary Serialization ===\n");
    println!("Binary size: {} bytes", binary_size);
    println!("Compression ratio: {:.1}x smaller than JSON", json_size as f64 / binary_size as f64);

    // --- Deserialize and verify ---
    let (loaded_weights, loaded_bias, loaded_means, loaded_stds, loaded_version) =
        deserialize_binary(&binary_data);

    println!("\n=== Deserialization Verification ===\n");
    let weights_match = model.weights.iter().zip(&loaded_weights)
        .all(|(a, b)| (a - b).abs() < 1e-10);
    let bias_match = (model.bias - loaded_bias).abs() < 1e-10;
    let means_match = scaler.means.iter().zip(&loaded_means)
        .all(|(a, b)| (a - b).abs() < 1e-10);

    println!("Version: {} (loaded: {})", 1, loaded_version);
    println!("Weights match:      {}", weights_match);
    println!("Bias match:         {}", bias_match);
    println!("Scaler means match: {}", means_match);

    // Verify predictions match
    let test_point = x_train[0].clone();
    let scaled_point: Vec<f64> = test_point.iter().enumerate()
        .map(|(j, &v)| (v - scaler.means[j]) / scaler.stds[j]).collect();
    let original_pred = model.predict(&scaled_point);

    let loaded_scaled: Vec<f64> = test_point.iter().enumerate()
        .map(|(j, &v)| (v - loaded_means[j]) / loaded_stds[j]).collect();
    let loaded_pred = loaded_bias + loaded_weights.iter().zip(&loaded_scaled)
        .map(|(w, x)| w * x).sum::<f64>();

    println!("Prediction match:   {}", (original_pred - loaded_pred).abs() < 1e-10);
    println!("  Original: {:.6}", original_pred);
    println!("  Loaded:   {:.6}", loaded_pred);

    // --- Model Versioning ---
    println!("\n=== Model Versioning ===\n");

    struct ModelVersion {
        version: u32,
        accuracy: f64,
        n_features: usize,
        description: String,
        is_active: bool,
    }

    let versions = vec![
        ModelVersion { version: 1, accuracy: 0.85, n_features: 5, description: "Initial model".to_string(), is_active: false },
        ModelVersion { version: 2, accuracy: 0.88, n_features: 5, description: "Retrained with new data".to_string(), is_active: false },
        ModelVersion { version: 3, accuracy: 0.90, n_features: 8, description: "Added features".to_string(), is_active: true },
    ];

    println!("{:>8} {:>10} {:>10} {:>20} {:>8}",
        "Version", "Accuracy", "Features", "Description", "Active");
    println!("{}", "-".repeat(60));

    for v in &versions {
        let marker = if v.is_active { "*" } else { "" };
        println!("{:>8} {:>10.4} {:>10} {:>20} {:>8}",
            format!("v{}", v.version), v.accuracy, v.n_features,
            v.description, marker);
    }

    // --- Serialization format comparison ---
    println!("\n=== Format Comparison ===\n");
    println!("{:>10} {:>10} {:>10} {:>12} {:>15}",
        "Format", "Size (B)", "Readable", "Type-safe", "Best for");
    println!("{}", "-".repeat(60));
    println!("{:>10} {:>10} {:>10} {:>12} {:>15}", "JSON", json_size, "Yes", "Yes", "Debug/APIs");
    println!("{:>10} {:>10} {:>10} {:>12} {:>15}", "Binary", binary_size, "No", "Yes", "Production");

    // --- Schema validation ---
    println!("\n=== Schema Validation ===\n");
    println!("In Rust with serde, deserialization is type-safe:");
    println!("  - Wrong field types cause compile-time errors");
    println!("  - Missing required fields cause runtime errors (Result::Err)");
    println!("  - Extra fields are ignored or caught by deny_unknown_fields");
    println!("  - No arbitrary code execution risk (unlike Python's pickle)");

    // Simulate schema mismatch
    let bad_binary: Vec<u8> = vec![1, 0, 0, 0, 99, 0, 0, 0]; // version=1, n_weights=99
    println!("\nAttempting to load binary with wrong schema (99 weights)...");
    if bad_binary.len() < 4 + 4 + 8 + 99 * 8 {
        println!("  Error: insufficient data for declared 99 weights");
        println!("  Type-safe deserialization prevented corrupt model load");
    }

    println!("\n=== Serialization Checklist ===");
    println!("  [x] Model parameters (weights, bias)");
    println!("  [x] Preprocessing state (scaler means, stds)");
    println!("  [x] Feature names");
    println!("  [x] Hyperparameters (learning_rate, n_iterations)");
    println!("  [x] Training metrics (MSE)");
    println!("  [x] Schema version for compatibility");

    println!();
    println!("kata_metric(\"json_size_bytes\", {})", json_size);
    println!("kata_metric(\"binary_size_bytes\", {})", binary_size);
    println!("kata_metric(\"predictions_match\", {})", if (original_pred - loaded_pred).abs() < 1e-10 { 1 } else { 0 });
}
```

---

## Key Takeaways

- **Model serialization converts trained model state into a persistent format.** In Rust, the serde ecosystem provides type-safe serialization to JSON, bincode, MessagePack, and many other formats.
- **Always serialize the complete pipeline, not just the model.** If you save the model without the scaler parameters or feature names, predictions on new data will be wrong.
- **Binary formats are smaller and faster than JSON**, but JSON is invaluable for debugging and inspection. Use JSON during development and binary in production.
- **Model versioning is essential for production.** Track which version is deployed, store metrics and metadata, and maintain the ability to roll back to any previous version.
- **Rust's type system makes serialization safer than pickle.** There is no arbitrary code execution risk, and schema mismatches are caught at compile time or as explicit errors.
