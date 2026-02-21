# Experiment Tracking

> Phase 11 — Productionizing ML | Kata 11.3

---

## Concept & Intuition

### What problem are we solving?

During model development, you run dozens or hundreds of experiments: different hyperparameters, different features, different algorithms. Without systematic tracking, you lose track of which experiment produced which results. You cannot reproduce your best model because you forgot the exact settings. This is the **reproducibility crisis** in ML, and **experiment tracking** solves it.

Experiment tracking records everything about each run: hyperparameters, metrics, code version, data version, artifacts (saved models, plots), and environment details. In Python, tools like MLflow, Weights & Biases, and Neptune provide this. In Rust, you would build a lightweight tracking system using structured logging, JSON metadata files, and a registry of runs.

Good experiment tracking answers questions like: "What hyperparameters gave the best validation accuracy?" "How did performance change when I added feature X?" "Can I reproduce the model from three weeks ago?" Without tracking, these questions are unanswerable.

### Why naive approaches fail

Relying on terminal output, Jupyter notebooks, or spreadsheets for tracking is fragile and incomplete. Terminal output gets lost when the session ends. Notebooks are not designed for systematic comparison across runs. Spreadsheets require manual entry and are error-prone. Automated tracking captures everything consistently, enables filtering and comparison, and integrates with model deployment pipelines.

### Mental models

- **Experiment tracking as a lab notebook**: scientists record every detail of every experiment. ML engineers should do the same, but automatically.
- **Runs as rows in a database**: each experiment is a row, with columns for hyperparameters and metrics. You query and sort to find the best configuration.
- **Reproducibility as insurance**: you will always need to re-run or explain past results. Tracking is the insurance policy that makes this possible.

### Visual explanations

```
Experiment Tracking System:

  Run ID | Algorithm  | LR     | Reg    | Val Acc | Test Acc | Status
  -------|------------|--------|--------|---------|----------|--------
  run_01 | LinReg     | 0.01   | 0.0    | 0.82    | 0.81     | done
  run_02 | LinReg     | 0.01   | 0.1    | 0.84    | 0.83     | done
  run_03 | Ridge      | 0.001  | 1.0    | 0.85    | 0.84     | done  <-- best
  run_04 | GBT        | 0.1    | 0.5    | 0.87    | 0.82     | overfit
  run_05 | GBT        | 0.05   | 1.0    | 0.86    | 0.85     | done

Each run stores:
  - params: {algorithm, learning_rate, regularization, ...}
  - metrics: {train_loss, val_accuracy, test_accuracy, ...}
  - artifacts: {model.bin, feature_importance.json, ...}
  - metadata: {timestamp, git_commit, data_version, ...}

Queries:
  "Best val_accuracy where algorithm=Ridge"  --> run_03
  "All runs where test_acc > 0.83"          --> run_03, run_05
  "Compare run_02 vs run_03"                --> +0.01 acc for +0.9 reg
```

---

## Hands-on Exploration

1. Build a simple experiment tracker that records hyperparameters and metrics for each run.
2. Run multiple experiments with different configurations. Use the tracker to find the best configuration.
3. Implement comparison: show how changing one hyperparameter affects metrics across runs.
4. Add metadata (timestamp, run ID) and artifact tracking (model parameters).

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

    println!("=== Experiment Tracking ===\n");

    // --- Simple experiment tracker ---
    struct Param {
        name: String,
        value: String,
    }

    struct Metric {
        name: String,
        value: f64,
    }

    struct ExperimentRun {
        run_id: String,
        params: Vec<Param>,
        metrics: Vec<Metric>,
        status: String,
        notes: String,
    }

    impl ExperimentRun {
        fn get_param(&self, name: &str) -> Option<&str> {
            self.params.iter().find(|p| p.name == name).map(|p| p.value.as_str())
        }
        fn get_metric(&self, name: &str) -> Option<f64> {
            self.metrics.iter().find(|m| m.name == name).map(|m| m.value)
        }
    }

    struct ExperimentTracker {
        experiment_name: String,
        runs: Vec<ExperimentRun>,
        next_id: usize,
    }

    impl ExperimentTracker {
        fn new(name: &str) -> Self {
            ExperimentTracker {
                experiment_name: name.to_string(),
                runs: Vec::new(),
                next_id: 1,
            }
        }

        fn log_run(&mut self, params: Vec<(&str, &str)>, metrics: Vec<(&str, f64)>,
                    status: &str, notes: &str) -> String {
            let run_id = format!("run_{:02}", self.next_id);
            self.next_id += 1;

            self.runs.push(ExperimentRun {
                run_id: run_id.clone(),
                params: params.iter().map(|(n, v)| Param {
                    name: n.to_string(), value: v.to_string()
                }).collect(),
                metrics: metrics.iter().map(|(n, v)| Metric {
                    name: n.to_string(), value: *v
                }).collect(),
                status: status.to_string(),
                notes: notes.to_string(),
            });
            run_id
        }

        fn best_run(&self, metric: &str) -> Option<&ExperimentRun> {
            self.runs.iter()
                .filter(|r| r.status == "done")
                .max_by(|a, b| {
                    let va = a.get_metric(metric).unwrap_or(f64::NEG_INFINITY);
                    let vb = b.get_metric(metric).unwrap_or(f64::NEG_INFINITY);
                    va.partial_cmp(&vb).unwrap()
                })
        }

        fn filter_runs(&self, param: &str, value: &str) -> Vec<&ExperimentRun> {
            self.runs.iter()
                .filter(|r| r.get_param(param) == Some(value))
                .collect()
        }
    }

    // --- Generate dataset ---
    let n_feat = 10;
    let n_train = 150;
    let n_test = 50;
    let true_weights: Vec<f64> = (0..n_feat).map(|i| if i < 5 { 1.0 - 0.2 * i as f64 } else { 0.0 }).collect();

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    let mut x_test: Vec<Vec<f64>> = Vec::new();
    let mut y_test: Vec<f64> = Vec::new();

    for _ in 0..n_train {
        let row: Vec<f64> = (0..n_feat).map(|_| rand_normal(&mut rng)).collect();
        let y: f64 = true_weights.iter().zip(&row).map(|(w, x)| w * x).sum::<f64>()
            + rand_normal(&mut rng) * 0.5;
        x_train.push(row);
        y_train.push(y);
    }
    for _ in 0..n_test {
        let row: Vec<f64> = (0..n_feat).map(|_| rand_normal(&mut rng)).collect();
        let y: f64 = true_weights.iter().zip(&row).map(|(w, x)| w * x).sum::<f64>()
            + rand_normal(&mut rng) * 0.5;
        x_test.push(row);
        y_test.push(y);
    }

    // --- Train models with different configurations ---
    fn train_ridge(x: &[Vec<f64>], y: &[f64], lr: f64, lambda: f64, n_iter: usize) -> Vec<f64> {
        let n = x.len();
        let n_feat = x[0].len();
        let mut w = vec![0.0; n_feat];

        for _ in 0..n_iter {
            let mut grad = vec![0.0; n_feat];
            for i in 0..n {
                let pred: f64 = w.iter().zip(&x[i]).map(|(wi, xi)| wi * xi).sum();
                let err = pred - y[i];
                for j in 0..n_feat {
                    grad[j] += err * x[i][j];
                }
            }
            for j in 0..n_feat {
                grad[j] = grad[j] / n as f64 + lambda * w[j];
                w[j] -= lr * grad[j];
            }
        }
        w
    }

    fn mse(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
        let n = x.len() as f64;
        x.iter().zip(y).map(|(row, &yi)| {
            let pred: f64 = w.iter().zip(row).map(|(wi, xi)| wi * xi).sum();
            (pred - yi).powi(2)
        }).sum::<f64>() / n
    }

    fn r_squared(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y).map(|(row, &yi)| {
            let pred: f64 = w.iter().zip(row).map(|(wi, xi)| wi * xi).sum();
            (yi - pred).powi(2)
        }).sum();
        1.0 - ss_res / ss_tot
    }

    let mut tracker = ExperimentTracker::new("ridge_regression_tuning");

    // Run experiments with different configurations
    let configs: Vec<(&str, f64, f64, usize)> = vec![
        ("LinReg",  0.01,  0.0,   500),
        ("LinReg",  0.001, 0.0,   1000),
        ("Ridge",   0.01,  0.01,  500),
        ("Ridge",   0.01,  0.1,   500),
        ("Ridge",   0.01,  1.0,   500),
        ("Ridge",   0.01,  10.0,  500),
        ("Ridge",   0.001, 0.1,   1000),
        ("Ridge",   0.001, 1.0,   1000),
    ];

    for (algo, lr, lambda, n_iter) in &configs {
        let w = train_ridge(&x_train, &y_train, *lr, *lambda, *n_iter);

        let train_mse = mse(&x_train, &y_train, &w);
        let test_mse = mse(&x_test, &y_test, &w);
        let train_r2 = r_squared(&x_train, &y_train, &w);
        let test_r2 = r_squared(&x_test, &y_test, &w);
        let w_norm: f64 = w.iter().map(|wi| wi.powi(2)).sum::<f64>().sqrt();

        let status = if test_mse > train_mse * 2.0 { "overfit" } else { "done" };

        tracker.log_run(
            vec![
                ("algorithm", algo),
                ("learning_rate", &format!("{}", lr)),
                ("lambda", &format!("{}", lambda)),
                ("n_iter", &format!("{}", n_iter)),
            ],
            vec![
                ("train_mse", train_mse),
                ("test_mse", test_mse),
                ("train_r2", train_r2),
                ("test_r2", test_r2),
                ("weight_norm", w_norm),
            ],
            status,
            "",
        );
    }

    // --- Display all runs ---
    println!("=== All Experiment Runs ===\n");
    println!("{:>8} {:>8} {:>6} {:>6} {:>6} {:>10} {:>10} {:>8} {:>8}",
        "Run", "Algo", "LR", "Lam", "Iter", "Train MSE", "Test MSE", "Test R2", "Status");
    println!("{}", "-".repeat(78));

    for run in &tracker.runs {
        println!("{:>8} {:>8} {:>6} {:>6} {:>6} {:>10.4} {:>10.4} {:>8.4} {:>8}",
            run.run_id,
            run.get_param("algorithm").unwrap_or("-"),
            run.get_param("learning_rate").unwrap_or("-"),
            run.get_param("lambda").unwrap_or("-"),
            run.get_param("n_iter").unwrap_or("-"),
            run.get_metric("train_mse").unwrap_or(0.0),
            run.get_metric("test_mse").unwrap_or(0.0),
            run.get_metric("test_r2").unwrap_or(0.0),
            run.status);
    }

    // --- Find best run ---
    if let Some(best) = tracker.best_run("test_r2") {
        println!("\n=== Best Run (by test R2) ===\n");
        println!("Run ID: {}", best.run_id);
        println!("Parameters:");
        for p in &best.params {
            println!("  {}: {}", p.name, p.value);
        }
        println!("Metrics:");
        for m in &best.metrics {
            println!("  {}: {:.6}", m.name, m.value);
        }
    }

    // --- Compare effect of regularization ---
    println!("\n=== Effect of Regularization (lambda) ===\n");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Lambda", "Train MSE", "Test MSE", "Test R2", "||w||");
    println!("{}", "-".repeat(52));

    let ridge_runs = tracker.filter_runs("algorithm", "Ridge");
    let mut lr_01_runs: Vec<&&ExperimentRun> = ridge_runs.iter()
        .filter(|r| r.get_param("learning_rate") == Some("0.01"))
        .collect();
    lr_01_runs.sort_by(|a, b| {
        let la: f64 = a.get_param("lambda").unwrap().parse().unwrap();
        let lb: f64 = b.get_param("lambda").unwrap().parse().unwrap();
        la.partial_cmp(&lb).unwrap()
    });

    for run in lr_01_runs {
        println!("{:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.3}",
            run.get_param("lambda").unwrap(),
            run.get_metric("train_mse").unwrap_or(0.0),
            run.get_metric("test_mse").unwrap_or(0.0),
            run.get_metric("test_r2").unwrap_or(0.0),
            run.get_metric("weight_norm").unwrap_or(0.0));
    }

    println!("\nAs lambda increases: train MSE rises (underfitting), weight norm decreases (regularization).");
    println!("Optimal lambda balances bias and variance.");

    // --- Serialize experiment log (simulated JSON) ---
    println!("\n=== Experiment Log (JSON format) ===\n");
    println!("{{");
    println!("  \"experiment\": \"{}\",", tracker.experiment_name);
    println!("  \"n_runs\": {},", tracker.runs.len());
    println!("  \"runs\": [");
    for (i, run) in tracker.runs.iter().enumerate() {
        let comma = if i < tracker.runs.len() - 1 { "," } else { "" };
        println!("    {{\"id\": \"{}\", \"algo\": \"{}\", \"lambda\": {}, \"test_r2\": {:.4}}}{}",
            run.run_id,
            run.get_param("algorithm").unwrap_or("-"),
            run.get_param("lambda").unwrap_or("0"),
            run.get_metric("test_r2").unwrap_or(0.0),
            comma);
    }
    println!("  ]");
    println!("}}");

    // --- Reproducibility checklist ---
    println!("\n=== Reproducibility Checklist ===\n");
    println!("  [x] Hyperparameters logged for each run");
    println!("  [x] Train and test metrics recorded");
    println!("  [x] Model status tracked (done, overfit, failed)");
    println!("  [x] Experiment comparison and filtering");
    println!("  [ ] Git commit hash (production systems)");
    println!("  [ ] Data version hash (production systems)");
    println!("  [ ] Environment info (Rust version, platform)");
    println!("  [ ] Random seed for exact reproducibility");

    let best_test_r2 = tracker.best_run("test_r2")
        .and_then(|r| r.get_metric("test_r2")).unwrap_or(0.0);
    let n_runs = tracker.runs.len();

    println!();
    println!("kata_metric(\"n_experiments\", {})", n_runs);
    println!("kata_metric(\"best_test_r2\", {:.4})", best_test_r2);
    let best_id = tracker.best_run("test_r2").map(|r| r.run_id.clone()).unwrap_or_default();
    println!("kata_metric(\"best_run_id\", \"{}\")", best_id);
}
```

---

## Key Takeaways

- **Experiment tracking is essential for reproducibility.** Without systematic logging, you cannot reproduce your best model or explain past results.
- **Track hyperparameters, metrics, and metadata for every run.** This enables filtering, comparison, and systematic analysis of what works and why.
- **Automated tracking beats manual logging.** Manual spreadsheets are error-prone and incomplete. Automated systems capture everything consistently.
- **Comparison across runs reveals insights.** Understanding how changing one hyperparameter affects multiple metrics guides model development more efficiently than random search.
