# Experiment Tracking

> Phase 11 — Productionizing ML | Kata 11.3

---

## Concept & Intuition

### What problem are we solving?

Machine learning development is inherently experimental. You try different features, hyperparameters, model architectures, and preprocessing steps. After a week of experiments, you find a great result — but which exact combination of settings produced it? Without systematic tracking, you will forget. You will struggle to reproduce results, compare approaches, or explain to stakeholders why you chose one model over another.

**Experiment tracking** solves this by logging every detail of every run: parameters, metrics, artifacts (model files, plots), code versions, and environment details. **MLflow** is the most widely adopted open-source tool for this. It provides a tracking API to log experiments, a UI to compare them, and a model registry to manage the lifecycle from development to production.

Even without MLflow, the discipline of experiment tracking matters. At minimum, you need a structured log of what you tried, what happened, and what you concluded. A spreadsheet works for small projects; a proper tracking system becomes essential as the team and project grow.

### Why naive approaches fail

Without tracking, ML development devolves into a mess of Jupyter notebooks with cryptic names ("model_v3_final_FINAL_v2.ipynb"), scattered results in Slack messages, and no reproducibility. When a stakeholder asks "why did you choose this model?", you scramble through old notebooks. When you need to reproduce a result from 3 months ago, the environment has changed and nothing works. Experiment tracking prevents this chaos.

### Mental models

- **Lab notebook for ML**: Just as a chemist records every experiment, an ML practitioner should log every run. The hypothesis, the procedure, and the results.
- **MLflow as a filing cabinet**: Each run is a folder with its parameters, metrics, and artifacts. The UI lets you search, sort, and compare across folders.
- **Model registry as a promotion pipeline**: Models move through stages (None -> Staging -> Production -> Archived), with approvals at each step.

### Visual explanations

```
Experiment tracking structure:

  Experiment: "churn_prediction"
    |
    +-- Run 1: RF, n_estimators=100, acc=0.85
    |     params: {n_estimators: 100, max_depth: 10}
    |     metrics: {accuracy: 0.85, f1: 0.78, auc: 0.91}
    |     artifacts: [model.joblib, confusion_matrix.png]
    |
    +-- Run 2: RF, n_estimators=200, acc=0.87
    |     params: {n_estimators: 200, max_depth: 15}
    |     metrics: {accuracy: 0.87, f1: 0.81, auc: 0.93}
    |     artifacts: [model.joblib, confusion_matrix.png]
    |
    +-- Run 3: XGBoost, acc=0.89
          params: {n_estimators: 150, learning_rate: 0.1}
          metrics: {accuracy: 0.89, f1: 0.84, auc: 0.95}
          artifacts: [model.joblib, confusion_matrix.png]

  Compare runs:
  +----------+--------+------+------+------+
  | Run      | Model  | Acc  | F1   | AUC  |
  +----------+--------+------+------+------+
  | Run 1    | RF     | 0.85 | 0.78 | 0.91 |
  | Run 2    | RF     | 0.87 | 0.81 | 0.93 |
  | Run 3    | XGBoost| 0.89 | 0.84 | 0.95 | <-- winner
  +----------+--------+------+------+------+

Model Registry stages:
  Development --> Staging --> Production --> Archived
                  (testing)   (serving)    (retired)
```

---

## Hands-on Exploration

1. Train 3 models with different hyperparameters. For each, manually record: parameters, accuracy, F1, and training time. How tedious is this?
2. Implement a simple tracking system using a dictionary and JSON file. Log each run automatically.
3. If MLflow is available, use its tracking API to log the same experiments. Launch the UI and compare runs.
4. Implement a basic model registry: save the best model with a "production" tag. When a better model appears, promote it and archive the old one.

---

## Live Code

```python
import numpy as np
import json
import time
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

np.random.seed(42)

# --- Generate dataset ---
X, y = make_classification(n_samples=1000, n_features=15, n_informative=10,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

# --- Simple Experiment Tracker (no external dependencies) ---
class ExperimentTracker:
    """A lightweight experiment tracker that logs to JSON."""

    def __init__(self, experiment_name, base_dir=None):
        self.experiment_name = experiment_name
        self.base_dir = base_dir or tempfile.mkdtemp()
        self.exp_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_path = os.path.join(self.exp_dir, "experiment_log.json")
        self.runs = self._load_log()
        self.current_run = None

    def _load_log(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return []

    def _save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.runs, f, indent=2)

    def start_run(self, run_name):
        self.current_run = {
            "run_name": run_name,
            "run_id": len(self.runs) + 1,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "status": "running",
        }
        return self

    def log_param(self, key, value):
        self.current_run["params"][key] = value

    def log_params(self, params_dict):
        self.current_run["params"].update(params_dict)

    def log_metric(self, key, value):
        self.current_run["metrics"][key] = round(value, 6)

    def log_metrics(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.log_metric(k, v)

    def log_artifact(self, obj, filename):
        artifact_dir = os.path.join(self.exp_dir, f"run_{self.current_run['run_id']}")
        os.makedirs(artifact_dir, exist_ok=True)
        path = os.path.join(artifact_dir, filename)
        joblib.dump(obj, path)
        self.current_run["artifacts"].append(filename)
        return path

    def end_run(self):
        self.current_run["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.current_run["status"] = "completed"
        self.runs.append(self.current_run)
        self._save_log()
        run = self.current_run
        self.current_run = None
        return run

    def compare_runs(self, metric="accuracy"):
        """Display all runs sorted by a metric."""
        sorted_runs = sorted(self.runs,
                             key=lambda r: r["metrics"].get(metric, 0),
                             reverse=True)
        return sorted_runs

    def get_best_run(self, metric="accuracy"):
        sorted_runs = self.compare_runs(metric)
        return sorted_runs[0] if sorted_runs else None


# --- Run experiments ---
tracker = ExperimentTracker("churn_prediction")

experiments = [
    {
        "name": "logistic_regression",
        "model_class": LogisticRegression,
        "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    },
    {
        "name": "random_forest_small",
        "model_class": RandomForestClassifier,
        "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
    },
    {
        "name": "random_forest_large",
        "model_class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 15, "random_state": 42},
    },
    {
        "name": "gradient_boosting",
        "model_class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                    "random_state": 42},
    },
]

print("=== Experiment Tracking ===\n")

for exp in experiments:
    # Start tracking
    tracker.start_run(exp["name"])
    tracker.log_param("model_type", exp["model_class"].__name__)
    tracker.log_params(exp["params"])

    # Train
    t0 = time.time()
    model = exp["model_class"](**exp["params"])
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Log metrics
    tracker.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "train_time_sec": train_time,
    })

    # Log model artifact
    tracker.log_artifact(model, "model.joblib")

    run = tracker.end_run()
    print(f"Run: {run['run_name']:>25s} | Acc: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# --- Compare all runs ---
print(f"\n=== Run Comparison (sorted by AUC) ===")
print(f"{'Run Name':>25}  {'Accuracy':>9}  {'F1':>7}  {'AUC':>7}  {'CV Mean':>8}  {'Time(s)':>8}")
print("-" * 72)

for run in tracker.compare_runs(metric="auc_roc"):
    m = run["metrics"]
    print(f"{run['run_name']:>25}  {m['accuracy']:>9.4f}  {m['f1_score']:>7.4f}  "
          f"{m['auc_roc']:>7.4f}  {m['cv_mean']:>8.4f}  {m['train_time_sec']:>8.3f}")

# --- Best run ---
best = tracker.get_best_run(metric="auc_roc")
print(f"\nBest run: {best['run_name']}")
print(f"  Parameters: {best['params']}")
print(f"  AUC-ROC: {best['metrics']['auc_roc']:.4f}")

# --- Simple Model Registry ---
print(f"\n=== Model Registry ===")

class ModelRegistry:
    """Simple model registry with staging workflow."""

    def __init__(self, registry_dir):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.registry_path = os.path.join(registry_dir, "registry.json")
        self.registry = self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register(self, name, model, metrics, stage="development"):
        version = len(self.registry["models"].get(name, {}).get("versions", [])) + 1
        model_path = os.path.join(self.registry_dir, f"{name}_v{version}.joblib")
        joblib.dump(model, model_path)

        if name not in self.registry["models"]:
            self.registry["models"][name] = {"versions": []}

        self.registry["models"][name]["versions"].append({
            "version": version,
            "stage": stage,
            "metrics": metrics,
            "path": model_path,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save()
        return version

    def promote(self, name, version, new_stage):
        versions = self.registry["models"][name]["versions"]
        # If promoting to production, archive current production model
        if new_stage == "production":
            for v in versions:
                if v["stage"] == "production":
                    v["stage"] = "archived"
        versions[version - 1]["stage"] = new_stage
        self._save()

    def get_production_model(self, name):
        versions = self.registry["models"][name]["versions"]
        for v in versions:
            if v["stage"] == "production":
                return joblib.load(v["path"]), v
        return None, None

    def list_versions(self, name):
        return self.registry["models"].get(name, {}).get("versions", [])


registry = ModelRegistry(os.path.join(tracker.base_dir, "registry"))

# Register models from our experiments
for run in tracker.runs:
    model_type = run["params"]["model_type"]
    # Retrain to get model object (in practice, load from artifact)
    exp = next(e for e in experiments if e["name"] == run["run_name"])
    model = exp["model_class"](**exp["params"])
    model.fit(X_train, y_train)
    version = registry.register("churn_model", model, run["metrics"])
    print(f"Registered: churn_model v{version} ({run['run_name']})")

# Promote best to staging, then production
registry.promote("churn_model", 4, "staging")
print(f"\nPromoted v4 to staging")
registry.promote("churn_model", 4, "production")
print(f"Promoted v4 to production")

# Show registry
print(f"\n{'Version':>8}  {'Stage':>12}  {'AUC':>7}  {'Accuracy':>9}")
print("-" * 40)
for v in registry.list_versions("churn_model"):
    print(f"v{v['version']:>7}  {v['stage']:>12}  {v['metrics']['auc_roc']:>7.4f}  {v['metrics']['accuracy']:>9.4f}")

# Load production model
prod_model, prod_info = registry.get_production_model("churn_model")
if prod_model:
    prod_acc = accuracy_score(y_test, prod_model.predict(X_test))
    print(f"\nProduction model: v{prod_info['version']}, accuracy: {prod_acc:.4f}")

# Cleanup
import shutil
shutil.rmtree(tracker.base_dir)
```

---

## Key Takeaways

- **Experiment tracking is non-negotiable for serious ML work.** Without it, you cannot reproduce results, compare approaches, or explain decisions to stakeholders.
- **Log everything: parameters, metrics, artifacts, and environment.** The cost of logging is tiny compared to the cost of losing a good result because you cannot remember how you got it.
- **A model registry manages the lifecycle from experiment to production.** Models move through stages (development, staging, production, archived) with clear ownership and versioning.
- **Even a simple JSON-based tracker beats no tracking at all.** Start simple; graduate to MLflow or similar tools as the project grows.
- **Comparison across runs reveals what actually matters.** Often, the big wins come from data preparation or feature engineering, not from model architecture — and you can only see this if you track systematically.
