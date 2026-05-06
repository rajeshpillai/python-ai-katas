# Model Serialization

> Phase 11 — Productionizing ML | Kata 11.1

---

## Concept & Intuition

### What problem are we solving?

You have trained a great model. Now what? You cannot retrain it every time you want to make a prediction. You need to **save** the model to disk and **load** it later — in a different script, on a different machine, or in a production server. This is **model serialization**: converting a trained model's state (parameters, hyperparameters, preprocessing steps) into a persistent format.

Python offers several serialization tools, each with trade-offs. **pickle** is Python's built-in serializer — it works with almost anything but is Python-specific and has security risks (loading a pickle file can execute arbitrary code). **joblib** is optimized for numpy arrays, making it faster for sklearn models. **ONNX** (Open Neural Network Exchange) is an open format that works across languages and frameworks — you can train in Python and deploy in C++ or JavaScript. Each format has a place in the ML lifecycle.

Beyond format choice, **model versioning** is critical. As you retrain models with new data or architectures, you need to track which version is deployed, reproduce past results, and roll back if a new model underperforms. Simple conventions like timestamped filenames are a start; tools like MLflow Model Registry provide a more robust solution.

### Why naive approaches fail

Saving just the model weights without metadata (training data version, hyperparameters, feature names, preprocessing steps) makes reproduction impossible. Using pickle without version control means you cannot roll back to a previous model. Deploying a pickle file across Python versions can cause mysterious failures — pickle format is not guaranteed stable across versions. ONNX solves the cross-language problem but requires conversion and may not support every model type.

### Mental models

- **Serialization = freezing the model in amber**: You capture the model at a specific moment, preserving everything needed to make predictions later.
- **pickle/joblib as snapshots**: Fast and easy, but Python-specific. Like saving a Word document — great if everyone uses Word, problematic if they do not.
- **ONNX as a universal translator**: It converts the model into a language-agnostic format. Like exporting to PDF — anyone can read it, anywhere.
- **Versioning as a time machine**: When the new model breaks in production, you need to instantly revert to the previous version. Without versioning, you are stuck.

### Visual explanations

```
Serialization formats:

  Format    Speed   Size   Cross-lang   Security   Best for
  ------    -----   ----   ----------   --------   --------
  pickle    Fast    Med    Python only  Risky      Quick experiments
  joblib    Fast+   Small  Python only  Risky      Sklearn models (numpy-heavy)
  ONNX      Med     Small  Yes          Safe       Production deployment

Model versioning:
  models/
    model_v1_2024-01-15.joblib    <-- initial model
    model_v2_2024-02-20.joblib    <-- retrained with new data
    model_v3_2024-03-10.joblib    <-- new architecture
    metadata.json                  <-- tracks which version is active

  metadata.json:
  {
    "active_version": "v3",
    "versions": {
      "v1": {"accuracy": 0.85, "data": "jan_2024", "created": "2024-01-15"},
      "v2": {"accuracy": 0.87, "data": "feb_2024", "created": "2024-02-20"},
      "v3": {"accuracy": 0.89, "data": "mar_2024", "created": "2024-03-10"}
    }
  }

Complete serialization checklist:
  [ ] Model parameters (weights, coefficients)
  [ ] Hyperparameters (learning_rate, n_estimators, etc.)
  [ ] Preprocessing pipeline (scaler, encoder, feature names)
  [ ] Training metadata (date, dataset version, metrics)
  [ ] Python/library versions (for reproducibility)
```

---

## Hands-on Exploration

1. Train a sklearn model and save it with pickle, joblib, and ONNX. Compare file sizes and save/load times.
2. Try loading a pickle file saved with a different sklearn version. What happens? (Hint: it might break.)
3. Save a complete pipeline (scaler + model) as a single serialized object. Verify that loading and predicting works correctly on new data.
4. Create a simple versioning scheme: save models with timestamps and a metadata JSON file. Implement a function that loads the "active" version.

---

## Live Code

```python
import numpy as np
import pickle
import joblib
import json
import time
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# --- Train a model ---
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

# Create a pipeline: scaler + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"=== Model Serialization ===")
print(f"Model: RandomForest (100 trees, 20 features)")
print(f"Accuracy: {accuracy:.4f}\n")

# --- Create temp directory for saving ---
save_dir = tempfile.mkdtemp()
print(f"Save directory: {save_dir}\n")

# --- Method 1: pickle ---
pickle_path = os.path.join(save_dir, "model.pkl")
t0 = time.time()
with open(pickle_path, 'wb') as f:
    pickle.dump(pipeline, f)
pickle_save_time = time.time() - t0
pickle_size = os.path.getsize(pickle_path)

t0 = time.time()
with open(pickle_path, 'rb') as f:
    loaded_pickle = pickle.load(f)
pickle_load_time = time.time() - t0
pickle_acc = loaded_pickle.score(X_test, y_test)

# --- Method 2: joblib ---
joblib_path = os.path.join(save_dir, "model.joblib")
t0 = time.time()
joblib.dump(pipeline, joblib_path, compress=3)
joblib_save_time = time.time() - t0
joblib_size = os.path.getsize(joblib_path)

t0 = time.time()
loaded_joblib = joblib.load(joblib_path)
joblib_load_time = time.time() - t0
joblib_acc = loaded_joblib.score(X_test, y_test)

# --- Method 3: ONNX (conversion) ---
onnx_available = True
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt

    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    onnx_path = os.path.join(save_dir, "model.onnx")
    t0 = time.time()
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    onnx_save_time = time.time() - t0
    onnx_size = os.path.getsize(onnx_path)

    t0 = time.time()
    sess = rt.InferenceSession(onnx_path)
    onnx_load_time = time.time() - t0

    # Predict with ONNX
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: X_test.astype(np.float32)})[0]
    onnx_acc = np.mean(onnx_preds == y_test)
except ImportError:
    onnx_available = False
    print("Note: skl2onnx or onnxruntime not installed. Skipping ONNX demo.\n")

# --- Comparison ---
print(f"{'Format':>10}  {'Size (KB)':>10}  {'Save (ms)':>10}  {'Load (ms)':>10}  {'Accuracy':>10}")
print("-" * 56)
print(f"{'pickle':>10}  {pickle_size/1024:>10.1f}  {pickle_save_time*1000:>10.1f}  "
      f"{pickle_load_time*1000:>10.1f}  {pickle_acc:>10.4f}")
print(f"{'joblib':>10}  {joblib_size/1024:>10.1f}  {joblib_save_time*1000:>10.1f}  "
      f"{joblib_load_time*1000:>10.1f}  {joblib_acc:>10.4f}")
if onnx_available:
    print(f"{'ONNX':>10}  {onnx_size/1024:>10.1f}  {onnx_save_time*1000:>10.1f}  "
          f"{onnx_load_time*1000:>10.1f}  {onnx_acc:>10.4f}")

# --- Verify loaded models produce identical predictions ---
print(f"\n=== Prediction Consistency ===")
preds_original = pipeline.predict(X_test[:5])
preds_pickle = loaded_pickle.predict(X_test[:5])
preds_joblib = loaded_joblib.predict(X_test[:5])
print(f"Original: {preds_original}")
print(f"Pickle:   {preds_pickle}")
print(f"Joblib:   {preds_joblib}")
print(f"Match:    {np.array_equal(preds_original, preds_pickle) and np.array_equal(preds_original, preds_joblib)}")

# --- Model Versioning ---
print(f"\n=== Model Versioning ===")

def save_model_version(model, save_dir, version, metrics, metadata_extra=None):
    """Save model with versioning metadata."""
    model_path = os.path.join(save_dir, f"model_v{version}.joblib")
    meta_path = os.path.join(save_dir, "metadata.json")

    # Save model
    joblib.dump(model, model_path, compress=3)

    # Load or create metadata
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"active_version": None, "versions": {}}

    # Update metadata
    version_info = {
        "path": model_path,
        "metrics": metrics,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_features": X_train.shape[1],
    }
    if metadata_extra:
        version_info.update(metadata_extra)

    metadata["versions"][f"v{version}"] = version_info
    metadata["active_version"] = f"v{version}"

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return model_path

def load_active_model(save_dir):
    """Load the currently active model version."""
    meta_path = os.path.join(save_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    active = metadata["active_version"]
    model_path = metadata["versions"][active]["path"]
    model = joblib.load(model_path)
    return model, active, metadata["versions"][active]

# Save multiple versions with different hyperparameters
version_dir = os.path.join(save_dir, "versioned")
os.makedirs(version_dir, exist_ok=True)

configs = [
    (50, 0.85, "Small forest"),
    (100, 0.89, "Medium forest"),
    (200, 0.90, "Large forest"),
]

for version, (n_est, acc_sim, desc) in enumerate(configs, 1):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=n_est, random_state=42))
    ])
    model.fit(X_train, y_train)
    actual_acc = model.score(X_test, y_test)

    path = save_model_version(
        model, version_dir, version,
        metrics={"accuracy": actual_acc, "n_estimators": n_est},
        metadata_extra={"description": desc}
    )
    print(f"Saved v{version}: {desc}, accuracy={actual_acc:.4f}")

# Load active version
loaded_model, active_v, info = load_active_model(version_dir)
print(f"\nActive model: {active_v}")
print(f"  Accuracy: {info['metrics']['accuracy']:.4f}")
print(f"  Created: {info['created']}")
print(f"  Description: {info['description']}")

# Show all versions
meta_path = os.path.join(version_dir, "metadata.json")
with open(meta_path, 'r') as f:
    all_meta = json.load(f)

print(f"\n=== All Model Versions ===")
print(f"{'Version':>8}  {'Accuracy':>10}  {'N_est':>6}  {'Description':>15}  {'Active':>8}")
print("-" * 55)
for v, info in all_meta["versions"].items():
    is_active = "*" if v == all_meta["active_version"] else ""
    print(f"{v:>8}  {info['metrics']['accuracy']:>10.4f}  "
          f"{info['metrics']['n_estimators']:>6}  {info['description']:>15}  {is_active:>8}")

# Cleanup
import shutil
shutil.rmtree(save_dir)
print(f"\nCleaned up temp directory.")
```

---

## Key Takeaways

- **pickle and joblib are the simplest serialization options for Python.** joblib is preferred for sklearn models because it handles numpy arrays more efficiently and supports compression.
- **ONNX enables cross-language deployment.** Train in Python, deploy in C++, Java, or JavaScript. It is the standard for production ML systems that need language independence.
- **Always serialize the complete pipeline, not just the model.** If you save the model without the scaler or encoder, predictions on new data will be wrong.
- **Model versioning is essential for production.** Track which version is deployed, store metrics and metadata, and maintain the ability to roll back to any previous version.
- **Never load pickle files from untrusted sources.** Pickle can execute arbitrary code during deserialization — it is a security vulnerability if the source is not trusted.
