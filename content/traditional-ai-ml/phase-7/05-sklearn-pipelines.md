# Sklearn Pipelines

> Phase 7 — Feature Engineering & Pipelines | Kata 7.5

---

## Concept & Intuition

### What problem are we solving?

A typical ML workflow involves multiple steps: impute missing values, scale features, encode categories, select features, then train a model. When these steps are done manually, there are two common problems. First, **data leakage**: if you fit the scaler on the full dataset before splitting, test set statistics leak into the training process. Second, **reproducibility**: keeping track of which transformations were applied in which order, with which parameters, becomes a nightmare when deploying or sharing models.

Sklearn's Pipeline solves both problems by chaining transformation steps into a single object. When you call `pipeline.fit(X_train)`, each step is fit only on the training data. When you call `pipeline.predict(X_test)`, each step transforms (not re-fits) the test data. This guarantees no data leakage and makes the entire workflow reproducible with a single object.

ColumnTransformer extends this idea for heterogeneous data: apply different transformations to different columns. Numerical columns might get imputed and scaled, while categorical columns get one-hot encoded. ColumnTransformer handles this routing automatically, and when placed inside a Pipeline, the entire process is leak-proof and serializable.

### Why naive approaches fail

The most dangerous mistake in ML is fitting a transformer on the entire dataset, then splitting into train/test. For example, if you standardize using the mean and standard deviation of all data, the test set's statistics influence the training process. This makes your cross-validation scores optimistically biased -- your model appears better than it actually is. Pipelines eliminate this by design: `fit_transform` is called only on training data, and `transform` is called on test data, using the parameters learned from training.

### Mental models

- **Assembly line**: raw materials (data) enter one end, pass through stations (imputer, scaler, encoder, model) in order, and a finished product (prediction) comes out the other end. Each station does one job.
- **Sealed black box**: the Pipeline encapsulates the entire workflow. You can pickle it, ship it to production, and it will apply the exact same transformations.
- **Train/test firewall**: the Pipeline enforces a strict separation. Fit parameters are learned only from training data, never from test data.

### Visual explanations

```
Manual Workflow (DANGER: leakage risk):
  scaler.fit(X_all)           # BUG: sees test data!
  X_all_scaled = scaler.transform(X_all)
  X_train, X_test = split(X_all_scaled)
  model.fit(X_train, y_train)

Pipeline Workflow (SAFE):
  pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("model", LogisticRegression())
  ])
  pipe.fit(X_train, y_train)     # scaler fits on X_train only
  pipe.predict(X_test)            # scaler transforms X_test using train params

ColumnTransformer:
  Numeric cols [age, income]  --> Imputer --> Scaler --\
                                                       +--> Concatenate --> Model
  Categ. cols [city, gender]  --> OneHotEncoder -------/
```

---

## Hands-on Exploration

1. Build a manual preprocessing workflow (impute, scale, encode) and intentionally create a data leakage bug. Measure cross-validation accuracy.
2. Rebuild the same workflow using Pipeline. Verify that the cross-validation accuracy is lower (and more honest) than the leaky version.
3. Use ColumnTransformer to handle mixed-type data: numerical columns get imputed and scaled, categorical columns get one-hot encoded.
4. Show that the Pipeline can be serialized (pickled) and reloaded, producing identical predictions on new data.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pickle
import tempfile
import os

np.random.seed(42)

# --- Create a realistic mixed-type dataset ---
n = 500
data = {
    "age": np.random.normal(40, 15, n).clip(18, 80),
    "income": np.random.exponential(50000, n),
    "credit_score": np.random.normal(650, 100, n).clip(300, 850),
    "city": np.random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], n),
    "employment": np.random.choice(["full-time", "part-time", "self-employed", "unemployed"], n),
}
df = pd.DataFrame(data)

# Inject some missing values
for col in ["age", "income", "credit_score"]:
    mask = np.random.random(n) < 0.08
    df.loc[mask, col] = np.nan

# Target: approval based on features
score = (
    0.01 * df["age"].fillna(40)
    + 0.00002 * df["income"].fillna(50000)
    + 0.005 * df["credit_score"].fillna(650)
    + (df["employment"] == "full-time").astype(float) * 0.5
    + np.random.normal(0, 0.5, n)
)
df["approved"] = (score > np.median(score)).astype(int)

print("=== Dataset ===")
print(f"  Shape: {df.shape}")
print(f"  Missing values:\n{df.isnull().sum().to_string()}\n")
print(df.head().to_string())

# --- Define column types ---
numeric_features = ["age", "income", "credit_score"]
categorical_features = ["city", "employment"]

X = df.drop("approved", axis=1)
y = df["approved"].values

# --- WRONG WAY: Manual preprocessing with data leakage ---
print("\n=== WRONG: Manual Preprocessing (data leakage) ===\n")
X_manual = X.copy()
for col in numeric_features:
    mean_val = X_manual[col].mean()  # Uses ALL data (train + test)
    X_manual[col].fillna(mean_val, inplace=True)
    X_manual[col] = (X_manual[col] - X_manual[col].mean()) / X_manual[col].std()

X_manual = pd.get_dummies(X_manual, columns=categorical_features)
scores_leaky = cross_val_score(
    LogisticRegression(max_iter=1000), X_manual, y, cv=5, scoring="accuracy"
)
print(f"  Leaky accuracy: {scores_leaky.mean():.4f} (optimistically biased!)")

# --- RIGHT WAY: Pipeline with ColumnTransformer ---
print("\n=== RIGHT: Pipeline + ColumnTransformer (no leakage) ===\n")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Full pipeline: preprocessing + model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

scores_safe = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"  Safe accuracy:  {scores_safe.mean():.4f} (honest estimate)")
print(f"  Difference:     {(scores_leaky.mean() - scores_safe.mean())*100:.2f} percentage points\n")

# --- Fit the pipeline and inspect ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

print("=== Pipeline Structure ===")
print(pipeline)
print(f"\n  Train accuracy: {pipeline.score(X_train, y_train):.4f}")
print(f"  Test accuracy:  {pipeline.score(X_test, y_test):.4f}")

# Show transformed feature names
cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_features = cat_encoder.named_steps["encoder"].get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_features)
print(f"\n  Transformed features ({len(all_feature_names)}):")
for f in all_feature_names:
    print(f"    {f}")

# --- Serialization: save and reload ---
print("\n=== Serialization ===\n")
tmp_path = os.path.join(tempfile.gettempdir(), "pipeline.pkl")
with open(tmp_path, "wb") as f:
    pickle.dump(pipeline, f)

with open(tmp_path, "rb") as f:
    loaded_pipeline = pickle.load(f)

pred_original = pipeline.predict(X_test)
pred_loaded = loaded_pipeline.predict(X_test)
print(f"  Predictions match after reload: {np.array_equal(pred_original, pred_loaded)}")
print(f"  Saved to: {tmp_path}")
print(f"  File size: {os.path.getsize(tmp_path) / 1024:.1f} KB")

# --- Compare multiple models in pipelines ---
print("\n=== Model Comparison via Pipelines ===\n")
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

model_results = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    model_results[name] = scores.mean()
    print(f"  {name:<25}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Leaky vs safe comparison
axes[0].bar(["Leaky\n(WRONG)", "Pipeline\n(CORRECT)"],
            [scores_leaky.mean(), scores_safe.mean()],
            color=["salmon", "steelblue"],
            yerr=[scores_leaky.std(), scores_safe.std()],
            capsize=10)
axes[0].set_ylabel("CV Accuracy")
axes[0].set_title("Data Leakage Effect")
axes[0].set_ylim(0.5, 0.85)

# Model comparison
axes[1].barh(list(model_results.keys()), list(model_results.values()),
             color="steelblue")
axes[1].set_xlabel("CV Accuracy")
axes[1].set_title("Model Comparison (via Pipeline)")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("sklearn_pipelines.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to sklearn_pipelines.png")
```

---

## Key Takeaways

- **Pipelines prevent data leakage by design.** Fit parameters are learned only from training data, even inside cross-validation loops.
- **ColumnTransformer handles mixed data types.** Route numerical, categorical, and text columns through different transformation pipelines, then concatenate the results.
- **Pipelines are serializable.** Pickle the entire pipeline and deploy it -- all preprocessing and model logic travel together.
- **Cross-validation with pipelines gives honest estimates.** Each fold fits the preprocessor from scratch on the training fold, preventing subtle information leakage.
- **Swapping models is trivial.** Change one line (the final step) to compare Logistic Regression, Random Forest, XGBoost, etc., with identical preprocessing.
