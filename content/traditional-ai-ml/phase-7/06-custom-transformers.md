# Custom Transformers

> Phase 7 — Feature Engineering & Pipelines | Kata 7.6

---

## Concept & Intuition

### What problem are we solving?

Sklearn's built-in transformers (StandardScaler, OneHotEncoder, etc.) cover common operations, but real-world feature engineering often requires custom logic: computing domain-specific ratios, applying business rules, extracting features from structured text, or performing conditional transformations. Custom transformers let you wrap any arbitrary Python function into the sklearn Pipeline ecosystem, gaining all the benefits of pipelines (no data leakage, serialization, reproducibility) for your custom code.

The key interface is the `TransformerMixin` and `BaseEstimator` classes. By inheriting from these and implementing `fit()` and `transform()` methods, your custom class becomes a first-class citizen in sklearn. It can be used in Pipelines, ColumnTransformers, cross-validation, and grid search -- just like any built-in transformer. The `fit()` method learns any necessary parameters from training data (e.g., column means for imputation), and `transform()` applies the transformation using those learned parameters.

For stateless transformations (where `fit()` learns nothing), sklearn provides `FunctionTransformer`, which wraps a plain function into a transformer. This is the quickest way to add custom logic to a pipeline without writing a full class.

### Why naive approaches fail

Writing custom preprocessing as ad-hoc functions outside the pipeline creates the same data leakage risks as any manual approach. If your custom function computes a normalization constant from the full dataset, it leaks test set information into training. By implementing the logic as a proper transformer with separate `fit()` and `transform()` methods, the pipeline ensures the function is fit only on training data and applied consistently to test data.

### Mental models

- **Fit/transform contract**: `fit()` learns from data (like a student studying), `transform()` applies what was learned (like taking the test). The separation is what prevents leakage.
- **Adapter pattern**: your custom logic is the core algorithm; the TransformerMixin interface wraps it so it can plug into sklearn's Pipeline system.
- **Stateful vs stateless**: if your transformation depends on learned parameters (e.g., percentile thresholds), you need `fit()`. If it is pure computation (e.g., log transform), use `FunctionTransformer`.

### Visual explanations

```
Custom Transformer Structure:

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=default):
        self.param1 = param1          # Hyperparameters set in __init__

    def fit(self, X, y=None):
        self.learned_param_ = ...     # Learn from training data
        return self                    # MUST return self

    def transform(self, X):
        # Use self.learned_param_ to transform X
        return X_transformed           # MUST return array-like

Pipeline Usage:
  pipe = Pipeline([
      ("custom", MyTransformer(param1=5)),
      ("scaler", StandardScaler()),
      ("model", LogisticRegression())
  ])

FunctionTransformer (stateless shortcut):
  from sklearn.preprocessing import FunctionTransformer
  log_transformer = FunctionTransformer(np.log1p)
  pipe = Pipeline([("log", log_transformer), ("model", Ridge())])
```

---

## Hands-on Exploration

1. Create a simple stateless transformer using `FunctionTransformer` that applies log1p to all features. Use it in a pipeline and verify it works with cross-validation.
2. Build a custom stateful transformer that clips outliers to the 5th and 95th percentile. The percentiles should be learned from training data in `fit()` and applied in `transform()`.
3. Create a domain-specific transformer that computes interaction features (ratios, products) from specific columns. Integrate it into a ColumnTransformer.
4. Use the custom transformer in a pipeline with GridSearchCV to tune both the transformer's parameters and the model's parameters simultaneously.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer

np.random.seed(42)

# ============================================================
# 1. FunctionTransformer (stateless, quick and easy)
# ============================================================
print("=== 1. FunctionTransformer (stateless) ===\n")

# Log transform for skewed features
log_transformer = FunctionTransformer(np.log1p, validate=True)

# Example: skewed income data
income = np.random.exponential(50000, 200)
income_log = log_transformer.fit_transform(income.reshape(-1, 1))
print(f"  Raw income:  mean={income.mean():.0f}, std={income.std():.0f}, "
      f"skew={pd.Series(income).skew():.2f}")
print(f"  Log income:  mean={income_log.mean():.2f}, std={income_log.std():.2f}, "
      f"skew={pd.Series(income_log.ravel()).skew():.2f}\n")


# ============================================================
# 2. Custom Stateful Transformer: OutlierClipper
# ============================================================
print("=== 2. Custom Stateful Transformer: OutlierClipper ===\n")


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips values to percentile-based bounds learned from training data."""

    def __init__(self, lower_pct=5, upper_pct=95):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_bounds_ = np.percentile(X, self.lower_pct, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_pct, axis=0)
        return self  # Must return self!

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            X[:, j] = np.clip(X[:, j], self.lower_bounds_[j], self.upper_bounds_[j])
        return X


# Demo
X_demo = np.random.normal(0, 1, (100, 3))
X_demo[0, 0] = 100  # outlier
X_demo[1, 1] = -50  # outlier

clipper = OutlierClipper(lower_pct=5, upper_pct=95)
clipper.fit(X_demo)
X_clipped = clipper.transform(X_demo)

print(f"  Before clipping: min={X_demo.min():.1f}, max={X_demo.max():.1f}")
print(f"  After clipping:  min={X_clipped.min():.2f}, max={X_clipped.max():.2f}")
print(f"  Learned bounds:  lower={clipper.lower_bounds_}, upper={clipper.upper_bounds_}\n")


# ============================================================
# 3. Domain-Specific Transformer: FeatureInteractor
# ============================================================
print("=== 3. Domain-Specific Transformer: FeatureInteractor ===\n")


class FeatureInteractor(BaseEstimator, TransformerMixin):
    """Creates interaction features (ratios and products) from specified column pairs."""

    def __init__(self, ratio_pairs=None, product_pairs=None):
        self.ratio_pairs = ratio_pairs or []
        self.product_pairs = product_pairs or []

    def fit(self, X, y=None):
        # Stateless: nothing to learn
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        new_features = [X]

        for i, j in self.ratio_pairs:
            ratio = X[:, i] / (X[:, j] + 1e-8)  # avoid division by zero
            new_features.append(ratio.reshape(-1, 1))

        for i, j in self.product_pairs:
            product = X[:, i] * X[:, j]
            new_features.append(product.reshape(-1, 1))

        return np.hstack(new_features)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = list(input_features)
        for i, j in self.ratio_pairs:
            names.append(f"{input_features[i]}/{input_features[j]}")
        for i, j in self.product_pairs:
            names.append(f"{input_features[i]}*{input_features[j]}")
        return names


# Demo with housing data
n = 300
sqft = np.random.uniform(800, 3000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
age = np.random.uniform(0, 50, n)
X_house = np.column_stack([sqft, bedrooms, bathrooms, age])
y_house = 150 * sqft + 10000 * (sqft / bedrooms) - 2000 * age + np.random.normal(0, 20000, n)

feature_names = ["sqft", "bedrooms", "bathrooms", "age"]

interactor = FeatureInteractor(
    ratio_pairs=[(0, 1), (0, 2)],   # sqft/bedrooms, sqft/bathrooms
    product_pairs=[(0, 3)]            # sqft*age
)

# Pipeline without interactions
pipe_raw = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
scores_raw = cross_val_score(pipe_raw, X_house, y_house, cv=5, scoring="r2")

# Pipeline with interactions
pipe_interact = make_pipeline(
    FeatureInteractor(ratio_pairs=[(0, 1), (0, 2)], product_pairs=[(0, 3)]),
    StandardScaler(),
    Ridge(alpha=1.0)
)
scores_interact = cross_val_score(pipe_interact, X_house, y_house, cv=5, scoring="r2")

print(f"  Raw features (4):     R2 = {scores_raw.mean():.4f}")
print(f"  With interactions (7): R2 = {scores_interact.mean():.4f}")

# Show feature names
interactor.fit(X_house)
print(f"  Feature names: {interactor.get_feature_names_out(feature_names)}\n")


# ============================================================
# 4. GridSearchCV with Custom Transformer Parameters
# ============================================================
print("=== 4. GridSearchCV with Custom Transformer ===\n")

pipe_grid = Pipeline([
    ("clipper", OutlierClipper()),
    ("interactor", FeatureInteractor()),
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

param_grid = {
    "clipper__lower_pct": [1, 5, 10],
    "clipper__upper_pct": [90, 95, 99],
    "interactor__ratio_pairs": [[], [(0, 1)], [(0, 1), (0, 2)]],
    "model__alpha": [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipe_grid, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_house, y_house)

print(f"  Best R2: {grid.best_score_:.4f}")
print(f"  Best params:")
for k, v in grid.best_params_.items():
    print(f"    {k}: {v}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Outlier clipping effect
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Add artificial outliers
X_train_noisy = X_train_h.copy()
X_train_noisy[:5, 0] *= 10  # extreme sqft outliers

pipe_no_clip = make_pipeline(StandardScaler(), Ridge())
pipe_no_clip.fit(X_train_noisy, y_train_h)
pred_no_clip = pipe_no_clip.predict(X_test_h)

pipe_clip = make_pipeline(OutlierClipper(), StandardScaler(), Ridge())
pipe_clip.fit(X_train_noisy, y_train_h)
pred_clip = pipe_clip.predict(X_test_h)

from sklearn.metrics import mean_squared_error
mse_no = mean_squared_error(y_test_h, pred_no_clip)
mse_clip = mean_squared_error(y_test_h, pred_clip)

axes[0].bar(["No Clipping", "OutlierClipper"], [mse_no, mse_clip],
            color=["salmon", "steelblue"])
axes[0].set_ylabel("Test MSE")
axes[0].set_title("Effect of Custom OutlierClipper")

# Feature interaction effect
axes[1].bar(["Raw Features\n(4 features)", "With Interactions\n(7 features)"],
            [scores_raw.mean(), scores_interact.mean()],
            color=["steelblue", "coral"])
axes[1].set_ylabel("R-squared (CV)")
axes[1].set_title("Effect of Custom FeatureInteractor")
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("custom_transformers.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to custom_transformers.png")
```

---

## Key Takeaways

- **Custom transformers wrap any Python logic into the sklearn ecosystem.** Inherit from BaseEstimator and TransformerMixin, implement fit() and transform(), and you are done.
- **The fit/transform separation prevents data leakage.** Parameters learned in fit() come only from training data, even for your custom logic.
- **FunctionTransformer is the shortcut for stateless operations.** No class needed -- wrap any function and use it in a pipeline.
- **Custom transformer parameters are tunable via GridSearchCV.** The naming convention "step_name__param_name" gives you free hyperparameter tuning for your custom logic.
- **Always return self from fit() and return array-like from transform().** These two rules are the entire contract that makes custom transformers work in pipelines.
