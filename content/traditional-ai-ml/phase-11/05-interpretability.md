# Interpretability

> Phase 11 — Productionizing ML | Kata 11.5

---

## Concept & Intuition

### What problem are we solving?

A model predicts that a loan applicant will default. Should the bank deny the loan? Before making that decision, the bank needs to know **why** the model made that prediction. Was it the applicant's income? Their credit history? Their zip code (which could be a proxy for race)? **Model interpretability** answers "why" — explaining which features drove a prediction and how.

There are two families of interpretability methods. **Global interpretability** explains the model as a whole: which features are generally most important, and how does each feature affect predictions on average? **Local interpretability** explains individual predictions: why did the model predict X for this specific person?

**SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) are the two most widely used model-agnostic methods. SHAP values come from game theory — they fairly distribute the "credit" for a prediction among features, with strong theoretical guarantees. LIME trains a simple interpretable model (like linear regression) locally around a specific prediction. Both are model-agnostic, meaning they work with any model: random forests, gradient boosting, neural networks, or anything else.

**Partial dependence plots (PDPs)** show the average relationship between a feature and the prediction, marginalizing over all other features. **Feature importance** ranks features by their overall contribution to model performance.

### Why naive approaches fail

Using a model's built-in feature importance (like random forest's `feature_importances_`) can be misleading — it measures impurity reduction, which is biased toward high-cardinality features and does not account for feature interactions. Trying to interpret a complex model by reading its rules or weights is hopeless for models with hundreds of features and thousands of trees. SHAP and LIME provide principled, model-agnostic explanations that work regardless of model complexity.

### Mental models

- **SHAP as fair credit assignment**: Imagine features as players on a team. SHAP values tell you each player's contribution to the team's score, accounting for all possible combinations of players.
- **LIME as local approximation**: Zoom in on a single prediction. Nearby, even a complex model looks approximately linear. LIME finds that local linear approximation.
- **PDP as a "what-if" analysis**: Hold everything else constant. What happens to the prediction as we vary just this one feature?

### Visual explanations

```
SHAP Values for a loan prediction:

  Feature          Value    SHAP Value   Effect
  ─────────        ─────    ──────────   ──────
  Income           $45K     -0.15        pushes toward "repay"
  Credit Score     580      +0.25        pushes toward "default"
  Debt-to-Income   0.45     +0.10        pushes toward "default"
  Employment       3 yrs    -0.05        pushes toward "repay"
  ─────────────────────────────────────────────
  Base prediction: 0.30
  Final prediction: 0.30 + (-0.15) + 0.25 + 0.10 + (-0.05) = 0.45

  SHAP values are additive: base_value + sum(SHAP) = prediction

Partial Dependence Plot:

  Prediction
      |
  0.8 |                          ****
      |                     *****
  0.6 |                *****
      |           *****
  0.4 |      *****
      |  ****
  0.2 |***
      +--+--+--+--+--+--+--+--+--> Credit Score
       500  550  600  650  700  750

  Interpretation: As credit score increases, default probability decreases.
  The relationship is roughly monotonic but nonlinear.
```

---

## Hands-on Exploration

1. Train a random forest on a classification problem. Look at `feature_importances_`. Now compute permutation importance. Do the rankings differ?
2. Compute SHAP values for a single prediction. Which feature pushed the prediction up? Which pushed it down? Do the SHAP values sum to the correct prediction?
3. Create a partial dependence plot for the most important feature. Is the relationship linear, monotonic, or more complex?
4. Use LIME to explain the same prediction you explained with SHAP. Do they agree on the most important features?

---

## Live Code

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, partial_dependence

np.random.seed(42)

# --- Generate a realistic dataset ---
n_samples = 1000
n_features = 8
feature_names = ['income', 'credit_score', 'debt_ratio', 'employment_years',
                 'num_accounts', 'recent_inquiries', 'loan_amount', 'age']

X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_informative=5, n_redundant=1, n_clusters_per_class=2,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

# Train a gradient boosting model
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"=== Model Interpretability ===")
print(f"Model: GradientBoosting, Accuracy: {accuracy:.4f}\n")

# --- Method 1: Built-in Feature Importance (Impurity-based) ---
print("=== 1. Impurity-Based Feature Importance ===")
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"{'Rank':>5}  {'Feature':>20}  {'Importance':>12}  {'Bar':>20}")
print("-" * 62)
for rank, idx in enumerate(sorted_idx):
    bar = '#' * int(importances[idx] * 50)
    print(f"{rank+1:>5}  {feature_names[idx]:>20}  {importances[idx]:>12.4f}  {bar}")

# --- Method 2: Permutation Importance ---
print(f"\n=== 2. Permutation Importance ===")
perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                   random_state=42)

sorted_idx_perm = np.argsort(perm_imp.importances_mean)[::-1]
print(f"{'Rank':>5}  {'Feature':>20}  {'Mean':>8}  {'Std':>8}")
print("-" * 45)
for rank, idx in enumerate(sorted_idx_perm):
    print(f"{rank+1:>5}  {feature_names[idx]:>20}  "
          f"{perm_imp.importances_mean[idx]:>8.4f}  {perm_imp.importances_std[idx]:>8.4f}")

# --- Method 3: SHAP Values (manual TreeSHAP approximation) ---
print(f"\n=== 3. SHAP Values (Approximation) ===")

def approximate_shap(model, X_background, x_instance, n_samples=200):
    """Approximate SHAP values using sampling-based approach."""
    n_features = X_background.shape[1]
    shap_values = np.zeros(n_features)

    base_pred = model.predict_proba(X_background)[:, 1].mean()

    for feature in range(n_features):
        marginal_contributions = []
        for _ in range(n_samples):
            # Random reference point
            ref_idx = np.random.randint(len(X_background))
            ref = X_background[ref_idx].copy()

            # Random subset of other features to include
            other_features = [f for f in range(n_features) if f != feature]
            n_include = np.random.randint(0, len(other_features) + 1)
            included = np.random.choice(other_features, size=n_include, replace=False) if n_include > 0 else []

            # With the feature
            x_with = ref.copy()
            x_with[feature] = x_instance[feature]
            for f in included:
                x_with[f] = x_instance[f]

            # Without the feature
            x_without = ref.copy()
            for f in included:
                x_without[f] = x_instance[f]

            pred_with = model.predict_proba(x_with.reshape(1, -1))[0, 1]
            pred_without = model.predict_proba(x_without.reshape(1, -1))[0, 1]
            marginal_contributions.append(pred_with - pred_without)

        shap_values[feature] = np.mean(marginal_contributions)

    return shap_values, base_pred

# Explain a specific prediction
sample_idx = 0
x_sample = X_test[sample_idx]
actual_pred = model.predict_proba(x_sample.reshape(1, -1))[0, 1]

shap_vals, base_value = approximate_shap(model, X_train[:100], x_sample, n_samples=100)

print(f"Explaining prediction for sample {sample_idx}")
print(f"Predicted probability (class 1): {actual_pred:.4f}")
print(f"Base value (average prediction): {base_value:.4f}")
print(f"Sum of SHAP values: {np.sum(shap_vals):+.4f}")
print(f"Base + SHAP sum: {base_value + np.sum(shap_vals):.4f}\n")

sorted_shap = np.argsort(np.abs(shap_vals))[::-1]
print(f"{'Feature':>20}  {'Value':>8}  {'SHAP':>10}  {'Effect':>20}")
print("-" * 62)
for idx in sorted_shap:
    direction = "pushes UP" if shap_vals[idx] > 0 else "pushes DOWN"
    bar = '+' * int(abs(shap_vals[idx]) * 50) if shap_vals[idx] > 0 else '-' * int(abs(shap_vals[idx]) * 50)
    print(f"{feature_names[idx]:>20}  {x_sample[idx]:>8.3f}  {shap_vals[idx]:>+10.4f}  {bar}")

# --- Method 4: Partial Dependence ---
print(f"\n=== 4. Partial Dependence Plots ===")
top_feature_idx = sorted_idx[0]
top_feature_name = feature_names[top_feature_idx]

pd_result = partial_dependence(model, X_train, features=[top_feature_idx],
                                kind='average', grid_resolution=10)

print(f"Partial dependence for '{top_feature_name}':")
print(f"{'Feature Value':>14}  {'Avg Prediction':>15}  {'Plot':>25}")
print("-" * 58)

pd_values = pd_result['average'][0]
grid_values = pd_result['grid_values'][0]
pd_min, pd_max = pd_values.min(), pd_values.max()

for val, pred in zip(grid_values, pd_values):
    bar_pos = int(25 * (pred - pd_min) / (pd_max - pd_min + 1e-10))
    bar = '.' * bar_pos + '#'
    print(f"{val:>14.3f}  {pred:>15.4f}  {bar}")

# --- Method 5: LIME-style local explanation ---
print(f"\n=== 5. LIME-style Local Explanation ===")

def lime_explain(model, x_instance, X_background, n_perturbations=500):
    """Simple LIME: perturb instance, fit local linear model."""
    from sklearn.linear_model import Ridge

    n_features = len(x_instance)
    # Generate perturbations around the instance
    perturbations = np.random.normal(0, 0.5, (n_perturbations, n_features))
    X_perturbed = x_instance + perturbations

    # Get model predictions for perturbations
    predictions = model.predict_proba(X_perturbed)[:, 1]

    # Weight by proximity (kernel)
    distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
    kernel_width = np.sqrt(n_features) * 0.75
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

    # Fit weighted linear model
    lime_model = Ridge(alpha=1.0)
    lime_model.fit(X_perturbed, predictions, sample_weight=weights)

    return lime_model.coef_, lime_model.intercept_

lime_coefs, lime_intercept = lime_explain(model, x_sample, X_train)

print(f"Local linear approximation around sample {sample_idx}:")
sorted_lime = np.argsort(np.abs(lime_coefs))[::-1]
print(f"{'Feature':>20}  {'LIME Coef':>10}  {'Importance':>10}")
print("-" * 44)
for idx in sorted_lime:
    print(f"{feature_names[idx]:>20}  {lime_coefs[idx]:>+10.4f}  {abs(lime_coefs[idx]):>10.4f}")

# --- Compare methods ---
print(f"\n=== Comparison: Feature Importance Rankings ===")
print(f"{'Rank':>5}  {'Impurity':>15}  {'Permutation':>15}  {'SHAP (abs)':>15}  {'LIME (abs)':>15}")
print("-" * 70)

rank_impurity = np.argsort(importances)[::-1]
rank_perm = np.argsort(perm_imp.importances_mean)[::-1]
rank_shap = np.argsort(np.abs(shap_vals))[::-1]
rank_lime = np.argsort(np.abs(lime_coefs))[::-1]

for rank in range(min(5, n_features)):
    print(f"{rank+1:>5}  {feature_names[rank_impurity[rank]]:>15}  "
          f"{feature_names[rank_perm[rank]]:>15}  "
          f"{feature_names[rank_shap[rank]]:>15}  "
          f"{feature_names[rank_lime[rank]]:>15}")
```

---

## Key Takeaways

- **SHAP values provide theoretically grounded, additive explanations.** They fairly distribute the prediction among features and always sum to the correct prediction (minus the base value).
- **Permutation importance is the most reliable global importance measure.** Unlike impurity-based importance, it is unbiased and model-agnostic.
- **LIME explains individual predictions with local linear approximations.** It is intuitive and model-agnostic, but its explanations can vary with random seeds and perturbation settings.
- **Partial dependence plots reveal feature-prediction relationships.** They show whether a feature has a linear, monotonic, or complex effect on predictions.
- **Different methods can give different rankings.** Global importance (which features matter overall) can differ from local importance (which features matter for this prediction). Use multiple methods and look for consensus.
