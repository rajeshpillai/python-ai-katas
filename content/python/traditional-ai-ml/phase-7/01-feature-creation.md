# Feature Creation

> Phase 7 — Feature Engineering & Pipelines | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

Raw data rarely comes in the form that machine learning algorithms need. A dataset might have "length" and "width" columns, but the model actually needs "area" (length x width) to make good predictions. Feature creation (also called feature engineering) is the art and science of transforming raw variables into representations that expose the underlying patterns to the algorithm.

There are three main strategies for creating features. First, **interaction terms**: multiplying features together to capture relationships the model cannot discover on its own (e.g., price-per-square-foot = price / sqft). Second, **polynomial features**: raising features to powers to capture nonlinear relationships (e.g., x^2, x^3). Third, **domain-driven features**: using your knowledge of the problem to create meaningful features (e.g., computing BMI from height and weight, or "days since last purchase" from timestamps).

Feature engineering is often the single biggest lever for improving model performance, especially with linear models. While tree-based models can discover some interactions on their own (by splitting on one feature then another), they do it inefficiently. Explicitly providing the interaction term lets the model learn faster and with less data.

### Why naive approaches fail

Throwing raw features into a model and hoping for the best is the most common mistake in machine learning. A linear model cannot learn that "area = length x width" from separate length and width columns -- it can only learn additive relationships. Even tree-based models struggle with multiplicative relationships because they need many splits to approximate a product. Creating the right features upfront can turn a mediocre model into an excellent one, often more effectively than switching to a fancier algorithm.

### Mental models

- **Translation for the algorithm**: you speak the language of domain knowledge; the algorithm speaks the language of numerical patterns. Feature engineering is the translation layer.
- **Polynomial expansion as curve fitting**: adding x^2, x^3 lets a linear model fit curves. Each polynomial term adds a new degree of freedom.
- **Domain features as distilled expertise**: a doctor does not feed raw lab values into their brain separately -- they compute ratios, differences, and composite scores. Feature engineering does the same for models.

### Visual explanations

```
Raw Features:               Engineered Features:

length  width               length  width  area   perimeter  aspect_ratio
  5      3                    5      3      15       16        1.67
  8      4                    8      4      32       24        2.00
  3      3                    3      3       9       12        1.00
  10     2                   10      2      20       24        5.00

Polynomial Expansion (degree=2):
  x1, x2  -->  x1, x2, x1^2, x1*x2, x2^2
  (2 features become 5)

Polynomial Expansion (degree=3):
  x1, x2  -->  x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
  (2 features become 9)

Warning: features grow combinatorially!
  10 features, degree 3 --> 286 features
```

---

## Hands-on Exploration

1. Start with a regression dataset with two features. Fit a linear model on raw features and measure R-squared.
2. Manually create interaction terms (product, ratio, difference) and add them as new features. Re-fit and compare R-squared.
3. Use sklearn's PolynomialFeatures to automatically generate polynomial and interaction terms. Observe how model performance improves but also how overfitting risk increases with higher degrees.
4. Create domain-motivated features for a realistic dataset (e.g., compute BMI from height/weight, or price-per-unit) and show the improvement.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

np.random.seed(42)

# --- Synthetic dataset where the TRUE relationship involves interactions ---
n = 200
length = np.random.uniform(2, 10, n)
width = np.random.uniform(1, 6, n)
# True target depends on area (length * width) and aspect ratio
y_true = 3.0 * (length * width) + 2.0 * (length / width) + 10
y = y_true + np.random.normal(0, 5, n)

X_raw = np.column_stack([length, width])
feature_names_raw = ["length", "width"]

# --- Model 1: Raw features only ---
lr_raw = LinearRegression()
scores_raw = cross_val_score(lr_raw, X_raw, y, cv=5, scoring="r2")
lr_raw.fit(X_raw, y)

print("=== Raw Features ===")
print(f"  Features: {feature_names_raw}")
print(f"  R-squared (CV): {scores_raw.mean():.4f} (+/- {scores_raw.std():.4f})")
print(f"  Coefficients: length={lr_raw.coef_[0]:.3f}, width={lr_raw.coef_[1]:.3f}\n")

# --- Model 2: Manual feature engineering ---
area = length * width
aspect_ratio = length / width
perimeter = 2 * (length + width)
diff = length - width

X_eng = np.column_stack([length, width, area, aspect_ratio, perimeter, diff])
feature_names_eng = ["length", "width", "area", "aspect_ratio", "perimeter", "diff"]

lr_eng = LinearRegression()
scores_eng = cross_val_score(lr_eng, X_eng, y, cv=5, scoring="r2")
lr_eng.fit(X_eng, y)

print("=== Manually Engineered Features ===")
print(f"  Features: {feature_names_eng}")
print(f"  R-squared (CV): {scores_eng.mean():.4f} (+/- {scores_eng.std():.4f})")
print(f"  Coefficients:")
for name, coef in zip(feature_names_eng, lr_eng.coef_):
    bar = "|" * int(abs(coef) * 2)
    print(f"    {name:<15}: {coef:+8.3f}  {bar}")

# --- Model 3: Polynomial features (automated) ---
print("\n=== Polynomial Feature Expansion ===\n")
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    n_features = X_poly.shape[1]

    # Use Ridge to handle potential multicollinearity
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X_poly, y, cv=5, scoring="r2")
    print(f"  Degree {degree}: {n_features:>3} features, "
          f"R-squared = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Show polynomial feature names for degree 2
poly2 = PolynomialFeatures(degree=2, include_bias=False)
poly2.fit(X_raw)
print(f"\n  Degree-2 feature names: {poly2.get_feature_names_out(['length', 'width']).tolist()}")

# --- Domain-driven example: house pricing ---
print("\n=== Domain-Driven Feature Engineering (House Example) ===\n")
n_houses = 300
sqft = np.random.uniform(800, 3000, n_houses)
bedrooms = np.random.randint(1, 6, n_houses)
bathrooms = np.random.randint(1, 4, n_houses)
age = np.random.uniform(0, 50, n_houses)

# True price depends on domain-meaningful combinations
price = (150 * sqft
         + 10000 * (sqft / bedrooms)   # sqft per bedroom matters
         - 2000 * age
         + 15000 * bathrooms
         + np.random.normal(0, 20000, n_houses))

X_house_raw = np.column_stack([sqft, bedrooms, bathrooms, age])
X_house_eng = np.column_stack([
    sqft, bedrooms, bathrooms, age,
    sqft / bedrooms,            # sqft per bedroom
    sqft / bathrooms,           # sqft per bathroom
    bedrooms / bathrooms,       # bed/bath ratio
    age * sqft,                 # age-size interaction
    np.log1p(sqft),             # log transform
])

scores_house_raw = cross_val_score(LinearRegression(), X_house_raw, price, cv=5, scoring="r2")
scores_house_eng = cross_val_score(LinearRegression(), X_house_eng, price, cv=5, scoring="r2")

print(f"  Raw features (4):        R-squared = {scores_house_raw.mean():.4f}")
print(f"  Engineered features (9): R-squared = {scores_house_eng.mean():.4f}")
print(f"  Improvement: {(scores_house_eng.mean() - scores_house_raw.mean()) * 100:.1f} percentage points")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Comparison bar chart
names = ["Raw\n(2 feat)", "Manual\nEngineering\n(6 feat)", "Poly deg=2\n(5 feat)", "Poly deg=3\n(9 feat)"]
poly_scores = []
for deg in [2, 3]:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_p = poly.fit_transform(X_raw)
    s = cross_val_score(Ridge(alpha=1.0), X_p, y, cv=5, scoring="r2")
    poly_scores.append(s.mean())

all_scores = [scores_raw.mean(), scores_eng.mean()] + poly_scores
colors = ["steelblue", "coral", "steelblue", "steelblue"]
axes[0].bar(names, all_scores, color=colors)
axes[0].set_ylabel("R-squared (CV)")
axes[0].set_title("Feature Engineering Impact")
axes[0].set_ylim(0, 1.05)

# Residual plot: raw vs engineered
lr_raw.fit(X_raw, y)
lr_eng.fit(X_eng, y)
residuals_raw = y - lr_raw.predict(X_raw)
residuals_eng = y - lr_eng.predict(X_eng)
axes[1].scatter(y, residuals_raw, s=10, alpha=0.5, label="Raw features")
axes[1].scatter(y, residuals_eng, s=10, alpha=0.5, label="Engineered features")
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_xlabel("True Value")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals: Raw vs Engineered")
axes[1].legend()

plt.tight_layout()
plt.savefig("feature_creation.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to feature_creation.png")
```

---

## Key Takeaways

- **Feature engineering is often the single biggest performance lever.** The right features can improve a simple model more than switching to a complex algorithm.
- **Interaction terms capture multiplicative relationships.** Linear models cannot learn "area = length x width" without being given the product feature explicitly.
- **Polynomial features automate expansion but grow combinatorially.** Degree 3 with 10 features creates 286 features. Use regularization (Ridge) to prevent overfitting.
- **Domain knowledge creates the best features.** Ratios, differences, log transforms, and composite scores inspired by the problem domain consistently outperform blind polynomial expansion.
- **Tree-based models can discover some interactions, but explicit features still help.** Even gradient boosting benefits from well-engineered features, especially when data is limited.
