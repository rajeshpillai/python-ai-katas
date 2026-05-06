# Encoding Categorical Variables

> Phase 1 — Data Wrangling | Kata 1.4

---

## Concept & Intuition

### What problem are we solving?

Machine learning algorithms work with numbers, but much of real-world data is categorical — colors, countries, product types, education levels. You cannot feed the string "Engineering" into a linear regression or distance calculation. **Encoding** is the process of converting categorical variables into numerical representations that algorithms can work with while preserving the information content of the original categories.

The challenge is that different encoding schemes make different assumptions about the data, and the wrong choice can introduce false relationships. If you encode {Red=1, Blue=2, Green=3}, a linear model will assume that Green is "more" than Red and that Blue is halfway between them — a completely meaningless ordinal relationship for colors. This is why one-hot encoding (creating binary columns for each category) is the default for nominal categories.

There are three main encoding strategies: **One-hot encoding** creates a binary column for each category (best for nominal data with no inherent order), **ordinal encoding** assigns integers to categories (appropriate only when there is a natural order like Low < Medium < High), and **target encoding** replaces each category with the mean of the target variable (powerful but prone to overfitting). Choosing the right strategy depends on the nature of the variable and the algorithm you plan to use.

### Why naive approaches fail

The most common mistake is using ordinal encoding (integers) for nominal categories. Assigning {Cat=0, Dog=1, Fish=2} tells a model that Dog is between Cat and Fish, and that the "distance" from Cat to Dog equals the distance from Dog to Fish. This is nonsensical for nominal data and can severely distort model behavior.

Conversely, one-hot encoding everything can cause problems too. If a categorical variable has 1,000 unique values (like zip codes or product IDs), one-hot encoding creates 1,000 new columns — a massive expansion called the "curse of dimensionality" that can make models slow, overfit, and difficult to interpret. High-cardinality categoricals require alternative approaches like target encoding, frequency encoding, or embedding.

### Mental models

- **Nominal vs. ordinal**: Nominal categories have no inherent order (colors, countries). Ordinal categories have a natural ranking (education level, customer satisfaction). The encoding must match the type
- **One-hot as a light switch panel**: Each category gets its own switch that is either ON (1) or OFF (0). Only one switch is on at a time. No switch is "greater than" another
- **Target encoding as cheating (a little)**: Target encoding leaks information from the target variable into the features. It is powerful but must be done carefully (with cross-validation) to avoid overfitting
- **The cardinality spectrum**: Low cardinality (2-10 categories) -> one-hot works great. Medium (10-100) -> consider target encoding. High (100+) -> definitely need target encoding, embeddings, or grouping

### Visual explanations

```
Encoding Methods Comparison
=============================

Original data:   Color = [Red, Blue, Green, Red, Blue]

1. ONE-HOT ENCODING (for nominal data)

   Color_Red  Color_Blue  Color_Green
       1          0           0
       0          1           0
       0          0           1
       1          0           0
       0          1           0

   No false ordinal relationships. Doubles/triples column count.


2. ORDINAL ENCODING (for ordinal data only!)

   Color_encoded
       0          (Red)
       1          (Blue)
       2          (Green)
       0          (Red)
       1          (Blue)

   DANGER: Implies Red < Blue < Green. Only use for truly ordered data!


3. TARGET ENCODING (mean of target per category)

   Target values:  [100, 200, 150, 120, 180]
   Mean by color:  Red=110, Blue=190, Green=150

   Color_encoded
      110         (Red)
      190         (Blue)
      150         (Green)
      110         (Red)
      190         (Blue)

   Uses target info. Risk of overfitting without cross-validation!


When to Use What
==================

  Variable Type        Example              Best Encoding
  ─────────────        ──────────           ──────────────
  Binary               Male/Female          Label: 0/1
  Nominal (low card.)  Color (3-5 values)   One-hot
  Nominal (high card.) Zip code (1000s)     Target / Frequency
  Ordinal              Education level      Ordinal (0,1,2,3)
```

---

## Hands-on Exploration

1. **One-hot encode and inspect**: Run the code below to see how one-hot encoding transforms categorical features. Count the total number of columns before and after encoding. Calculate the expansion factor.

2. **The ordinal trap**: Look at how using ordinal encoding for a nominal variable (department) creates a false ordering that a linear model can exploit. Compare model coefficients with one-hot vs. ordinal encoding.

3. **Target encoding in practice**: Implement target encoding for the department variable. Compare the encoded values — do departments with higher average target values get higher encoded values?

4. **High cardinality challenge**: Imagine a "city" column with 500 unique values. What would happen with one-hot encoding? Calculate the resulting number of columns and discuss why target encoding is preferable.

---

## Live Code

```python
"""
Encoding Categorical Variables — One-hot, ordinal, and target encoding.

This code demonstrates different encoding strategies, their appropriate
use cases, and the pitfalls of choosing the wrong encoding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# ============================================================
# Part 1: Create Dataset with Categorical Features
# ============================================================

np.random.seed(42)
n = 500

# Department (nominal — no natural order)
departments = np.random.choice(
    ["Engineering", "Sales", "Marketing", "HR", "Finance", "Legal"],
    size=n, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10],
)

# Education level (ordinal — natural order)
education = np.random.choice(
    ["High School", "Bachelor", "Master", "PhD"],
    size=n, p=[0.15, 0.40, 0.30, 0.15],
)

# Numeric features
years_exp = np.random.randint(0, 30, size=n)
age = 22 + years_exp + np.random.randint(0, 10, size=n)

# Target: salary (depends on department, education, and experience)
dept_bonus = {"Engineering": 15000, "Finance": 12000, "Sales": 8000,
              "Marketing": 7000, "Legal": 13000, "HR": 6000}
edu_bonus = {"High School": 0, "Bachelor": 10000, "Master": 20000, "PhD": 30000}

salary = (40000
          + np.array([dept_bonus[d] for d in departments])
          + np.array([edu_bonus[e] for e in education])
          + 1500 * years_exp
          + np.random.normal(0, 5000, n))

df = pd.DataFrame({
    "department": departments,
    "education": education,
    "years_experience": years_exp,
    "age": age,
    "salary": salary,
})

print("=" * 60)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nCategorical columns:")
for col in ["department", "education"]:
    print(f"  {col}: {df[col].nunique()} categories -> {df[col].unique().tolist()}")

# ============================================================
# Part 2: One-Hot Encoding (for nominal data)
# ============================================================

print("\n" + "=" * 60)
print("ONE-HOT ENCODING (Department — Nominal)")
print("=" * 60)

df_onehot = pd.get_dummies(df, columns=["department"], prefix="dept", dtype=int)
print(f"\nColumns before: {len(df.columns)}")
print(f"Columns after:  {len(df_onehot.columns)}")
print(f"\nNew columns: {[c for c in df_onehot.columns if c.startswith('dept_')]}")
print(f"\nSample (first 5 rows):")
print(df_onehot[[c for c in df_onehot.columns if c.startswith("dept_")]].head().to_string())

# ============================================================
# Part 3: Ordinal Encoding (for ordinal data)
# ============================================================

print("\n" + "=" * 60)
print("ORDINAL ENCODING (Education — Ordinal)")
print("=" * 60)

edu_order = ["High School", "Bachelor", "Master", "PhD"]
ord_encoder = OrdinalEncoder(categories=[edu_order])
df["education_ordinal"] = ord_encoder.fit_transform(df[["education"]])

print("\nEncoding mapping:")
for i, level in enumerate(edu_order):
    print(f"  {level} -> {i}")

print(f"\nSample:")
print(df[["education", "education_ordinal"]].head(10).to_string())

# ============================================================
# Part 4: Target Encoding
# ============================================================

print("\n" + "=" * 60)
print("TARGET ENCODING (Department -> Mean Salary)")
print("=" * 60)

target_means = df.groupby("department")["salary"].mean()
df["department_target_encoded"] = df["department"].map(target_means)

print("\nTarget encoding map (department -> mean salary):")
for dept, mean_sal in target_means.sort_values(ascending=False).items():
    print(f"  {dept:<15} -> ${mean_sal:,.0f}")

# ============================================================
# Part 5: The Ordinal Trap — Danger of Wrong Encoding
# ============================================================

print("\n" + "=" * 60)
print("THE ORDINAL TRAP — Wrong Encoding for Nominal Data")
print("=" * 60)

# Wrong: ordinal encode a nominal feature
label_enc = LabelEncoder()
df["department_wrong"] = label_enc.fit_transform(df["department"])
print("\nWrong ordinal encoding for department:")
for i, dept in enumerate(label_enc.classes_):
    print(f"  {dept} -> {i}")
print("This implies: Engineering < Finance < HR < Legal < Marketing < Sales")
print("This ordering is MEANINGLESS for nominal data!")

# Compare model performance
from sklearn.model_selection import cross_val_score

# Model with one-hot encoding
X_onehot = df_onehot[["years_experience", "age"] +
                       [c for c in df_onehot.columns if c.startswith("dept_")]].copy()
X_onehot["education_ordinal"] = df["education_ordinal"]

# Model with wrong ordinal encoding
X_wrong = df[["years_experience", "age", "department_wrong", "education_ordinal"]].copy()

# Model with target encoding
X_target = df[["years_experience", "age", "department_target_encoded", "education_ordinal"]].copy()

y = df["salary"]

for name, X in [("One-hot (correct)", X_onehot),
                ("Ordinal (WRONG for dept)", X_wrong),
                ("Target encoding", X_target)]:
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"\n  {name}:")
    print(f"    R-squared: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================================
# Part 6: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean salary by department
ax = axes[0, 0]
dept_means = df.groupby("department")["salary"].mean().sort_values()
colors = plt.cm.Set2(np.linspace(0, 1, len(dept_means)))
ax.barh(dept_means.index, dept_means.values, color=colors, edgecolor="black")
ax.set_xlabel("Mean Salary ($)")
ax.set_title("Mean Salary by Department\n(This IS the target encoding)", fontweight="bold")

# One-hot encoding visualization
ax = axes[0, 1]
sample = df_onehot[[c for c in df_onehot.columns if c.startswith("dept_")]].head(10)
ax.imshow(sample.values, cmap="YlGn", aspect="auto")
ax.set_xticks(range(sample.shape[1]))
ax.set_xticklabels([c.replace("dept_", "") for c in sample.columns], rotation=45, ha="right")
ax.set_ylabel("Row Index")
ax.set_title("One-Hot Encoding Matrix\n(One 1 per row)", fontweight="bold")
for i in range(sample.shape[0]):
    for j in range(sample.shape[1]):
        ax.text(j, i, str(sample.values[i, j]), ha="center", va="center", fontsize=10)

# Education ordinal encoding
ax = axes[1, 0]
edu_counts = df.groupby("education_ordinal")["salary"].mean()
ax.bar(range(len(edu_order)), [edu_counts.get(i, 0) for i in range(len(edu_order))],
       color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"], edgecolor="black")
ax.set_xticks(range(len(edu_order)))
ax.set_xticklabels(edu_order, rotation=15)
ax.set_ylabel("Mean Salary ($)")
ax.set_title("Ordinal Encoding (Education)\nNatural order preserved!", fontweight="bold")

# Encoding comparison: model R-squared
ax = axes[1, 1]
methods = ["One-Hot\n(correct)", "Ordinal\n(WRONG)", "Target\nEncoding"]
r2_scores = []
for X in [X_onehot, X_wrong, X_target]:
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    r2_scores.append(scores.mean())

colors = ["#2ecc71", "#e74c3c", "#3498db"]
ax.bar(methods, r2_scores, color=colors, edgecolor="black", alpha=0.8)
ax.set_ylabel("R-squared (CV)")
ax.set_title("Encoding Method vs Model Performance", fontweight="bold")
ax.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()

print("\nKey insight: The encoding you choose determines what your model can learn.")
print("One-hot encoding is safe for nominal data; ordinal encoding is for ordered data.")
print("Target encoding is powerful but must be used with cross-validation to avoid leakage.")
```

---

## Key Takeaways

- **One-hot encoding is the safe default for nominal categories** — it creates binary columns that introduce no false ordinal relationships, but it increases dimensionality
- **Ordinal encoding preserves natural order** and is appropriate ONLY for variables with a meaningful ranking (e.g., education level, size categories)
- **Target encoding is powerful but dangerous**: It uses the target variable mean for each category, which can leak information and cause overfitting without proper cross-validation
- **The wrong encoding can hurt more than no encoding**: Using ordinal encoding for nominal data creates meaningless numerical relationships that models will exploit, leading to poor generalization
- **High-cardinality categoricals need special treatment**: One-hot encoding a column with hundreds of unique values creates curse-of-dimensionality problems; use target encoding, frequency encoding, or embeddings instead
