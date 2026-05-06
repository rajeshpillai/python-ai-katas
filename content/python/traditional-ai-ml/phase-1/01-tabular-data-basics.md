# Tabular Data Basics

> Phase 1 — Data Wrangling | Kata 1.1

---

## Concept & Intuition

### What problem are we solving?

Most real-world data comes in tables — rows of observations and columns of features. Whether it is patient records, sales transactions, sensor readings, or survey responses, tabular data is the bread and butter of data science. Before you can build any model, you must be able to **load, inspect, and understand** your data. This kata introduces the fundamental operations for working with tabular data using pandas DataFrames.

A DataFrame is like a supercharged spreadsheet. Each column has a specific data type (dtype) — integers, floats, strings, dates, booleans — and the dtype determines what operations you can perform. Getting the dtypes right is often the first battle in any data project. A column of numbers stored as strings cannot be used in calculations; a date column stored as plain text cannot be sorted chronologically.

Understanding the shape, structure, and content of your data before doing anything else is called **data profiling**. It answers critical questions: How many rows and columns? What are the dtypes? Are there missing values? What do the first and last rows look like? Skipping this step is like driving without looking at the road — you might get lucky, but you will eventually crash.

### Why naive approaches fail

A common mistake is jumping straight into modeling without inspecting the data. This leads to subtle bugs: a "price" column that contains the string "$1,200" instead of the number 1200, a "gender" column with 15 different spellings of "Male" and "Female", or an ID column that a model treats as a numeric feature. These issues silently corrupt your results.

Another pitfall is assuming that large datasets are too big to inspect. In fact, even a quick look at `head()`, `tail()`, `info()`, and `describe()` can reveal problems that would take hours to debug later. Five minutes of profiling saves five hours of debugging.

### Mental models

- **The spreadsheet analogy**: A DataFrame is a programmatic spreadsheet where each column is a Series (like a single column in Excel) with a consistent data type
- **Rows are observations, columns are features**: Each row represents one "thing" you measured (a patient, a transaction, a time point). Each column represents one attribute you measured about that thing
- **dtypes are contracts**: The dtype of a column is a contract about what operations are valid. Treating a categorical column as numeric (or vice versa) violates the contract and produces nonsense
- **Shape tells the story**: A dataset with 1 million rows and 5 columns is very different from one with 100 rows and 5,000 columns. The shape immediately tells you about the problem structure

### Visual explanations

```
Anatomy of a DataFrame
========================

          col_0    col_1    col_2    col_3
         (int64) (float64) (object) (bool)
        +--------+---------+--------+-------+
row 0   |   42   |  3.14   | "cat"  | True  |
row 1   |   17   |  2.72   | "dog"  | False |
row 2   |   99   |   NaN   | "cat"  | True  |  <- missing value!
row 3   |   23   |  1.41   | "bird" | True  |
        +--------+---------+--------+-------+

        Index     Columns (with dtypes)
        (rows)

Key Properties:
  .shape    -> (4, 4)        # rows x columns
  .dtypes   -> int64, float64, object, bool
  .columns  -> ['col_0', 'col_1', 'col_2', 'col_3']
  .index    -> [0, 1, 2, 3]


Common dtypes
===============
  int64    -> Whole numbers (age, count)
  float64  -> Decimal numbers (price, temperature)
  object   -> Text/strings (name, category)
  bool     -> True/False (is_active, has_discount)
  datetime -> Dates and times (created_at, birth_date)
  category -> Categorical data (efficient for repeated strings)
```

---

## Hands-on Exploration

1. **Profile the dataset**: Run the code below to load a synthetic dataset. Examine the output of `shape`, `dtypes`, `head()`, `tail()`, `describe()`, and `info()`. Write down three observations about the data before reading further.

2. **Fix the dtypes**: Notice that some columns have incorrect dtypes (e.g., a numeric column stored as `object`). Use `astype()` or `pd.to_numeric()` to fix them. Verify with `dtypes` that the fixes worked.

3. **Select and filter**: Practice selecting specific columns, filtering rows by conditions (e.g., all rows where age > 30), and combining conditions with `&` and `|`. These are the most common operations you will perform daily.

4. **Summarize by group**: Use `groupby()` to compute statistics (mean, count, max) for different subgroups. This is the foundation of exploratory data analysis.

---

## Live Code

```python
"""
Tabular Data Basics — Loading, inspecting, and understanding DataFrames.

This code creates a realistic synthetic dataset and demonstrates
essential pandas operations for data profiling and manipulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Part 1: Create a Realistic Synthetic Dataset
# ============================================================

np.random.seed(42)
n = 200

data = {
    "id": range(1, n + 1),
    "age": np.random.randint(18, 75, size=n),
    "income": np.random.lognormal(mean=10.5, sigma=0.6, size=n).round(2),
    "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], size=n),
    "years_experience": np.random.randint(0, 35, size=n),
    "satisfaction_score": np.random.uniform(1, 10, size=n).round(1),
    "is_remote": np.random.choice([True, False], size=n, p=[0.4, 0.6]),
    "performance_rating": np.random.choice(["Low", "Medium", "High", "Excellent"], size=n,
                                           p=[0.1, 0.3, 0.4, 0.2]),
}

df = pd.DataFrame(data)

# Introduce some realistic messiness
# Some incomes stored as strings with dollar signs
mask = np.random.choice(n, size=15, replace=False)
df.loc[mask, "income"] = df.loc[mask, "income"].apply(lambda x: f"${x:,.2f}")

# Some missing values
df.loc[np.random.choice(n, size=10, replace=False), "satisfaction_score"] = np.nan
df.loc[np.random.choice(n, size=5, replace=False), "years_experience"] = np.nan

print("=" * 60)
print("TABULAR DATA BASICS — Employee Dataset")
print("=" * 60)

# ============================================================
# Part 2: Essential Inspection Commands
# ============================================================

print("\n--- df.shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- df.dtypes ---")
print(df.dtypes)

print("\n--- df.head() (first 5 rows) ---")
print(df.head().to_string())

print("\n--- df.tail(3) (last 3 rows) ---")
print(df.tail(3).to_string())

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() (numeric columns only) ---")
print(df.describe().round(2).to_string())

print("\n--- df.describe(include='object') (non-numeric columns) ---")
print(df.describe(include="object").to_string())

# ============================================================
# Part 3: Identifying and Fixing Problems
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY ISSUES")
print("=" * 60)

# Check for mixed types in the income column
print("\n1. Income column dtype:", df["income"].dtype)
print("   Sample values:", df["income"].head(10).tolist())
print("   Problem: Some values are strings like '$45,234.56'!")

# Fix: convert income to numeric
df["income_clean"] = pd.to_numeric(
    df["income"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce"
)
print(f"\n   After cleaning: dtype = {df['income_clean'].dtype}")
print(f"   Income range: ${df['income_clean'].min():,.2f} — ${df['income_clean'].max():,.2f}")

# Check for missing values
print("\n2. Missing values per column:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
missing_report = pd.DataFrame({"Count": missing, "Percent": missing_pct})
print(missing_report[missing_report["Count"] > 0].to_string())

# Check unique values in categorical columns
print("\n3. Unique values in categorical columns:")
for col in ["department", "performance_rating", "is_remote"]:
    print(f"   {col}: {df[col].nunique()} unique -> {df[col].unique().tolist()}")

# ============================================================
# Part 4: Selection and Filtering
# ============================================================

print("\n" + "=" * 60)
print("SELECTION AND FILTERING")
print("=" * 60)

# Select specific columns
print("\n--- Select columns: age, department, income_clean ---")
subset = df[["age", "department", "income_clean"]]
print(subset.head().to_string())

# Filter rows
print("\n--- Filter: age > 50 AND department == 'Engineering' ---")
filtered = df[(df["age"] > 50) & (df["department"] == "Engineering")]
print(f"   {len(filtered)} rows match this condition")
print(filtered[["age", "department", "income_clean"]].head().to_string())

# Group by and aggregate
print("\n--- Group by department: mean income and satisfaction ---")
grouped = df.groupby("department").agg({
    "income_clean": "mean",
    "satisfaction_score": "mean",
    "age": "count",
}).rename(columns={"age": "count"}).round(2)
print(grouped.to_string())

# ============================================================
# Part 5: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of age
ax = axes[0, 0]
ax.hist(df["age"], bins=20, color="#3498db", edgecolor="black", alpha=0.7)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution", fontweight="bold")
ax.axvline(df["age"].mean(), color="red", linestyle="--", label=f"Mean: {df['age'].mean():.1f}")
ax.legend()

# Income by department (box plot)
ax = axes[0, 1]
departments = df["department"].unique()
dept_incomes = [df[df["department"] == d]["income_clean"].dropna() for d in departments]
bp = ax.boxplot(dept_incomes, labels=departments, patch_artist=True)
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Income ($)")
ax.set_title("Income by Department", fontweight="bold")
ax.tick_params(axis="x", rotation=30)

# dtypes breakdown
ax = axes[1, 0]
dtype_counts = df.dtypes.astype(str).value_counts()
ax.barh(dtype_counts.index, dtype_counts.values, color="#2ecc71", edgecolor="black")
ax.set_xlabel("Number of Columns")
ax.set_title("Column Data Types", fontweight="bold")

# Missing values
ax = axes[1, 1]
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    ax.bar(missing_cols.index, missing_cols.values, color="#e74c3c", edgecolor="black", alpha=0.7)
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
else:
    ax.text(0.5, 0.5, "No missing values!", ha="center", va="center", fontsize=14)
    ax.set_title("Missing Values", fontweight="bold")

plt.tight_layout()
plt.show()

print("\nKey insight: Always profile your data before modeling.")
print("Five minutes of .shape, .dtypes, .head(), .describe(), and .info()")
print("can save hours of debugging mysterious model behavior.")
```

---

## Key Takeaways

- **Always profile before modeling**: Use `shape`, `dtypes`, `head()`, `tail()`, `info()`, and `describe()` to understand your data before doing anything else
- **dtypes matter enormously**: A numeric column stored as `object` (string) will silently break calculations; always verify and fix dtypes early
- **Missing values are everywhere**: Real datasets almost always have missing values; detecting them early with `isnull().sum()` prevents surprises later
- **Selection and filtering are your daily tools**: Mastering `df[columns]`, `df[condition]`, and `df.groupby()` covers the vast majority of data manipulation needs
- **Visualization complements statistics**: A histogram or box plot often reveals patterns (skew, outliers, clusters) that summary statistics alone cannot capture
