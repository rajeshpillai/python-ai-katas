# What Is Data?

> Phase 0 — Foundations | Kata 0.1

---

## Concept & Intuition

### What problem are we solving?

Before we can build any model that "learns," we need to understand what it learns *from*. Data is the raw material of machine learning. In its simplest form, data is a collection of observations — measurements, counts, labels, or descriptions of things in the world. Every row in a dataset represents one observation, and every column represents one attribute of that observation.

Understanding data means understanding its shape, its types, and its quirks. Numerical data (heights, temperatures, prices) can be continuous or discrete. Categorical data (colors, species, countries) represents membership in groups. Before any algorithm touches your data, you need to know what you are working with, because the structure of the data determines which tools and techniques are appropriate.

In Rust, we represent data using basic types: `f64` for continuous values, `i64` for discrete counts, `String` or enums for categories, and `Vec<T>` for collections. There are no magic DataFrames here — we build our understanding from first principles.

### Why naive approaches fail

A common mistake is to throw data directly into an algorithm without first examining it. If you do not know the range, distribution, or type of each column, you cannot choose appropriate preprocessing steps. Missing values, outliers, and mixed types can silently corrupt results. The first step is always: look at your data.

### Mental models

- **Data as a table**: Rows are observations, columns are attributes. This tabular metaphor applies even when data is more complex (images, text) — you can always flatten structure into rows and columns.
- **Data as evidence**: Each data point is one piece of evidence about the world. More evidence (more rows) gives you more confidence; more attributes (more columns) give you more perspectives.
- **Data types determine tools**: Continuous data can be averaged; categorical data can be counted. Know your types before choosing your tools.

### Visual explanations

```
  Dataset: House prices
  ┌──────────┬──────────┬────────┬─────────┐
  │ sq_feet  │ bedrooms │  city  │  price  │
  ├──────────┼──────────┼────────┼─────────┤
  │  1400.0  │    3     │  "A"   │ 250000  │  ← one observation
  │  1800.0  │    4     │  "B"   │ 320000  │
  │  1100.0  │    2     │  "A"   │ 195000  │
  │   ...    │   ...    │  ...   │   ...   │
  └──────────┴──────────┴────────┴─────────┘
     continuous  discrete  categorical  target

  Each column has a TYPE that determines how we process it.
```

---

## Hands-on Exploration

1. Create a small dataset of numeric observations and compute basic statistics (min, max, mean).
2. Separate the data into numerical and categorical components and process each appropriately.
3. Observe how summary statistics give you a compressed view of the full dataset.

---

## Live Code

```rust
fn main() {
    // === What is data? ===
    // Data is a collection of observations with attributes.
    // Let's build a tiny dataset from scratch.

    // Our dataset: [sq_feet, bedrooms, price]
    let data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 250000.0],
        vec![1800.0, 4.0, 320000.0],
        vec![1100.0, 2.0, 195000.0],
        vec![2200.0, 5.0, 410000.0],
        vec![1600.0, 3.0, 275000.0],
        vec![900.0,  1.0, 150000.0],
        vec![2000.0, 4.0, 380000.0],
        vec![1300.0, 2.0, 220000.0],
    ];

    let column_names = ["sq_feet", "bedrooms", "price"];
    let n_rows = data.len();
    let n_cols = data[0].len();

    println!("=== Dataset Overview ===");
    println!("Number of observations (rows): {}", n_rows);
    println!("Number of attributes (columns): {}", n_cols);
    println!();

    // Print the data table
    println!("  {:>10} {:>10} {:>10}", column_names[0], column_names[1], column_names[2]);
    println!("  {:->10} {:->10} {:->10}", "", "", "");
    for row in &data {
        println!("  {:>10.0} {:>10.0} {:>10.0}", row[0], row[1], row[2]);
    }
    println!();

    // Compute column statistics
    println!("=== Column Statistics ===");
    for col_idx in 0..n_cols {
        let column: Vec<f64> = data.iter().map(|row| row[col_idx]).collect();

        let min = column.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = column.iter().sum();
        let mean = sum / column.len() as f64;

        // Variance and standard deviation
        let variance: f64 = column.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>() / column.len() as f64;
        let std_dev = variance.sqrt();

        println!("{:>10}: min={:>10.1}  max={:>10.1}  mean={:>10.1}  std={:>10.1}",
            column_names[col_idx], min, max, mean, std_dev);
    }
    println!();

    // Demonstrate data types
    println!("=== Data Types ===");
    println!("sq_feet  : continuous (can take any positive real value)");
    println!("bedrooms : discrete   (integer counts: 1, 2, 3, ...)");
    println!("price    : continuous (target variable we might predict)");
    println!();

    // Categorical data example
    let cities = vec!["Austin", "Boston", "Austin", "Chicago", "Boston", "Austin", "Chicago", "Boston"];
    println!("=== Categorical Data: cities ===");

    // Count occurrences
    let mut unique_cities: Vec<&str> = cities.clone();
    unique_cities.sort();
    unique_cities.dedup();

    for city in &unique_cities {
        let count = cities.iter().filter(|&&c| c == *city).count();
        println!("  {:>10}: {} occurrences", city, count);
    }

    println!();
    println!("Key insight: Data = observations x attributes.");
    println!("Know your types (continuous, discrete, categorical) before modeling.");
}
```

---

## Key Takeaways

- Data is a structured collection of observations, where each observation has attributes (features) and potentially a target value.
- Understanding data types (continuous, discrete, categorical) is essential because it determines which operations and models are appropriate.
- Summary statistics (min, max, mean, standard deviation) compress a full column into a few numbers that reveal its character.
- Always examine your data before modeling — its shape, types, and distribution tell you what tools to reach for.
