# Tabular Data Basics

> Phase 1 — Data Wrangling | Kata 1.01

---

## Concept & Intuition

### What problem are we solving?

Most real-world machine learning begins with tabular data: rows of observations and columns of features, much like a spreadsheet. Before we can build any model, we need to load, inspect, and manipulate this data programmatically. In Rust, without external crates, this means building a simple data frame abstraction from scratch using vectors and structs.

Understanding your data is the first and most critical step in any ML pipeline. What are the feature types (numeric, categorical, boolean)? What are the ranges and distributions? Are there patterns or anomalies? Data scientists spend 60-80% of their time on data preparation, and poor data quality is the number one cause of failed ML projects.

In this kata, we build a minimal data frame structure in Rust that supports loading data, inspecting columns, computing basic statistics, and selecting subsets. This foundation will be used throughout the remaining katas.

### Why naive approaches fail

Storing data as a `Vec<Vec<f64>>` seems simple but loses all column metadata: names, types, and semantics. You end up passing around magic column indices and hoping they stay consistent. Real data also has mixed types — some columns are numeric, others are strings. A flat numeric array cannot represent "color = red". Without proper abstractions, data manipulation code becomes brittle and error-prone.

### Mental models

- **Data frame as a table**: Rows are observations (samples), columns are features (variables). Each column has a name and a consistent type.
- **Statistics as summaries**: Mean, median, min, max, and standard deviation compress an entire column into a few numbers that characterize its distribution.
- **Data types matter**: Numeric features can be averaged and scaled. Categorical features can be counted and encoded. Mixing them up leads to nonsensical results.

### Visual explanations

```
  Data Frame:
  +-------+------+--------+----------+
  | Name  | Age  | Income | Category |
  +-------+------+--------+----------+
  | Alice |  30  | 50000  |    A     |  <- Row 0 (observation)
  | Bob   |  25  | 45000  |    B     |  <- Row 1
  | Carol |  35  | 60000  |    A     |  <- Row 2
  +-------+------+--------+----------+
    ^col0   ^col1  ^col2    ^col3
    string  f64    f64      string
```

---

## Hands-on Exploration

1. Define a simple data frame struct with column names and row data.
2. Load sample data into the frame and display it as a formatted table.
3. Compute basic statistics for numeric columns: mean, min, max, std dev.
4. Select subsets of rows and columns.

---

## Live Code

```rust
fn main() {
    println!("=== Tabular Data Basics ===\n");

    // Create a dataset: house features
    let columns = vec![
        "sqft", "bedrooms", "bathrooms", "age_years", "price",
    ];

    let data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 2.0, 10.0, 250000.0],
        vec![1600.0, 3.0, 2.0, 5.0, 310000.0],
        vec![1700.0, 4.0, 2.5, 8.0, 340000.0],
        vec![1100.0, 2.0, 1.0, 25.0, 180000.0],
        vec![2100.0, 4.0, 3.0, 2.0, 420000.0],
        vec![1500.0, 3.0, 2.0, 15.0, 275000.0],
        vec![1800.0, 4.0, 2.5, 3.0, 365000.0],
        vec![950.0,  2.0, 1.0, 30.0, 150000.0],
        vec![2400.0, 5.0, 3.5, 1.0, 500000.0],
        vec![1300.0, 3.0, 1.5, 20.0, 230000.0],
    ];

    let df = DataFrame::new(columns, data);

    // Display the table
    println!("--- Full Dataset ---");
    df.display();

    // Basic info
    println!("\n--- Dataset Info ---");
    println!("Rows: {}", df.n_rows());
    println!("Columns: {}", df.n_cols());
    println!("Column names: {:?}", df.column_names());

    // Statistics for each column
    println!("\n--- Column Statistics ---");
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Column", "Mean", "Std Dev", "Min", "Max", "Median"
    );
    println!("{}", "-".repeat(62));

    for col_name in df.column_names() {
        if let Some(col) = df.column(col_name) {
            let mean = mean(&col);
            let std = std_dev(&col);
            let min = min_val(&col);
            let max = max_val(&col);
            let med = median(&col);
            println!(
                "{:<12} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
                col_name, mean, std, min, max, med
            );
        }
    }

    // Select specific columns
    println!("\n--- Select: sqft and price ---");
    let subset = df.select_columns(&["sqft", "price"]);
    subset.display();

    // Filter rows: houses with > 3 bedrooms
    println!("\n--- Filter: bedrooms > 3 ---");
    let filtered = df.filter("bedrooms", |v| *v > 3.0);
    filtered.display();

    // Correlation between sqft and price
    println!("\n--- Correlation Analysis ---");
    let sqft = df.column("sqft").unwrap();
    let price = df.column("price").unwrap();
    let corr = correlation(&sqft, &price);
    println!("Correlation(sqft, price) = {:.4}", corr);

    let age = df.column("age_years").unwrap();
    let corr_age = correlation(&age, &price);
    println!("Correlation(age_years, price) = {:.4}", corr_age);

    kata_metric("n_rows", df.n_rows() as f64);
    kata_metric("n_cols", df.n_cols() as f64);
    kata_metric("price_mean", mean(&price));
    kata_metric("sqft_price_correlation", corr);
    kata_metric("age_price_correlation", corr_age);
}

struct DataFrame {
    columns: Vec<String>,
    data: Vec<Vec<f64>>,
}

impl DataFrame {
    fn new(columns: Vec<&str>, data: Vec<Vec<f64>>) -> Self {
        DataFrame {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            data,
        }
    }

    fn n_rows(&self) -> usize {
        self.data.len()
    }

    fn n_cols(&self) -> usize {
        self.columns.len()
    }

    fn column_names(&self) -> &[String] {
        &self.columns
    }

    fn column(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.columns.iter().position(|c| c == name)?;
        Some(self.data.iter().map(|row| row[idx]).collect())
    }

    fn display(&self) {
        // Print header
        for col in &self.columns {
            print!("{:>12}", col);
        }
        println!();
        println!("{}", "-".repeat(12 * self.columns.len()));

        // Print rows
        for row in &self.data {
            for val in row {
                if *val == val.floor() && val.abs() < 1e6 {
                    print!("{:>12.0}", val);
                } else {
                    print!("{:>12.1}", val);
                }
            }
            println!();
        }
    }

    fn select_columns(&self, names: &[&str]) -> DataFrame {
        let indices: Vec<usize> = names
            .iter()
            .filter_map(|name| self.columns.iter().position(|c| c == name))
            .collect();

        let new_cols: Vec<&str> = indices
            .iter()
            .map(|&i| self.columns[i].as_str())
            .collect();

        let new_data: Vec<Vec<f64>> = self
            .data
            .iter()
            .map(|row| indices.iter().map(|&i| row[i]).collect())
            .collect();

        DataFrame::new(new_cols, new_data)
    }

    fn filter(&self, column: &str, predicate: fn(&f64) -> bool) -> DataFrame {
        let idx = self.columns.iter().position(|c| c == column).unwrap();
        let cols: Vec<&str> = self.columns.iter().map(|s| s.as_str()).collect();
        let new_data: Vec<Vec<f64>> = self
            .data
            .iter()
            .filter(|row| predicate(&row[idx]))
            .cloned()
            .collect();
        DataFrame::new(cols, new_data)
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    let variance = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    variance.sqrt()
}

fn min_val(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn max_val(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn median(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x == 0.0 || den_y == 0.0 {
        return 0.0;
    }
    num / (den_x.sqrt() * den_y.sqrt())
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Tabular data is the most common format in traditional ML: rows are observations, columns are features.
- Basic statistics (mean, std dev, min, max, median) provide a quick summary of each feature's distribution.
- Correlation analysis reveals linear relationships between features — essential for feature selection and understanding your data.
- Building a data frame abstraction in Rust teaches you what libraries like Polars and pandas do under the hood.
