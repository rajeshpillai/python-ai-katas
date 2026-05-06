# Encoding Categorical Variables

> Phase 1 — Data Wrangling | Kata 1.04

---

## Concept & Intuition

### What problem are we solving?

Machine learning algorithms operate on numbers, but real-world data often contains categories: colors ("red", "blue", "green"), sizes ("small", "medium", "large"), countries, product types, and so on. We need a principled way to convert these categorical values into numeric representations that algorithms can process without introducing false relationships.

The encoding strategy matters enormously. Assigning red=1, blue=2, green=3 implies an ordering (blue is between red and green) and a magnitude (green is "three times" red), neither of which makes sense for colors. This is called label encoding and it works only for ordinal categories where order is meaningful (e.g., small < medium < large). For nominal categories (no natural order), we need one-hot encoding: each category becomes its own binary column.

In this kata, we implement label encoding, one-hot encoding, and target encoding from scratch, and demonstrate when each is appropriate.

### Why naive approaches fail

Label encoding nominal variables introduces spurious ordinal relationships. A model might learn that "green" (3) is more similar to "blue" (2) than to "red" (1), even though colors have no natural ordering. This can cause distance-based algorithms (KNN, SVM) and linear models to produce nonsensical results. One-hot encoding avoids this but can create very high-dimensional data when a feature has many categories (e.g., zip codes), leading to the "curse of dimensionality."

### Mental models

- **Label encoding = ordinal mapping**: Good for categories with natural order (education levels, shirt sizes). Bad for categories without order.
- **One-hot encoding = binary indicator columns**: Each category gets its own 0/1 column. No false ordinal relationships, but dimensionality grows with cardinality.
- **Target encoding = category-as-average**: Replace each category with the mean of the target variable for that category. Compact but risks data leakage.

### Visual explanations

```
  Original:         Label Encoding:     One-Hot Encoding:
  Color             Color               Color_Red  Color_Blue  Color_Green
  -----             -----               ---------  ----------  -----------
  Red               0                   1          0           0
  Blue              1                   0          1           0
  Green             2                   0          0           1
  Red               0                   1          0           0
  Blue              1                   0          1           0

  Label: 1 column, false ordering     One-Hot: N columns, no ordering
```

---

## Hands-on Exploration

1. Create a dataset with both nominal and ordinal categorical features.
2. Implement label encoding for ordinal features (size: S < M < L < XL).
3. Implement one-hot encoding for nominal features (color, city).
4. Implement target encoding and discuss the data leakage risk.

---

## Live Code

```rust
fn main() {
    println!("=== Encoding Categorical Variables ===\n");

    // Sample dataset: product catalog
    let colors = vec!["red", "blue", "green", "red", "blue", "green", "red", "blue"];
    let sizes = vec!["M", "L", "S", "XL", "M", "L", "S", "M"];
    let cities = vec!["NYC", "LA", "NYC", "CHI", "LA", "NYC", "CHI", "LA"];
    let prices = vec![25.0, 45.0, 15.0, 55.0, 40.0, 35.0, 20.0, 38.0];

    println!("--- Original Data ---");
    println!("{:<8} {:<6} {:<6} {:<8}", "Color", "Size", "City", "Price");
    println!("{}", "-".repeat(28));
    for i in 0..colors.len() {
        println!("{:<8} {:<6} {:<6} {:<8.0}", colors[i], sizes[i], cities[i], prices[i]);
    }

    // 1. Label Encoding (for ordinal: size)
    println!("\n--- Label Encoding (Size: ordinal) ---");
    let size_order = vec!["S", "M", "L", "XL"];
    let size_encoded: Vec<f64> = sizes.iter().map(|s| {
        size_order.iter().position(|&o| o == *s).unwrap() as f64
    }).collect();
    println!("Mapping: S=0, M=1, L=2, XL=3");
    println!("Encoded: {:?}", size_encoded);

    // 2. One-Hot Encoding (for nominal: color, city)
    println!("\n--- One-Hot Encoding (Color: nominal) ---");
    let color_categories = unique_values(&colors);
    let color_onehot = one_hot_encode(&colors, &color_categories);
    print_onehot("color", &color_categories, &color_onehot);

    println!("\n--- One-Hot Encoding (City: nominal) ---");
    let city_categories = unique_values(&cities);
    let city_onehot = one_hot_encode(&cities, &city_categories);
    print_onehot("city", &city_categories, &city_onehot);

    // 3. Target Encoding (color -> mean price)
    println!("\n--- Target Encoding (Color -> mean Price) ---");
    let color_target = target_encode(&colors, &prices);
    println!("Category means:");
    let color_cats = unique_values(&colors);
    for cat in &color_cats {
        let vals: Vec<f64> = colors.iter().zip(prices.iter())
            .filter(|(c, _)| **c == cat.as_str())
            .map(|(_, p)| *p)
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        println!("  {} -> {:.2}", cat, mean);
    }
    println!("Encoded: {:?}", color_target.iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>());

    // 4. Build the final numeric dataset
    println!("\n--- Final Numeric Dataset ---");

    // Combine all features
    let mut header = vec!["size_enc".to_string()];
    for cat in &color_categories {
        header.push(format!("color_{}", cat));
    }
    for cat in &city_categories {
        header.push(format!("city_{}", cat));
    }
    header.push("price".to_string());

    // Print header
    for h in &header {
        print!("{:>12}", h);
    }
    println!();
    println!("{}", "-".repeat(12 * header.len()));

    for i in 0..colors.len() {
        print!("{:>12.0}", size_encoded[i]);
        for j in 0..color_categories.len() {
            print!("{:>12.0}", color_onehot[i][j]);
        }
        for j in 0..city_categories.len() {
            print!("{:>12.0}", city_onehot[i][j]);
        }
        print!("{:>12.0}", prices[i]);
        println!();
    }

    // Dimensionality analysis
    let original_cols = 3; // color, size, city
    let encoded_cols = 1 + color_categories.len() + city_categories.len();
    println!("\n--- Dimensionality Impact ---");
    println!("Original categorical columns: {}", original_cols);
    println!("Encoded numeric columns: {}", encoded_cols);
    println!("Expansion factor: {:.1}x", encoded_cols as f64 / original_cols as f64);

    kata_metric("original_columns", original_cols as f64);
    kata_metric("encoded_columns", encoded_cols as f64);
    kata_metric("color_categories", color_categories.len() as f64);
    kata_metric("city_categories", city_categories.len() as f64);
}

fn unique_values(data: &[&str]) -> Vec<String> {
    let mut unique: Vec<String> = Vec::new();
    for &val in data {
        let s = val.to_string();
        if !unique.contains(&s) {
            unique.push(s);
        }
    }
    unique.sort();
    unique
}

fn one_hot_encode(data: &[&str], categories: &[String]) -> Vec<Vec<f64>> {
    data.iter().map(|&val| {
        categories.iter().map(|cat| {
            if cat == val { 1.0 } else { 0.0 }
        }).collect()
    }).collect()
}

fn target_encode(categories: &[&str], targets: &[f64]) -> Vec<f64> {
    let unique = unique_values(categories);
    let mut means: Vec<(String, f64)> = Vec::new();

    for cat in &unique {
        let vals: Vec<f64> = categories.iter().zip(targets.iter())
            .filter(|(c, _)| **c == cat.as_str())
            .map(|(_, t)| *t)
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        means.push((cat.clone(), mean));
    }

    categories.iter().map(|&cat| {
        means.iter().find(|(c, _)| c == cat).unwrap().1
    }).collect()
}

fn print_onehot(prefix: &str, categories: &[String], encoded: &[Vec<f64>]) {
    // Header
    for cat in categories {
        print!("{:>12}", format!("{}_{}", prefix, cat));
    }
    println!();
    println!("{}", "-".repeat(12 * categories.len()));

    for row in encoded {
        for val in row {
            print!("{:>12.0}", val);
        }
        println!();
    }
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Categorical variables must be encoded to numbers before feeding into ML algorithms, but the encoding method matters.
- Label encoding works only for ordinal features where the numerical ordering is meaningful (e.g., small < medium < large).
- One-hot encoding is safe for nominal features but increases dimensionality, potentially causing issues with high-cardinality features.
- Target encoding is compact and powerful but requires careful handling to avoid data leakage (using the target during training in a way that would not be available at prediction time).
