# Linear Regression

> Phase 1 — What Does It Mean to Learn? | Kata 1.4

---

## Concept & Intuition

### What problem are we solving?

Linear regression is the most fundamental parametric model in machine learning. It assumes the relationship between features and target is a weighted sum: y = w1*x1 + w2*x2 + ... + b, where the weights (w) and bias (b) are the parameters the model learns. Unlike k-NN, which stores all training data, linear regression compresses the entire dataset into a handful of numbers.

The "learning" process in linear regression means finding the weights that minimize the sum of squared errors between predictions and actual values. This has a beautiful closed-form solution (the Normal Equation): w = (X^T X)^(-1) X^T y. No iteration needed — just matrix algebra. This makes linear regression a perfect starting point for understanding what it means for a model to "learn" parameters from data.

Linear regression illustrates the core machine learning loop: choose a model family (linear functions), define a loss function (squared error), and find the parameters that minimize that loss. Every more sophisticated model follows this same pattern — they just use more expressive model families and more complex optimization.

### Why naive approaches fail

Solving the normal equation requires inverting a matrix, which fails if features are perfectly correlated (multicollinear). It also assumes the relationship is truly linear — if the real pattern is curved, a straight line will systematically under-predict and over-predict in different regions. Understanding the limitations of linearity is key to knowing when to reach for more powerful models.

### Mental models

- **Best-fit line**: Linear regression finds the line (or hyperplane) that minimizes the total squared distance from all data points to the line.
- **Weights as importance**: Each weight tells you how much the target changes when that feature increases by one unit, holding all other features constant.
- **From memorization to generalization**: Unlike a lookup table, linear regression compresses the data into a compact formula that can predict for inputs it has never seen.

### Visual explanations

```
  Linear regression fits a line through the data:

  y │        .
    │      . /  .
    │    . / .
    │  . /.
    │ /. .
    │/.
    └──────────── x

  The line y = wx + b minimizes the sum of
  squared vertical distances (residuals).
```

---

## Hands-on Exploration

1. Implement the Normal Equation to solve for optimal weights.
2. Fit a linear model to data and examine the learned weights.
3. Compute residuals and R-squared to evaluate the fit quality.

---

## Live Code

```rust
fn main() {
    // === Linear Regression from Scratch ===
    // Find weights that minimize squared error: y = Xw + b

    // Dataset: [sq_feet, bedrooms] → price
    let x_data: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0], vec![1800.0, 4.0], vec![1100.0, 2.0],
        vec![2200.0, 5.0], vec![1600.0, 3.0], vec![900.0, 1.0],
        vec![2000.0, 4.0], vec![1300.0, 2.0], vec![1700.0, 3.0],
        vec![1500.0, 3.0], vec![2100.0, 4.0], vec![1000.0, 2.0],
        vec![1900.0, 4.0], vec![1200.0, 2.0], vec![1650.0, 3.0],
    ];
    let y_data: Vec<f64> = vec![
        250000.0, 320000.0, 195000.0, 410000.0, 275000.0, 150000.0,
        380000.0, 220000.0, 310000.0, 265000.0, 400000.0, 180000.0,
        350000.0, 205000.0, 290000.0,
    ];

    let n = x_data.len();
    let m = x_data[0].len(); // number of features

    // Add bias column (column of 1s) → [sq_feet, bedrooms, 1.0]
    let x_aug: Vec<Vec<f64>> = x_data.iter()
        .map(|row| {
            let mut r = row.clone();
            r.push(1.0);
            r
        })
        .collect();
    let cols = m + 1;

    // === Matrix operations (from scratch) ===

    // Transpose
    let transpose = |mat: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let rows = mat.len();
        let cols = mat[0].len();
        let mut t = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j][i] = mat[i][j];
            }
        }
        t
    };

    // Matrix multiply
    let matmul = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let rows = a.len();
        let cols = b[0].len();
        let inner = b.len();
        let mut result = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    };

    // Matrix-vector multiply
    let matvec = |mat: &Vec<Vec<f64>>, v: &Vec<f64>| -> Vec<f64> {
        mat.iter().map(|row| {
            row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
        }).collect()
    };

    // 3x3 matrix inverse (for our small system)
    let invert_3x3 = |m: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        let inv_det = 1.0 / det;

        vec![
            vec![
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            vec![
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            vec![
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ]
    };

    // === Normal Equation: w = (X^T X)^(-1) X^T y ===
    println!("=== Linear Regression via Normal Equation ===\n");

    let xt = transpose(&x_aug);
    let xtx = matmul(&xt, &x_aug);
    let xtx_inv = invert_3x3(&xtx);

    // X^T y
    let xty: Vec<f64> = (0..cols).map(|j| {
        (0..n).map(|i| x_aug[i][j] * y_data[i]).sum()
    }).collect();

    // w = (X^T X)^(-1) X^T y
    let weights = matvec(&xtx_inv, &xty);

    println!("  Learned parameters:");
    println!("    w_sqfeet   = {:.2} (price per sqft)", weights[0]);
    println!("    w_bedrooms = {:.2} (price per bedroom)", weights[1]);
    println!("    bias       = {:.2}", weights[2]);
    println!();
    println!("  Model: price = {:.2} * sqft + {:.2} * beds + {:.2}\n",
        weights[0], weights[1], weights[2]);

    // === Predictions and residuals ===
    println!("=== Predictions vs Actual ===\n");
    println!("  {:>8} {:>6} {:>10} {:>10} {:>10}",
        "sq_feet", "beds", "actual", "predicted", "residual");
    println!("  {:->8} {:->6} {:->10} {:->10} {:->10}", "", "", "", "", "");

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;

    for i in 0..n {
        let predicted: f64 = x_aug[i].iter().zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        let residual = y_data[i] - predicted;
        ss_res += residual * residual;
        ss_tot += (y_data[i] - y_mean) * (y_data[i] - y_mean);

        println!("  {:>8.0} {:>6.0} {:>10.0} {:>10.0} {:>+10.0}",
            x_data[i][0], x_data[i][1], y_data[i], predicted, residual);
    }

    let r_squared = 1.0 - ss_res / ss_tot;
    let mse = ss_res / n as f64;
    let rmse = mse.sqrt();

    println!();
    println!("=== Model Evaluation ===\n");
    println!("  MSE:       {:.0}", mse);
    println!("  RMSE:      ${:.0}", rmse);
    println!("  R-squared: {:.4}", r_squared);
    println!("  (R²=1 is perfect, R²=0 is no better than mean predictor)\n");

    // === Predictions on new data ===
    println!("=== Predictions on New Data ===\n");
    let new_houses = vec![
        vec![1550.0, 3.0],
        vec![2500.0, 5.0],
        vec![800.0, 1.0],
    ];

    for house in &new_houses {
        let predicted = house[0] * weights[0] + house[1] * weights[1] + weights[2];
        println!("  [{:.0} sqft, {} beds] → predicted price: ${:.0}",
            house[0], house[1] as i32, predicted);
    }

    // === Interpretation of weights ===
    println!("\n=== Weight Interpretation ===\n");
    println!("  Each extra sqft adds ~${:.0} to the price.", weights[0]);
    println!("  Each extra bedroom adds ~${:.0} to the price.", weights[1]);
    println!("  The baseline price (0 sqft, 0 beds) is ${:.0} (the bias).", weights[2]);
    println!("  (The bias is a mathematical artifact — not a real scenario.)");

    println!();
    println!("Key insight: Linear regression compresses data into weights.");
    println!("Each weight tells you how important a feature is.");
}
```

---

## Key Takeaways

- Linear regression fits a weighted sum y = w1*x1 + w2*x2 + b by finding weights that minimize the sum of squared errors.
- The Normal Equation provides a closed-form solution: no iteration needed, just matrix algebra.
- R-squared measures how much variance the model explains — it tells you how much better you are than the mean predictor.
- The learned weights are interpretable: each weight tells you the effect of a one-unit increase in that feature on the prediction.
