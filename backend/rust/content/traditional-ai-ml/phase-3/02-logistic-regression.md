# Logistic Regression

> Phase 3 — Supervised Learning: Classification | Kata 3.02

---

## Concept & Intuition

### What problem are we solving?

Logistic regression is the workhorse of binary classification. Despite its name, it is a classification algorithm, not a regression algorithm. It models the probability that an input belongs to class 1 using the logistic (sigmoid) function: P(y=1|x) = 1 / (1 + exp(-(w*x + b))). The sigmoid maps any real number to the range (0, 1), making it a natural probability estimate.

What makes logistic regression powerful is its interpretability. Each coefficient tells you how much a one-unit increase in that feature changes the log-odds of the positive class. In medical diagnosis, a doctor can understand exactly why the model flagged a patient as high-risk. In credit scoring, regulators require this kind of transparency. Logistic regression combines the simplicity of linear models with the ability to output probabilities.

In this kata, we implement logistic regression from scratch using gradient descent to maximize the log-likelihood (or equivalently, minimize the binary cross-entropy loss).

### Why naive approaches fail

You cannot use linear regression for classification because it predicts unbounded real values, not probabilities. A linear model might predict P(spam) = -0.3 or P(spam) = 1.5, neither of which makes sense. The sigmoid function fixes this by squashing the linear output into (0, 1). Using MSE loss for classification also fails because the loss landscape has poor gradients when predictions are confidently wrong. Cross-entropy loss provides much better gradients for learning.

### Mental models

- **Sigmoid as a soft threshold**: The sigmoid function is a smooth, differentiable version of a step function. Below the threshold, probability is near 0; above, it is near 1. The transition width depends on the coefficient magnitudes.
- **Log-odds as the linear part**: Logistic regression is linear in log-odds space. If log(P/(1-P)) = w*x + b, then each unit increase in x adds w to the log-odds. This is the link between the linear world and the probability world.
- **Decision boundary**: The decision boundary is where P(y=1) = 0.5, which corresponds to w*x + b = 0. In 2D, this is a straight line.

### Visual explanations

```
  Sigmoid Function:              Decision Boundary (2D):

  P(y=1)                          x2
  1.0 |          ------           |  + + + +
      |        /                  |  + + +/+
  0.5 |------/--------            | + + /+ +
      |    /                      | + /- - -
  0.0 |---/                       |  /- - - -
      +------------- z            +------------- x1
          z = w*x + b              / = boundary (w1*x1 + w2*x2 + b = 0)
```

---

## Hands-on Exploration

1. Implement the sigmoid function and binary cross-entropy loss.
2. Train logistic regression using gradient descent.
3. Visualize the decision boundary on a 2D dataset.
4. Analyze the model's probability outputs and coefficient interpretations.

---

## Live Code

```rust
fn main() {
    println!("=== Logistic Regression ===\n");

    // Generate 2D binary classification dataset
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    // Class 0: exam scores of failing students
    for _ in 0..25 {
        rng = lcg(rng);
        let hours = 1.0 + (rng as f64 / u64::MAX as f64) * 4.0;
        rng = lcg(rng);
        let score = 30.0 + (rng as f64 / u64::MAX as f64) * 30.0;
        features.push(vec![hours, score]);
        labels.push(0.0);
    }

    // Class 1: exam scores of passing students
    for _ in 0..25 {
        rng = lcg(rng);
        let hours = 4.0 + (rng as f64 / u64::MAX as f64) * 5.0;
        rng = lcg(rng);
        let score = 55.0 + (rng as f64 / u64::MAX as f64) * 35.0;
        features.push(vec![hours, score]);
        labels.push(1.0);
    }

    // Standardize features
    let (scaled, means, stds) = standardize(&features);

    // Split
    let n = scaled.len();
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 77);
    let train_n = 38;

    let x_train: Vec<Vec<f64>> = indices[..train_n].iter().map(|&i| scaled[i].clone()).collect();
    let y_train: Vec<f64> = indices[..train_n].iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = indices[train_n..].iter().map(|&i| scaled[i].clone()).collect();
    let y_test: Vec<f64> = indices[train_n..].iter().map(|&i| labels[i]).collect();

    println!("Features: hours_studied, prev_exam_score");
    println!("Train: {}, Test: {}\n", train_n, n - train_n);

    // Train logistic regression
    let lr = 0.1;
    let epochs = 1000;
    let (weights, bias, loss_history) = train_logistic(&x_train, &y_train, lr, epochs);

    println!("--- Trained Model ---");
    println!("  Weights: [{:.4}, {:.4}]", weights[0], weights[1]);
    println!("  Bias: {:.4}", bias);

    // Loss convergence
    println!("\n--- Training Loss Convergence ---");
    let checkpoints = vec![0, 1, 10, 50, 100, 500, 999];
    for &ep in &checkpoints {
        if ep < loss_history.len() {
            let bar = "#".repeat(((loss_history[ep] / loss_history[0]) * 30.0).min(30.0) as usize);
            println!("  Epoch {:>4}: loss = {:.6} |{}", ep, loss_history[ep], bar);
        }
    }

    // Evaluate
    println!("\n--- Evaluation ---");
    let train_probs: Vec<f64> = x_train.iter().map(|x| predict_proba(x, &weights, bias)).collect();
    let test_probs: Vec<f64> = x_test.iter().map(|x| predict_proba(x, &weights, bias)).collect();

    let train_preds: Vec<f64> = train_probs.iter().map(|&p| if p >= 0.5 { 1.0 } else { 0.0 }).collect();
    let test_preds: Vec<f64> = test_probs.iter().map(|&p| if p >= 0.5 { 1.0 } else { 0.0 }).collect();

    let train_acc = accuracy(&y_train, &train_preds);
    let test_acc = accuracy(&y_test, &test_preds);
    let train_loss = binary_cross_entropy(&y_train, &train_probs);
    let test_loss = binary_cross_entropy(&y_test, &test_probs);

    println!("  Train accuracy: {:.1}%, loss: {:.4}", train_acc * 100.0, train_loss);
    println!("  Test accuracy:  {:.1}%, loss: {:.4}", test_acc * 100.0, test_loss);

    // Probability outputs
    println!("\n--- Test Predictions with Probabilities ---");
    println!("{:<6} {:>8} {:>8} {:>8} {:>10}", "Idx", "Actual", "P(y=1)", "Pred", "Correct");
    println!("{}", "-".repeat(42));
    for i in 0..x_test.len() {
        let correct = if test_preds[i] == y_test[i] { "yes" } else { "NO" };
        println!("{:<6} {:>8.0} {:>8.3} {:>8.0} {:>10}",
            i, y_test[i], test_probs[i], test_preds[i], correct);
    }

    // Coefficient interpretation
    println!("\n--- Coefficient Interpretation ---");
    println!("  hours_studied weight:  {:.4}", weights[0]);
    println!("  prev_score weight:     {:.4}", weights[1]);
    println!("  A 1-std increase in hours_studied changes log-odds by {:.4}", weights[0]);
    println!("  A 1-std increase in prev_score changes log-odds by {:.4}", weights[1]);

    let more_important = if weights[0].abs() > weights[1].abs() {
        "hours_studied"
    } else {
        "prev_exam_score"
    };
    println!("  Most important feature: {}", more_important);

    // Threshold analysis
    println!("\n--- Threshold Analysis ---");
    println!("{:<12} {:>10} {:>10} {:>10}", "Threshold", "Accuracy", "Pos Pred", "Neg Pred");
    println!("{}", "-".repeat(42));
    for t in &[0.3, 0.4, 0.5, 0.6, 0.7] {
        let preds: Vec<f64> = test_probs.iter().map(|&p| if p >= *t { 1.0 } else { 0.0 }).collect();
        let acc = accuracy(&y_test, &preds);
        let pos = preds.iter().filter(|&&p| p == 1.0).count();
        let neg = preds.iter().filter(|&&p| p == 0.0).count();
        println!("{:<12.1} {:>9.1}% {:>10} {:>10}", t, acc * 100.0, pos, neg);
    }

    kata_metric("train_accuracy", train_acc);
    kata_metric("test_accuracy", test_acc);
    kata_metric("final_loss", *loss_history.last().unwrap());
    kata_metric("weight_hours", weights[0]);
    kata_metric("weight_score", weights[1]);
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn predict_proba(x: &[f64], weights: &[f64], bias: f64) -> f64 {
    let z: f64 = x.iter().zip(weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + bias;
    sigmoid(z)
}

fn binary_cross_entropy(y: &[f64], probs: &[f64]) -> f64 {
    let n = y.len() as f64;
    let eps = 1e-15;
    y.iter().zip(probs.iter()).map(|(&yi, &pi)| {
        let p = pi.max(eps).min(1.0 - eps);
        -(yi * p.ln() + (1.0 - yi) * (1.0 - p).ln())
    }).sum::<f64>() / n
}

fn train_logistic(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    epochs: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut weights = vec![0.0; p];
    let mut bias = 0.0;
    let mut loss_history = Vec::new();

    for _ in 0..epochs {
        let mut grad_w = vec![0.0; p];
        let mut grad_b = 0.0;
        let mut total_loss = 0.0;

        for i in 0..x.len() {
            let prob = predict_proba(&x[i], &weights, bias);
            let error = prob - y[i];

            for j in 0..p {
                grad_w[j] += error * x[i][j];
            }
            grad_b += error;

            let eps = 1e-15;
            let p_safe = prob.max(eps).min(1.0 - eps);
            total_loss += -(y[i] * p_safe.ln() + (1.0 - y[i]) * (1.0 - p_safe).ln());
        }

        for j in 0..p {
            weights[j] -= lr * grad_w[j] / n;
        }
        bias -= lr * grad_b / n;
        loss_history.push(total_loss / n);
    }

    (weights, bias, loss_history)
}

fn accuracy(actual: &[f64], predicted: &[f64]) -> f64 {
    let correct = actual.iter().zip(predicted.iter())
        .filter(|(a, p)| (**a - **p).abs() < 0.5).count();
    correct as f64 / actual.len() as f64
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    let p = data[0].len();
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        means[j] = col.iter().sum::<f64>() / n;
        stds[j] = (col.iter().map(|x| (x - means[j]).powi(2)).sum::<f64>() / n).sqrt();
    }
    let scaled = data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            if stds[j].abs() < 1e-10 { 0.0 } else { (v - means[j]) / stds[j] }
        }).collect()
    }).collect();
    (scaled, means, stds)
}

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() {
        rng = lcg(rng);
        let j = (rng % (i as u64 + 1)) as usize;
        arr.swap(i, j);
    }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Logistic regression models the probability of the positive class using the sigmoid function, making it ideal for binary classification.
- The model is trained by minimizing binary cross-entropy loss via gradient descent, which provides better gradients than MSE for classification.
- Coefficients are directly interpretable: each represents the change in log-odds per unit increase in the corresponding feature.
- The classification threshold (default 0.5) can be tuned to trade off between precision and recall based on the application's requirements.
