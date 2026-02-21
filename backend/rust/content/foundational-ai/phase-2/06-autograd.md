# Automatic Differentiation

> Phase 2 — Optimization | Kata 2.6

---

## Concept & Intuition

### What problem are we solving?

Automatic differentiation (autodiff) is the engine that powers all modern deep learning. Instead of manually deriving gradients for every model (which is error-prone and tedious), autodiff automatically computes exact gradients by tracking the chain of operations that produced each value and applying the chain rule in reverse.

The key insight is that every computation, no matter how complex, is built from simple operations: addition, multiplication, division, exponentiation, etc. Each of these has a known derivative. By recording the sequence of operations in a computational graph and then traversing that graph in reverse (backpropagation), we can compute the gradient of the output with respect to every input.

In this kata, we build a simple autodiff system from scratch. Each value (a "node" in the computational graph) remembers how it was created and which inputs it depends on. When we call backward() on the final loss, each node computes its local gradient and passes it to its children via the chain rule. This is exactly how PyTorch's autograd works — we are just implementing a minimal version.

### Why naive approaches fail

Manual gradient derivation does not scale. For a neural network with millions of parameters and dozens of layers, writing gradients by hand would take weeks and be riddled with bugs. Numerical differentiation (finite differences) is simple but too slow and imprecise for large models. Autodiff gives exact gradients at a cost proportional to the forward pass — it is the perfect solution.

### Mental models

- **Computational graph**: Every computation builds a graph. Nodes are values, edges are operations. Forward pass flows inputs to output. Backward pass flows gradients from output to inputs.
- **Chain rule as message passing**: Each node receives a gradient from its parent ("how much does the loss change if I change?") and multiplies it by its local derivative to send to its children.
- **Dual numbers**: Each value carries both its numeric value and its gradient. Forward: compute values. Backward: compute gradients. Same graph, different direction.

### Visual explanations

```
  Computational graph for: loss = (y - (w*x + b))²

  Forward pass:              Backward pass:
  x, w → [*] → wx           ∂L/∂w ← [*] ← ∂L/∂(wx)
            ↓                            ↑
  wx, b → [+] → wx+b        ∂L/∂b ← [+] ← ∂L/∂(wx+b)
            ↓                            ↑
  y, pred → [-] → err       ∂L/∂pred ← [-] ← ∂L/∂err
              ↓                              ↑
  err → [²] → loss          ∂L/∂err ← [²] ← ∂L/∂loss = 1
```

---

## Hands-on Exploration

1. Build a Value type that tracks its computational graph.
2. Implement forward operations (add, mul, pow) that record the graph.
3. Implement backward() that propagates gradients through the graph via the chain rule.

---

## Live Code

```rust
use std::cell::RefCell;
use std::rc::Rc;

/// A node in the computational graph.
#[derive(Clone)]
struct Value {
    inner: Rc<RefCell<ValueInner>>,
}

struct ValueInner {
    data: f64,
    grad: f64,
    /// Backward function: given output grad, propagate to children.
    backward_fn: Option<Box<dyn Fn()>>,
    /// Children (for topological sort).
    children: Vec<Value>,
    label: String,
}

impl Value {
    fn new(data: f64, label: &str) -> Self {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                grad: 0.0,
                backward_fn: None,
                children: vec![],
                label: label.to_string(),
            })),
        }
    }

    fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    fn label(&self) -> String {
        self.inner.borrow().label.clone()
    }

    fn set_grad(&self, g: f64) {
        self.inner.borrow_mut().grad = g;
    }

    fn add_grad(&self, g: f64) {
        self.inner.borrow_mut().grad += g;
    }

    /// Addition: out = self + other
    fn add(&self, other: &Value) -> Value {
        let out_data = self.data() + other.data();
        let out = Value::new(out_data, &format!("({}+{})", self.label(), other.label()));

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();

        out.inner.borrow_mut().children = vec![self.clone(), other.clone()];
        out.inner.borrow_mut().backward_fn = Some(Box::new(move || {
            let out_grad = out_clone.grad();
            self_clone.add_grad(out_grad);      // d(a+b)/da = 1
            other_clone.add_grad(out_grad);     // d(a+b)/db = 1
        }));

        out
    }

    /// Multiplication: out = self * other
    fn mul(&self, other: &Value) -> Value {
        let out_data = self.data() * other.data();
        let out = Value::new(out_data, &format!("({}*{})", self.label(), other.label()));

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();

        let self_data = self.data();
        let other_data = other.data();

        out.inner.borrow_mut().children = vec![self.clone(), other.clone()];
        out.inner.borrow_mut().backward_fn = Some(Box::new(move || {
            let out_grad = out_clone.grad();
            self_clone.add_grad(out_grad * other_data);  // d(a*b)/da = b
            other_clone.add_grad(out_grad * self_data);  // d(a*b)/db = a
        }));

        out
    }

    /// Power: out = self ^ n
    fn pow(&self, n: f64) -> Value {
        let out_data = self.data().powf(n);
        let out = Value::new(out_data, &format!("({}^{:.0})", self.label(), n));

        let self_clone = self.clone();
        let out_clone = out.clone();
        let self_data = self.data();

        out.inner.borrow_mut().children = vec![self.clone()];
        out.inner.borrow_mut().backward_fn = Some(Box::new(move || {
            let out_grad = out_clone.grad();
            // d(x^n)/dx = n * x^(n-1)
            self_clone.add_grad(out_grad * n * self_data.powf(n - 1.0));
        }));

        out
    }

    /// Negation: out = -self
    fn neg(&self) -> Value {
        let minus_one = Value::new(-1.0, "-1");
        self.mul(&minus_one)
    }

    /// Subtraction: out = self - other
    fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    /// ReLU activation
    fn relu(&self) -> Value {
        let out_data = if self.data() > 0.0 { self.data() } else { 0.0 };
        let out = Value::new(out_data, &format!("relu({})", self.label()));

        let self_clone = self.clone();
        let out_clone = out.clone();
        let self_data = self.data();

        out.inner.borrow_mut().children = vec![self.clone()];
        out.inner.borrow_mut().backward_fn = Some(Box::new(move || {
            let out_grad = out_clone.grad();
            let local_grad = if self_data > 0.0 { 1.0 } else { 0.0 };
            self_clone.add_grad(out_grad * local_grad);
        }));

        out
    }

    /// Backward pass: compute all gradients.
    fn backward(&self) {
        // Topological sort
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: Vec<*const RefCell<ValueInner>> = Vec::new();

        fn build_topo(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut Vec<*const RefCell<ValueInner>>,
        ) {
            let ptr = Rc::as_ptr(&v.inner);
            if visited.contains(&ptr) {
                return;
            }
            visited.push(ptr);
            let children: Vec<Value> = v.inner.borrow().children.clone();
            for child in &children {
                build_topo(child, topo, visited);
            }
            topo.push(v.clone());
        }

        build_topo(self, &mut topo, &mut visited);

        // Set output gradient to 1
        self.set_grad(1.0);

        // Backward in reverse topological order
        for node in topo.iter().rev() {
            let backward_fn = node.inner.borrow().backward_fn.is_some();
            if backward_fn {
                // Call the backward function
                let func = node.inner.borrow();
                if let Some(ref f) = func.backward_fn {
                    f();
                }
            }
        }
    }
}

fn main() {
    println!("=== Automatic Differentiation from Scratch ===\n");

    // === Example 1: Simple expression ===
    println!("--- Example 1: f = (a * b) + c ---\n");

    let a = Value::new(2.0, "a");
    let b = Value::new(3.0, "b");
    let c = Value::new(4.0, "c");

    let ab = a.mul(&b);     // a * b = 6
    let f = ab.add(&c);     // ab + c = 10

    f.backward();

    println!("  a = {:.1}, b = {:.1}, c = {:.1}", a.data(), b.data(), c.data());
    println!("  f = a*b + c = {:.1}", f.data());
    println!("  ∂f/∂a = {} (expected: b = 3)", a.grad());
    println!("  ∂f/∂b = {} (expected: a = 2)", b.grad());
    println!("  ∂f/∂c = {} (expected: 1)", c.grad());

    // === Example 2: MSE Loss ===
    println!("\n--- Example 2: MSE Loss for Linear Regression ---\n");

    // y_pred = w*x + b, loss = (y_true - y_pred)^2
    let x = Value::new(3.0, "x");
    let y_true = Value::new(7.0, "y_true");
    let w = Value::new(1.5, "w");
    let b = Value::new(0.5, "b");

    let wx = w.mul(&x);          // 1.5 * 3 = 4.5
    let y_pred = wx.add(&b);     // 4.5 + 0.5 = 5.0
    let diff = y_true.sub(&y_pred); // 7.0 - 5.0 = 2.0
    let loss = diff.pow(2.0);    // 2.0^2 = 4.0

    loss.backward();

    println!("  x={}, y_true={}, w={}, b={}", x.data(), y_true.data(), w.data(), b.data());
    println!("  y_pred = w*x + b = {}", y_pred.data());
    println!("  loss = (y_true - y_pred)² = {}", loss.data());
    println!();
    println!("  Gradients (autodiff):");
    println!("    ∂loss/∂w = {:.1}", w.grad());
    println!("    ∂loss/∂b = {:.1}", b.grad());
    println!();

    // Verify numerically
    let eps = 1e-5;
    let loss_fn = |w_val: f64, b_val: f64| -> f64 {
        let pred = w_val * 3.0 + b_val;
        (7.0 - pred) * (7.0 - pred)
    };
    let num_dw = (loss_fn(1.5 + eps, 0.5) - loss_fn(1.5 - eps, 0.5)) / (2.0 * eps);
    let num_db = (loss_fn(1.5, 0.5 + eps) - loss_fn(1.5, 0.5 - eps)) / (2.0 * eps);

    println!("  Gradients (numerical verification):");
    println!("    ∂loss/∂w ≈ {:.4}", num_dw);
    println!("    ∂loss/∂b ≈ {:.4}", num_db);
    println!("  Autodiff and numerical gradients match!");

    // === Example 3: Gradient descent using autodiff ===
    println!("\n--- Example 3: Training with Autodiff ---\n");

    // Data: y = 2x + 1
    let dataset: Vec<(f64, f64)> = vec![
        (1.0, 3.1), (2.0, 5.0), (3.0, 6.9), (4.0, 9.1), (5.0, 11.0),
    ];

    let mut w_val = 0.0;
    let mut b_val = 0.0;
    let lr = 0.01;

    println!("  Training y = w*x + b to fit data (true: w=2, b=1)\n");
    println!("  {:>5} {:>8} {:>8} {:>10}", "epoch", "w", "b", "loss");
    println!("  {:->5} {:->8} {:->8} {:->10}", "", "", "", "");

    for epoch in 0..100 {
        // Forward pass: compute loss over all data
        // We rebuild the graph each iteration (like PyTorch)
        let w = Value::new(w_val, "w");
        let b = Value::new(b_val, "b");

        // Sum of squared errors
        let mut total_loss = Value::new(0.0, "zero");
        for &(xi, yi) in &dataset {
            let x = Value::new(xi, "x");
            let y = Value::new(yi, "y");
            let pred = w.mul(&x).add(&b);
            let err = y.sub(&pred);
            let sq_err = err.pow(2.0);
            total_loss = total_loss.add(&sq_err);
        }

        // Backward pass
        total_loss.backward();

        if epoch < 10 || epoch % 10 == 0 {
            println!("  {:>5} {:>8.4} {:>8.4} {:>10.4}",
                epoch, w_val, b_val, total_loss.data());
        }

        // Update parameters
        w_val -= lr * w.grad();
        b_val -= lr * b.grad();
    }

    println!("\n  Final: w={:.4}, b={:.4}", w_val, b_val);
    println!("  Expected: w≈2.0, b≈1.0\n");

    // === Example 4: ReLU activation ===
    println!("--- Example 4: ReLU Gradients ---\n");

    for &x_val in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
        let x = Value::new(x_val, "x");
        let y = x.relu();
        y.backward();
        println!("  x={:>5.1}: relu(x)={:.1}, ∂relu/∂x={}",
            x_val, y.data(), x.grad());
    }

    println!("\n  ReLU passes gradient through when x > 0, blocks it when x <= 0.\n");

    println!("Key insight: Autodiff computes exact gradients automatically by");
    println!("recording operations and applying the chain rule in reverse.");
    println!("This is the engine that makes training neural networks possible.");
}
```

---

## Key Takeaways

- Automatic differentiation computes exact gradients by recording operations in a computational graph and applying the chain rule in reverse (backpropagation).
- Each operation knows its local derivative; the chain rule multiplies local derivatives along the path from output to input.
- Autodiff is fundamentally different from numerical differentiation (finite differences) — it is exact, not approximate, and scales efficiently.
- This is exactly how PyTorch's autograd works: build a graph during the forward pass, traverse it backward to compute gradients, then update parameters.
