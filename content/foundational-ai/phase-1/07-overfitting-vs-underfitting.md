# Overfitting vs Underfitting

> Phase 1 — What Does It Mean to Learn? | Kata 1.7

---

## Concept & Intuition

### What problem are we solving?

A model can fail in two opposite ways. **Underfitting** means the model is too simple to capture the real pattern -- like drawing a straight line through curved data. **Overfitting** means the model is too complex and memorizes noise instead of learning the pattern -- like drawing a wiggly line that passes through every training point but predicts terribly on new data.

This is the **bias-variance tradeoff**, the central tension in all of machine learning. A degree-1 polynomial (straight line) has high bias -- it cannot bend to fit a curve. A degree-9 polynomial has high variance -- it bends to fit every random fluctuation. Somewhere in between is the sweet spot where the model is complex enough to capture the real pattern but simple enough to generalize.

The key insight: **training error always decreases as model complexity increases**, because a more flexible model can always fit the training data better. But **test error follows a U-shape** -- it decreases at first (reducing underfitting), hits a minimum, then increases (overfitting kicks in). The gap between train and test error is the fingerprint of overfitting.

### Why naive approaches fail

If you only look at training error, you will always choose the most complex model -- and overfit badly. A degree-9 polynomial can perfectly interpolate 10 points (zero training error!), but its predictions between and beyond those points will be wildly wrong. The only way to detect overfitting is to evaluate on data the model has never seen. This is why train/test splits are non-negotiable.

### Mental models

- **Studying for an exam**: Underfitting is skimming the textbook (you don't know enough). Overfitting is memorizing the practice test answers verbatim (you can't handle new questions). Good learning is understanding the concepts (you can generalize).
- **Fitting a suit**: Too loose (underfit) looks sloppy. Too tight (overfit to your exact measurements on one day) rips when you move. A good fit has some room to breathe.
- **Connect-the-dots**: A straight line through dots is underfitting. A crazy curve through every dot is overfitting. The true shape is somewhere in between.

### Visual explanations

```
Polynomial degree vs error:

Error
  |  X                               X
  |   X                            X
  |    X                         X
  |     X                      X       <-- test error (U-shape)
  |      X                   X
  |        X     X   X    X
  |          X X   X   X
  |  *                                 <-- train error (always drops)
  |    *  *
  |        *  *
  |             *  *  *
  |                      *  *  *  *
  +----+----+----+----+----+----+----> Degree
       1    2    3    5    7    9

  Sweet spot: where test error is minimized (degree ~3)
  Overfit zone: train error near 0 but test error explodes
```

---

## Hands-on Exploration

1. Generate data from a known cubic function with noise
2. Fit polynomials of degree 1, 3, 5, 7, 9 on the training set
3. Compute train and test error for each -- watch the U-shaped test curve appear

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- True function: cubic with noise ---
def true_fn(x):
    return 0.5 * x ** 3 - 2 * x ** 2 + x + 3

x_all = np.linspace(-1, 4, 30)
y_all = true_fn(x_all) + np.random.randn(30) * 2.0

# --- Train/test split ---
indices = np.random.permutation(30)
x_train, y_train = x_all[indices[:20]], y_all[indices[:20]]
x_test, y_test   = x_all[indices[20:]], y_all[indices[20:]]

print(f"Training points: {len(x_train)}, Test points: {len(x_test)}\n")

# --- Fit polynomials of increasing degree ---
degrees = [1, 2, 3, 5, 7, 9]
train_errors = []
test_errors = []

print(f"{'Degree':>6} {'Train MSE':>12} {'Test MSE':>12} {'Status':<15}")
print("=" * 50)

for deg in degrees:
    # Fit polynomial using least squares
    coeffs = np.polyfit(x_train, y_train, deg)
    poly = np.poly1d(coeffs)

    train_pred = poly(x_train)
    test_pred  = poly(x_test)

    train_mse = ((y_train - train_pred) ** 2).mean()
    test_mse  = ((y_test - test_pred) ** 2).mean()

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    # Diagnose
    if test_mse > train_mse * 3 and deg > 3:
        status = "OVERFIT"
    elif train_mse > 10:
        status = "UNDERFIT"
    else:
        status = "good"

    print(f"{deg:>6} {train_mse:>12.3f} {test_mse:>12.3f} {status:<15}")

# --- Visual: train vs test error bar chart ---
print(f"\n=== TRAIN vs TEST ERROR ===")
print(f"{'Deg':<5} {'Train':<30} {'Test':<30}")
scale = max(max(train_errors), max(test_errors))
# Cap display for readability
cap = min(scale, 200)
for i, deg in enumerate(degrees):
    tr = min(train_errors[i], cap)
    te = min(test_errors[i], cap)
    tr_bar = "#" * int(tr / cap * 25)
    te_bar = "#" * int(te / cap * 25)
    tr_display = f"{train_errors[i]:.1f}"
    te_display = f"{test_errors[i]:.1f}"
    if test_errors[i] > cap:
        te_display += "!"
    print(f"  {deg:<3} T {tr_bar:<25} {tr_display}")
    print(f"      V {te_bar:<25} {te_display}")

# --- Show the gap ---
print(f"\n=== OVERFIT DIAGNOSTIC (gap = test - train) ===")
for i, deg in enumerate(degrees):
    gap = test_errors[i] - train_errors[i]
    gap_bar = "#" * min(int(abs(gap) / 2), 40)
    print(f"  Degree {deg}: gap = {gap:>+10.2f}  {gap_bar}")

# --- Key comparison ---
best_idx = np.argmin(test_errors)
print(f"\nBest model: degree {degrees[best_idx]} "
      f"(test MSE = {test_errors[best_idx]:.3f})")
print(f"Worst overfit: degree {degrees[-1]} "
      f"(train={train_errors[-1]:.3f}, test={test_errors[-1]:.3f})")
print(f"\nLesson: degree-{degrees[-1]} memorizes training data "
      f"but fails on new data.")
```

---

## Key Takeaways

- **Underfitting = too simple, overfitting = too complex.** Both produce bad predictions, but for opposite reasons.
- **Training error always decreases with complexity.** This is why training error alone is a misleading guide.
- **Test error follows a U-shape.** It decreases as you reduce underfitting, then increases as overfitting kicks in.
- **The train-test gap is the fingerprint of overfitting.** A large gap means the model memorized training noise.
- **The bias-variance tradeoff is the central tension of ML.** Every model choice -- complexity, regularization, data size -- is a negotiation between these two forces.
