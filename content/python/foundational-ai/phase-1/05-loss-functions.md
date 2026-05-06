# Loss Functions

> Phase 1 — What Does It Mean to Learn? | Kata 1.5

---

## Concept & Intuition

### What problem are we solving?

In Kata 1.4, we used squared error to fit a line, but we never asked: **why squared error?** The loss function is the lens through which a model sees its mistakes. Different lenses highlight different things. Mean Squared Error (MSE) panics about large errors because squaring amplifies them. Mean Absolute Error (MAE) treats all errors proportionally. Huber loss is a hybrid -- it acts like MSE for small errors and MAE for large ones.

The choice of loss function directly shapes what your model optimizes for. If your data has outliers (a mansion in a neighborhood of cottages), MSE will distort the entire fit to reduce that one huge squared error. MAE will shrug it off. Huber lets you tune the threshold where you stop caring exponentially. Same data, same model, three different "best" fits.

This kata forces you to see that the loss function is not a technical footnote -- it is a **design decision** that encodes your priorities about what errors matter most.

### Why naive approaches fail

A common beginner mistake is to always use MSE because it's the default in textbooks. But MSE is the right choice only when all errors are equally unacceptable and outliers are rare. In real datasets with noise, sensor glitches, or extreme values, MSE lets a single outlier hijack the entire model. Without understanding the alternatives, you cannot make an informed choice -- you're just using the default and hoping.

### Mental models

- **Grading rubrics**: MSE is like a teacher who grades on the square of your mistakes -- get one answer very wrong and your grade tanks. MAE is a teacher who deducts points linearly. Huber is a teacher who says "small mistakes count normally, but I cap the penalty for huge blunders."
- **Pain tolerance**: MSE has zero tolerance for outliers (pain grows quadratically). MAE has constant sensitivity (pain grows linearly). Huber has a "pain threshold" -- beyond delta, pain grows only linearly.
- **Volume knob**: MSE turns the volume up on loud errors. MAE keeps the volume flat. Huber has a limiter that clips the peaks.

### Visual explanations

```
Error (x) vs Loss (y) for each function:

          MSE (x^2)           MAE (|x|)          Huber (delta=1)
    Loss                 Loss                 Loss
     9 |      *   *      3 |   *       *      2 |    *       *
     4 |   *         *   2 |  *         *     1 | *           *
     1 | *             * 1 | *           *   .5|*             *
     0 *        *        0 *       *         0 *       *
      -3  -1  0  1  3    -3  -1  0  1  3     -3  -1  0  1  3

MSE: steep walls = outliers dominate
MAE: V-shape = steady penalty
Huber: rounded bottom + gentle slopes = best of both
```

---

## Hands-on Exploration

1. Compute MSE, MAE, and Huber loss on the same set of prediction errors
2. Add a single outlier and watch how each loss function responds
3. Compare which loss function is most and least sensitive to the outlier

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Dataset: predictions vs actuals ---
actual  = np.array([200, 220, 250, 270, 300, 310, 330, 350, 380, 400])
predict = np.array([210, 215, 245, 280, 295, 320, 325, 360, 370, 390])
errors  = actual - predict

print("=== CLEAN DATA (no outliers) ===")
print(f"Errors: {errors}\n")

# --- Define loss functions ---
def mse(errors):
    return (errors ** 2).mean()

def mae(errors):
    return np.abs(errors).mean()

def huber(errors, delta=15.0):
    abs_err = np.abs(errors)
    quadratic = 0.5 * errors ** 2
    linear = delta * abs_err - 0.5 * delta ** 2
    return np.where(abs_err <= delta, quadratic, linear).mean()

for name, fn in [("MSE", mse), ("MAE", mae), ("Huber(d=15)", huber)]:
    print(f"  {name:<16} = {fn(errors):>8.2f}")

# --- Now add a single outlier ---
actual_out  = np.append(actual, 300)
predict_out = np.append(predict, 600)   # wildly wrong prediction
errors_out  = actual_out - predict_out

print(f"\n=== WITH OUTLIER (true=300, predicted=600) ===")
print(f"Errors: {errors_out}\n")

losses_clean = [mse(errors), mae(errors), huber(errors)]
losses_dirty = [mse(errors_out), mae(errors_out), huber(errors_out)]

print(f"{'Loss':<16} {'Clean':>8} {'Outlier':>8} {'Increase':>10}")
print("=" * 46)
for name, c, d in zip(["MSE","MAE","Huber(d=15)"], losses_clean, losses_dirty):
    pct = (d - c) / c * 100
    bar = "#" * int(pct / 20)
    print(f"{name:<16} {c:>8.2f} {d:>8.2f} {pct:>+9.1f}%  {bar}")

# --- Per-error loss comparison ---
print(f"\n=== PER-ERROR LOSS (see how outlier dominates MSE) ===")
print(f"{'Error':>7} {'MSE':>10} {'MAE':>10} {'Huber':>10}")
print("-" * 40)
test_errors = np.array([-5, -10, 10, -300])
for e in test_errors:
    e_arr = np.array([e], dtype=float)
    print(f"{e:>+7d} {mse(e_arr):>10.1f} {mae(e_arr):>10.1f} {huber(e_arr):>10.1f}")

print("\nNotice: error of -300 costs 90000 in MSE but only 4387.5 in Huber")
```

---

## Key Takeaways

- **MSE amplifies large errors quadratically.** One outlier with error 10x bigger produces 100x more loss -- it dominates the optimization.
- **MAE treats all errors linearly.** It's robust to outliers but its gradient is discontinuous at zero, making optimization trickier.
- **Huber loss is a practical compromise.** Quadratic for small errors (smooth, easy to optimize), linear for large errors (outlier-resistant).
- **The loss function encodes your priorities.** Choosing MSE means "large errors are unacceptable." Choosing MAE means "all errors matter equally."
- **Always test with outliers.** A loss function that looks fine on clean data may fail catastrophically when real-world noise appears.
