# Error Surfaces

> Phase 1 — What Does It Mean to Learn? | Kata 1.6

---

## Concept & Intuition

### What problem are we solving?

In linear regression (y = mx + b), we found the optimal m and b using the normal equation. But what does "optimal" actually look like? If you computed the loss for every possible combination of (m, b), you would get a **surface** -- a landscape of error values. The optimal parameters sit at the bottom of a bowl-shaped valley. This kata makes that landscape visible.

Understanding error surfaces is the gateway to understanding optimization. Gradient descent, which we cover in Phase 2, is literally the process of rolling downhill on this surface. If you can see the surface, you can understand why learning rates matter, why some problems are harder than others, and why getting stuck in local minima is a real concern for non-convex problems. For linear regression with MSE, the surface is always a smooth bowl (convex), which is why the normal equation works -- there is exactly one minimum.

Visualizing the error surface also reveals how sensitive your model is to each parameter. A steep surface in the m direction means small changes in slope cause big changes in loss. A flat surface in the b direction means the intercept barely matters. This intuition carries directly into understanding gradients.

### Why naive approaches fail

Without seeing the error surface, optimization is a black box. Students often wonder: "Why does gradient descent converge? Why does the learning rate matter? Why does my model get stuck?" These questions only make sense when you can picture the terrain. Trying to understand optimization without error surfaces is like trying to navigate a mountain range with your eyes closed.

### Mental models

- **Hiking a bowl-shaped valley**: you're blindfolded in a bowl. You feel the slope under your feet (the gradient) and step downhill. The bottom of the bowl is the optimal parameters. Steeper slopes mean you're far from the minimum.
- **Topographic map**: the contour lines show regions of equal loss. Tightly packed contours mean the loss changes rapidly. The center of the innermost contour is the minimum.
- **Hot/cold game**: each (m, b) pair gets a temperature (its loss). You want to find the coldest spot. The surface tells you the temperature everywhere.

### Visual explanations

```
Error Surface for y = mx + b (top-down contour view):

        b (intercept)
        ^
   60   |  . . . . . . 800 . . . . . .
   40   |  . . . . 400 . . . 400 . . .
   20   |  . . . 200 . . 100 . 200 . .
    0   |  . . 200 . . [MIN] . . 200 .
  -20   |  . . . 200 . . 100 . 200 . .
  -40   |  . . . . 400 . . . 400 . . .
  -60   |  . . . . . . 800 . . . . . .
        +------------------------------> m (slope)
          0.0   0.1   0.2   0.3   0.4

  Concentric ellipses of equal loss surround the minimum.
  The minimum is where gradient = (0, 0).
```

---

## Hands-on Exploration

1. Create a grid of (m, b) values and compute the MSE at each point
2. Find the minimum loss on the grid and compare with the normal equation solution
3. Display the error surface as a text-based contour map

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Dataset ---
sqft = np.array([1100, 1300, 1400, 1600, 1700, 1800, 2100, 2400, 1500, 1900])
prices = np.array([190, 230, 250, 310, 340, 350, 420, 480, 280, 370])

# --- Exact solution via normal equation ---
X = np.column_stack([sqft, np.ones(len(sqft))])
w = np.linalg.inv(X.T @ X) @ X.T @ prices
m_opt, b_opt = w[0], w[1]
print(f"Normal equation solution: m={m_opt:.4f}, b={b_opt:.2f}")

# --- Build error surface over grid of (m, b) ---
m_vals = np.linspace(0.05, 0.35, 50)
b_vals = np.linspace(-100, 100, 50)
loss_grid = np.zeros((len(b_vals), len(m_vals)))

for i, b in enumerate(b_vals):
    for j, m in enumerate(m_vals):
        preds = m * sqft + b
        loss_grid[i, j] = ((prices - preds) ** 2).mean()

# --- Find grid minimum ---
min_idx = np.unravel_index(loss_grid.argmin(), loss_grid.shape)
grid_b = b_vals[min_idx[0]]
grid_m = m_vals[min_idx[1]]
print(f"Grid search minimum:     m={grid_m:.4f}, b={grid_b:.2f}")
print(f"Min MSE on grid:         {loss_grid.min():.2f}\n")

# --- Text-based contour map ---
print("=== ERROR SURFACE (contour map) ===")
print("  Rows = intercept (b), Cols = slope (m)")
print(f"  m range: [{m_vals[0]:.2f}, {m_vals[-1]:.2f}]")
print(f"  b range: [{b_vals[0]:.0f}, {b_vals[-1]:.0f}]")
print(f"  '*' = minimum region\n")

# Downsample to 20x30 for display
rows = np.linspace(0, len(b_vals)-1, 20, dtype=int)
cols = np.linspace(0, len(m_vals)-1, 35, dtype=int)
levels = [200, 500, 1000, 2500, 5000, 10000, 25000]
symbols = "*o.+xX#@"

header = "  b\\m  " + "".join(f"{m_vals[c]:.2f} "[::5] for c in cols[::5])
print(f"       {'m -->':^35}")
for i in rows[::-1]:
    row = ""
    for j in cols:
        v = loss_grid[i, j]
        sym = symbols[-1]
        for k, lev in enumerate(levels):
            if v < lev:
                sym = symbols[k]
                break
        row += sym
    print(f"  {b_vals[i]:>+5.0f} |{row}|")
print(f"  b    +{'-'*35}+")
print(f"  {'':7}m = {m_vals[cols[0]]:.2f}{'':>20}m = {m_vals[cols[-1]]:.2f}")

print(f"\n  Legend: * <200  o <500  . <1k  + <2.5k  x <5k  X <10k  # <25k  @ >=25k")

# --- Loss along a slice (fix b at optimal, sweep m) ---
print(f"\n=== SLICE: fix b={b_opt:.0f}, vary m ===")
m_slice = np.linspace(0.05, 0.35, 15)
for m in m_slice:
    preds = m * sqft + b_opt
    loss = ((prices - preds) ** 2).mean()
    bar = "#" * int(loss / 300)
    marker = " <-- min" if abs(m - m_opt) < 0.02 else ""
    print(f"  m={m:.3f}  MSE={loss:>8.1f} {bar}{marker}")
```

---

## Key Takeaways

- **The error surface maps every parameter combination to a loss value.** For y = mx + b, it's a 2D surface over (m, b) space.
- **MSE for linear models is always convex (bowl-shaped).** There is exactly one minimum, which is why the normal equation works.
- **The minimum of the surface is the optimal solution.** Gradient descent is the process of walking downhill on this surface.
- **Contour lines show equal-loss regions.** Tight contours mean the loss is sensitive to that parameter; wide contours mean it's not.
- **Grid search is brute-force optimization.** It works but scales terribly -- gradient descent is the efficient alternative you'll learn in Phase 2.
