# Autograd

> Phase 2 — Learning & Optimization | Kata 2.6

---

## Concept & Intuition

### What problem are we solving?

In Phase 1, we computed gradients by hand -- taking the derivative of the loss function, plugging in values, and coding the formulas ourselves. This works for simple models, but for a neural network with millions of parameters, manually deriving and coding gradients is impossible. **Autograd** is PyTorch's automatic differentiation engine. You write the forward pass (compute the prediction and loss), call `.backward()`, and PyTorch computes all the gradients for you.

Autograd works by recording every operation you perform on tensors with `requires_grad=True` into a **computation graph**. When you call `.backward()` on the final loss, PyTorch walks backward through this graph, applying the chain rule at each node to compute the derivative of the loss with respect to every parameter. This is **backpropagation** -- the algorithm that makes training neural networks feasible.

The mental shift here is profound: you no longer think about gradients. You think about the **forward computation** -- how inputs become predictions become loss. PyTorch handles the rest. This is what makes deep learning practical: you can experiment with any architecture and the gradients come for free.

### Why naive approaches fail

Manually computing gradients for even a 2-layer network requires pages of calculus and careful index bookkeeping. One sign error and the model trains in the wrong direction. One missing chain rule term and a layer stops learning. Autograd eliminates this entire class of bugs. It also handles complex operations (like softmax or batch normalization) whose gradients are non-trivial to derive by hand.

### Mental models

- **Recording tape**: when `requires_grad=True`, PyTorch records every operation on a tape. `.backward()` plays the tape in reverse, computing derivatives at each step. After use, the tape is erased (to save memory).
- **Chain rule machine**: if `loss = f(g(h(x)))`, autograd computes `dloss/dx = f'(g(h(x))) * g'(h(x)) * h'(x)` by chaining derivatives backward through each operation. You just write `f(g(h(x)))` and call `.backward()`.
- **Dependency tracker**: the computation graph is a map of "what depends on what." The gradient of the loss with respect to parameter p is computed by following all paths from loss back to p and accumulating contributions.

### Visual explanations

```
Forward pass (you write this):

  x -----> [* w] -----> [+ b] -----> pred -----> [(pred-y)^2] -----> loss
  (input)   (multiply)   (add)                    (MSE)

Backward pass (autograd does this):

  loss <--- d/d(pred) <--- d/d(sum) <--- d/d(w), d/d(b) <--- chain rule
                                          |           |
                                          v           v
                                       w.grad       b.grad

  You call:   loss.backward()
  You read:   w.grad, b.grad
  That's it.
```

---

## Hands-on Exploration

1. Create tensors with `requires_grad=True` and perform a forward pass
2. Call `.backward()` and inspect the `.grad` attributes
3. Verify autograd's answers by computing gradients manually

---

## Live Code

```python
import torch

torch.manual_seed(42)

# --- Simple example: y = x^2 at x=3 ---
print("=== BASIC AUTOGRAD: y = x^2 ===")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # forward pass
y.backward()       # compute dy/dx
print(f"x = {x.item():.1f}")
print(f"y = x^2 = {y.item():.1f}")
print(f"dy/dx (autograd) = {x.grad.item():.1f}")
print(f"dy/dx (manual)   = 2*x = {2*x.item():.1f}\n")

# --- Chain rule: z = (x^2 + 1)^3 ---
print("=== CHAIN RULE: z = (x^2 + 1)^3 ===")
x = torch.tensor(2.0, requires_grad=True)
u = x ** 2 + 1    # u = 5
z = u ** 3         # z = 125
z.backward()
# dz/dx = 3*u^2 * 2*x = 3*25*4 = 300
print(f"x = {x.item():.1f}, u = x^2+1 = {u.item():.1f}, z = u^3 = {z.item():.1f}")
print(f"dz/dx (autograd) = {x.grad.item():.1f}")
print(f"dz/dx (manual)   = 3*u^2 * 2*x = {3 * u.item()**2 * 2 * x.item():.1f}\n")

# --- Linear regression with autograd ---
print("=== LINEAR REGRESSION VIA AUTOGRAD ===")
# Data
sqft = torch.tensor([1.1, 1.3, 1.6, 1.8, 2.1, 2.4], dtype=torch.float32)
price = torch.tensor([190., 230., 310., 350., 420., 480.])

# Parameters (learnable)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.05

print(f"{'Step':>4} {'Loss':>10} {'w':>8} {'b':>8} {'w.grad':>8} {'b.grad':>8}")
print("=" * 52)

for step in range(8):
    # Forward pass
    pred = w * sqft + b
    loss = ((pred - price) ** 2).mean()

    # Backward pass (computes gradients)
    loss.backward()

    # Print before updating
    print(f"{step:>4} {loss.item():>10.2f} {w.item():>8.3f} "
          f"{b.item():>8.3f} {w.grad.item():>8.1f} {b.grad.item():>8.1f}")

    # Update parameters (gradient descent)
    with torch.no_grad():       # don't track this update
        w -= lr * w.grad
        b -= lr * b.grad

    # CRITICAL: zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

print(f"\nFinal: price = {w.item():.2f} * sqft + {b.item():.2f}")

# --- Why zero_() matters ---
print("\n=== WHY .zero_() IS CRITICAL ===")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"After 1st backward: x.grad = {x.grad.item():.1f}")
y = x ** 2
y.backward()
print(f"After 2nd backward: x.grad = {x.grad.item():.1f}  (accumulated!)")
x.grad.zero_()
y = x ** 2
y.backward()
print(f"After zero + backward: x.grad = {x.grad.item():.1f}  (correct)")
```

---

## Key Takeaways

- **`requires_grad=True` tells PyTorch to track operations.** Every computation on this tensor is recorded in a computation graph.
- **`.backward()` applies the chain rule automatically.** It walks the graph in reverse and fills in `.grad` for every leaf tensor.
- **You must zero gradients between iterations.** PyTorch accumulates gradients by default -- calling `.zero_()` prevents stale gradients from corrupting updates.
- **`with torch.no_grad()` disables tracking.** Use it for parameter updates so they don't become part of the computation graph.
- **Autograd eliminates manual calculus.** You define the forward pass; PyTorch derives and computes all gradients. This is what makes deep learning practical.
