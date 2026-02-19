# PyTorch Tensors

> Phase 2 — Learning & Optimization | Kata 2.5

---

## Concept & Intuition

### What problem are we solving?

So far we have used numpy arrays for everything. Numpy is great for CPU-based math, but it has two critical limitations: it cannot run on GPUs, and it cannot automatically compute gradients. **PyTorch tensors** are the replacement -- they look and feel almost identical to numpy arrays, but they unlock GPU acceleration and automatic differentiation (autograd), which is the foundation of all modern deep learning.

A PyTorch tensor is a multi-dimensional array, just like a numpy array. You can add them, multiply them, reshape them, and index into them with nearly identical syntax. The key difference is that tensors carry extra machinery under the hood: a device (CPU or GPU), a dtype, and optionally a computation graph for tracking gradients. Think of tensors as numpy arrays that went to graduate school.

This kata is about building fluency. You need to create tensors, convert between numpy and torch, perform operations, and understand the small but important differences. This is the vocabulary you will use for every kata from here forward.

### Why naive approaches fail

Students often try to mix numpy and PyTorch carelessly -- passing a numpy array to a function that expects a tensor, or vice versa. This causes subtle bugs. A numpy operation in the middle of a PyTorch computation graph silently breaks gradient tracking. Understanding the boundary between the two worlds, and how to cross it cleanly, prevents hours of debugging later.

### Mental models

- **Numpy with superpowers**: a tensor is a numpy array that can also track its own history (for gradients) and live on a GPU. If you know numpy, you already know 90% of tensor operations.
- **Bilingual container**: tensors can speak both numpy and torch. `.numpy()` converts to numpy, `torch.from_numpy()` converts back. But the translation has rules -- shared memory means changing one changes the other.
- **Upgrade path**: Phase 1 used numpy because it's simpler. Phase 2 upgrades to PyTorch because we need gradients. The math is the same; only the container changes.

### Visual explanations

```
Numpy Array                     PyTorch Tensor
+---+---+---+                   +---+---+---+
| 1 | 2 | 3 |  np.array(...)   | 1 | 2 | 3 |  torch.tensor(...)
+---+---+---+                   +---+---+---+
    |                               |
    | .mean(), +, @, reshape        | .mean(), +, @, reshape
    |                               |
    v                               v
  Result (numpy)                  Result (tensor)
                                    + device (cpu/cuda)
                                    + dtype (float32)
                                    + grad tracking

  Conversion:
  np_array ----torch.from_numpy()----> tensor  (shared memory!)
  tensor   ----.numpy()--------------> np_array (shared memory!)
  tensor   ----.detach().numpy()-----> np_array (safe copy path)
```

---

## Hands-on Exploration

1. Create tensors from scratch and from numpy arrays -- compare the syntax
2. Perform arithmetic, matrix multiplication, and reshaping on tensors
3. Convert back and forth between numpy and torch, observing shared memory behavior

---

## Live Code

```python
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

# --- Creating tensors ---
t1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
t2 = torch.zeros(3, 4)
t3 = torch.randn(3, 3)       # normal distribution
t4 = torch.arange(0, 10, 2)  # like np.arange

print("=== TENSOR CREATION ===")
print(f"t1: {t1}  shape={t1.shape}  dtype={t1.dtype}")
print(f"t2 (zeros 3x4):\n{t2}")
print(f"t3 (randn 3x3):\n{t3}")
print(f"t4 (arange): {t4}\n")

# --- Numpy <-> Torch conversion ---
print("=== NUMPY <-> TORCH ===")
np_arr = np.array([10.0, 20.0, 30.0])
t_from_np = torch.from_numpy(np_arr)
print(f"numpy:  {np_arr}  (dtype={np_arr.dtype})")
print(f"tensor: {t_from_np}  (dtype={t_from_np.dtype})")

# Shared memory: modifying one changes the other
np_arr[0] = 999.0
print(f"After np_arr[0]=999: tensor={t_from_np}  (shared memory!)")

# Safe conversion (no shared memory)
t_safe = torch.tensor(np_arr)  # copies data
np_arr[0] = 0.0
print(f"After np_arr[0]=0:   safe tensor={t_safe}  (independent copy)\n")

# --- Arithmetic operations (same as numpy) ---
print("=== ARITHMETIC ===")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a + b   = {a + b}")
print(f"a * b   = {a * b}       (element-wise)")
print(f"a @ b   = {a @ b:.1f}         (dot product)")
print(f"a.sum() = {a.sum():.1f}")
print(f"a.mean()= {a.mean():.1f}\n")

# --- Matrix operations ---
print("=== MATRIX OPS ===")
M = torch.randn(3, 4)
v = torch.randn(4)
result = M @ v     # matrix-vector multiply
print(f"M (3x4) @ v (4,) = shape {result.shape}")
print(f"Result: {result}\n")

# --- Reshaping ---
print("=== RESHAPING ===")
flat = torch.arange(12, dtype=torch.float32)
grid = flat.reshape(3, 4)
print(f"flat:        {flat}")
print(f"grid (3x4):\n{grid}")
print(f"grid.T:\n{grid.T}\n")

# --- Key differences from numpy ---
print("=== KEY DIFFERENCES ===")
print(f"Default float dtype:  numpy={np.array([1.0]).dtype}, "
      f"torch={torch.tensor([1.0]).dtype}")
print(f"Device:               {t1.device}")
print(f"Requires grad:        {t1.requires_grad} (default off, turn on for autograd)")

t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"With requires_grad:   {t_grad.requires_grad} (ready for Kata 2.6!)")
```

---

## Key Takeaways

- **Tensors are numpy arrays with superpowers.** Same math, same syntax, plus GPU support and gradient tracking.
- **`torch.from_numpy()` shares memory.** Changes to the numpy array affect the tensor and vice versa. Use `torch.tensor()` for an independent copy.
- **Default dtypes differ.** Numpy defaults to float64, PyTorch to float32. This matters for precision and GPU performance.
- **`requires_grad=True` enables gradient tracking.** This is the bridge to autograd (Kata 2.6) -- it tells PyTorch to record operations for backpropagation.
- **The API is nearly identical to numpy.** If you can write `np.mean(x)`, you can write `x.mean()` in PyTorch. The learning curve is minimal.
