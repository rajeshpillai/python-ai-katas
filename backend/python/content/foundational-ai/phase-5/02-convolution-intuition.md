# Convolution Intuition

> Phase 5 — Convolutional Neural Networks | Kata 5.2

---

## Concept & Intuition

### What problem are we solving?

We need a way to process images that respects their spatial structure. Instead of connecting every pixel to every neuron, we want a small "window" that slides across the image, examining one local patch at a time. This operation is called convolution. A small matrix (the kernel or filter) is placed on top of a patch of the image, element-wise multiplied, and summed to produce a single output value. The kernel then slides to the next patch and repeats. The result is a new 2D grid called a feature map.

The key insight is parameter sharing: the same kernel is used at every position. This means if the kernel learns to detect a vertical edge, it can detect that vertical edge anywhere in the image -- top-left, center, or bottom-right. We go from needing 100,000+ parameters (dense) to just 9 parameters (a 3x3 kernel) that work everywhere.

Locality is the other crucial property. Each output pixel depends only on a small local region of the input, not the entire image. This matches how visual features actually work: an edge is a local phenomenon defined by neighboring pixels, not by pixels on the opposite side of the image.

### Why naive approaches fail

Without convolution, we are left with two bad options. First, we can use dense layers, which throw away spatial information and waste parameters (as Kata 5.1 showed). Second, we can try hand-crafting position-specific detectors, which is impossibly tedious and does not generalize.

Convolution elegantly solves both problems: it preserves spatial relationships (the output is still a 2D grid that maps to input positions) and it shares parameters (one small kernel covers the entire image). This is not just an engineering shortcut -- it encodes the prior knowledge that visual patterns can appear anywhere in an image.

### Mental models

- **Magnifying glass sliding over a page.** You examine one small area at a time, using the same eyes (same kernel) at every position.
- **Rubber stamp.** The kernel is like a stamp that "tests" each patch of the image: high output means "this patch matches my pattern," low output means "no match here."
- **Template matching.** The kernel is a tiny template. Convolution computes a similarity score between the template and every local region of the image.
- **Stencil over graph paper.** Place a 3x3 stencil on the grid, multiply and sum what you see through the holes, write down the answer, slide the stencil one cell over, repeat.

### Visual explanations

```
  2D Convolution step by step (3x3 kernel on 5x5 image):

  Image (5x5):              Kernel (3x3):
  ┌─────────────────┐       ┌─────────┐
  │  1  0  1  0  1  │       │  1  0 -1 │
  │  0  1  0  1  0  │       │  1  0 -1 │
  │  1  0  1  0  1  │       │  1  0 -1 │
  │  0  1  0  1  0  │       └─────────┘
  │  1  0  1  0  1  │
  └─────────────────┘

  Step 1: Kernel at position (0,0)      Step 2: Kernel at position (0,1)
  ┌───────┐                              ┌───────┐
  │ 1  0  1│ 0  1                     1 │ 0  1  0│ 1
  │ 0  1  0│ 1  0                     0 │ 1  0  1│ 0
  │ 1  0  1│ 0  1                     1 │ 0  1  0│ 1
    0  1  0  1  0                        0  1  0  1  0
    1  0  1  0  1                        1  0  1  0  1

  Multiply element-wise        Multiply element-wise
  and sum: 1+0-1+0+0+0         and sum: 0+0+0+1+0-1
           +1+0-1 = 0                    +0+0+0 = 0

  Output (3x3):   ← The output shrinks because the kernel
  ┌─────────┐       cannot extend beyond the image edges
  │ 0  0  0 │       Output size = (5-3+1) x (5-3+1) = 3x3
  │ 0  0  0 │
  │ 0  0  0 │
  └─────────┘
```

```
  Parameter sharing visualized:

  SAME 9 weights used at EVERY position:

  Position (0,0)  Position (0,1)  Position (0,2) ...
  [w1 w2 w3]     [w1 w2 w3]     [w1 w2 w3]
  [w4 w5 w6]     [w4 w5 w6]     [w4 w5 w6]
  [w7 w8 w9]     [w7 w8 w9]     [w7 w8 w9]

  Total learnable parameters: just 9 (plus 1 bias = 10)
```

---

## Hands-on Exploration

1. Implement 2D convolution from scratch using nested loops, then verify it produces the same result as sliding and summing.

2. Apply a vertical edge detection kernel `[[1,0,-1],[1,0,-1],[1,0,-1]]` to a simple image with a clear vertical boundary and observe the output.

3. Change the kernel size from 3x3 to 5x5 and notice how the output shrinks more. Understand the formula: output_size = input_size - kernel_size + 1.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

def convolve2d(image, kernel):
    """2D convolution from scratch -- no libraries, pure numpy indexing."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1   # output dimensions
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i:i+kh, j:j+kw]   # extract local region
            output[i, j] = np.sum(patch * kernel)  # element-wise multiply & sum
    return output

def print_grid(arr, label="", fmt="{:5.1f}"):
    if label:
        print(f"\n{label}:")
    for row in arr:
        print("  " + " ".join(fmt.format(v) for v in row))

# --- Create a simple image with a vertical boundary ---
image = np.zeros((7, 7))
image[:, :3] = 1.0    # left half is white, right half is black

print_grid(image, "Input image (left=bright, right=dark)", fmt="{:4.0f}")

# --- Vertical edge detection kernel ---
kernel_v = np.array([[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]], dtype=float)
print_grid(kernel_v, "Vertical edge kernel", fmt="{:4.0f}")

# --- Convolve ---
output_v = convolve2d(image, kernel_v)
print_grid(output_v, "Output: vertical edges detected", fmt="{:5.1f}")
print("  --> Column of 3s marks the vertical boundary!")

# --- Horizontal edge detection ---
kernel_h = np.array([[ 1,  1,  1],
                     [ 0,  0,  0],
                     [-1, -1, -1]], dtype=float)

image_h = np.zeros((7, 7))
image_h[:3, :] = 1.0    # top half bright, bottom half dark
output_h = convolve2d(image_h, kernel_h)
print_grid(image_h, "Input image (top=bright, bottom=dark)", fmt="{:4.0f}")
print_grid(output_h, "Output: horizontal edges detected", fmt="{:5.1f}")
print("  --> Row of 3s marks the horizontal boundary!")

# --- Output size formula ---
print("\n--- Output Size Formula: (input - kernel + 1) ---")
for img_size in [7, 14, 28]:
    for k_size in [3, 5]:
        out_size = img_size - k_size + 1
        print(f"  Image {img_size}x{img_size} with {k_size}x{k_size} kernel -> Output {out_size}x{out_size}")

# --- Parameter sharing proof ---
print("\n--- Parameter sharing ---")
print(f"Kernel has {kernel_v.size} parameters")
print(f"Applied to {output_v.shape[0]}x{output_v.shape[1]} = {output_v.size} positions")
print(f"Dense equivalent would need: {image.size * output_v.size} parameters")
print(f"Convolution needs:           {kernel_v.size} parameters (shared everywhere)")
```

---

## Key Takeaways

- **Convolution is sliding dot products.** A small kernel slides across the image, computing a weighted sum at each position to produce an output feature map.
- **Parameter sharing is the key advantage.** The same kernel weights are reused at every spatial position, so a 3x3 kernel has just 9 parameters regardless of image size.
- **Locality means each output depends only on nearby inputs.** The kernel examines a small patch, matching how real visual features (edges, corners) are local phenomena.
- **Output size shrinks.** For a valid convolution (no padding), the output dimensions follow: output = input - kernel + 1.
- **Convolution preserves spatial structure.** The output is still a 2D grid where position (i,j) corresponds to a specific region of the input, unlike flattening for dense layers.
