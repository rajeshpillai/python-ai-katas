# Filters as Pattern Detectors

> Phase 5 — Convolutional Neural Networks | Kata 5.3

---

## Concept & Intuition

### What problem are we solving?

A convolutional filter (also called a kernel) is a small matrix that slides across an image, computing a dot product at every position. Different filters detect different local patterns. A horizontal edge filter responds strongly wherever pixel intensities change sharply from top to bottom. A vertical edge filter catches left-to-right transitions. A corner filter fires at intersections of edges. A blur filter averages neighborhoods to smooth out noise.

Understanding how hand-crafted filters work is essential before moving to learned filters in CNNs. In a trained convolutional network, the first layer's filters look remarkably similar to these classical filters -- edges at various orientations, blobs, and color gradients. The difference is that the network discovers them automatically from data. By building them by hand first, you gain intuition for exactly what a convolution operation computes and why different kernels produce different responses.

This kata takes a simple binary image, applies several classic filters to it, and displays the results as text grids. You will see that each filter acts as a pattern detector: it produces large values where its target pattern exists and near-zero values elsewhere.

### Why naive approaches fail

Without convolutions, the simplest approach to detecting an edge would be to check every pixel and compare it to its neighbors with handwritten if-else logic. This is fragile, verbose, and does not generalize. You would need separate code for horizontal edges, vertical edges, diagonal edges, corners, and every other pattern -- and the thresholds would be different for every image.

A fully connected neural network approach is equally problematic. It would flatten the image into a 1D vector, destroying all spatial structure. Pixel (0,0) would be connected to pixel (99,99) with the same weight structure as to its immediate neighbor. The network would need to independently learn spatial relationships that are trivially captured by a 3x3 convolution kernel. The number of parameters would explode, and the model would fail to generalize across spatial positions.

### Mental models

- **Stencil.** A filter is like a stencil you place over a small patch of the image. The stencil has weights that say "I am looking for THIS pattern." The dot product between the stencil and the image patch measures how well the patch matches the pattern.
- **Template matching.** Each filter is a tiny template. High response means "this patch looks like my template." Sliding the filter across the image produces a response map showing where the template matches.
- **Weighted voting.** At each position, the filter weights vote on whether the pattern is present. Positive weights vote "yes" where they overlap bright pixels; negative weights vote "yes" where they overlap dark pixels.
- **Musical chord detection.** Just as a tuning fork resonates only with its matching frequency, each filter "resonates" only with its matching spatial pattern.

### Visual explanations

```
How a 3x3 filter slides across a 5x5 image:

  Image (5x5):          Filter (3x3):         Output (3x3):
  +---+---+---+---+---+
  | 0 | 0 | 0 | 1 | 1 |  +---------+          +---+---+---+
  +---+---+---+---+---+  | -1  0  1 |          |   |   |   |
  | 0 | 0 | 0 | 1 | 1 |  | -1  0  1 |    ==>  |   |   |   |
  +---+---+---+---+---+  | -1  0  1 |          |   |   |   |
  | 0 | 0 | 0 | 1 | 1 |  +---------+          +---+---+---+
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |  (vertical edge
  +---+---+---+---+---+   detector)
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

  At position (1,1): sum of element-wise products of the
  3x3 patch centered at (1,1) with the filter weights.

Classic filter types:

  Horizontal Edge:    Vertical Edge:    Sharpen:        Blur:
  [-1 -1 -1]         [-1  0  1]        [ 0 -1  0]     [1/9 1/9 1/9]
  [ 0  0  0]         [-1  0  1]        [-1  5 -1]     [1/9 1/9 1/9]
  [ 1  1  1]         [-1  0  1]        [ 0 -1  0]     [1/9 1/9 1/9]
```

---

## Hands-on Exploration

1. Create a simple 8x8 binary image containing a recognizable shape (a cross/plus sign) with clear horizontal and vertical edges.
2. Implement a 2D convolution function from scratch that slides a 3x3 kernel across the image and computes the dot product at each position.
3. Apply four different filters (horizontal edge, vertical edge, sharpen, blur) and display each result as a text grid to see which parts of the image each filter responds to.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Create an 8x8 image with a cross/plus pattern ---
image = np.zeros((8, 8))
image[3, 1:7] = 1.0   # horizontal bar
image[1:7, 3] = 1.0   # vertical bar

print("=== Original Image (8x8) ===")
for row in image:
    print("  " + " ".join(f"{int(v)}" for v in row))

# --- 2D Convolution (valid mode, no padding) ---
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    oh = img.shape[0] - kh + 1
    ow = img.shape[1] - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(patch * kernel)
    return output

# --- Define classic filters ---
filters = {
    "Horizontal Edge": np.array([[-1, -1, -1],
                                  [ 0,  0,  0],
                                  [ 1,  1,  1]], dtype=float),
    "Vertical Edge":   np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=float),
    "Sharpen":         np.array([[ 0, -1,  0],
                                  [-1,  5, -1],
                                  [ 0, -1,  0]], dtype=float),
    "Blur (3x3 avg)":  np.ones((3, 3)) / 9.0,
}

# --- Apply each filter and display results ---
for name, kernel in filters.items():
    result = convolve2d(image, kernel)
    print(f"\n=== {name} Filter ===")
    print(f"  Kernel:")
    for row in kernel:
        print("    [" + " ".join(f"{v:5.2f}" for v in row) + "]")

    print(f"  Response map ({result.shape[0]}x{result.shape[1]}):")
    for row in result:
        line = "  "
        for val in row:
            if val > 0.5:
                line += " + "    # strong positive response
            elif val < -0.5:
                line += " - "    # strong negative response
            else:
                line += " . "    # weak/no response
        print(line)

    print(f"  Raw values (showing where filter fires):")
    for row in result:
        print("    " + " ".join(f"{v:5.1f}" for v in row))

# --- Summary statistics ---
print("\n=== Filter Response Summary ===")
for name, kernel in filters.items():
    result = convolve2d(image, kernel)
    print(f"  {name:20s} | max: {result.max():+5.2f}"
          f" | min: {result.min():+5.2f}"
          f" | active cells: {np.sum(np.abs(result) > 0.5)}")

print("\n=== What each filter detected ===")
print("  Horizontal Edge: responds at top/bottom of the cross bars")
print("  Vertical Edge:   responds at left/right of the cross bars")
print("  Sharpen:         enhances the cross shape, suppresses flat areas")
print("  Blur:            smooths everything, cross becomes softer")
```

---

## Key Takeaways

- **A convolution is a sliding dot product.** The filter slides across the image, and at each position the element-wise product is summed into a single number measuring pattern match strength.
- **Different filters detect different patterns.** Edge filters respond to intensity transitions, blur filters average neighborhoods, and sharpen filters enhance local contrast. Each is just a different set of weights.
- **Filter response indicates pattern location.** The output feature map is a spatial map of "where does this pattern occur?" Large positive values mean the pattern is strongly present; large negative values mean the opposite pattern is present.
- **CNNs learn these filters automatically.** In a trained CNN, the first convolutional layer learns filters nearly identical to these classic ones. Deeper layers combine simple patterns into complex ones.
- **Spatial structure is preserved.** Unlike fully connected layers, convolutions maintain the 2D arrangement of the input, so the output map tells you both what was detected and where.
