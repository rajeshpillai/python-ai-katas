# Pooling Effects

> Phase 5 — Convolutional Neural Networks | Kata 5.5

---

## Concept & Intuition

### What problem are we solving?

After convolution, feature maps retain nearly the same spatial resolution as the input. An image of 224x224 pixels convolved with a 3x3 filter still produces a 222x222 map. This creates three problems. First, the sheer number of values is enormous -- 16 filters produce 16 x 222 x 222 = 786,048 activations, and stacking multiple layers compounds this. Second, the feature maps are sensitive to the exact position of patterns: a cat shifted two pixels to the right produces a completely different activation pattern, even though it is the same cat. Third, subsequent layers would need huge receptive fields to capture global structure because each neuron still sees only a tiny local neighborhood.

**Pooling** solves all three problems by downsampling the feature maps. A 2x2 max pooling operation divides the map into non-overlapping 2x2 blocks, and for each block keeps only the maximum value. This halves the spatial dimensions (and quarters the total values), introduces tolerance to small translations, and effectively doubles the receptive field of subsequent layers.

Average pooling takes the mean of each block instead of the maximum. Both reduce spatial resolution, but they have different characteristics: max pooling preserves the strongest activations (best for detecting whether a pattern exists), while average pooling preserves the overall energy distribution (better for texture-like features).

### Why naive approaches fail

Without pooling, you might try to reduce spatial dimensions by using strided convolutions (skipping positions). This works but is rigid -- the stride is fixed and the downsampling is tightly coupled to the feature extraction. You lose the modularity of having separate "detect" and "summarize" stages.

Another approach is simply using the full-resolution feature maps everywhere. This fails at scale: a network processing a 224x224 image with 512 filters at full resolution would need to store and compute on 512 x 224 x 224 = 25 million activations per layer. The memory and compute costs are prohibitive, and the network is fragile to tiny shifts in the input. Pooling provides a principled way to progressively reduce resolution while retaining the most important information.

### Mental models

- **Summarizing a paragraph.** Pooling is like reading a paragraph and keeping only the key sentence from each section. Max pooling keeps the loudest signal (the strongest feature detection), while average pooling keeps the general tone.
- **Zooming out.** Each pooling layer is like stepping back from a painting. Close up, you see individual brushstrokes (pixels). Step back once, you see shapes. Step back again, you see the whole scene. You lose detail but gain perspective.
- **Binning sensor data.** If a temperature sensor reports every second but you only need hourly data, you might take the max of each hour (to catch spikes) or the average (for general trends). Pooling does the same for spatial data.
- **Translation tolerance.** If you ask "is there an edge somewhere in this 4x4 region?", the answer should be the same whether the edge is at position (0,0) or (1,1) within that region. Max pooling naturally provides this: it fires as long as the pattern exists anywhere in the pooling window.

### Visual explanations

```
Max Pooling (2x2, stride 2):

  Input Feature Map (4x4):        Output (2x2):
  +----+----+----+----+           +----+----+
  |  1 |  3 |  2 |  0 |          |    |    |
  +----+----+----+----+    ==>   |  5 |  8 |
  |  5 |  2 |  8 |  1 |          |    |    |
  +----+----+----+----+          +----+----+
  |  0 |  4 |  3 |  7 |          |    |    |
  +----+----+----+----+    ==>   |  4 |  7 |
  |  1 |  2 |  0 |  6 |          |    |    |
  +----+----+----+----+          +----+----+

  Each 2x2 block -> take the max value

Average Pooling (2x2, stride 2):

  Same input:                     Output (2x2):
  +----+----+----+----+          +------+------+
  |  1 |  3 |  2 |  0 |         |      |      |
  +----+----+----+----+   ==>   | 2.75 | 2.75 |
  |  5 |  2 |  8 |  1 |         |      |      |
  +----+----+----+----+         +------+------+
  |  0 |  4 |  3 |  7 |         |      |      |
  +----+----+----+----+   ==>   | 1.75 | 4.00 |
  |  1 |  2 |  0 |  6 |         |      |      |
  +----+----+----+----+         +------+------+

  Each 2x2 block -> take the mean value

Translation invariance:

  Original:        Shifted 1px right:
  [0, 5, 0, 0]     [0, 0, 5, 0]

  Max pool (2x2):   Max pool (2x2):
  [  5  ,  0  ]     [  0  ,  5  ]      <- different!

  But if we ask "is 5 present?" the answer is YES in both cases.
  The WHAT is preserved, the exact WHERE is discarded.
```

---

## Hands-on Exploration

1. Create a small feature map and apply 2x2 max pooling and average pooling side by side. Compare how each preserves different kinds of information.
2. Demonstrate translation invariance: shift a pattern within the input and show that the pooled output stays nearly the same, while the raw feature map changes completely.
3. Apply pooling multiple times to see how spatial dimensions shrink rapidly and how the total number of parameters in subsequent layers decreases dramatically.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Pooling operations ---
def max_pool2d(x, size=2):
    h, w = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            block = x[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = np.max(block)
    return out

def avg_pool2d(x, size=2):
    h, w = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            block = x[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = np.mean(block)
    return out

def show_grid(name, grid, fmt=".1f"):
    print(f"\n  {name} ({grid.shape[0]}x{grid.shape[1]}):")
    for row in grid:
        print("    " + " ".join(f"{v:{fmt}}" for v in row))

# --- 1. Basic pooling comparison ---
fmap = np.array([[1, 3, 2, 0, 5, 1],
                 [5, 2, 8, 1, 0, 3],
                 [0, 4, 3, 7, 2, 1],
                 [1, 2, 0, 6, 4, 0],
                 [3, 8, 1, 0, 2, 5],
                 [0, 1, 4, 3, 7, 2]], dtype=float)

print("=== 1. Max Pooling vs Average Pooling ===")
show_grid("Original feature map", fmap, ".0f")
show_grid("After MAX pool (2x2)", max_pool2d(fmap), ".1f")
show_grid("After AVG pool (2x2)", avg_pool2d(fmap), ".1f")

print("\n  Observation: Max pool keeps peak activations (strongest signal).")
print("  Avg pool keeps the general distribution (smoothed signal).")

# --- 2. Translation invariance ---
print("\n=== 2. Translation Invariance ===")
img_a = np.zeros((8, 8))
img_a[2:4, 2:4] = np.array([[9, 7], [6, 8]])  # pattern at top-left

img_b = np.zeros((8, 8))
img_b[2:4, 4:6] = np.array([[9, 7], [6, 8]])  # same pattern shifted right

print("\n  Pattern at position (2,2):")
for row in img_a:
    print("    " + " ".join(f"{int(v):1d}" for v in row))

print("\n  Same pattern shifted to (2,4):")
for row in img_b:
    print("    " + " ".join(f"{int(v):1d}" for v in row))

pool_a = max_pool2d(img_a)
pool_b = max_pool2d(img_b)
show_grid("Max pooled (original)", pool_a, ".0f")
show_grid("Max pooled (shifted)", pool_b, ".0f")

diff = np.sum(np.abs(pool_a - pool_b))
print(f"\n  Total absolute difference after pooling: {diff:.1f}")
print("  Pooling absorbs small spatial shifts!")

# --- 3. Repeated pooling: dimension reduction ---
print("\n=== 3. Repeated Pooling (Dimension Reduction) ===")
big_map = np.random.rand(16, 16) * 10

current = big_map
print(f"  Layer 0: {current.shape[0]:3d}x{current.shape[1]:<3d}"
      f" = {current.size:5d} values | range [{current.min():.1f}, {current.max():.1f}]")

for layer in range(1, 5):
    current = max_pool2d(current)
    print(f"  Layer {layer}: {current.shape[0]:3d}x{current.shape[1]:<3d}"
          f" = {current.size:5d} values | range [{current.min():.1f}, {current.max():.1f}]")

print("\n  16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1")
print("  256 values reduced to 1 in just 4 pooling operations!")
print(f"  Final value (global max): {current[0,0]:.1f}")

# --- 4. Parameter savings ---
print("\n=== 4. Impact on Network Parameters ===")
filters = 64
for size_label, size in [("No pooling", 224), ("After pool 1", 112),
                          ("After pool 2", 56), ("After pool 3", 28)]:
    activations = filters * size * size
    next_layer_params = filters * filters * 3 * 3  # 3x3 conv
    print(f"  {size_label:16s}: {size:3d}x{size:3d}x{filters}"
          f" = {activations:>9,d} activations")

print(f"\n  Pooling reduces memory by 4x at each stage!")
print(f"  3 pools: {224*224} -> {112*112} -> {56*56} -> {28*28}"
      f" ({224*224/28/28:.0f}x reduction)")
```

---

## Key Takeaways

- **Max pooling keeps the strongest signal.** From each spatial block, it retains only the maximum activation, preserving whether a feature was detected regardless of its exact position within the block.
- **Average pooling keeps the general trend.** It computes the mean of each block, providing a smoother, more distributed summary that preserves overall activation levels.
- **Pooling provides translation invariance.** Small shifts in the input are absorbed by the pooling window, so the output remains stable. This is critical for recognition: a cat shifted a few pixels should still be classified as a cat.
- **Spatial dimensions shrink exponentially.** Each 2x2 pooling halves height and width, quartering the total activations. Three pooling layers reduce a 224x224 map to 28x28 -- a 64x reduction in spatial values.
- **Pooling is a design choice.** Max pooling is standard in classification (detecting presence of features). Average pooling is common in the final layer (global average pooling). Modern architectures sometimes replace pooling with strided convolutions, but understanding pooling remains foundational.
