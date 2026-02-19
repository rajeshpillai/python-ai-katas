# Feature Maps

> Phase 5 — Convolutional Neural Networks | Kata 5.4

---

## Concept & Intuition

### What problem are we solving?

A single convolutional filter detects a single pattern -- one edge orientation, one blob type, one texture. But real images contain many patterns simultaneously: horizontal edges AND vertical edges AND curves AND corners. A convolutional layer solves this by applying multiple filters in parallel, each producing its own **feature map**. If we apply 16 filters to an image, we get 16 feature maps, each highlighting a different pattern.

The real power emerges when we stack convolutional layers. The first layer's filters operate on raw pixels and detect simple patterns like edges. The second layer's filters operate on the first layer's feature maps, effectively combining simple patterns into more complex ones: two edges meeting at a point become a "corner detector," parallel edges become a "stripe detector," and so on. By layer three or four, the network is detecting high-level concepts like textures, shapes, or even object parts -- all built hierarchically from simple local patterns.

Understanding feature maps is understanding the core mechanism of CNNs: each layer transforms a stack of spatial maps into another stack of spatial maps, with each successive stack encoding progressively more abstract and complex features.

### Why naive approaches fail

If you tried to detect complex patterns directly with a single convolution layer, you would need enormous filters. A filter that detects an entire "face" would need to be as large as the face itself, defeating the purpose of local pattern matching. You would need separate giant filters for every pose, every scale, and every position -- an impossible combinatorial explosion.

Without stacking layers, you also cannot build compositional representations. A single layer can detect "horizontal edge" and "vertical edge" separately but cannot combine them into "corner" or "T-junction." Hierarchy is essential: simple features compose into complex features, which compose into even more complex features. This compositional structure mirrors how visual systems actually work -- both biological and artificial.

### Mental models

- **Building blocks.** Feature maps at layer 1 are like individual LEGO bricks (edges, spots). Layer 2 combines bricks into small structures (corners, curves). Layer 3 assembles structures into recognizable shapes (circles, rectangles). Each layer builds on what came before.
- **Multi-channel photography.** Think of feature maps like the channels of a multispectral camera. A regular camera captures 3 channels (R, G, B). A CNN layer might produce 64 "channels," each tuned to a different visual feature rather than a color wavelength.
- **Committee of experts.** Each filter in a layer is a specialist that looks for one specific pattern. The collection of feature maps is like a committee report: expert 1 found edges here, expert 2 found blobs there, expert 3 found textures over there.
- **Depth = abstraction.** The spatial dimensions shrink through the network (due to pooling), but the channel dimension grows. You trade spatial resolution for semantic richness: fewer pixels, but each pixel describes a more abstract concept.

### Visual explanations

```
Single layer with 3 filters:

  Input Image    Filter 1       Filter 2       Filter 3
   (8x8x1)   (horiz edge)   (vert edge)     (diagonal)
      |            |              |              |
      v            v              v              v
  +--------+  +--------+     +--------+     +--------+
  |        |  | ------ |     | | | |  |     | /  /   |
  |  ++    |  |        |     | | | |  |     |  /  /  |
  |  ++    |  | ------ |     |        |     |   /  / |
  +--------+  +--------+     +--------+     +--------+
              Feature Map 1   Feature Map 2  Feature Map 3

  Output: 3 feature maps stacked => (6x6x3) volume

Stacking two convolutional layers:

  Input        Layer 1              Layer 2
  (8x8x1)     (8x8x1) -> (6x6x3)  (6x6x3) -> (4x4x4)

  Raw          Simple patterns      Combinations
  pixels  -->  edges, spots    -->  corners, crosses
               (each map has        (each map combines
                ONE pattern)         MULTIPLE layer-1 maps)

  Layer 2 Filter shape: (3 x 3 x 3)
                         ^   ^   ^
                height width input_channels
                         (matches layer 1 output depth)
```

---

## Hands-on Exploration

1. Create a small input image, apply multiple filters in one layer to produce a stack of feature maps, and show how each map highlights a different pattern.
2. Feed the first layer's feature maps into a second convolutional layer where each filter spans all input channels, producing composite pattern detections.
3. Observe how layer 2 feature maps respond to combinations of layer 1 features -- detecting complex patterns built from simpler ones.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Create a 9x9 image with an "L" shape ---
image = np.zeros((9, 9))
image[1:7, 2] = 1.0   # vertical bar
image[6, 2:7] = 1.0   # horizontal bar
print("=== Input Image (9x9) - An 'L' shape ===")
for row in image:
    print("  " + " ".join("#" if v > 0.5 else "." for v in row))

# --- 2D convolution ---
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

def relu(x): return np.maximum(0, x)

# --- LAYER 1: 3 filters on raw image -> 3 feature maps ---
filters_L1 = {
    "horizontal": np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=float),
    "vertical":   np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=float),
    "diagonal":   np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=float),
}

print("\n=== LAYER 1: Applying 3 filters to raw image ===")
layer1_maps = []
for name, kernel in filters_L1.items():
    fmap = relu(convolve2d(image, kernel))
    layer1_maps.append(fmap)
    print(f"\n  Feature Map: {name} edge detector ({fmap.shape[0]}x{fmap.shape[1]})")
    for row in fmap:
        print("    " + " ".join("#" if v > 0.8 else "+" if v > 0.3 else "." for v in row))

layer1_stack = np.stack(layer1_maps, axis=-1)  # (7, 7, 3)
print(f"\n  Layer 1 output shape: {layer1_stack.shape}  (height x width x channels)")

# --- LAYER 2: 2 filters that combine all 3 layer-1 maps ---
# Multi-channel convolution: filter is (3, 3, num_input_channels)
def convolve2d_multichannel(vol, kernel_3d):
    kh, kw, kc = kernel_3d.shape
    oh = vol.shape[0] - kh + 1
    ow = vol.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = vol[i:i+kh, j:j+kw, :]  # (kh, kw, kc)
            out[i, j] = np.sum(patch * kernel_3d)
    return out

# Filter A: detects corners (horiz + vert edges together)
kernel_a = np.zeros((3, 3, 3))
kernel_a[:, :, 0] = np.array([[0,0,0],[0,1,0],[0,1,0]])  # horiz response
kernel_a[:, :, 1] = np.array([[0,0,0],[0,1,1],[0,0,0]])  # vert response
# Filter B: detects the vertical part only (strong vertical, ignore others)
kernel_b = np.zeros((3, 3, 3))
kernel_b[:, :, 1] = np.array([[0,1,0],[0,1,0],[0,1,0]])  # vert only

print("\n=== LAYER 2: Combining layer 1 feature maps ===")
layer2_names = ["corner detector (horiz+vert)", "vertical segment detector"]
layer2_kernels = [kernel_a, kernel_b]
layer2_maps = []
for name, kernel in zip(layer2_names, layer2_kernels):
    fmap = relu(convolve2d_multichannel(layer1_stack, kernel))
    layer2_maps.append(fmap)
    print(f"\n  Feature Map: {name} ({fmap.shape[0]}x{fmap.shape[1]})")
    for row in fmap:
        print("    " + " ".join("#" if v > 1.5 else "+" if v > 0.5 else "." for v in row))

# --- Summary ---
print("\n=== Hierarchical Feature Summary ===")
print(f"  Input:   {image.shape}          (raw pixels)")
print(f"  Layer 1: {layer1_stack.shape}  (3 simple feature maps)")
l2 = np.stack(layer2_maps, axis=-1)
print(f"  Layer 2: {l2.shape}  (2 composite feature maps)")
print(f"\n  Layer 1 detects: edges (simple, local patterns)")
print(f"  Layer 2 detects: corners, segments (combinations of edges)")
print(f"  Each deeper layer captures more abstract patterns!")

# --- Show that layer 2 corner detector fires at the L's corner ---
corner_map = layer2_maps[0]
max_pos = np.unravel_index(np.argmax(corner_map), corner_map.shape)
print(f"\n  Corner detector peak at position: {max_pos}")
print(f"  This corresponds to the bend in the 'L' shape!")
```

---

## Key Takeaways

- **Multiple filters produce multiple feature maps.** Each convolutional layer applies N filters, generating N spatial maps that each highlight a different pattern in the input.
- **Stacking layers builds hierarchy.** Layer 1 detects edges, layer 2 combines edges into corners and curves, layer 3 combines those into shapes. Complexity builds compositionally.
- **Multi-channel filters span all input maps.** A filter in layer 2 has shape (height, width, input_channels), so it can learn which combinations of layer-1 features are important.
- **Spatial dimensions shrink, channel dimensions grow.** Through the network, images go from large-spatial/few-channels to small-spatial/many-channels, trading pixel resolution for semantic richness.
- **This is how CNNs see.** The progression from pixels to edges to shapes to objects is not just a metaphor -- it is literally what the learned feature maps show when you visualize them in trained networks.
