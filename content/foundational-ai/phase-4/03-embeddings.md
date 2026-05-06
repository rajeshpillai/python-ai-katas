# Embeddings

> Phase 4 — Representation Learning | Kata 4.3

---

## Concept & Intuition

### What problem are we solving?

In machine learning, we constantly deal with discrete, categorical data: words in a vocabulary, product IDs in a catalog, user IDs in a recommendation system, or amino acids in a protein sequence. These items have no inherent numerical relationship -- "cat" is not greater than "dog," and user #42 is not twice user #21. Yet neural networks operate on continuous numbers and need meaningful numerical inputs to learn patterns.

The embedding problem asks: how do we represent discrete items as continuous vectors such that the geometry of the vector space captures meaningful relationships? We want items that behave similarly (appear in similar contexts, get purchased together, or serve similar functions) to end up close together in vector space, while dissimilar items land far apart.

Embeddings are one of the most important ideas in modern AI. Word embeddings like Word2Vec showed that arithmetic on word vectors captures semantic meaning (king - man + woman = queen). This same principle extends to recommendation systems, graph neural networks, and the token embeddings that power every large language model.

### Why naive approaches fail

The most obvious way to represent N discrete items numerically is **one-hot encoding**: give each item a vector of length N with a single 1 and the rest 0s. This fails for three reasons. First, every pair of one-hot vectors has exactly the same distance from every other pair -- "cat" is just as far from "kitten" as it is from "spaceship." There is zero semantic information in the geometry. Second, these vectors are enormous: a vocabulary of 50,000 words produces 50,000-dimensional vectors, most of which are zeros. Third, one-hot vectors cannot generalize -- learning something about "cat" teaches the model nothing about "kitten" because their representations share nothing in common.

You might try assigning sequential integers (cat=1, dog=2, fish=3), but this imposes a false ordering. The network would conclude that "dog" is between "cat" and "fish" in some meaningful sense, which is nonsense. Embeddings solve this by learning a dense, low-dimensional vector for each item where the geometry emerges from the training objective itself.

### Mental models

- **Address book analogy.** One-hot encoding is like identifying people by their row number in a massive spreadsheet -- it tells you nothing about who they are. An embedding is like describing each person with a few key traits (age, height, interests) that let you compute meaningful similarity.
- **Map coordinates.** Just as latitude and longitude place cities in a 2D space where nearby cities are geographically close, embeddings place items in an N-dimensional space where semantically similar items are geometrically close.
- **Lookup table.** An embedding layer is fundamentally just a matrix where row i is the vector for item i. "Forward pass" means indexing into a row. "Training" means adjusting those rows with gradients so the geometry becomes useful.
- **Compression.** One-hot vectors for 10,000 items live in 10,000 dimensions. Embeddings compress them into perhaps 32 dimensions, forcing the network to discover which features actually matter.

### Visual explanations

```
One-hot encoding (5 items, 5 dims -- no similarity info):

  cat:       [1, 0, 0, 0, 0]
  kitten:    [0, 1, 0, 0, 0]    distance(cat, kitten) = sqrt(2)
  dog:       [0, 0, 1, 0, 0]    distance(cat, dog)    = sqrt(2)
  spaceship: [0, 0, 0, 1, 0]    distance(cat, spaceship) = sqrt(2)
  rocket:    [0, 0, 0, 0, 1]         ALL DISTANCES EQUAL!

Learned embeddings (5 items, 2 dims -- similarity emerges):

  cat:       [0.8, 0.3]
  kitten:    [0.9, 0.4]    distance(cat, kitten)    = 0.14  (close!)
  dog:       [0.7, 0.5]    distance(cat, dog)        = 0.22
  spaceship: [-0.6, 0.8]   distance(cat, spaceship)  = 1.48  (far!)
  rocket:    [-0.5, 0.9]   distance(cat, rocket)     = 1.37  (far!)

Embedding space (2D):

    furry ^
      1.0 |
      0.5 |  dog
      0.4 |   kitten                 spaceship
      0.3 |  cat                      rocket
      0.0 +-----|---------|---------|----->
         -1.0  -0.5      0.0   0.5   1.0
                                    vehicle-ness
```

---

## Hands-on Exploration

1. Build an embedding lookup table as a simple numpy matrix (vocab_size x embed_dim) and retrieve vectors by integer index.
2. Train embeddings using a co-occurrence objective: items that appear together in "contexts" should have similar embeddings (high dot product), items that do not should have dissimilar embeddings (low dot product).
3. After training, measure cosine similarity between all pairs of items and observe that items from the same category cluster together, even though we never told the model about categories.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Vocabulary: 8 items from 2 implicit categories ---
vocab = ["cat", "dog", "kitten", "puppy",       # animals
         "car", "truck", "bicycle", "motorcycle"] # vehicles
vocab_size = len(vocab)
embed_dim = 4

# --- Embedding table: just a matrix! Row i = vector for item i ---
embeddings = np.random.randn(vocab_size, embed_dim) * 0.1

# --- Training data: co-occurrence pairs (items seen together) ---
# Animals co-occur with animals, vehicles with vehicles
animal_ids = [0, 1, 2, 3]
vehicle_ids = [4, 5, 6, 7]
pairs = []
for group in [animal_ids, vehicle_ids]:
    for i in group:
        for j in group:
            if i != j:
                pairs.append((i, j, 1.0))   # positive pair
        # negative pairs: pick items from the OTHER group
        other = vehicle_ids if group == animal_ids else animal_ids
        for j in other[:2]:
            pairs.append((i, j, 0.0))        # negative pair

lr = 0.05

# --- Train: push co-occurring items together, others apart ---
print("=== Training Embeddings ===")
for epoch in range(200):
    total_loss = 0.0
    np.random.shuffle(pairs)
    for (i, j, target) in pairs:
        vi = embeddings[i]
        vj = embeddings[j]
        # Sigmoid of dot product -> predicted similarity
        dot = np.dot(vi, vj)
        dot = np.clip(dot, -10, 10)
        pred = 1.0 / (1.0 + np.exp(-dot))
        loss = -(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))
        total_loss += loss
        # Gradients
        grad = pred - target
        embeddings[i] -= lr * grad * vj
        embeddings[j] -= lr * grad * vi
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss:.4f}")

# --- Cosine similarity matrix ---
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("\n=== Cosine Similarity Matrix ===")
header = "          " + "".join(f"{w:>10}" for w in vocab)
print(header)
for i in range(vocab_size):
    row = f"{vocab[i]:>10}"
    for j in range(vocab_size):
        sim = cosine_sim(embeddings[i], embeddings[j])
        row += f"{sim:10.2f}"
    print(row)

# --- Show clustering ---
print("\n=== Embedding Vectors (learned) ===")
for i, word in enumerate(vocab):
    vec_str = ", ".join(f"{v:+.3f}" for v in embeddings[i])
    print(f"  {word:>12}: [{vec_str}]")

# --- Average within-group vs between-group similarity ---
animal_sims, vehicle_sims, cross_sims = [], [], []
for i in animal_ids:
    for j in animal_ids:
        if i < j: animal_sims.append(cosine_sim(embeddings[i], embeddings[j]))
for i in vehicle_ids:
    for j in vehicle_ids:
        if i < j: vehicle_sims.append(cosine_sim(embeddings[i], embeddings[j]))
for i in animal_ids:
    for j in vehicle_ids:
        cross_sims.append(cosine_sim(embeddings[i], embeddings[j]))

print(f"\n=== Similarity Summary ===")
print(f"  Avg animal-animal similarity:   {np.mean(animal_sims):+.3f}")
print(f"  Avg vehicle-vehicle similarity: {np.mean(vehicle_sims):+.3f}")
print(f"  Avg animal-vehicle similarity:  {np.mean(cross_sims):+.3f}")
print(f"\n  Items within the same category are MUCH more similar!")
```

---

## Key Takeaways

- **Embeddings convert discrete items to continuous vectors.** An embedding layer is simply a lookup table (matrix) where each row is a learned vector for one item.
- **Geometry encodes meaning.** After training, the distances and angles between embedding vectors reflect semantic relationships -- similar items cluster together.
- **One-hot encoding wastes dimensions.** It gives every pair equal distance and scales linearly with vocabulary size. Embeddings compress items into a small, dense space where similarity is meaningful.
- **Training shapes the space.** The embedding vectors are initialized randomly, but the training objective (co-occurrence, next-word prediction, etc.) pushes related items together and unrelated items apart.
- **Embeddings are foundational.** Every transformer, recommendation system, and language model begins by embedding discrete tokens into continuous space. Understanding this lookup-and-learn mechanism is essential.
