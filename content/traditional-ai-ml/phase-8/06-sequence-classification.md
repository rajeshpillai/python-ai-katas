# Sequence Classification

> Phase 8 — Time Series & Sequential Data | Kata 8.6

---

## Concept & Intuition

### What problem are we solving?

Not all sequential data is about forecasting. Sometimes you have a sequence of observations and you want to classify the *entire sequence* into a category. Is this accelerometer trace from walking, running, or sitting? Is this heartbeat signal normal or arrhythmic? Is this series of web requests from a bot or a human? This is sequence classification -- assigning a label to a whole time series rather than predicting its next value.

The challenge is that standard ML classifiers expect a fixed-length feature vector, but sequences can vary in length and the relevant information is often in the shape, trend, or statistics of the sequence rather than in individual values. The solution is **feature extraction from sequences**: transform each variable-length sequence into a fixed-length vector of summary statistics, then feed those vectors into any standard classifier.

Common features include statistical summaries (mean, std, min, max, skewness, kurtosis), temporal features (slope, autocorrelation, number of peaks), frequency-domain features (dominant frequency from FFT), and shape features (number of zero-crossings, longest increasing run). The **sliding window** approach converts a long sequence into many overlapping windows, each classified independently, which is useful for detecting events within a continuous stream.

### Why naive approaches fail

Feeding raw sequence values directly into a classifier fails for two reasons. First, sequences of different lengths produce different-sized feature vectors, which most classifiers cannot handle. Second, even for fixed-length sequences, the raw values are too high-dimensional and noisy. A 1000-step sequence creates 1000 features, most of which are redundant. Feature extraction compresses the sequence into a handful of informative statistics (e.g., 20 features), making the classification problem tractable and the model more robust.

### Mental models

- **Sequence summary as fingerprint**: each activity (walking, running) produces a characteristic statistical fingerprint. Mean acceleration, variance, dominant frequency -- these statistics differ systematically between activities.
- **Sliding window as microscope**: slide a small window across a long sequence, extract features from each window, and classify it. This turns a stream-level problem into many frame-level classifications.
- **Feature engineering as domain expertise**: the best features come from understanding the domain. For audio, frequency features dominate. For motion, acceleration statistics. For financial data, return distributions.

### Visual explanations

```
Sequence Classification Pipeline:

  Raw sequence (variable length):
    Walking:  [0.2, 0.5, 0.1, 0.4, 0.6, 0.2, 0.5, ...]  (N=1200)
    Running:  [0.8, 1.2, 0.3, 1.0, 1.5, ...]              (N=800)
    Sitting:  [0.01, 0.02, 0.01, 0.02, ...]                (N=1500)

  Feature extraction:
    Walking --> [mean=0.38, std=0.17, max=0.62, freq=2.1, peaks=12, ...]
    Running --> [mean=0.96, std=0.42, max=1.50, freq=3.5, peaks=18, ...]
    Sitting --> [mean=0.015, std=0.005, max=0.03, freq=0.0, peaks=0, ...]

  Fixed-length feature vector --> Standard classifier (RF, SVM, etc.)

Sliding Window:
  Sequence: |====================================|
  Window 1: |===|
  Window 2:   |===|
  Window 3:     |===|  ... (overlap)
  Each window --> features --> classify
```

---

## Hands-on Exploration

1. Generate synthetic sequences for three activity classes (walking, running, sitting) with distinct statistical properties. Visualize examples from each class.
2. Implement feature extraction: compute statistical, temporal, and frequency-domain features from each sequence. Inspect the feature distributions per class.
3. Train a Random Forest classifier on the extracted features and evaluate using cross-validation.
4. Implement a sliding window approach on a long continuous sequence and visualize the per-window classification results.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# ============================================================
# Generate synthetic activity sequences
# ============================================================

def generate_walking(n_steps=200):
    """Regular oscillation, moderate amplitude."""
    t = np.arange(n_steps)
    return (0.4 * np.sin(2 * np.pi * t / 50)
            + 0.1 * np.sin(2 * np.pi * t / 25)
            + np.random.normal(0, 0.08, n_steps))

def generate_running(n_steps=200):
    """Faster oscillation, higher amplitude."""
    t = np.arange(n_steps)
    return (0.9 * np.sin(2 * np.pi * t / 30)
            + 0.3 * np.sin(2 * np.pi * t / 15)
            + np.random.normal(0, 0.15, n_steps))

def generate_sitting(n_steps=200):
    """Nearly flat, very low amplitude."""
    return np.random.normal(0, 0.03, n_steps)

# Generate dataset
n_samples_per_class = 100
sequences = []
labels = []

for _ in range(n_samples_per_class):
    n_steps = np.random.randint(150, 250)
    sequences.append(generate_walking(n_steps))
    labels.append("walking")

    n_steps = np.random.randint(150, 250)
    sequences.append(generate_running(n_steps))
    labels.append("running")

    n_steps = np.random.randint(150, 250)
    sequences.append(generate_sitting(n_steps))
    labels.append("sitting")

labels = np.array(labels)
print(f"=== Dataset: {len(sequences)} sequences, 3 classes ===")
print(f"  Sequence lengths: {min(len(s) for s in sequences)} to "
      f"{max(len(s) for s in sequences)}\n")

# ============================================================
# Feature extraction
# ============================================================

def extract_features(seq):
    """Extract fixed-length feature vector from a variable-length sequence."""
    features = {}

    # Statistical features
    features["mean"] = np.mean(seq)
    features["std"] = np.std(seq)
    features["min"] = np.min(seq)
    features["max"] = np.max(seq)
    features["range"] = np.max(seq) - np.min(seq)
    features["median"] = np.median(seq)
    features["skewness"] = float(pd.Series(seq).skew())
    features["kurtosis"] = float(pd.Series(seq).kurtosis())

    # Percentiles
    features["q25"] = np.percentile(seq, 25)
    features["q75"] = np.percentile(seq, 75)
    features["iqr"] = features["q75"] - features["q25"]

    # Temporal features
    diff = np.diff(seq)
    features["mean_abs_diff"] = np.mean(np.abs(diff))
    features["std_diff"] = np.std(diff)
    features["max_abs_diff"] = np.max(np.abs(diff))

    # Zero crossings
    zero_crossings = np.sum(np.diff(np.sign(seq)) != 0)
    features["zero_crossings"] = zero_crossings

    # Number of peaks (simple: local maxima)
    peaks = np.sum((seq[1:-1] > seq[:-2]) & (seq[1:-1] > seq[2:]))
    features["n_peaks"] = peaks

    # Autocorrelation at lag 1
    if len(seq) > 1:
        features["autocorr_1"] = np.corrcoef(seq[:-1], seq[1:])[0, 1]
    else:
        features["autocorr_1"] = 0

    # Frequency domain (dominant frequency from FFT)
    fft_vals = np.abs(np.fft.rfft(seq))
    fft_freqs = np.fft.rfftfreq(len(seq))
    features["dominant_freq"] = fft_freqs[np.argmax(fft_vals[1:]) + 1]
    features["spectral_energy"] = np.sum(fft_vals ** 2)
    features["spectral_entropy"] = -np.sum(
        (fft_vals / fft_vals.sum()) * np.log(fft_vals / fft_vals.sum() + 1e-10)
    )

    # Energy
    features["rms"] = np.sqrt(np.mean(seq ** 2))
    features["abs_energy"] = np.sum(seq ** 2)

    return features


# Extract features for all sequences
print("=== Feature Extraction ===\n")
feature_dicts = [extract_features(s) for s in sequences]
X = pd.DataFrame(feature_dicts)
feature_names = list(X.columns)

print(f"  Extracted {len(feature_names)} features per sequence:")
for f in feature_names:
    print(f"    {f}")

# ============================================================
# Classification
# ============================================================
print("\n=== Classification ===\n")

clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# Cross-validation
scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
print(f"  5-fold CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})\n")

# Train/test split for detailed analysis
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("  Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
rf_model = clf.named_steps["randomforestclassifier"]
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("  Top 10 Features:")
for rank, idx in enumerate(sorted_idx[:10]):
    bar = "|" * int(importances[idx] * 100)
    print(f"    {rank+1}. {feature_names[idx]:<20} {importances[idx]:.3f}  {bar}")

# ============================================================
# Sliding window approach
# ============================================================
print("\n=== Sliding Window Classification ===\n")

# Generate a long continuous sequence with changing activities
long_seq = np.concatenate([
    generate_walking(300),
    generate_running(200),
    generate_sitting(250),
    generate_walking(200),
    generate_running(150),
])
true_labels_long = (
    ["walking"] * 300 + ["running"] * 200 + ["sitting"] * 250 +
    ["walking"] * 200 + ["running"] * 150
)

# Sliding window
window_size = 100
step_size = 20
window_preds = []
window_centers = []

for start in range(0, len(long_seq) - window_size, step_size):
    window = long_seq[start:start + window_size]
    features = extract_features(window)
    features_df = pd.DataFrame([features])
    pred = clf.predict(features_df)[0]
    window_preds.append(pred)
    window_centers.append(start + window_size // 2)

print(f"  Long sequence length: {len(long_seq)}")
print(f"  Window size: {window_size}, Step: {step_size}")
print(f"  Number of windows: {len(window_preds)}")

# Accuracy (approximate, using center point label)
window_true = [true_labels_long[c] for c in window_centers]
window_acc = np.mean([p == t for p, t in zip(window_preds, window_true)])
print(f"  Window classification accuracy: {window_acc:.4f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Example sequences
for i, (cls, gen_fn, color) in enumerate([
    ("Walking", generate_walking, "blue"),
    ("Running", generate_running, "red"),
    ("Sitting", generate_sitting, "green"),
]):
    seq = gen_fn(200)
    axes[0, 0].plot(seq + i * 2, color=color, linewidth=0.8,
                    label=cls, alpha=0.8)
axes[0, 0].set_title("Example Sequences by Activity")
axes[0, 0].legend()
axes[0, 0].set_xlabel("Time step")

# Feature distributions
feat_to_show = "std"
for cls in ["walking", "running", "sitting"]:
    mask = labels == cls
    vals = X.loc[mask, feat_to_show]
    axes[0, 1].hist(vals, bins=20, alpha=0.5, label=cls)
axes[0, 1].set_title(f"Feature Distribution: '{feat_to_show}'")
axes[0, 1].legend()
axes[0, 1].set_xlabel(feat_to_show)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["walking", "running", "sitting"])
im = axes[1, 0].imshow(cm, cmap="Blues")
axes[1, 0].set_xticks(range(3))
axes[1, 0].set_yticks(range(3))
axes[1, 0].set_xticklabels(["walk", "run", "sit"])
axes[1, 0].set_yticklabels(["walk", "run", "sit"])
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_title("Confusion Matrix")
for i in range(3):
    for j in range(3):
        axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

# Sliding window classification
color_map = {"walking": "blue", "running": "red", "sitting": "green"}
axes[1, 1].plot(long_seq, linewidth=0.5, color="gray", alpha=0.5)
for i, (center, pred) in enumerate(zip(window_centers, window_preds)):
    axes[1, 1].axvspan(
        center - window_size // 2, center + window_size // 2,
        alpha=0.03, color=color_map[pred]
    )
# Add legend patches
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, alpha=0.3, label=l) for l, c in color_map.items()]
axes[1, 1].legend(handles=legend_patches, fontsize=8)
axes[1, 1].set_title("Sliding Window Classification")
axes[1, 1].set_xlabel("Time step")

plt.tight_layout()
plt.savefig("sequence_classification.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to sequence_classification.png")
```

---

## Key Takeaways

- **Sequence classification assigns a label to an entire time series, not a single time step.** The challenge is converting variable-length sequences into fixed-length feature vectors.
- **Feature extraction is the critical step.** Statistical (mean, std, skewness), temporal (autocorrelation, peaks), and frequency (FFT) features capture different aspects of the sequence's shape and behavior.
- **Any standard classifier works once features are extracted.** Random Forests, SVMs, and logistic regression all work well on the extracted feature vectors.
- **Sliding windows enable real-time classification of continuous streams.** Slide a window across the stream, extract features, classify each window, and detect activity transitions.
- **Domain knowledge drives the best features.** Generic statistical features are a good starting point, but domain-specific features (e.g., step frequency for activity recognition, return volatility for financial classification) often provide the biggest accuracy gains.
