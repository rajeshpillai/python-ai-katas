# Text Features

> Phase 7 — Feature Engineering & Pipelines | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

Machine learning algorithms operate on numerical vectors, but much of the world's data is text. To use text as input to a model, we need to convert it into numbers -- and the conversion method matters enormously. The simplest approach, Bag-of-Words (BoW), counts how many times each word appears in a document. TF-IDF improves on this by downweighting common words (like "the") and upweighting rare, informative words. N-grams extend single-word features to capture short phrases (bigrams like "not good" carry different meaning than "not" and "good" separately).

These representations are the workhorses of traditional text classification, sentiment analysis, and information retrieval. They are simple, fast, and surprisingly effective. A logistic regression on TF-IDF features can match or beat neural approaches on many text classification tasks, especially with limited data.

The key limitation is that these methods ignore word order (bag-of-words literally means "throw all words in a bag"). "Dog bites man" and "Man bites dog" have identical BoW representations. N-grams partially address this by capturing local word order, but at the cost of a much larger feature space.

### Why naive approaches fail

The most naive approach -- using raw text strings as features -- does not work because algorithms cannot do math on strings. Even converting each unique word to an integer ID (label encoding) fails because it implies an ordering (word 5 is "closer" to word 6 than to word 100), which is meaningless. BoW and TF-IDF create proper numerical representations where each dimension has a clear meaning (presence/frequency of a specific word), enabling standard ML algorithms to work directly on text.

### Mental models

- **Bag of words as a word histogram**: for each document, count how many times each word from the vocabulary appears. Two documents with similar word distributions are likely about similar topics.
- **TF-IDF as importance weighting**: Term Frequency says "this word appears a lot in this document." Inverse Document Frequency says "this word is rare across all documents." TF-IDF = TF x IDF highlights words that are both frequent locally and rare globally -- the most informative words.
- **N-grams as phrase detection**: unigrams lose "not good" (looks positive if "good" is positive). Bigrams capture "not good" as a single negative feature.

### Visual explanations

```
Document: "the cat sat on the mat"
Vocabulary: [cat, mat, on, sat, the]

Bag of Words (raw counts):
  cat=1  mat=1  on=1  sat=1  the=2

TF-IDF:
  cat=high  mat=high  on=low  sat=medium  the=very low
  ("the" appears in every document, so IDF crushes it)

N-grams (bigrams):
  "the cat"  "cat sat"  "sat on"  "on the"  "the mat"
  (captures word pairs, 5 bigram features)

Feature Matrix (BoW, 3 documents):
                cat  dog  mat  on  sat  the
  doc1 "cat sat"  1    0    0   0   1    0
  doc2 "dog sat"  0    1    0   0   1    0
  doc3 "cat mat"  1    0    1   0   0    0

  --> sparse matrix (mostly zeros), high-dimensional
```

---

## Hands-on Exploration

1. Take a small collection of text documents and manually build a Bag-of-Words matrix. Verify the counts match your hand calculation.
2. Compute TF-IDF scores and observe how common words (the, is, a) get low scores while rare topic words get high scores.
3. Compare classification accuracy of a model trained on BoW features vs TF-IDF features vs bigram features.
4. Examine the highest-weighted features in a trained model to understand which words (or word pairs) are most predictive for each class.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups

np.random.seed(42)

# --- Small example to understand the mechanics ---
docs = [
    "the cat sat on the mat",
    "the dog played in the park",
    "the cat chased the dog",
    "a dog sat on a mat in the park",
]

print("=== Bag of Words (small example) ===\n")
bow = CountVectorizer()
X_bow = bow.fit_transform(docs)
vocab = bow.get_feature_names_out()

print(f"  Vocabulary ({len(vocab)} words): {list(vocab)}\n")
print(f"  {'Document':<35} ", end="")
for word in vocab:
    print(f"{word:>7}", end="")
print()
print("  " + "-" * (35 + 7 * len(vocab)))
for i, doc in enumerate(docs):
    row = X_bow[i].toarray()[0]
    print(f"  {doc:<35} ", end="")
    for val in row:
        print(f"{val:>7}", end="")
    print()

print("\n=== TF-IDF (small example) ===\n")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)

print(f"  {'Document':<35} ", end="")
for word in vocab[:6]:
    print(f"{word:>8}", end="")
print()
print("  " + "-" * (35 + 8 * 6))
for i, doc in enumerate(docs):
    row = X_tfidf[i].toarray()[0]
    print(f"  {doc:<35} ", end="")
    for val in row[:6]:
        print(f"{val:>8.3f}", end="")
    print()

# --- Real classification task: 20 newsgroups ---
print("\n=== Text Classification: 20 Newsgroups ===\n")
categories = ["rec.sport.baseball", "sci.space", "talk.politics.misc", "comp.graphics"]
newsgroups = fetch_20newsgroups(
    subset="train", categories=categories, random_state=42,
    remove=("headers", "footers", "quotes")
)
X_text = newsgroups.data
y_text = newsgroups.target
class_names = newsgroups.target_names

print(f"  {len(X_text)} documents, {len(categories)} categories")
print(f"  Categories: {class_names}\n")

# --- Compare BoW vs TF-IDF vs Bigrams ---
configs = {
    "BoW (unigrams)": CountVectorizer(max_features=5000),
    "TF-IDF (unigrams)": TfidfVectorizer(max_features=5000),
    "TF-IDF (uni+bigrams)": TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
    "TF-IDF (bigrams only)": TfidfVectorizer(max_features=5000, ngram_range=(2, 2)),
}

results = {}
lr = LogisticRegression(max_iter=1000, random_state=42)

for name, vectorizer in configs.items():
    X_vec = vectorizer.fit_transform(X_text)
    scores = cross_val_score(lr, X_vec, y_text, cv=5, scoring="accuracy")
    results[name] = scores.mean()
    n_feats = X_vec.shape[1]
    sparsity = 1 - X_vec.nnz / (X_vec.shape[0] * X_vec.shape[1])
    print(f"  {name:<25}: accuracy={scores.mean():.4f} "
          f"(+/- {scores.std():.4f}), {n_feats} features, "
          f"{sparsity*100:.1f}% sparse")

# --- Most predictive features ---
print("\n=== Most Predictive Features (TF-IDF + Logistic Regression) ===\n")
best_vec = TfidfVectorizer(max_features=5000)
X_best = best_vec.fit_transform(X_text)
lr.fit(X_best, y_text)
feature_names = best_vec.get_feature_names_out()

for i, class_name in enumerate(class_names):
    coefs = lr.coef_[i]
    top_idx = np.argsort(coefs)[-8:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    top_scores = [coefs[j] for j in top_idx]
    print(f"  {class_name}:")
    for word, score in zip(top_words, top_scores):
        bar = "|" * int(score * 10)
        print(f"    {word:<20} {score:+.3f}  {bar}")
    print()

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Method comparison
names = list(results.keys())
accs = [results[n] for n in names]
axes[0].barh(names, accs, color="steelblue")
axes[0].set_xlabel("CV Accuracy")
axes[0].set_title("Text Representation Comparison")
axes[0].set_xlim(min(accs) - 0.02, max(accs) + 0.02)
axes[0].invert_yaxis()

# Top features per class
class_idx = 0
coefs = lr.coef_[class_idx]
top_n = 12
top_idx = np.argsort(np.abs(coefs))[-top_n:]
axes[1].barh(
    [feature_names[j] for j in top_idx],
    [coefs[j] for j in top_idx],
    color=["coral" if coefs[j] > 0 else "steelblue" for j in top_idx]
)
axes[1].set_xlabel("Coefficient")
axes[1].set_title(f"Top Features for '{class_names[class_idx]}'")
axes[1].axvline(0, color="k", linewidth=0.5)

plt.tight_layout()
plt.savefig("text_features.png", dpi=100, bbox_inches="tight")
plt.show()
print("Plot saved to text_features.png")
```

---

## Key Takeaways

- **Bag-of-Words converts text to word-count vectors.** Simple and effective, but ignores word order and treats all words equally.
- **TF-IDF improves BoW by downweighting common words.** Words that appear everywhere ("the", "is") get low scores; rare topic words get high scores.
- **N-grams capture local word order.** Bigrams like "not good" or "machine learning" carry meaning that individual words do not.
- **Text feature matrices are extremely sparse.** Most documents use only a tiny fraction of the vocabulary. Sparse matrix storage is essential.
- **Simple models on good text features are hard to beat.** Logistic regression on TF-IDF features remains a strong baseline even in the era of deep learning, especially with limited data.
