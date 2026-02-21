# Naive Bayes

> Phase 3 — Supervised Learning: Classification | Kata 3.5

---

## Concept & Intuition

### What problem are we solving?

Given evidence (features), what is the probability that an item belongs to each class? Naive Bayes flips this question using **Bayes' theorem**: instead of directly computing P(class|features), it computes P(features|class) * P(class), which is often much easier to estimate. The "naive" part comes from assuming that all features are **conditionally independent** given the class -- each feature contributes independently to the classification.

This assumption is almost always wrong in practice (words in a document are correlated, pixel values in images are correlated), yet Naive Bayes works remarkably well anyway. It is particularly dominant in **text classification** -- spam detection, sentiment analysis, document categorization -- where the feature space is enormous (thousands of words) and the independence assumption, while incorrect, provides a useful approximation.

Naive Bayes is extraordinarily fast to train (a single pass through the data), handles high-dimensional data gracefully, and works well with small training sets. It serves as an excellent baseline for any classification problem and is often the first model tried in NLP pipelines.

### Why naive approaches fail

Without Bayes' theorem, you would need to estimate the full joint probability distribution P(feature1, feature2, ..., featureN | class). With N features, this requires an exponentially large table of probabilities. Even a modest dataset with 1000 binary features would need 2^1000 probability entries per class -- more than the number of atoms in the universe.

The independence assumption dramatically reduces this to just N probabilities per class. The tradeoff is that Naive Bayes can underestimate or overestimate confidence when features are correlated. For example, if "free" and "winner" both appear in an email, a Naive Bayes spam filter counts their evidence independently, even though they rarely appear together in legitimate emails.

### Mental models

- **Medical diagnosis by symptoms**: A doctor estimates disease probability by considering each symptom independently. "Fever makes flu 3x more likely. Cough makes it 2x more likely." The joint estimate (6x) is approximate but useful.
- **Bag of words**: In text classification, we dump all words into a bag, losing their order. Each word independently votes for a class. "Free" votes for spam; "meeting" votes for ham.
- **Prior as default belief**: Before seeing any evidence, Naive Bayes starts with the base rate (prior). If 80% of emails are ham, a new email starts as 80% likely ham. Evidence from features then updates this belief.

### Visual explanations

```
Bayes' Theorem:

  P(class | features) = P(features | class) * P(class)
                         ----------------------------
                               P(features)

  posterior  =  likelihood * prior  /  evidence

Example: Spam Classification

  Word "free" appears:
    P(free | spam)  = 0.8     (80% of spam contains "free")
    P(free | ham)   = 0.1     (10% of ham contains "free")
    P(spam)         = 0.3     (30% of emails are spam)
    P(ham)          = 0.7

    P(spam | free) = (0.8 * 0.3) / (0.8*0.3 + 0.1*0.7)
                   = 0.24 / 0.31
                   = 0.77  -->  77% likely spam!


Gaussian Naive Bayes (continuous features):

  Class 0:  Feature 1 ~ N(mu=2, sigma=1)
  Class 1:  Feature 1 ~ N(mu=5, sigma=1.5)

  Probability
    |   /\          /\
    |  /  \        /  \
    | / C0 \      / C1 \
    |/      \    /      \
    +---+----+--+---+-----> Feature 1
        2        5
```

---

## Hands-on Exploration

1. Implement Bayes' theorem by hand for a simple spam classification example. Compute P(spam | "free", "winner") step by step using the conditional independence assumption.
2. Train a `GaussianNB` on the Iris dataset. Plot the class-conditional distributions for each feature and see how the Gaussian assumption fits the actual data.
3. Build a text classifier using `MultinomialNB` on the 20 Newsgroups dataset. Inspect the top 10 most indicative words for each class using `feature_log_prob_`.
4. Compare Naive Bayes with Logistic Regression on a text dataset. Naive Bayes often wins with very little training data; Logistic Regression catches up with more data.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# Part 1: Bayes' Theorem by Hand
# ==========================================
print("=== Part 1: Manual Bayes' Theorem ===")
# Spam example
p_spam = 0.3
p_ham = 0.7
p_free_given_spam = 0.8
p_free_given_ham = 0.1
p_winner_given_spam = 0.6
p_winner_given_ham = 0.05

# P(spam | free, winner) using independence assumption
likelihood_spam = p_free_given_spam * p_winner_given_spam
likelihood_ham = p_free_given_ham * p_winner_given_ham

posterior_spam = (likelihood_spam * p_spam)
posterior_ham = (likelihood_ham * p_ham)
normalizer = posterior_spam + posterior_ham

p_spam_given_evidence = posterior_spam / normalizer
print(f"P(spam | 'free', 'winner') = {p_spam_given_evidence:.4f}")
print(f"P(ham  | 'free', 'winner') = {1 - p_spam_given_evidence:.4f}")

# ==========================================
# Part 2: Gaussian Naive Bayes on 2D data
# ==========================================
print("\n=== Part 2: Gaussian Naive Bayes ===")
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Gaussian NB Accuracy: {accuracy_score(y_test, y_pred):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Decision boundary
ax = axes[0]
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=30)
ax.set_title('Gaussian NB Decision Boundary')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Class-conditional distributions for Feature 1
ax = axes[1]
for cls in [0, 1]:
    mu = gnb.theta_[cls, 0]
    sigma = np.sqrt(gnb.var_[cls, 0])
    x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma)**2)
    ax.plot(x_range, pdf, linewidth=2, label=f'Class {cls} (mu={mu:.1f}, sig={sigma:.1f})')
    ax.hist(X_train[y_train == cls, 0], bins=15, density=True, alpha=0.3)
ax.set_title('Class-Conditional Distributions (Feature 1)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# ==========================================
# Part 3: Text Classification
# ==========================================
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.guns', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
news_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_text = vectorizer.fit_transform(newsgroups.data)
X_test_text = vectorizer.transform(news_test.data)

mnb = MultinomialNB()
mnb.fit(X_train_text, newsgroups.target)
text_acc = accuracy_score(news_test.target, mnb.predict(X_test_text))
print(f"\nMultinomial NB on 20 Newsgroups: Accuracy={text_acc:.3f}")

# Show top words per class
ax = axes[2]
feature_names = np.array(vectorizer.get_feature_names_out())
top_n = 8
class_labels = newsgroups.target_names
y_positions = []
y_labels = []

for i, (cls_name, log_prob) in enumerate(zip(class_labels, mnb.feature_log_prob_)):
    top_idx = np.argsort(log_prob)[-top_n:]
    top_words = feature_names[top_idx]
    top_probs = np.exp(log_prob[top_idx])
    pos = np.arange(top_n) + i * (top_n + 1)
    ax.barh(pos, top_probs, height=0.8, label=cls_name)
    for j, (p, w) in enumerate(zip(pos, top_words)):
        ax.text(top_probs[j], p, f' {w}', va='center', fontsize=7)

ax.set_title('Top Words per Class (Multinomial NB)')
ax.set_xlabel('Probability')
ax.set_yticks([])
ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()

print(f"\n--- Classification Report ---")
print(classification_report(news_test.target, mnb.predict(X_test_text),
                            target_names=class_labels))
```

---

## Key Takeaways

- **Naive Bayes applies Bayes' theorem with conditional independence.** It decomposes P(class|features) into a product of per-feature likelihoods, making estimation tractable even with thousands of features.
- **The "naive" independence assumption is usually wrong but still works well.** In practice, the ranking of class probabilities is often correct even when the absolute probabilities are miscalibrated.
- **Different variants handle different feature types.** GaussianNB for continuous data, MultinomialNB for counts (text), BernoulliNB for binary features.
- **Naive Bayes excels in text classification.** It is fast, handles high-dimensional sparse data naturally, and performs well even with small training sets.
- **Training is a single pass through the data.** Naive Bayes simply counts frequencies, making it one of the fastest classifiers to train and an ideal baseline model.
