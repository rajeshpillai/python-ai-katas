# Responsible AI

> Phase 11 — Productionizing ML | Kata 11.6

---

## Concept & Intuition

### What problem are we solving?

Machine learning models increasingly make decisions that affect people's lives: who gets a loan, who gets hired, who gets paroled, what news you see, what medical treatment you receive. These decisions carry enormous ethical weight. A model trained on historically biased data will perpetuate and even amplify those biases. A model that performs well on average may systematically fail for minority groups. **Responsible AI** is the discipline of detecting, measuring, and mitigating these harms.

**Fairness** in ML means that a model's predictions do not systematically disadvantage people based on protected attributes like race, gender, age, or disability. But "fair" can mean many things mathematically — and different fairness definitions are often mutually incompatible. **Demographic parity** requires equal positive prediction rates across groups. **Equalized odds** requires equal true positive and false positive rates. **Calibration** requires that a predicted probability of 70% means the same thing for all groups. You cannot satisfy all three simultaneously (except in trivial cases), so choosing a fairness metric is itself an ethical decision.

**Bias detection** involves auditing a model for disparities across groups. **Bias mitigation** involves fixing those disparities, through preprocessing (rebalancing data), in-processing (adding fairness constraints to the loss function), or post-processing (adjusting thresholds per group). Each approach has trade-offs between fairness and accuracy.

### Why naive approaches fail

Removing protected attributes (like race or gender) from the input does not make a model fair. Other features — zip code, name, purchasing habits — can be correlated with protected attributes and serve as proxies. This is called "fairness through unawareness," and it is widely recognized as insufficient. You must measure outcomes across groups to detect bias, not just check whether the model has access to protected attributes.

### Mental models

- **Bias in, bias out**: A model trained on historical hiring data where women were underrepresented in leadership will learn to penalize women. The model is faithfully learning patterns in the data — but those patterns reflect historical injustice, not ground truth.
- **Fairness as a multi-stakeholder problem**: Different stakeholders (applicants, lenders, regulators) may prefer different fairness definitions. There is no universal "correct" definition — it depends on the context and values.
- **The impossibility theorem**: You cannot simultaneously achieve demographic parity, equalized odds, and calibration (except when base rates are equal across groups). Choosing a fairness metric is a value judgment, not a technical decision.

### Visual explanations

```
Fairness Metrics:

  Demographic Parity:
    P(Y_hat=1 | group=A) = P(Y_hat=1 | group=B)
    "Equal acceptance rates regardless of group"

  Equalized Odds:
    P(Y_hat=1 | Y=1, group=A) = P(Y_hat=1 | Y=1, group=B)  (equal TPR)
    P(Y_hat=1 | Y=0, group=A) = P(Y_hat=1 | Y=0, group=B)  (equal FPR)
    "Equal accuracy regardless of group"

  Calibration:
    P(Y=1 | Y_hat=p, group=A) = P(Y=1 | Y_hat=p, group=B) = p
    "Predicted probabilities mean the same thing for all groups"

  Example of conflict:
    Group A (base rate 30%): if 30% are truly positive
    Group B (base rate 10%): if 10% are truly positive

    A calibrated model will naturally predict higher rates for Group A
    --> Violates demographic parity (unequal prediction rates)
    --> But satisfies calibration (probabilities are accurate)

Sources of bias in ML:
  Data collection    --> Who is in the dataset? Who is missing?
  Labeling           --> Are labels equally accurate for all groups?
  Feature selection  --> Do features serve as proxies for protected attributes?
  Model training     --> Does the model amplify existing disparities?
  Deployment         --> Are all groups equally served by the system?
```

---

## Hands-on Exploration

1. Generate a synthetic dataset with a protected attribute (e.g., group A and group B). Train a classifier. Compute accuracy separately for each group — are they the same?
2. Compute demographic parity, equalized odds, and calibration. Does the model satisfy all three? (Hint: it almost certainly does not.)
3. Apply a simple mitigation: adjust the classification threshold separately for each group to achieve demographic parity. What happens to overall accuracy?
4. Discuss: if the base rates are genuinely different between groups (e.g., different disease prevalence), is it "fair" to force equal prediction rates? What are the arguments on each side?

---

## Live Code

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

# --- Generate synthetic data with a protected attribute ---
n_samples = 2000

# Protected attribute: group (0 = Group A, 1 = Group B)
group = np.random.binomial(1, 0.4, n_samples)  # 40% Group B

# Features: some correlated with group (potential proxies)
# @param bias_strength float 0.0 2.0 1.0
bias_strength = 1.0  # how much the group affects outcomes

income = np.random.normal(50, 15, n_samples) + group * 5  # Group B slightly higher income
credit = np.random.normal(650, 80, n_samples) - group * bias_strength * 20  # Group B lower credit (bias)
education = np.random.normal(14, 3, n_samples) + group * 1
experience = np.random.normal(10, 5, n_samples)
debt_ratio = np.random.normal(0.35, 0.15, n_samples)

X = np.column_stack([income, credit, education, experience, debt_ratio])
feature_names = ['income', 'credit_score', 'education', 'experience', 'debt_ratio']

# True label: influenced by features AND group (systemic bias)
logit = (0.02 * (credit - 650) + 0.01 * (income - 50) + 0.05 * (education - 14)
         - 0.5 * debt_ratio - bias_strength * 0.3 * group)
prob = 1 / (1 + np.exp(-logit))
y = np.random.binomial(1, prob)

print(f"=== Responsible AI: Fairness Analysis ===")
print(f"Samples: {n_samples}, Bias strength: {bias_strength}")
print(f"Group A: {np.sum(group==0)}, Group B: {np.sum(group==1)}")
print(f"Base rate Group A: {np.mean(y[group==0]):.3f}")
print(f"Base rate Group B: {np.mean(y[group==1]):.3f}\n")

# --- Train model (without group as a feature — "fairness through unawareness") ---
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    X, y, group, test_size=0.3, random_state=42
)

model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# --- Fairness Metrics ---
def compute_fairness_metrics(y_true, y_pred, y_prob, group, group_names=('A', 'B')):
    """Compute fairness metrics across groups."""
    metrics = {}
    for g, name in enumerate(group_names):
        mask = group == g
        n = np.sum(mask)
        if n == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()

        metrics[name] = {
            'n': n,
            'base_rate': np.mean(y_t),
            'positive_rate': np.mean(y_p),
            'accuracy': accuracy_score(y_t, y_p),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'avg_prob': np.mean(y_pr),
        }
    return metrics

metrics = compute_fairness_metrics(y_test, y_pred, y_prob, group_test)

print("=== Per-Group Performance ===")
print(f"{'Metric':>20}  {'Group A':>10}  {'Group B':>10}  {'Disparity':>10}")
print("-" * 55)

metric_labels = [
    ('n', 'Sample Size'),
    ('base_rate', 'Base Rate'),
    ('positive_rate', 'Positive Rate'),
    ('accuracy', 'Accuracy'),
    ('tpr', 'True Positive Rate'),
    ('fpr', 'False Positive Rate'),
    ('avg_prob', 'Avg Pred Probability'),
]

for key, label in metric_labels:
    a_val = metrics['A'][key]
    b_val = metrics['B'][key]
    disp = abs(a_val - b_val)
    flag = " !" if disp > 0.05 and key not in ['n', 'base_rate'] else ""
    print(f"{label:>20}  {a_val:>10.4f}  {b_val:>10.4f}  {disp:>10.4f}{flag}")

# --- Formal Fairness Definitions ---
print(f"\n=== Fairness Definitions ===")

# Demographic Parity
dp_diff = abs(metrics['A']['positive_rate'] - metrics['B']['positive_rate'])
dp_ratio = min(metrics['A']['positive_rate'], metrics['B']['positive_rate']) / \
           max(metrics['A']['positive_rate'], metrics['B']['positive_rate']) if \
           max(metrics['A']['positive_rate'], metrics['B']['positive_rate']) > 0 else 0

print(f"1. Demographic Parity:")
print(f"   P(Y_hat=1|A) = {metrics['A']['positive_rate']:.4f}")
print(f"   P(Y_hat=1|B) = {metrics['B']['positive_rate']:.4f}")
print(f"   Difference:   {dp_diff:.4f} (ideal: 0)")
print(f"   Ratio:        {dp_ratio:.4f} (4/5 rule threshold: 0.80)")
print(f"   Status:       {'FAIR' if dp_ratio >= 0.80 else 'UNFAIR (violates 4/5 rule)'}")

# Equalized Odds
tpr_diff = abs(metrics['A']['tpr'] - metrics['B']['tpr'])
fpr_diff = abs(metrics['A']['fpr'] - metrics['B']['fpr'])

print(f"\n2. Equalized Odds:")
print(f"   TPR Group A: {metrics['A']['tpr']:.4f}, TPR Group B: {metrics['B']['tpr']:.4f} (diff: {tpr_diff:.4f})")
print(f"   FPR Group A: {metrics['A']['fpr']:.4f}, FPR Group B: {metrics['B']['fpr']:.4f} (diff: {fpr_diff:.4f})")
print(f"   Status:       {'FAIR' if tpr_diff < 0.05 and fpr_diff < 0.05 else 'UNFAIR'}")

# Predictive Parity
print(f"\n3. Calibration Check:")
for g_name in ['A', 'B']:
    mask = group_test == (0 if g_name == 'A' else 1)
    high_prob = y_prob[mask] > 0.5
    if np.sum(high_prob) > 0:
        actual_rate = np.mean(y_test[mask][high_prob])
        print(f"   Group {g_name}: Among high-confidence predictions, actual positive rate = {actual_rate:.4f}")

# --- Bias Mitigation: Threshold Adjustment ---
print(f"\n=== Bias Mitigation: Threshold Adjustment ===")

def find_threshold_for_rate(y_prob, target_rate):
    """Find threshold that achieves a target positive prediction rate."""
    thresholds = np.linspace(0, 1, 1000)
    for t in thresholds:
        if np.mean(y_prob >= t) <= target_rate:
            return t
    return 0.5

# Target: match Group A's positive rate
target_rate = metrics['A']['positive_rate']

# Find per-group thresholds
mask_A = group_test == 0
mask_B = group_test == 1

threshold_A = 0.5  # keep Group A threshold at 0.5
threshold_B = find_threshold_for_rate(y_prob[mask_B], target_rate)

y_pred_fair = np.zeros_like(y_pred)
y_pred_fair[mask_A] = (y_prob[mask_A] >= threshold_A).astype(int)
y_pred_fair[mask_B] = (y_prob[mask_B] >= threshold_B).astype(int)

print(f"Original threshold: 0.5 for both groups")
print(f"Adjusted thresholds: Group A = {threshold_A:.3f}, Group B = {threshold_B:.3f}")

# Recompute metrics with adjusted thresholds
metrics_fair = compute_fairness_metrics(y_test, y_pred_fair, y_prob, group_test)

print(f"\n{'Metric':>20}  {'Before':>10}  {'After':>10}")
print("-" * 45)

comparison_metrics = [
    ('Accuracy (overall)', accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_fair)),
    ('Pos Rate A', metrics['A']['positive_rate'], metrics_fair['A']['positive_rate']),
    ('Pos Rate B', metrics['B']['positive_rate'], metrics_fair['B']['positive_rate']),
    ('DP Difference', dp_diff,
     abs(metrics_fair['A']['positive_rate'] - metrics_fair['B']['positive_rate'])),
]

for label, before, after in comparison_metrics:
    improved = "better" if (label == 'DP Difference' and after < before) else \
               ("worse" if label == 'Accuracy (overall)' and after < before else "")
    print(f"{label:>20}  {before:>10.4f}  {after:>10.4f}  {improved}")

# --- Proxy Detection ---
print(f"\n=== Proxy Detection ===")
print("Correlation between features and protected attribute (group):")
print(f"{'Feature':>15}  {'Correlation':>12}  {'Potential Proxy?':>18}")
print("-" * 50)
for i, name in enumerate(feature_names):
    corr = np.corrcoef(X_test[:, i], group_test)[0, 1]
    is_proxy = "YES" if abs(corr) > 0.1 else "no"
    print(f"{name:>15}  {corr:>+12.4f}  {is_proxy:>18}")

print(f"\nNote: Even without 'group' as a feature, the model can learn")
print(f"to discriminate through proxy features correlated with group membership.")

# --- Ethical Considerations Checklist ---
print(f"\n=== Responsible AI Checklist ===")
checklist = [
    ("Data representativeness", "Are all groups adequately represented in training data?"),
    ("Label quality", "Are labels equally accurate/available across groups?"),
    ("Feature audit", "Do any features serve as proxies for protected attributes?"),
    ("Fairness metrics", "Which fairness definition is appropriate for this use case?"),
    ("Disparate impact", "Does the model's error rate differ across groups?"),
    ("Mitigation strategy", "What interventions are appropriate if bias is found?"),
    ("Transparency", "Can affected individuals understand why a decision was made?"),
    ("Recourse", "Can affected individuals contest or appeal a decision?"),
    ("Monitoring", "Is fairness monitored continuously in production?"),
    ("Stakeholder input", "Have affected communities been consulted?"),
]

for i, (item, question) in enumerate(checklist, 1):
    print(f"  {i:>2}. [{' '}] {item}")
    print(f"       {question}")
```

---

## Key Takeaways

- **Removing protected attributes does not make a model fair.** Other features can serve as proxies for race, gender, or other protected characteristics. You must measure outcomes across groups.
- **Fairness has multiple, often conflicting mathematical definitions.** Demographic parity, equalized odds, and calibration cannot all be satisfied simultaneously (when base rates differ). Choosing a fairness metric is a value judgment.
- **Bias mitigation involves trade-offs.** Adjusting thresholds to achieve demographic parity may reduce overall accuracy. Pre-processing, in-processing, and post-processing methods each have different trade-off profiles.
- **The four-fifths rule is a practical heuristic.** If the positive prediction rate for any group is less than 80% of the rate for the most favored group, disparate impact may be present.
- **Responsible AI is not just a technical problem.** It requires stakeholder engagement, transparency, recourse mechanisms, and ongoing monitoring. Technical tools like fairness metrics are necessary but not sufficient.
