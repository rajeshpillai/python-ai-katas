from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.kata import Kata, KataList

router = APIRouter()

CONTENT_DIR = Path(__file__).resolve().parents[2] / "content"

FOUNDATIONAL_KATAS: list[dict] = [
    # Phase 0 — Foundations
    {"id": "what-is-data", "title": "What is data?", "phase": 0, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "what-is-a-feature", "title": "What is a feature?", "phase": 0, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "what-is-noise", "title": "What is noise?", "phase": 0, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "what-is-signal", "title": "What is signal?", "phase": 0, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "visualizing-distributions", "title": "Visualizing distributions", "phase": 0, "sequence": 5, "track_id": "foundational-ai"},
    # Phase 1 — What Does It Mean to Learn?
    {"id": "constant-predictor", "title": "Constant predictor", "phase": 1, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "mean-predictor", "title": "Mean predictor", "phase": 1, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "distance-based-prediction", "title": "Distance-based prediction", "phase": 1, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "linear-regression", "title": "Linear regression", "phase": 1, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "loss-functions", "title": "Loss functions", "phase": 1, "sequence": 5, "track_id": "foundational-ai"},
    {"id": "error-surfaces", "title": "Error surfaces", "phase": 1, "sequence": 6, "track_id": "foundational-ai"},
    {"id": "overfitting-vs-underfitting", "title": "Overfitting vs underfitting", "phase": 1, "sequence": 7, "track_id": "foundational-ai"},
    # Phase 2 — Optimization
    {"id": "manual-gradient-descent", "title": "Manual gradient descent", "phase": 2, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "learning-rate-experiments", "title": "Learning rate experiments", "phase": 2, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "convergence-vs-divergence", "title": "Convergence vs divergence", "phase": 2, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "visualizing-loss-curves", "title": "Visualizing loss curves", "phase": 2, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "pytorch-tensors", "title": "PyTorch tensors", "phase": 2, "sequence": 5, "track_id": "foundational-ai"},
    {"id": "autograd", "title": "Autograd", "phase": 2, "sequence": 6, "track_id": "foundational-ai"},
    # Phase 3 — Artificial Neural Networks
    {"id": "single-neuron", "title": "Single neuron", "phase": 3, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "multi-layer-perceptron", "title": "Multi-layer perceptron", "phase": 3, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "activation-functions", "title": "Activation functions", "phase": 3, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "vanishing-gradients", "title": "Vanishing gradients", "phase": 3, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "dead-neurons", "title": "Dead neurons", "phase": 3, "sequence": 5, "track_id": "foundational-ai"},
    # Phase 4 — Representation Learning
    {"id": "handcrafted-vs-learned-features", "title": "Handcrafted vs learned features", "phase": 4, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "pca-intuition", "title": "PCA intuition", "phase": 4, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "embeddings", "title": "Embeddings", "phase": 4, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "visualizing-hidden-layers", "title": "Visualizing hidden layers", "phase": 4, "sequence": 4, "track_id": "foundational-ai"},
    # Phase 5 — CNN
    {"id": "why-dense-networks-fail-on-images", "title": "Why dense networks fail on images", "phase": 5, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "convolution-intuition", "title": "Convolution intuition", "phase": 5, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "filters-as-pattern-detectors", "title": "Filters as pattern detectors", "phase": 5, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "feature-maps", "title": "Feature maps", "phase": 5, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "pooling-effects", "title": "Pooling effects", "phase": 5, "sequence": 5, "track_id": "foundational-ai"},
    # Phase 6 — Sequence Models
    {"id": "n-grams", "title": "N-grams", "phase": 6, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "rnn-intuition", "title": "RNN intuition", "phase": 6, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "lstm-limitations", "title": "LSTM limitations", "phase": 6, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "why-recurrence-struggles-at-scale", "title": "Why recurrence struggles at scale", "phase": 6, "sequence": 4, "track_id": "foundational-ai"},
    # Phase 7 — Attention & Transformers
    {"id": "attention-as-weighted-averaging", "title": "Attention as weighted averaging", "phase": 7, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "self-attention-visualization", "title": "Self-attention visualization", "phase": 7, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "positional-encoding", "title": "Positional encoding", "phase": 7, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "why-transformers-scale", "title": "Why transformers scale", "phase": 7, "sequence": 4, "track_id": "foundational-ai"},
    # Phase 8 — LLMs
    {"id": "tokenization", "title": "Tokenization", "phase": 8, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "next-token-prediction", "title": "Next-token prediction", "phase": 8, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "sampling-strategies", "title": "Sampling strategies", "phase": 8, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "temperature-top-k-top-p", "title": "Temperature, top-k, top-p", "phase": 8, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "emergence-through-scale", "title": "Emergence through scale", "phase": 8, "sequence": 5, "track_id": "foundational-ai"},
    # Phase 9 — Reasoning Models
    {"id": "chain-of-thought-as-latent-variables", "title": "Chain-of-thought as latent variables", "phase": 9, "sequence": 1, "track_id": "foundational-ai"},
    {"id": "scratchpads", "title": "Scratchpads", "phase": 9, "sequence": 2, "track_id": "foundational-ai"},
    {"id": "tool-usage", "title": "Tool usage", "phase": 9, "sequence": 3, "track_id": "foundational-ai"},
    {"id": "planning-vs-prediction", "title": "Planning vs prediction", "phase": 9, "sequence": 4, "track_id": "foundational-ai"},
    {"id": "failure-modes-of-llm-reasoning", "title": "Failure modes of LLM reasoning", "phase": 9, "sequence": 5, "track_id": "foundational-ai"},
]

TRADITIONAL_ML_KATAS: list[dict] = [
    # Phase 0 — What is AI?
    {"id": "what-is-ai", "title": "What is AI?", "phase": 0, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "rule-based-systems", "title": "Rule-based systems", "phase": 0, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "search-algorithms", "title": "Search algorithms", "phase": 0, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "heuristics-and-cost", "title": "Heuristics and cost", "phase": 0, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "knowledge-representation", "title": "Knowledge representation", "phase": 0, "sequence": 5, "track_id": "traditional-ai-ml"},
    # Phase 1 — Data Wrangling
    {"id": "tabular-data-basics", "title": "Tabular data basics", "phase": 1, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "missing-values", "title": "Missing values", "phase": 1, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "outlier-detection", "title": "Outlier detection", "phase": 1, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "encoding-categorical-variables", "title": "Encoding categorical variables", "phase": 1, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "feature-scaling", "title": "Feature scaling", "phase": 1, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "train-test-split", "title": "Train-test split", "phase": 1, "sequence": 6, "track_id": "traditional-ai-ml"},
    {"id": "exploratory-data-analysis", "title": "Exploratory data analysis", "phase": 1, "sequence": 7, "track_id": "traditional-ai-ml"},
    # Phase 2 — Supervised Learning: Regression
    {"id": "simple-linear-regression", "title": "Simple linear regression", "phase": 2, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "multiple-linear-regression", "title": "Multiple linear regression", "phase": 2, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "polynomial-regression", "title": "Polynomial regression", "phase": 2, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "regularization-ridge", "title": "Regularization: Ridge", "phase": 2, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "regularization-lasso", "title": "Regularization: Lasso", "phase": 2, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "elastic-net", "title": "Elastic Net", "phase": 2, "sequence": 6, "track_id": "traditional-ai-ml"},
    {"id": "regression-diagnostics", "title": "Regression diagnostics", "phase": 2, "sequence": 7, "track_id": "traditional-ai-ml"},
    # Phase 3 — Supervised Learning: Classification
    {"id": "k-nearest-neighbors", "title": "K-nearest neighbors", "phase": 3, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "logistic-regression", "title": "Logistic regression", "phase": 3, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "decision-trees", "title": "Decision trees", "phase": 3, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "support-vector-machines", "title": "Support vector machines", "phase": 3, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "naive-bayes", "title": "Naive Bayes", "phase": 3, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "multiclass-strategies", "title": "Multiclass strategies", "phase": 3, "sequence": 6, "track_id": "traditional-ai-ml"},
    {"id": "imbalanced-classes", "title": "Imbalanced classes", "phase": 3, "sequence": 7, "track_id": "traditional-ai-ml"},
    # Phase 4 — Model Evaluation & Selection
    {"id": "accuracy-and-its-limits", "title": "Accuracy and its limits", "phase": 4, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "precision-recall-f1", "title": "Precision, recall, F1", "phase": 4, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "roc-and-auc", "title": "ROC and AUC", "phase": 4, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "cross-validation", "title": "Cross-validation", "phase": 4, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "hyperparameter-tuning", "title": "Hyperparameter tuning", "phase": 4, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "bias-variance-tradeoff", "title": "Bias-variance tradeoff", "phase": 4, "sequence": 6, "track_id": "traditional-ai-ml"},
    # Phase 5 — Unsupervised Learning
    {"id": "k-means-clustering", "title": "K-means clustering", "phase": 5, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "hierarchical-clustering", "title": "Hierarchical clustering", "phase": 5, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "dbscan", "title": "DBSCAN", "phase": 5, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "pca-for-dimensionality-reduction", "title": "PCA for dimensionality reduction", "phase": 5, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "t-sne-visualization", "title": "t-SNE visualization", "phase": 5, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "anomaly-detection", "title": "Anomaly detection", "phase": 5, "sequence": 6, "track_id": "traditional-ai-ml"},
    # Phase 6 — Ensemble Methods
    {"id": "bagging-intuition", "title": "Bagging intuition", "phase": 6, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "random-forests", "title": "Random forests", "phase": 6, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "boosting-intuition", "title": "Boosting intuition", "phase": 6, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "adaboost", "title": "AdaBoost", "phase": 6, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "gradient-boosting", "title": "Gradient boosting", "phase": 6, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "xgboost", "title": "XGBoost", "phase": 6, "sequence": 6, "track_id": "traditional-ai-ml"},
    {"id": "stacking", "title": "Stacking", "phase": 6, "sequence": 7, "track_id": "traditional-ai-ml"},
    # Phase 7 — Feature Engineering & Pipelines
    {"id": "feature-creation", "title": "Feature creation", "phase": 7, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "feature-selection", "title": "Feature selection", "phase": 7, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "text-features", "title": "Text features", "phase": 7, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "datetime-features", "title": "Datetime features", "phase": 7, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "sklearn-pipelines", "title": "Scikit-learn pipelines", "phase": 7, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "custom-transformers", "title": "Custom transformers", "phase": 7, "sequence": 6, "track_id": "traditional-ai-ml"},
    # Phase 8 — Time Series & Sequential Data
    {"id": "time-series-basics", "title": "Time series basics", "phase": 8, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "autocorrelation", "title": "Autocorrelation", "phase": 8, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "arima", "title": "ARIMA", "phase": 8, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "exponential-smoothing", "title": "Exponential smoothing", "phase": 8, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "forecasting-evaluation", "title": "Forecasting evaluation", "phase": 8, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "sequence-classification", "title": "Sequence classification", "phase": 8, "sequence": 6, "track_id": "traditional-ai-ml"},
    # Phase 9 — Reinforcement Learning
    {"id": "markov-decision-processes", "title": "Markov decision processes", "phase": 9, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "value-functions", "title": "Value functions", "phase": 9, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "policy-iteration", "title": "Policy iteration", "phase": 9, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "q-learning", "title": "Q-learning", "phase": 9, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "sarsa", "title": "SARSA", "phase": 9, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "multi-armed-bandits", "title": "Multi-armed bandits", "phase": 9, "sequence": 6, "track_id": "traditional-ai-ml"},
    {"id": "environment-design", "title": "Environment design", "phase": 9, "sequence": 7, "track_id": "traditional-ai-ml"},
    # Phase 10 — Probabilistic & Bayesian Methods
    {"id": "bayesian-thinking", "title": "Bayesian thinking", "phase": 10, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "maximum-likelihood-estimation", "title": "Maximum likelihood estimation", "phase": 10, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "map-estimation", "title": "MAP estimation", "phase": 10, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "gaussian-mixture-models", "title": "Gaussian mixture models", "phase": 10, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "bayesian-optimization", "title": "Bayesian optimization", "phase": 10, "sequence": 5, "track_id": "traditional-ai-ml"},
    # Phase 11 — Productionizing ML
    {"id": "model-serialization", "title": "Model serialization", "phase": 11, "sequence": 1, "track_id": "traditional-ai-ml"},
    {"id": "data-drift-detection", "title": "Data drift detection", "phase": 11, "sequence": 2, "track_id": "traditional-ai-ml"},
    {"id": "experiment-tracking", "title": "Experiment tracking", "phase": 11, "sequence": 3, "track_id": "traditional-ai-ml"},
    {"id": "ab-testing-for-models", "title": "A/B testing for models", "phase": 11, "sequence": 4, "track_id": "traditional-ai-ml"},
    {"id": "interpretability", "title": "Interpretability", "phase": 11, "sequence": 5, "track_id": "traditional-ai-ml"},
    {"id": "responsible-ai", "title": "Responsible AI", "phase": 11, "sequence": 6, "track_id": "traditional-ai-ml"},
]

FOUNDATIONAL_PHASE_NAMES = {
    0: "Foundations (Before ML)",
    1: "What Does It Mean to Learn?",
    2: "Optimization",
    3: "Artificial Neural Networks (ANN)",
    4: "Representation Learning",
    5: "Convolutional Neural Networks (CNN)",
    6: "Sequence Models",
    7: "Attention & Transformers",
    8: "Large Language Models (LLMs)",
    9: "Reasoning Models",
}

TRADITIONAL_PHASE_NAMES = {
    0: "What is AI?",
    1: "Data Wrangling",
    2: "Supervised Learning: Regression",
    3: "Supervised Learning: Classification",
    4: "Model Evaluation & Selection",
    5: "Unsupervised Learning",
    6: "Ensemble Methods",
    7: "Feature Engineering & Pipelines",
    8: "Time Series & Sequential Data",
    9: "Reinforcement Learning",
    10: "Probabilistic & Bayesian Methods",
    11: "Productionizing ML",
}

TRACK_REGISTRY = {
    "foundational-ai": {
        "katas": FOUNDATIONAL_KATAS,
        "phases": FOUNDATIONAL_PHASE_NAMES,
    },
    "traditional-ai-ml": {
        "katas": TRADITIONAL_ML_KATAS,
        "phases": TRADITIONAL_PHASE_NAMES,
    },
}


@router.get("/tracks/{track_id}/katas")
async def list_katas(track_id: str):
    track = TRACK_REGISTRY.get(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
    return {
        "katas": track["katas"],
        "phases": track["phases"],
    }


@router.get("/tracks/{track_id}/katas/{phase_id}/{kata_id}/content", response_class=PlainTextResponse)
async def get_kata_content(track_id: str, phase_id: int, kata_id: str):
    track = TRACK_REGISTRY.get(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")

    katas = track["katas"]
    kata = next(
        (k for k in katas if k["id"] == kata_id and k["phase"] == phase_id),
        None,
    )
    if not kata:
        raise HTTPException(status_code=404, detail=f"Kata '{kata_id}' not found in phase {phase_id}")

    phase_dir = CONTENT_DIR / track_id / f"phase-{phase_id}"
    seq = kata["sequence"]
    filename = f"{seq:02d}-{kata_id}.md"
    filepath = phase_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Content file not found: {filename}")

    return filepath.read_text(encoding="utf-8")
