use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct Kata {
    pub id: &'static str,
    pub title: &'static str,
    pub phase: u32,
    pub sequence: u32,
    pub track_id: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Serialize)]
pub struct KataListResponse {
    pub katas: Vec<&'static Kata>,
    pub phases: HashMap<&'static str, &'static str>,
}

pub struct TrackRegistry {
    pub katas: &'static [Kata],
    pub phases: &'static [(&'static str, &'static str)],
}

pub static FOUNDATIONAL_KATAS: &[Kata] = &[
    // Phase 0 — Foundations
    Kata { id: "what-is-data", title: "What is data?", phase: 0, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "what-is-a-feature", title: "What is a feature?", phase: 0, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "what-is-noise", title: "What is noise?", phase: 0, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "what-is-signal", title: "What is signal?", phase: 0, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "visualizing-distributions", title: "Visualizing distributions", phase: 0, sequence: 5, track_id: "foundational-ai", description: "" },
    // Phase 1 — What Does It Mean to Learn?
    Kata { id: "constant-predictor", title: "Constant predictor", phase: 1, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "mean-predictor", title: "Mean predictor", phase: 1, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "distance-based-prediction", title: "Distance-based prediction", phase: 1, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "linear-regression", title: "Linear regression", phase: 1, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "loss-functions", title: "Loss functions", phase: 1, sequence: 5, track_id: "foundational-ai", description: "" },
    Kata { id: "error-surfaces", title: "Error surfaces", phase: 1, sequence: 6, track_id: "foundational-ai", description: "" },
    Kata { id: "overfitting-vs-underfitting", title: "Overfitting vs underfitting", phase: 1, sequence: 7, track_id: "foundational-ai", description: "" },
    // Phase 2 — Optimization
    Kata { id: "manual-gradient-descent", title: "Manual gradient descent", phase: 2, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "learning-rate-experiments", title: "Learning rate experiments", phase: 2, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "convergence-vs-divergence", title: "Convergence vs divergence", phase: 2, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "visualizing-loss-curves", title: "Visualizing loss curves", phase: 2, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "pytorch-tensors", title: "Tensors from scratch", phase: 2, sequence: 5, track_id: "foundational-ai", description: "" },
    Kata { id: "autograd", title: "Automatic differentiation", phase: 2, sequence: 6, track_id: "foundational-ai", description: "" },
    // Phase 3 — Artificial Neural Networks
    Kata { id: "single-neuron", title: "Single neuron", phase: 3, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "multi-layer-perceptron", title: "Multi-layer perceptron", phase: 3, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "activation-functions", title: "Activation functions", phase: 3, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "vanishing-gradients", title: "Vanishing gradients", phase: 3, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "dead-neurons", title: "Dead neurons", phase: 3, sequence: 5, track_id: "foundational-ai", description: "" },
    // Phase 4 — Representation Learning
    Kata { id: "handcrafted-vs-learned-features", title: "Handcrafted vs learned features", phase: 4, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "pca-intuition", title: "PCA intuition", phase: 4, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "embeddings", title: "Embeddings", phase: 4, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "visualizing-hidden-layers", title: "Visualizing hidden layers", phase: 4, sequence: 4, track_id: "foundational-ai", description: "" },
    // Phase 5 — CNN
    Kata { id: "why-dense-networks-fail-on-images", title: "Why dense networks fail on images", phase: 5, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "convolution-intuition", title: "Convolution intuition", phase: 5, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "filters-as-pattern-detectors", title: "Filters as pattern detectors", phase: 5, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "feature-maps", title: "Feature maps", phase: 5, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "pooling-effects", title: "Pooling effects", phase: 5, sequence: 5, track_id: "foundational-ai", description: "" },
    // Phase 6 — Sequence Models
    Kata { id: "n-grams", title: "N-grams", phase: 6, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "rnn-intuition", title: "RNN intuition", phase: 6, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "lstm-limitations", title: "LSTM limitations", phase: 6, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "why-recurrence-struggles-at-scale", title: "Why recurrence struggles at scale", phase: 6, sequence: 4, track_id: "foundational-ai", description: "" },
    // Phase 7 — Attention & Transformers
    Kata { id: "attention-as-weighted-averaging", title: "Attention as weighted averaging", phase: 7, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "self-attention-visualization", title: "Self-attention visualization", phase: 7, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "positional-encoding", title: "Positional encoding", phase: 7, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "why-transformers-scale", title: "Why transformers scale", phase: 7, sequence: 4, track_id: "foundational-ai", description: "" },
    // Phase 8 — LLMs
    Kata { id: "tokenization", title: "Tokenization", phase: 8, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "next-token-prediction", title: "Next-token prediction", phase: 8, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "sampling-strategies", title: "Sampling strategies", phase: 8, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "temperature-top-k-top-p", title: "Temperature, top-k, top-p", phase: 8, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "emergence-through-scale", title: "Emergence through scale", phase: 8, sequence: 5, track_id: "foundational-ai", description: "" },
    // Phase 9 — Reasoning Models
    Kata { id: "chain-of-thought-as-latent-variables", title: "Chain-of-thought as latent variables", phase: 9, sequence: 1, track_id: "foundational-ai", description: "" },
    Kata { id: "scratchpads", title: "Scratchpads", phase: 9, sequence: 2, track_id: "foundational-ai", description: "" },
    Kata { id: "tool-usage", title: "Tool usage", phase: 9, sequence: 3, track_id: "foundational-ai", description: "" },
    Kata { id: "planning-vs-prediction", title: "Planning vs prediction", phase: 9, sequence: 4, track_id: "foundational-ai", description: "" },
    Kata { id: "failure-modes-of-llm-reasoning", title: "Failure modes of LLM reasoning", phase: 9, sequence: 5, track_id: "foundational-ai", description: "" },
];

pub static TRADITIONAL_ML_KATAS: &[Kata] = &[
    // Phase 0 — What is AI?
    Kata { id: "what-is-ai", title: "What is AI?", phase: 0, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "rule-based-systems", title: "Rule-based systems", phase: 0, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "search-algorithms", title: "Search algorithms", phase: 0, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "heuristics-and-cost", title: "Heuristics and cost", phase: 0, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "knowledge-representation", title: "Knowledge representation", phase: 0, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    // Phase 1 — Data Wrangling
    Kata { id: "tabular-data-basics", title: "Tabular data basics", phase: 1, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "missing-values", title: "Missing values", phase: 1, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "outlier-detection", title: "Outlier detection", phase: 1, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "encoding-categorical-variables", title: "Encoding categorical variables", phase: 1, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "feature-scaling", title: "Feature scaling", phase: 1, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "train-test-split", title: "Train-test split", phase: 1, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "exploratory-data-analysis", title: "Exploratory data analysis", phase: 1, sequence: 7, track_id: "traditional-ai-ml", description: "" },
    // Phase 2 — Supervised Learning: Regression
    Kata { id: "simple-linear-regression", title: "Simple linear regression", phase: 2, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "multiple-linear-regression", title: "Multiple linear regression", phase: 2, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "polynomial-regression", title: "Polynomial regression", phase: 2, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "regularization-ridge", title: "Regularization: Ridge", phase: 2, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "regularization-lasso", title: "Regularization: Lasso", phase: 2, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "elastic-net", title: "Elastic Net", phase: 2, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "regression-diagnostics", title: "Regression diagnostics", phase: 2, sequence: 7, track_id: "traditional-ai-ml", description: "" },
    // Phase 3 — Supervised Learning: Classification
    Kata { id: "k-nearest-neighbors", title: "K-nearest neighbors", phase: 3, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "logistic-regression", title: "Logistic regression", phase: 3, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "decision-trees", title: "Decision trees", phase: 3, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "support-vector-machines", title: "Support vector machines", phase: 3, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "naive-bayes", title: "Naive Bayes", phase: 3, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "multiclass-strategies", title: "Multiclass strategies", phase: 3, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "imbalanced-classes", title: "Imbalanced classes", phase: 3, sequence: 7, track_id: "traditional-ai-ml", description: "" },
    // Phase 4 — Model Evaluation & Selection
    Kata { id: "accuracy-and-its-limits", title: "Accuracy and its limits", phase: 4, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "precision-recall-f1", title: "Precision, recall, F1", phase: 4, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "roc-and-auc", title: "ROC and AUC", phase: 4, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "cross-validation", title: "Cross-validation", phase: 4, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "hyperparameter-tuning", title: "Hyperparameter tuning", phase: 4, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "bias-variance-tradeoff", title: "Bias-variance tradeoff", phase: 4, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    // Phase 5 — Unsupervised Learning
    Kata { id: "k-means-clustering", title: "K-means clustering", phase: 5, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "hierarchical-clustering", title: "Hierarchical clustering", phase: 5, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "dbscan", title: "DBSCAN", phase: 5, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "pca-for-dimensionality-reduction", title: "PCA for dimensionality reduction", phase: 5, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "t-sne-visualization", title: "t-SNE visualization", phase: 5, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "anomaly-detection", title: "Anomaly detection", phase: 5, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    // Phase 6 — Ensemble Methods
    Kata { id: "bagging-intuition", title: "Bagging intuition", phase: 6, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "random-forests", title: "Random forests", phase: 6, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "boosting-intuition", title: "Boosting intuition", phase: 6, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "adaboost", title: "AdaBoost", phase: 6, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "gradient-boosting", title: "Gradient boosting", phase: 6, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "xgboost", title: "XGBoost", phase: 6, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "stacking", title: "Stacking", phase: 6, sequence: 7, track_id: "traditional-ai-ml", description: "" },
    // Phase 7 — Feature Engineering & Pipelines
    Kata { id: "feature-creation", title: "Feature creation", phase: 7, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "feature-selection", title: "Feature selection", phase: 7, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "text-features", title: "Text features", phase: 7, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "datetime-features", title: "Datetime features", phase: 7, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "sklearn-pipelines", title: "ML pipelines", phase: 7, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "custom-transformers", title: "Custom transformers", phase: 7, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    // Phase 8 — Time Series & Sequential Data
    Kata { id: "time-series-basics", title: "Time series basics", phase: 8, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "autocorrelation", title: "Autocorrelation", phase: 8, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "arima", title: "ARIMA", phase: 8, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "exponential-smoothing", title: "Exponential smoothing", phase: 8, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "forecasting-evaluation", title: "Forecasting evaluation", phase: 8, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "sequence-classification", title: "Sequence classification", phase: 8, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    // Phase 9 — Reinforcement Learning
    Kata { id: "markov-decision-processes", title: "Markov decision processes", phase: 9, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "value-functions", title: "Value functions", phase: 9, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "policy-iteration", title: "Policy iteration", phase: 9, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "q-learning", title: "Q-learning", phase: 9, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "sarsa", title: "SARSA", phase: 9, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "multi-armed-bandits", title: "Multi-armed bandits", phase: 9, sequence: 6, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "environment-design", title: "Environment design", phase: 9, sequence: 7, track_id: "traditional-ai-ml", description: "" },
    // Phase 10 — Probabilistic & Bayesian Methods
    Kata { id: "bayesian-thinking", title: "Bayesian thinking", phase: 10, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "maximum-likelihood-estimation", title: "Maximum likelihood estimation", phase: 10, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "map-estimation", title: "MAP estimation", phase: 10, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "gaussian-mixture-models", title: "Gaussian mixture models", phase: 10, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "bayesian-optimization", title: "Bayesian optimization", phase: 10, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    // Phase 11 — Productionizing ML
    Kata { id: "model-serialization", title: "Model serialization", phase: 11, sequence: 1, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "data-drift-detection", title: "Data drift detection", phase: 11, sequence: 2, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "experiment-tracking", title: "Experiment tracking", phase: 11, sequence: 3, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "ab-testing-for-models", title: "A/B testing for models", phase: 11, sequence: 4, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "interpretability", title: "Interpretability", phase: 11, sequence: 5, track_id: "traditional-ai-ml", description: "" },
    Kata { id: "responsible-ai", title: "Responsible AI", phase: 11, sequence: 6, track_id: "traditional-ai-ml", description: "" },
];

pub static FOUNDATIONAL_PHASES: &[(&str, &str)] = &[
    ("0", "Foundations (Before ML)"),
    ("1", "What Does It Mean to Learn?"),
    ("2", "Optimization"),
    ("3", "Artificial Neural Networks (ANN)"),
    ("4", "Representation Learning"),
    ("5", "Convolutional Neural Networks (CNN)"),
    ("6", "Sequence Models"),
    ("7", "Attention & Transformers"),
    ("8", "Large Language Models (LLMs)"),
    ("9", "Reasoning Models"),
];

pub static TRADITIONAL_PHASES: &[(&str, &str)] = &[
    ("0", "What is AI?"),
    ("1", "Data Wrangling"),
    ("2", "Supervised Learning: Regression"),
    ("3", "Supervised Learning: Classification"),
    ("4", "Model Evaluation & Selection"),
    ("5", "Unsupervised Learning"),
    ("6", "Ensemble Methods"),
    ("7", "Feature Engineering & Pipelines"),
    ("8", "Time Series & Sequential Data"),
    ("9", "Reinforcement Learning"),
    ("10", "Probabilistic & Bayesian Methods"),
    ("11", "Productionizing ML"),
];

pub fn get_track(track_id: &str) -> Option<TrackRegistry> {
    match track_id {
        "foundational-ai" => Some(TrackRegistry {
            katas: FOUNDATIONAL_KATAS,
            phases: FOUNDATIONAL_PHASES,
        }),
        "traditional-ai-ml" => Some(TrackRegistry {
            katas: TRADITIONAL_ML_KATAS,
            phases: TRADITIONAL_PHASES,
        }),
        _ => None,
    }
}
