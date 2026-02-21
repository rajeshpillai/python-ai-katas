export const THEME_STORAGE_KEY = "theme";
export const API_BASE = "/api";

export const LANGUAGES = [
  {
    id: "python",
    name: "Python",
    description:
      "NumPy, PyTorch, scikit-learn, and matplotlib — the most popular AI/ML ecosystem.",
  },
  {
    id: "rust",
    name: "Rust",
    description:
      "High-performance, safe AI/ML with ndarray, tch-rs, and zero-cost abstractions.",
  },
] as const;

export const TRACKS = {
  FOUNDATIONAL_AI: {
    id: "foundational-ai",
    name: "Foundational AI",
    description:
      "Build deep intuition for AI from first principles — from data basics through neural networks to LLMs and reasoning models.",
  },
  TRADITIONAL_AI_ML: {
    id: "traditional-ai-ml",
    name: "Traditional AI/ML",
    description:
      "Classical AI, supervised and unsupervised learning, reinforcement learning, and traditional ML algorithms.",
  },
} as const;

export const PHASE_NAMES: Record<string, Record<number, string>> = {
  "foundational-ai": {
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
  },
  "traditional-ai-ml": {
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
  },
};
