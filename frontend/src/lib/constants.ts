export const THEME_STORAGE_KEY = "theme";
export const API_BASE = "/api";

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

export const PHASE_NAMES: Record<number, string> = {
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
};
