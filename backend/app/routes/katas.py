from fastapi import APIRouter, HTTPException

from app.models.kata import Kata, KataList

router = APIRouter()

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

PHASE_NAMES = {
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


@router.get("/tracks/{track_id}/katas")
async def list_katas(track_id: str):
    if track_id == "foundational-ai":
        return {
            "katas": FOUNDATIONAL_KATAS,
            "phases": PHASE_NAMES,
        }
    if track_id == "traditional-ai-ml":
        raise HTTPException(status_code=404, detail="Traditional AI/ML track is coming soon")
    raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
