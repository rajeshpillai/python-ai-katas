# Python AI Katas — Todo

---

## Project Setup

- [x] Create CLAUDE.MD
- [x] Create .gitignore
- [x] Create todo.md
- [x] Set up FastAPI backend project structure
- [x] Set up SolidJS frontend project structure
- [x] Enforce lowercase-hyphenated file and folder names
- [x] Create kata markdown content files (all 50 katas)

---

## Landing Page

- [x] Landing page with two track cards
- [x] Foundational AI card (active, links to foundational track)
- [x] Traditional AI/ML card (coming soon / disabled until foundational is complete)
- [x] Routing from landing page into selected track

---

## Kata Structure (Applies to Every Kata)

- [x] Concept & intuition section (problem, why naïve approaches fail, mental models, visuals)
- [x] Interactive experiment (sliders for learning rate, epochs, layers, batch size)
- [x] Live plots (loss, accuracy, gradients) with incremental visual feedback
- [x] Live code editor (editable Python code, working example, step-by-step execution)
- [x] Reset to original functionality
- [ ] Save versions for logged-in users

---

# Foundational AI Track

---

## PHASE 0 — Foundations (Before ML)

- [x] Kata: What is data?
- [x] Kata: What is a feature?
- [x] Kata: What is noise?
- [x] Kata: What is signal?
- [x] Kata: Visualizing distributions

---

## PHASE 1 — What Does It Mean to Learn?

- [x] Kata: Constant predictor
- [x] Kata: Mean predictor
- [x] Kata: Distance-based prediction
- [x] Kata: Linear regression (from scratch, no PyTorch)
- [x] Teach: Loss functions
- [x] Teach: Error surfaces
- [x] Teach: Overfitting vs underfitting

---

## PHASE 2 — Optimization

- [x] Kata: Manual gradient descent
- [x] Kata: Learning rate experiments
- [x] Kata: Convergence vs divergence
- [x] Kata: Visualizing loss curves
- [x] Introduce: PyTorch tensors
- [x] Introduce: Autograd

---

## PHASE 3 — Artificial Neural Networks (ANN)

- [x] Kata: Single neuron
- [x] Kata: Multi-layer perceptron
- [x] Kata: Activation functions
- [x] Kata: Vanishing gradients
- [x] Kata: Dead neurons

---

## PHASE 4 — Representation Learning

- [x] Kata: Handcrafted vs learned features
- [x] Kata: PCA intuition
- [x] Kata: Embeddings
- [x] Kata: Visualizing hidden layers

---

## PHASE 5 — Convolutional Neural Networks (CNN)

- [x] Teach: Why dense networks fail on images
- [x] Teach: Convolution intuition
- [x] Teach: Filters as pattern detectors
- [x] Teach: Feature maps
- [x] Teach: Pooling effects
- [x] Dataset: MNIST integration
- [x] Dataset: CIFAR-10 integration

---

## PHASE 6 — Sequence Models

- [x] Teach: N-grams
- [x] Teach: RNN intuition
- [x] Teach: LSTM limitations
- [x] Teach: Why recurrence struggles at scale

---

## PHASE 7 — Attention & Transformers

- [x] Teach: Attention as weighted averaging
- [x] Teach: Self-attention visualization
- [x] Teach: Positional encoding
- [x] Teach: Why transformers scale

---

## PHASE 8 — Large Language Models (LLMs)

- [x] Teach: Tokenization
- [x] Teach: Next-token prediction
- [x] Teach: Sampling strategies
- [x] Teach: Temperature, top-k, top-p
- [x] Teach: Emergence through scale

---

## PHASE 9 — Reasoning Models

- [x] Teach: Chain-of-thought as latent variables
- [x] Teach: Scratchpads
- [x] Teach: Tool usage
- [x] Teach: Planning vs prediction
- [x] Teach: Failure modes of LLM reasoning

---

## Frontend (SolidJS)

- [x] Stream training progress (SSE streaming endpoint + Batch/Stream toggle)
- [x] Visualize tensors and matrices (kata_tensor heatmap rendering)
- [x] Show intermediate states (streaming mode shows output line-by-line)
- [x] Code and output side-by-side layout
- [x] Resizable code and output windows with maximise option
- [x] Sidebar with sequence numbers and collapsible burger menu
- [x] Dark/light theme toggle with persistent preference

---

## Backend (FastAPI)

- [ ] Dataset loading
- [x] Secure code execution (sandboxed, not in main process)
- [x] CPU limits enforcement
- [x] Memory limits enforcement
- [x] Execution timeout enforcement
- [ ] Disable filesystem and network access in sandbox
- [x] Return metrics, visual outputs, and educational error messages
- [x] Progress tracking
- [x] Code versioning
- [ ] Simple authentication
- [ ] Anonymous users: temporary session, no persistence
- [ ] Logged-in users: saved progress, restored experiments, version history

---

# Traditional AI/ML Track (Deferred)

> Will be designed and built after the Foundational AI track is complete.

- [ ] Define curriculum (classical AI, supervised, unsupervised, reinforcement learning, etc.)
- [ ] Create kata sequence for Traditional AI/ML track
