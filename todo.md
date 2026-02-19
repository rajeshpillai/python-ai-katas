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

- [ ] Concept & intuition section (problem, why naïve approaches fail, mental models, visuals)
- [ ] Interactive experiment (sliders for learning rate, epochs, layers, batch size)
- [ ] Live plots (loss, accuracy, gradients) with incremental visual feedback
- [ ] Live code editor (editable PyTorch code, working example, step-by-step execution)
- [ ] Reset to original functionality
- [ ] Save versions for logged-in users

---

# Foundational AI Track

---

## PHASE 0 — Foundations (Before ML)

- [ ] Kata: What is data?
- [ ] Kata: What is a feature?
- [ ] Kata: What is noise?
- [ ] Kata: What is signal?
- [ ] Kata: Visualizing distributions

---

## PHASE 1 — What Does It Mean to Learn?

- [ ] Kata: Constant predictor
- [ ] Kata: Mean predictor
- [ ] Kata: Distance-based prediction
- [ ] Kata: Linear regression (from scratch, no PyTorch)
- [ ] Teach: Loss functions
- [ ] Teach: Error surfaces
- [ ] Teach: Overfitting vs underfitting

---

## PHASE 2 — Optimization

- [ ] Kata: Manual gradient descent
- [ ] Kata: Learning rate experiments
- [ ] Kata: Convergence vs divergence
- [ ] Kata: Visualizing loss curves
- [ ] Introduce: PyTorch tensors
- [ ] Introduce: Autograd

---

## PHASE 3 — Artificial Neural Networks (ANN)

- [ ] Kata: Single neuron
- [ ] Kata: Multi-layer perceptron
- [ ] Kata: Activation functions
- [ ] Kata: Vanishing gradients
- [ ] Kata: Dead neurons

---

## PHASE 4 — Representation Learning

- [ ] Kata: Handcrafted vs learned features
- [ ] Kata: PCA intuition
- [ ] Kata: Embeddings
- [ ] Kata: Visualizing hidden layers

---

## PHASE 5 — Convolutional Neural Networks (CNN)

- [ ] Teach: Why dense networks fail on images
- [ ] Teach: Convolution intuition
- [ ] Teach: Filters as pattern detectors
- [ ] Teach: Feature maps
- [ ] Teach: Pooling effects
- [ ] Dataset: MNIST integration
- [ ] Dataset: CIFAR-10 integration

---

## PHASE 6 — Sequence Models

- [ ] Teach: N-grams
- [ ] Teach: RNN intuition
- [ ] Teach: LSTM limitations
- [ ] Teach: Why recurrence struggles at scale

---

## PHASE 7 — Attention & Transformers

- [ ] Teach: Attention as weighted averaging
- [ ] Teach: Self-attention visualization
- [ ] Teach: Positional encoding
- [ ] Teach: Why transformers scale

---

## PHASE 8 — Large Language Models (LLMs)

- [ ] Teach: Tokenization
- [ ] Teach: Next-token prediction
- [ ] Teach: Sampling strategies
- [ ] Teach: Temperature, top-k, top-p
- [ ] Teach: Emergence through scale

---

## PHASE 9 — Reasoning Models

- [ ] Teach: Chain-of-thought as latent variables
- [ ] Teach: Scratchpads
- [ ] Teach: Tool usage
- [ ] Teach: Planning vs prediction
- [ ] Teach: Failure modes of LLM reasoning

---

## Frontend (SolidJS)

- [ ] Stream training progress
- [ ] Visualize tensors and matrices
- [ ] Show intermediate states
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
- [ ] Return metrics, visual outputs, and educational error messages
- [x] Progress tracking
- [x] Code versioning
- [x] Simple authentication
- [ ] Anonymous users: temporary session, no persistence
- [ ] Logged-in users: saved progress, restored experiments, version history

---

# Traditional AI/ML Track (Deferred)

> Will be designed and built after the Foundational AI track is complete.

- [ ] Define curriculum (classical AI, supervised, unsupervised, reinforcement learning, etc.)
- [ ] Create kata sequence for Traditional AI/ML track
