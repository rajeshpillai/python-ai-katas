
Every model — linear regression, CNN, Transformer, LLM — must be explained through this lens.

---

## Application Structure

The app has a **landing page** with two track cards:

### 1. Foundational AI (Active)
Covers Phases 0–9 below. This is the current focus and must be built first.

### 2. Traditional AI/ML (Deferred)
Covers classical AI, supervised learning, unsupervised learning, reinforcement learning, and traditional ML algorithms (decision trees, SVMs, k-means, etc.). **This track will be designed and built after the Foundational AI track is complete.**

The landing page routes into the selected track. Each track has its own sidebar, kata sequence, and progress tracking.

---

## Foundational AI — Learning Sequence (MANDATORY ORDER)

You must follow this progression **even if it contradicts popular courses**.

---

### PHASE 0 — Foundations (Before ML)

**Goal:** Build data intuition

Teach:
- What is data?
- What is a feature?
- What is noise?
- What is signal?
- Visualizing distributions

No models yet.

---

### PHASE 1 — What Does It Mean to Learn?

**Goal:** Learning = minimizing error

Katas:
- Constant predictor
- Mean predictor
- Distance-based prediction
- Linear regression (from scratch, no PyTorch)

Focus:
- Loss functions
- Error surfaces
- Overfitting vs underfitting

---

### PHASE 2 — Optimization

**Goal:** How models improve themselves

Katas:
- Manual gradient descent
- Learning rate experiments
- Convergence vs divergence
- Visualizing loss curves

Only **after this**, introduce:
- PyTorch tensors
- Autograd

---

### PHASE 3 — Artificial Neural Networks (ANN)

**Goal:** Models as composed functions

Katas:
- Single neuron
- Multi-layer perceptron
- Activation functions
- Vanishing gradients
- Dead neurons

No CNNs yet.

---

### PHASE 4 — Representation Learning

**Goal:** Features can be learned

Katas:
- Handcrafted vs learned features
- PCA intuition
- Embeddings
- Visualizing hidden layers

---

### PHASE 5 — Convolutional Neural Networks (CNN)

**Goal:** Structure matters

Teach:
- Why dense networks fail on images
- Convolution intuition
- Filters as pattern detectors
- Feature maps
- Pooling effects

Datasets:
- MNIST → CIFAR-10

---

### PHASE 6 — Sequence Models

**Goal:** Order matters

Teach:
- N-grams
- RNN intuition
- LSTM limitations
- Why recurrence struggles at scale

---

### PHASE 7 — Attention & Transformers

**Goal:** Learn what to focus on

Teach:
- Attention as weighted averaging
- Self-attention visualization
- Positional encoding
- Why transformers scale

---

### PHASE 8 — Large Language Models (LLMs)

**Goal:** Scale changes behavior

Teach:
- Tokenization
- Next-token prediction
- Sampling strategies
- Temperature, top-k, top-p
- Emergence through scale

No hype. Only mechanics.

---

### PHASE 9 — Reasoning Models

**Goal:** Reasoning is structured prediction

Teach:
- Chain-of-thought as latent variables
- Scratchpads
- Tool usage
- Planning vs prediction
- Failure modes of LLM reasoning

---

## Kata Structure (MANDATORY)

Each kata must contain:

### 1. Concept & Intuition
- What problem are we solving?
- Why naïve approaches fail
- Mental models
- Visual explanations

### 2. Interactive Experiment
- Sliders for:
  - learning rate
  - epochs
  - layers
  - batch size
- Live plots:
  - loss
  - accuracy
  - gradients
- Visual feedback must update incrementally

### 3. Live Code
- Editable PyTorch code
- Working example
- Step-by-step execution (where applicable)
- Reset to original always available
- Save versions for logged-in users

---

## Code Execution Rules (STRICT)

You must:
- Never execute user code in the main process
- Enforce:
  - CPU limits
  - Memory limits
  - Execution timeout
- Disable filesystem and network access
- Return:
  - Metrics
  - Visual outputs
  - Educational error messages

Errors must explain *why* something failed.

---
## Coding Conventions
- File and folder names must be lowercase-hyphenated
- The code and output window must be resizable and should have maximise option 
- The sidebar for katas should have sequence numbers and should be collapsible (burge menu)

---


## Frontend Responsibilities (SolidJS)

The frontend acts as a **reactive ML instrument panel**.

It must:
- Stream training progress
- Visualize tensors and matrices
- Show intermediate states
- Keep code and output side-by-side

It must NOT:
- Hide training steps
- Auto-fix errors
- Abstract away learning dynamics

---

## Backend Responsibilities (FastAPI)

Backend handles:
- Dataset loading
- Secure code execution
- Progress tracking
- Code versioning
- Simple authentication

Anonymous users:
- Temporary session
- No persistence

Logged-in users:
- Saved progress
- Restored experiments
- Version history

---

## Teaching Rules (VERY IMPORTANT)

You must:

- Explain *why* something works
- Show failure cases
- Compare alternatives
- Encourage experimentation
- Build intuition before math

You must NOT:

- Jump to best practices without context
- Hide complexity
- Use unexplained abstractions
- Pretend models “understand”

---

## Output Expectations From You (the AI)

When responding, you must:

- Proceed step-by-step
- Avoid large code dumps
- Explain design decisions
- Prefer incremental builds
- Use diagrams and visual metaphors

Favor:
- Intuition
- Debuggability
- Transparency

---

## Success Criteria

This system is successful if learners can:

- Debug a failing model
- Explain why CNNs work
- Understand why LLMs scale
- Recognize limits of reasoning models
- Build mental models transferable across architectures

---

## Final Instruction

Teach AI as an **engineering discipline**, not magic.

Create/update todo.md in the root and checkmark all items completed.

When in doubt:
- Choose intuition over abstraction
- Choose transparency over convenience
- Choose understanding over speed

Proceed deliberately.  
Explain everything.  
Never assume.
