# 🧠 AI Katas

> *Learn AI as an engineering discipline, not magic. Build intuition through hands-on experimentation — at your own pace, in your own order.*

<p align="center">
  <img src="https://img.shields.io/badge/tracks-2-blue" alt="2 tracks">
  <img src="https://img.shields.io/badge/katas-125-green" alt="125 katas">
  <img src="https://img.shields.io/badge/path-non--linear-purple" alt="Non-linear">
  <img src="https://img.shields.io/badge/backends-Python%20%7C%20Rust-orange" alt="Python or Rust">
  <img src="https://img.shields.io/badge/frontend-SolidJS-yellow" alt="SolidJS">
  <img src="https://img.shields.io/badge/purpose-learning-lightgrey" alt="Learning">
</p>

---

## 📖 What Is This?

A live, interactive playground of **125 small katas** that teach AI from the ground up. Each kata is a focused, runnable experiment — read the concept, edit the code, hit run, see what happens. No black boxes. No hand-waving.

You don't watch lectures. You **break things on purpose** and learn from what falls apart.

---

## 🧭 Two Tracks, One Goal

We give you **two complementary lenses** on AI. Both are first-class. Pick one, run both in parallel, or hop between them — whatever maps onto how *you* learn.

| 🧬 Foundational AI | 🛠️ Traditional AI/ML |
|---|---|
| The journey from raw data → reasoning models | Classical AI, statistical learning, production ML |
| 10 phases · ~50 katas | 12 phases · ~75 katas |
| Builds toward Transformers and LLMs | Decision trees, SVMs, ensembles, time series, RL |
| For: people curious how ChatGPT *actually* works | For: people shipping models in production |

---

## 🔀 Pick Your Own Path — The Learning is Not Tied to an Order

This is the most important thing to know about this project:

> **🚦 There is no "correct" sequence. Tracks, phases, and katas are independent units you can pick, swap, skip, or revisit at any time.**

Every kata is **self-contained**: it states what it teaches, what you need to know first (if anything), and what you'll walk away with. You're not locked into a curriculum. Treat the tracks like a buffet, not a textbook.

| 🎒 If you want to... | 🛣️ Try this path |
|---|---|
| Understand modern LLMs end-to-end | Foundational AI · Phases 0 → 9, sequentially |
| Ship a regression / classification model | Traditional · Phases 1 → 4, then 7 |
| Master *just* attention and Transformers | Foundational · Phase 7 (jump straight in) |
| Learn the math intuition behind ML | Traditional · Phase 0 + Foundational · Phases 1–2 |
| Sample broadly before committing | One kata from each phase, then double down |
| Already know the basics — skip ahead | Start anywhere. The kata will tell you if you're missing context |

> 💡 **Order is a suggestion, not a constraint.** Your sidebar in the app lets you navigate freely between any kata in any phase in either track.

---

## 🗺️ Tracks at a Glance

### 🧬 Foundational AI

The full arc from "what is data?" to "how does a reasoning model think?"

| Phase | Topic | Katas |
|-------|-------|-------|
| 0 | Foundations (Before ML) | 5 |
| 1 | What Does It Mean to Learn? | 7 |
| 2 | Optimization | 6 |
| 3 | Artificial Neural Networks | 5 |
| 4 | Representation Learning | 4 |
| 5 | Convolutional Neural Networks | 5 |
| 6 | Sequence Models | 4 |
| 7 | Attention & Transformers | 4 |
| 8 | Large Language Models | 5 |
| 9 | Reasoning Models | 5 |

### 🛠️ Traditional AI/ML

Classical, statistical, and production-grade ML.

| Phase | Topic | Katas |
|-------|-------|-------|
| 0 | What is AI? | 5 |
| 1 | Data Wrangling | 7 |
| 2 | Supervised Learning: Regression | 7 |
| 3 | Supervised Learning: Classification | 7 |
| 4 | Model Evaluation & Selection | 6 |
| 5 | Unsupervised Learning | 6 |
| 6 | Ensemble Methods | 7 |
| 7 | Feature Engineering & Pipelines | 6 |
| 8 | Time Series & Sequential Data | 6 |
| 9 | Reinforcement Learning | 7 |
| 10 | Probabilistic & Bayesian Methods | 5 |
| 11 | Productionizing ML | 6 |

---

## 🧩 What's in a Kata?

Every kata follows the same shape — predictable so you can focus on the ideas, not the format.

| Step | Format | Purpose |
|---|---|---|
| 1️⃣ **Concept & Intuition** | Plain-English framing + visuals | What problem are we solving? Why do naive approaches fail? |
| 2️⃣ **Hands-on Exploration** | Guided experiments | Build an intuitive mental model before formalizing it |
| 3️⃣ **Live Code** | Editable, runnable, streaming output | Touch the levers. See what changes. |

---

## 🏗️ Tech Stack

| Layer | Stack |
|---|---|
| **Frontend** | SolidJS · Vite · TypeScript (one frontend, swappable backends) |
| **Python Backend** | FastAPI · PyTorch · NumPy · Matplotlib · scikit-learn · pandas |
| **Rust Backend** | Axum · Tokio (every algorithm written from scratch — no ML crates) |

The frontend speaks the same API to either backend. **Pick the language you want to learn in.**

---

## 🚀 Quick Start

> 💡 We use [**uv**](https://github.com/astral-sh/uv) for the Python backend — it's a 10-100× faster drop-in replacement for `pip` + `venv`. If you prefer plain `pip`, the section below tells you how.

### 0. Install prerequisites

```bash
# Node.js 18+      → https://nodejs.org
# uv (Python)      → curl -LsSf https://astral.sh/uv/install.sh | sh
# Rust (optional)  → https://rustup.rs
```

### 1. Frontend (always required)

```bash
cd frontend
npm install
npm run dev                                # http://localhost:3000
```

### 2a. Python backend (recommended for most learners)

```bash
cd backend/python
uv venv                                    # create .venv/
source .venv/bin/activate                  # Mac/Linux
# .venv\Scripts\activate                   # Windows
uv pip install -r requirements.txt         # fast, deterministic install
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Python backend → http://localhost:8000

> 🐢 **Prefer plain pip?** Swap `uv venv` for `python3 -m venv .venv` and `uv pip install` for `pip install`. Same result, slower.

### 2b. Rust backend (if you want algorithms from scratch)

```bash
cd backend/rust
cargo run                                  # http://localhost:8001
```

### 3. Pick a backend in the frontend

Both backends expose the same API. To switch which one the UI talks to, point the API base URL in [api-client.ts](frontend/src/lib/api-client.ts) at port `8000` (Python) or `8001` (Rust).

---

## 🌐 Deploy to GitHub Pages

The app ships with a fully static build mode that runs entirely in the browser — Python katas execute via [Pyodide](https://pyodide.org) (Python compiled to WebAssembly), so no server is needed. Rust katas are read-only on the static build (run them locally if you want execution).

### Prerequisites

- The Python backend venv is set up (the build script reads `TRACK_REGISTRY` from it). Run the Python backend Quick Start once — that's enough.
- A GitHub repository with Pages enabled (Settings → Pages).

### Build and publish

```bash
# 1. Dry run — produces frontend/dist/ and prints the tree
./scripts/publish-gh-pages.sh

# 2. Push to the gh-pages branch on origin
./scripts/publish-gh-pages.sh --push

# 3. Configure GitHub: Settings → Pages → Source: gh-pages branch / root
```

That's it — your live site appears at `https://<user>.github.io/<repo>/`.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GHPAGES_REMOTE` | `origin` | Git remote (name or URL) to push to |
| `GHPAGES_BRANCH` | `gh-pages` | Branch on the remote |
| `GHPAGES_BASE_PATH` | derived from `origin` repo name | Asset base path. Set to `/` for custom-domain deploys. |
| `GHPAGES_MESSAGE` | `Deploy static build <date>` | Custom commit message |

### Test the static build locally

```bash
./scripts/build-static.sh
python3 -m http.server -d frontend/dist 4173
# open http://localhost:4173
```

> 💡 **PyTorch on Pyodide** is experimental — torch katas (foundational phase 2 onward) lazy-load the wheel at first run. If a kata feels slow or fails, clone the repo and run it locally.

---

## 🎯 Skills You'll Gain

- ✅ Build mental models for *every* layer of an AI system, from raw data to reasoning
- ✅ Implement core algorithms by hand: gradient descent, attention, decision trees, k-means
- ✅ Read modern ML papers without panic — you'll recognize the components
- ✅ Diagnose training failures (loss spikes, dead neurons, overfitting) from intuition
- ✅ Compare classical and deep approaches and pick the right tool for a real problem
- ✅ Ship a model: feature pipelines, evaluation, deployment patterns

---

## 📁 Project Structure

```
python-ai-katas/
├── backend/
│   ├── python/                   # FastAPI server (port 8000)
│   │   ├── main.py
│   │   ├── requirements.txt      # uv pip install -r requirements.txt
│   │   ├── app/                  # routes, models, services, middleware
│   │   └── content/              # markdown kata files
│   │       ├── foundational-ai/
│   │       └── traditional-ai-ml/
│   └── rust/                     # Axum server (port 8001)
│       ├── Cargo.toml
│       ├── src/                  # routes, models, services, sandbox
│       └── content/
│           ├── foundational-ai/
│           └── traditional-ai-ml/
├── frontend/                     # SolidJS client (shared)
│   └── src/
│       ├── app.tsx               # root + routing
│       ├── pages/                # route pages
│       ├── components/           # UI
│       └── lib/                  # API client, constants
└── scripts/                      # utility scripts (OSS publish, etc.)
```

---

## ⚙️ Environment Variables

All backend settings use the `KATAS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `KATAS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `KATAS_SANDBOX_TIMEOUT_SECONDS` | `30` | Max execution time per kata run |
| `KATAS_PORT` | `8000` (Python) / `8001` (Rust) | Server port |

---

## 📚 Conventions

- File and folder names are **lowercase-hyphenated**
- Kata content lives in markdown under `backend/*/content/<track>/<phase>/`
- Frontend proxies `/api` requests to the backend in development
- Both backends serve the **same API shape** — keep parity if you contribute

---

<p align="center">
  <i>"Tell me and I forget. Teach me and I may remember. Involve me and I learn." — Xun Kuang</i>
</p>

<p align="center">
  <sub>⭐ Star if useful · 🐛 Issues & PRs welcome · 📖 Happy learning</sub>
</p>
