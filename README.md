# AI Katas

Learn AI as an engineering discipline, not magic. Build intuition through hands-on experimentation.

## Tracks

### Foundational AI

10 phases covering the full journey from raw data to reasoning models:

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

### Traditional AI/ML

12 phases covering classical AI, supervised/unsupervised learning, and production ML:

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

## Tech Stack

- **Frontend**: SolidJS + Vite + TypeScript (shared across all backends)
- **Python Backend**: FastAPI + PyTorch + NumPy + Matplotlib
- **Rust Backend**: Axum + Tokio (all algorithms implemented from scratch, no ML crates)

## Getting Started

### Prerequisites

- Node.js 18+
- **For Python backend**: Python 3.12+
- **For Rust backend**: Rust toolchain (`rustc`, `cargo`)

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

### Python Backend

```bash
cd backend/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

Python backend runs at http://localhost:8000

### Rust Backend

```bash
cd backend/rust
cargo build
cargo run
```

Rust backend runs at http://localhost:8001

### Choosing a Backend

The frontend connects to whichever backend is running. Both backends expose the same API shape, so the frontend works identically with either one.

| | Python Backend | Rust Backend |
|---|---|---|
| **Port** | 8000 | 8001 |
| **Kata code** | Python (PyTorch, NumPy) | Rust (from scratch, no crates) |
| **Sandbox** | `python` subprocess | `rustc` compile + run |
| **Content** | 125 katas | 125 katas |

To switch backends, update the API base URL in the frontend's [api-client.ts](frontend/src/lib/api-client.ts) to point at the desired port.

## Project Structure

```
ai-katas/
├── backend/
│   ├── python/                  # FastAPI server (port 8000)
│   │   ├── main.py              # App entry point
│   │   ├── requirements.txt
│   │   ├── app/                 # Routes, models, services
│   │   └── content/             # Python kata content (markdown)
│   │       ├── foundational-ai/
│   │       └── traditional-ai-ml/
│   └── rust/                    # Axum server (port 8001)
│       ├── Cargo.toml
│       ├── src/                 # Routes, models, services, sandbox
│       └── content/             # Rust kata content (markdown)
│           ├── foundational-ai/
│           └── traditional-ai-ml/
├── frontend/                    # SolidJS client (shared)
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── app.tsx              # Root + routing
│       ├── context/             # Theme provider
│       ├── components/          # UI components
│       ├── pages/               # Route pages
│       └── lib/                 # API client, constants
└── scripts/                     # Utility scripts
```

## Kata Structure

Each kata contains:

1. **Concept & Intuition** — what problem we're solving, why naive approaches fail, mental models, visuals
2. **Hands-on Exploration** — guided experiments to build understanding
3. **Live Code** — editable code with run, reset, and streaming output

## Environment Variables

All backend settings use the `KATAS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `KATAS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `KATAS_SANDBOX_TIMEOUT_SECONDS` | `30` | Max execution time per run |
| `KATAS_PORT` | `8000` (Python) / `8001` (Rust) | Server port |

## Conventions

- File and folder names are **lowercase-hyphenated**
- Kata content lives in markdown files under `backend/*/content/`
- Frontend proxies `/api` requests to the backend in development
