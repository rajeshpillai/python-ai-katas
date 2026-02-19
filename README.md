# Python AI Katas

Learn AI as an engineering discipline, not magic. Build intuition through hands-on experimentation.

## Tracks

### Foundational AI (Active)

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

### Traditional AI/ML (Coming Soon)

Classical AI, supervised/unsupervised learning, and traditional ML algorithms.

## Tech Stack

- **Frontend**: SolidJS + Vite + TypeScript
- **Backend**: FastAPI + Python
- **ML**: PyTorch + NumPy + Matplotlib

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Backend runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

## Project Structure

```
python-ai-katas/
├── backend/                  # FastAPI server
│   ├── main.py               # App entry point
│   ├── requirements.txt
│   └── app/
│       ├── config.py          # Settings
│       ├── routes/            # API endpoints
│       ├── services/          # Business logic
│       ├── models/            # Pydantic schemas
│       └── middleware/        # CORS etc.
├── frontend/                 # SolidJS client
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── app.tsx            # Root + routing
│       ├── context/           # Theme provider
│       ├── components/        # UI components
│       ├── pages/             # Route pages
│       └── lib/               # API client, constants
└── content/                  # Kata content (markdown)
    └── foundational-ai/
        ├── phase-0/
        ├── phase-1/
        └── ...
```

## Kata Structure

Each kata contains:

1. **Concept & Intuition** — what problem we're solving, why naive approaches fail, mental models, visuals
2. **Interactive Experiment** — sliders for hyperparameters, live plots for loss/accuracy/gradients
3. **Live Code** — editable Python code with run, reset, and version save

## Conventions

- File and folder names are **lowercase-hyphenated**
- Kata content lives in markdown files under `content/`
- Frontend proxies `/api` requests to the backend in development
