# Neural Akinator: Dynamic Constraint & Decision Tree Engine

An enterprise-grade, high-performance **Client-Server Hybrid AI Deduction Engine** designed to solve the classic 20-questions game with mathematical precision. Unlike naive LLM prompts that suffer from hallucinations and inconsistent logic, Neural Akinator combines **deterministic constraint logic**, **Shannon Entropy (Information Gain)**, and **LLM belief-state reasoning** into a unified inference architecture.

---

## 🚀 Architectural Overview & Engineering Philosophy

Modern LLMs struggle with multi-turn deductive reasoning when maintaining state over long conversation horizons. Neural Akinator solves this by decoupling **logical deduction** from **natural language generation**:

1. **Dynamic Constraint Ledger:** Maintains a structured belief state across 225+ candidate entities. Every user response (`Yes`, `No`, `I Don't Know`) applies an immutable mathematical filter to the candidate pool.
2. **Information Gain Optimization (Shannon Entropy):** Before asking a question, the engine evaluates the candidate space and calculates the exact expected information gain (entropy reduction) for all possible traits, dynamically selecting the question that splits the remaining search space most efficiently (closest to a 50/50 split).
3. **Hybrid Client-Server Validation:** Combines lightning-fast local tree pruning with server-side LLM validation and Server-Sent Events (SSE) streaming, ensuring zero latency spikes while eliminating logical contradictions and hallucinations.
4. **Self-Learning Architecture:** Features an automated ingestion pipeline where new entities taught by users are analyzed, decomposed into boolean trait vectors, and merged into the canonical constraint ledger.

---

## 🛠️ Tech Stack & System Components

- **Backend & Inference:** Python 3.10+, FastAPI, Pydantic v2, Uvicorn (Async SSE Streaming)
- **Mathematical Engine:** Custom Shannon Entropy & Information Gain algorithm (Scipy / Numpy logic)
- **LLM Integration:** OpenAI-compatible API bridge (Z.ai / GLM-5 / Groq / OpenAI) for natural language formatting and edge-case resolution
- **Frontend Integration:** Next.js 16, React 19, TypeScript, Tailwind CSS, Framer Motion

---

## 📦 Repository Structure

```text
├── app/
│   ├── main.py              # FastAPI Application & SSE Streaming Routes
│   ├── engine.py            # Core Deduction & Information Gain Mathematical Engine
│   ├── ledger.py            # Dynamic 225-Animal Constraint Database & Trait Matrix
│   └── models.py            # Pydantic Schemas & State Validators
├── simulate_engine_logic.py # Offline mathematical verification & entropy simulation suite
├── simulate_fixed_targets.py# Benchmark tests against fixed target paths
├── test_game.py             # Automated end-to-end multi-turn regression tests
└── requirements.txt         # Production dependencies
```

---

## ⚡ Quick Start (Local Development)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Mohammed-Darrige/ai-akinator.git
cd ai-akinator
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file and set your LLM provider credentials:
```bash
cp .env.example .env
```
```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.z.ai/api/coding/paas/v4
LLM_MODEL_NAME=glm-5-turbo
```

### 3. Launch Inference Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API endpoints and Swagger UI documentation will be available at `http://localhost:8000/docs`.

---

## 🧪 Verification & Simulation Suite

To verify the mathematical correctness of the Shannon Entropy pruning without consuming LLM API tokens, run the included simulation suite:
```bash
python simulate_engine_logic.py
python test_game.py
```

---
*Engineered by **Mohammed Darrige** — Fırat University AI & Data Engineering.*
