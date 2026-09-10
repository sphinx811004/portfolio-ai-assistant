# AI-Driven Portfolio Optimization & Risk Management Assistant

Project by **Lakshita Chandrakar (23115053)** and **Pratyu Dehariya (23115073)**
Department of Computer Science and Engineering, NIT Raipur

See `docs/` for full project documentation and architecture diagram.

## Structure

- `src/agent/` — central orchestrator that ties every module together
- `src/market_data/` — price/fundamental data fetching
- `src/features/` — technical indicators and feature pipeline
- `src/rag/` — retrieval-augmented generation over news/filings
- `src/fingpt_signals/` — FinGPT-based sentiment and event extraction
- `src/user_profile/` — risk tolerance, horizon, capital capture and encoding
- `src/optimization/baseline/` — Markowitz / Black-Litterman optimizer
- `src/optimization/rl_agent/` — PPO agent via FinRL
- `src/risk_checker/` — rule-based validation gate on proposed allocations
- `src/decision_engine/` — converts target weights into Buy/Sell/Rebalance actions
- `src/explanation/` — LLM-generated plain-language explanations
- `backtesting/` — walk-forward evaluation, performance metrics, benchmarks
- `api/` — FastAPI backend
- `app/` — Streamlit demo UI
- `notebooks/` — exploration and experiments
- `tests/` — unit tests
- `configs/` — YAML configs (risk limits, model/env settings)
- `data/` — raw, processed, and cached data (gitignored)
- `docs/` — project documentation and diagrams

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env            # fill in API keys
```

## Run

```bash
python scripts/run_pipeline.py      # end-to-end pipeline
uvicorn api.main:app --reload       # backend API
streamlit run app/streamlit_app.py  # demo UI
```
