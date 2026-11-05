JST APEX NEXUS V20.3

JUMP STREET TRADING — Hybrid Quant Research Engine

XAU/USD | M5 Primary | Multi-Timeframe Analysis

Research, signal generation, quantitative analysis, and paper-trading environment. No broker execution and no automatic order placement.

Repository Files

- "app.py" — Streamlit dashboard and system interface.
- "jst_engine.py" — JST APEX NEXUS V20.3 research and signal engine.
- "requirements.txt" — Core Python dependencies.

Core Capabilities

1. Signal Engine

- Multi-timeframe market analysis.
- H1 trend context.
- M15 structure and supply/demand zones.
- M5 primary signal analysis.
- M1 entry timing.
- Signal fusion and execution-readiness assessment.

2. Quantitative Research

- Linear algebra and PCA.
- Statistical analysis and optimization.
- Bayesian probability estimation.
- Time-series analysis and econometrics.
- Volatility and regime analysis.
- Monte Carlo simulation.
- Risk measurement and tail-risk diagnostics.
- Physics-inspired mathematical diagnostics.

3. Financial Mathematics

- Black–Scholes theoretical option pricing.
- Option Greeks: Delta, Gamma, Vega, Theta, and Rho.

4. Model Validation and Research

- Chronological walk-forward validation baseline.
- Rolling hedge-ratio statistical-arbitrage research.
- Machine-learning research ensemble.
- LSTM trade-context research.

These modules are research tools. Their presence does not imply proven predictive performance or a validated trading edge.

Dashboard

The Streamlit dashboard provides:

- Signal and Scan output.
- Paper Trading Journal.
- Open and closed paper positions.
- Performance statistics.
- Runtime capability information.
- Research module information.
- Manual scan and session-based auto scan.

Installation

Install the dependencies listed in "requirements.txt".

Run locally:

streamlit run app.py

Streamlit Cloud Secrets

Configure the API key in:

Streamlit Cloud → App Settings → Secrets

TWELVE_DATA_API_KEY = "YOUR_TWELVE_DATA_API_KEY"

Do not publish API keys in source code or commit them to GitHub.

Important Limitations

- Market data is sourced through Twelve Data.
- No direct COMEX order-book or footprint feed is connected.
- Auto scan operates within the active Streamlit session; it is not a guaranteed 24/7 background worker.
- Streamlit Cloud local filesystem storage is not guaranteed to persist through rebuilds or restarts.
- Optional machine-learning libraries may be unavailable if they are not installed.
- Walk-forward and statistical-arbitrage research require further robustness and out-of-sample testing.
- Theoretical option pricing is not a substitute for live options-market data.
- Estimated transaction costs must be calibrated against actual spread, commission, and slippage.

Risk and Execution Policy

PAPER / RESEARCH ONLY

- No broker connection.
- No automatic order execution.
- Research outputs are not guaranteed trading signals.
- Validate all models, data quality, transaction costs, and risk assumptions before considering live deployment.

---

JUMP STREET TRADING — JST APEX NEXUS V20.2
