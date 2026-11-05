# JST APEX NEXUS V20.3
## JUMP STREET TRADING — Hybrid Quant Research Engine

**Instrument:** XAU/USD  
**Primary timeframe:** M5  
**Context:** D1 / H4 / H1 / M15  
**Entry timing:** M1  
**Data:** Twelve Data OHLC  
**Mode:** Signal generation + paper research only. No broker connection and no automatic order execution.

## Repository files
- `app.py` — Streamlit dashboard, scan controls, paper journal and runtime status.
- `jst_engine.py` — V20.3 signal, market-structure, supply/demand and quantitative research engine.
- `requirements.txt` — core runtime dependencies.

## V20.3 additions
- Multi-timeframe market structure diagnostics across D1, H4, H1, M15, M5 and M1.
- Confirmed swing-high / swing-low structure and BOS diagnostics.
- Estimated resistance/support trendline state: break, retest/testing, above/below line.
- OHLC-based RBR, DBR, RBD and DBD base-zone approximation.
- Zone distance, current interaction (inside/above/below), age and departure-strength proxy.
- Existing candidate/adaptive/fusion workflow, paper execution guardrails, journals, ML and quant research modules retained.

## Important interpretation limits
- Trendline calculations are linear estimates from the two latest confirmed pivots, not manually drawn discretionary trendlines.
- Supply/demand zones are OHLC pattern approximations, not verified institutional order blocks or actual resting orders.
- BOS and candle reaction are diagnostics; they do not guarantee continuation or reversal.
- Market-structure diagnostics are informational and do not automatically alter the signal score or paper execution decision.
- Twelve Data OHLC does not provide COMEX footprint, full order book depth or broker-specific execution data.
- ML outputs are advisory and should not be treated as calibrated probabilities or proven alpha.
- Risk, reward and estimated transaction costs are research assumptions; replace cost assumptions with measured spread, commission and slippage.
- Paper results are not live trading results and do not guarantee future performance.

## Installation
Install the core packages:
```bash
pip install -r requirements.txt
```

Optional research packages (install only if needed and supported by your hosting environment):
- `tensorflow` — LSTM research module; large dependency and may not be practical on Streamlit Community Cloud.
- `lightgbm`
- `catboost`
- `xgboost`

The engine handles unavailable optional packages by disabling the corresponding research module where supported.

## Twelve Data API key
For Streamlit Community Cloud, set this in **App settings → Secrets**:
```toml
TWELVE_DATA_API_KEY = "YOUR_TWELVE_DATA_API_KEY"
```
For Pydroid, set the `TWELVE_DATA_API_KEY` environment variable or paste the key into `PASTE_API_KEY_HERE` locally. Never commit a populated API key to a public repository. Rotate the key if it has been exposed.

## Running
- Streamlit Cloud: set the app entry point to `app.py`.
- Pydroid: run `jst_engine.py` for the console scanner.
- The Streamlit auto-scan runs only while the app session remains active; it is not a 24/7 background worker.
- Respect Twelve Data plan limits. Candle caching reduces repeat requests but does not guarantee avoidance of HTTP 429 responses.

## Journal continuity
The engine retains the existing V11.1-named paper journal/state filenames so existing paper history can be reused when those files are present in the same runtime filesystem. Back up CSV/JSON files before deploying. Streamlit Community Cloud local files are not guaranteed to survive rebuilds, restarts or sleeping sessions; use durable external storage for long-term records.

## Validation checklist
1. Confirm the dashboard displays V20.3 for both app and engine.
2. Confirm `jst_engine.py` is in the same repository directory as `app.py`.
3. Confirm the Twelve Data secret is configured.
4. Run one manual scan and check all six timeframe diagnostics.
5. Verify paper journal history before and after deployment.
6. Test on historical/out-of-sample data before interpreting any diagnostic or strategy performance.
