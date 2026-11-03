# JST APEX NEXUS V20.0 — Streamlit starter

Research and paper mode only. No broker execution.

## Files
- `app.py`: Streamlit dashboard wrapper.
- `jst_engine.py`: original V20.0 engine copied without changing its logic.
- `requirements.txt`: core dependencies.
- `.gitignore`: excludes secrets and local journal/state files.

## Streamlit secrets
Set in Streamlit Cloud → App settings → Secrets:

```toml
TWELVE_DATA_API_KEY = "YOUR_TWELVE_DATA_API_KEY"
```

## Important limitations
- This starter runs one scan cycle only when the user presses **SCAN SEKARANG**.
- It is not a 24/7 background worker.
- Community Cloud local files are not durable storage; CSV journal/state can be lost on rebuild/restart/sleep.
- TensorFlow, LightGBM, CatBoost and XGBoost are optional and are not installed by the minimal requirements. Optional modules may report unavailable.
- Validate data, paper journal persistence, dependencies and signal behavior before relying on it.
