# =====================================================================
# JUMP STREET TRADING | JST APEX NEXUS V20.0
# XAU/USD | HYBRID (Pydroid 3 + Colab) | Twelve Data | SIGNAL + PAPER RESEARCH ONLY
# NO AUTO-ORDER | NO BROKER EXECUTION | NO PROFIT GUARANTEE
#
# REQUIRED: requests, pandas, numpy
# OPTIONAL ML: scikit-learn
#
# API KEY SETUP (Pydroid 3):
# Option A (recommended): set environment variable TWELVE_DATA_API_KEY
# before running this script, if your Pydroid setup supports environment
# variables.
# Option B (simple): paste your key into PASTE_API_KEY_HERE below.
# Keep the key private. Never send it in chat or publish this file with
# the key filled in. If exposed, revoke/rotate it at Twelve Data.
# =====================================================================

from __future__ import annotations

import os
import time
import math
import csv
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone

try:
    import requests
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Package belum lengkap. Di Pydroid 3, buka Pip dan install: "
        "requests, pandas, numpy. Detail: " + str(exc)
    )


def _trapezoid_compat(y, dx=1.0):
    """Trapezoidal integration compatible with NumPy 1.x and 2.x.

    The current Quant Math call integrates a one-dimensional return series
    over unit-spaced observations. Prefer NumPy's public trapezoid API when
    available; use a NumPy-only mathematical fallback otherwise.
    """
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("_trapezoid_compat currently expects a 1-D series")
    if values.size < 2:
        return 0.0
    trapezoid_fn = getattr(np, "trapezoid", None)
    if callable(trapezoid_fn):
        return float(trapezoid_fn(values, dx=dx))
    # Compatibility fallback for older NumPy versions; does not rely on
    # np.trapz, which may be absent in newer NumPy builds.
    return float(np.sum((values[:-1] + values[1:]) * (0.5 * dx)))

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    SKLEARN_OK = True
    try:
        from xgboost import XGBClassifier
        XGBOOST_OK = True
    except Exception:
        XGBClassifier = None
        XGBOOST_OK = False
    try:
        from statsmodels.tsa.stattools import coint, adfuller
        STATSMODELS_OK = True
    except Exception:
        coint = adfuller = None
        STATSMODELS_OK = False
except Exception:
    SKLEARN_OK = False
    XGBOOST_OK = False
    STATSMODELS_OK = False


# V20 optional research dependencies; missing packages do not stop the scanner.
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_OK = True
except Exception:
    tf = Sequential = Input = LSTM = Dense = Dropout = EarlyStopping = None
    TENSORFLOW_OK = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_OK = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_OK = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except Exception:
    CatBoostClassifier = None
    CATBOOST_OK = False

# ----------------------------- CONFIG ---------------------------------
APP_NAME = "JST APEX NEXUS V20.0 — HYBRID ML + DEEP LEARNING"
API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
PASTE_API_KEY_HERE = ""  # TEMPEL API KEY TWELVE DATA DI SINI.
if not API_KEY:
    API_KEY = PASTE_API_KEY_HERE.strip()

SYMBOL = "XAU/USD"
API_SYMBOL = "XAU/USD"  # Harus tersedia pada paket Twelve Data milik Anda.
BASE_URL = "https://api.twelvedata.com/time_series"
REQUEST_TIMEOUT = 20
SCAN_SECONDS = 60
ENGINE_MODE = "DUAL_INDEPENDENT"
ARCHITECTURE = "CANDIDATE + ADAPTIVE + FUSION"
ENVIRONMENT_MODE = "HYBRID"

TIMEFRAMES = {"H1": "1h", "M15": "15min", "M5": "5min", "M1": "1min"}
BARS = {"H1": 260, "M15": 260, "M5": 320, "M1": 320}
MIN_BARS = 80
CACHE_SECONDS = {"H1": 3300, "M15": 840, "M5": 240, "M1": 45}

# Research defaults only; not optimized/proven.
RISK_PERCENT = 0.25
DAILY_LOSS_LIMIT_PERCENT = 1.0
MIN_SCORE = 68
MIN_SCORE_GAP = 12
MIN_RR = 1.80
TP_R_MULTIPLIER = 1.80
ATR_SL_MULTIPLIER = 1.50
MIN_SL_PRICE = 1.00
MAX_SL_PRICE = 12.00
MAX_PAPER_OPEN = 1
COOLDOWN_MINUTES = 15
MAX_PAPER_SIGNALS_PER_DAY = 12
MAX_HOLD_MINUTES = 240
PAPER_MODE = True

# Placeholder: replace with measured spread + commission + slippage expressed in R.
ESTIMATED_COST_R = 0.10

ML_MIN_ROWS = 100
ML_MIN_PER_CLASS = 15
ML_HARD_FILTER = False
ML_MIN_PROBABILITY = 0.53
ML_MAX_UNCERTAINTY = 0.30

# V19 ML ensemble research. Models remain advisory until chronological validation.
ML_ENSEMBLE_ENABLED = True
ML_ENSEMBLE_MIN_ROWS = 120
ML_ENSEMBLE_MIN_PER_CLASS = 20
ML_ENSEMBLE_HOLDOUT_FRACTION = 0.20
ML_ENSEMBLE_RANDOM_STATE = 19
DL_ENABLED = True
DL_SEQUENCE_LENGTH = 12
DL_MIN_ROWS = 320
DL_MIN_PER_CLASS = 50
DL_EPOCHS = 30
DL_BATCH_SIZE = 32
DL_RANDOM_STATE = 20


# Statistical arbitrage is optional because each extra symbol consumes API quota.
STAT_ARB_ENABLED = False  # Enable only after reference symbol is accessible on your Twelve Data plan
STAT_ARB_REFERENCE_SYMBOL = "XAG/USD"
STAT_ARB_INTERVAL = "5min"
STAT_ARB_OUTPUTSIZE = 320
STAT_ARB_MIN_OBSERVATIONS = 100
STAT_ARB_ENTRY_Z = 2.0
STAT_ARB_EXIT_Z = 0.5
STAT_ARB_STOP_Z = 3.5
STAT_ARB_COST_BPS_PER_LEG = 2.0

ROOT = Path(__file__).resolve().parent
TRADE_JOURNAL = ROOT / "jst_v11_1_paper_journal.csv"  # preserve V15.1 history
CANDIDATE_JOURNAL = ROOT / "jst_v11_1_candidates.csv"
STATE_FILE = ROOT / "jst_v11_1_state.json"

FEATURES = [
    "f_trend", "f_momentum", "f_rsi_centered", "f_robust_z",
    "f_efficiency_ratio", "f_volatility_ratio", "f_atr_pct",
    "f_body_ratio", "f_breakout", "f_direction", "f_m1_trigger",
    "f_structure", "f_zone", "f_liquidity", "f_engulfing", "f_math_bias"
]
TRADE_COLUMNS = [
    "id", "signal_time", "entry_candle_time", "last_processed_candle",
    "close_time", "symbol", "direction", "entry", "sl", "tp",
    "risk_percent", "score", "rr", "status", "outcome_r", "close_price",
    "close_reason", "regime", "ml_probability", "ev_r",
    "structure", "zone", "liquidity", "engulfing", "math_bias",
    "adaptive_regime", "adaptive_mode", "adaptive_min_score", *FEATURES
]
CANDIDATE_COLUMNS = [
    "id", "time", "candle_time", "symbol", "direction", "entry", "sl", "tp",
    "score", "score_gap", "rr", "decision", "reason", "ml_probability",
    "uncertainty", "historical_win_rate", "historical_expectancy_r",
    "structure", "zone", "liquidity", "engulfing", "math_bias",
    "adaptive_regime", "adaptive_mode", "adaptive_min_score", *FEATURES
]

_DATA_CACHE = {}
_RESEARCH_CACHE = {"mtime": None, "report": None}


# ----------------------------- UTILITIES -------------------------------
def utc_now():
    return datetime.now(timezone.utc)


def stamp():
    return utc_now().isoformat(timespec="seconds")


def local_stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def ensure_csv(path, columns):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def append_csv(path, columns, row):
    ensure_csv(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerow({key: row.get(key, "") for key in columns})


def load_state():
    state = {
        "last_candidate_key": "",
        "last_signal_time": "",
        "daily_signal_count": 0,
        "daily_date": utc_now().date().isoformat(),
    }
    try:
        saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            state.update(saved)
    except Exception:
        pass
    today = utc_now().date().isoformat()
    if state.get("daily_date") != today:
        state["daily_date"] = today
        state["daily_signal_count"] = 0
    return state


def save_state(state):
    try:
        temp = STATE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(STATE_FILE)
    except Exception as exc:
        print("[WARN] State belum tersimpan:", exc)


def migrate_journal_if_needed():
    """Use V11.1-specific files, and safely add missing columns if file exists."""
    ensure_csv(TRADE_JOURNAL, TRADE_COLUMNS)
    ensure_csv(CANDIDATE_JOURNAL, CANDIDATE_COLUMNS)
    try:
        df = pd.read_csv(TRADE_JOURNAL)
        changed = False
        for col in TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
                changed = True
        if changed:
            df = df[TRADE_COLUMNS]
            backup = TRADE_JOURNAL.with_suffix(".pre_migration.csv")
            if not backup.exists():
                TRADE_JOURNAL.replace(backup)
                df.to_csv(TRADE_JOURNAL, index=False)
            else:
                df.to_csv(TRADE_JOURNAL, index=False)
    except Exception as exc:
        print("[WARN] Journal migration check:", exc)


# ----------------------------- MARKET DATA -----------------------------
def fetch_candles(name, interval, outputsize):
    now_mono = time.monotonic()
    cached = _DATA_CACHE.get(name)
    if cached and now_mono - cached[0] < CACHE_SECONDS.get(name, 0):
        return cached[1].copy()

    if not API_KEY:
        raise RuntimeError(
            "API key belum diisi. Isi PASTE_API_KEY_HERE di bagian CONFIG "
            "atau atur TWELVE_DATA_API_KEY."
        )

    response = requests.get(
        BASE_URL,
        params={
            "symbol": API_SYMBOL, "interval": interval,
            "outputsize": outputsize, "apikey": API_KEY,
            "format": "JSON", "timezone": "UTC"
        },
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("values"), list):
        msg = payload.get("message", payload) if isinstance(payload, dict) else payload
        raise RuntimeError(f"Twelve Data {name}: {msg}")

    df = pd.DataFrame(payload["values"])
    required = {"datetime", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{name}: kolom data hilang: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "volume":
            df[col] = 0.0

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    valid = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
        (df["low"] <= df[["open", "close", "high"]].min(axis=1)) &
        (df["high"] >= df["low"]) &
        (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    )
    df = df[valid].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Exclude the latest returned bar because it may still be forming.
    if not df.empty:
        df = df.iloc[:-1].copy()
    if len(df) < MIN_BARS:
        raise RuntimeError(
            f"Candle valid {name} hanya {len(df)}; minimum {MIN_BARS}. "
            "Cek simbol, interval, paket API, kuota, dan koneksi."
        )

    df = df.reset_index(drop=True)
    _DATA_CACHE[name] = (now_mono, df.copy())
    return df


# ----------------------------- INDICATORS ------------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def efficiency_ratio(close, period=10):
    net = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period).sum().replace(0, np.nan)
    return (net / path).fillna(0).clip(0, 1)


def robust_z(series, period=50):
    w = series.tail(period)
    med = w.median()
    mad = (w - med).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        return 0.0
    return safe_float((series.iloc[-1] - med) / (1.4826 * mad))


def quant_snapshot(df):
    close = df["close"]
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    a = safe_float(atr(df).iloc[-1])
    e20 = safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    e50 = safe_float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    e200 = safe_float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    trend = 1 if e20 > e50 > e200 else (-1 if e20 < e50 < e200 else 0)
    short_vol = safe_float(returns.tail(10).std())
    long_vol = safe_float(returns.tail(50).std())
    vol_ratio = short_vol / long_vol if long_vol > 1e-12 else 1.0
    er = safe_float(efficiency_ratio(close).iloc[-1])
    prior_high = safe_float(df["high"].iloc[-21:-1].max())
    prior_low = safe_float(df["low"].iloc[-21:-1].min())
    price = safe_float(close.iloc[-1])
    breakout = 1 if price > prior_high else (-1 if price < prior_low else 0)
    rng = max(safe_float(df["high"].iloc[-1] - df["low"].iloc[-1]), 1e-9)
    body_ratio = abs(safe_float(df["close"].iloc[-1] - df["open"].iloc[-1])) / rng
    regime = "TREND" if er >= 0.30 else ("MIXED" if er >= 0.15 else "RANGE")
    regime += "/HIGH_VOL" if vol_ratio >= 2.2 else ("/LOW_VOL" if vol_ratio <= 0.55 else "/NORMAL_VOL")
    return {
        "price": price, "atr": a, "rsi": safe_float(rsi(close).iloc[-1], 50),
        "z": robust_z(close), "er": er, "vol_ratio": vol_ratio,
        "momentum": safe_float((price / max(safe_float(close.iloc[-6]), 1e-9) - 1) * 100),
        "body_ratio": body_ratio, "breakout": breakout, "trend": trend,
        "ema20": e20, "ema50": e50, "ema200": e200,
        "prior_high": prior_high, "prior_low": prior_low, "regime": regime
    }


def adaptive_regime(q, physics):
    """Transparent rule-based regime label; diagnostic, not a trained HMM."""
    er = safe_float(q.get("er"))
    vr = safe_float(q.get("vol_ratio"), 1.0)
    shift = safe_float(physics.get("change_score"))
    if vr >= 2.5:
        return "HIGH_VOLATILITY", "REDUCE_EXPOSURE"
    if shift >= 1.8:
        return "TRANSITION", "REQUIRE_STRONG_CONFIRMATION"
    if er >= 0.35:
        return "DIRECTIONAL", "TREND_CONTINUATION"
    if er <= 0.18:
        return "RANGE", "MEAN_REVERSION_RESEARCH_ONLY"
    return "MIXED", "SELECTIVE"


def adaptive_policy(q, physics, direction, base_score=MIN_SCORE):
    """Conservative adaptive thresholds; never relax below baseline."""
    regime, mode = adaptive_regime(q, physics)
    threshold = int(base_score)
    reasons = []
    if regime == "HIGH_VOLATILITY":
        threshold += 10; reasons.append("high volatility: threshold +10")
    elif regime == "TRANSITION":
        threshold += 8; reasons.append("regime transition: threshold +8")
    elif regime == "MIXED":
        threshold += 4; reasons.append("mixed regime: threshold +4")
    elif regime == "RANGE":
        threshold += 6; reasons.append("range regime: trend signal threshold +6")
    # In a directional regime, do not loosen the baseline score requirement.
    if safe_float(physics.get("change_score")) >= 1.2:
        threshold = max(threshold, base_score + 5)
        reasons.append("distribution shift watch")
    return {"regime": regime, "mode": mode, "min_score": threshold,
            "reasons": reasons, "abstain": regime == "HIGH_VOLATILITY" and safe_float(q.get("vol_ratio")) >= 3.5}


def bayesian_win_rate_summary():
    """Beta(1,1) posterior summary from closed paper outcomes; descriptive only."""
    df = read_journal()
    if df.empty or "outcome_r" not in df:
        return {"n": 0, "alpha": 1.0, "beta": 1.0, "mean": 0.5, "lo": 0.0, "hi": 1.0}
    closed = df[df.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("CLOSED")]
    r = pd.to_numeric(closed.get("outcome_r", pd.Series(dtype=float)), errors="coerce").dropna()
    wins = int((r > 0).sum()); losses = int((r <= 0).sum())
    a, b = 1.0 + wins, 1.0 + losses
    # Normal approximation is intentionally withheld for tiny samples; quantiles use beta distribution if SciPy exists.
    mean = a / (a + b)
    lo, hi = 0.0, 1.0
    try:
        from scipy.stats import beta as beta_dist
        lo, hi = (float(x) for x in beta_dist.ppf([0.025, 0.975], a, b))
    except Exception:
        pass
    return {"n": len(r), "alpha": a, "beta": b, "mean": mean, "lo": lo, "hi": hi}


def m1_trigger(df, direction):
    last, prev = df.iloc[-1], df.iloc[-2]
    sign = 1 if direction == "BUY" else -1
    body = safe_float(last["close"] - last["open"])
    rng = max(safe_float(last["high"] - last["low"]), 1e-9)
    score, reasons = 0, []
    if body * sign > 0 and abs(body) / rng >= 0.45:
        score += 1
        reasons.append("M1 candle tegas searah")
    if direction == "BUY" and last["close"] > prev["high"]:
        score += 1
        reasons.append("M1 micro-breakout naik")
    elif direction == "SELL" and last["close"] < prev["low"]:
        score += 1
        reasons.append("M1 micro-breakout turun")
    return score, reasons


# ----------------------------- CONFLUENCE ------------------------------
def detect_structure(df, lookback=20):
    if len(df) < lookback + 2:
        return "NONE"
    prior, last = df.iloc[-lookback-1:-1], df.iloc[-1]
    if last["close"] > prior["high"].max():
        return "BOS_BULLISH"
    if last["close"] < prior["low"].min():
        return "BOS_BEARISH"
    return "NONE"


def detect_zone(df):
    # Simple candle-shape proxy, not a validated order-block detector.
    if len(df) < 5:
        return "NONE"
    a, base, departure = df.iloc[-4], df.iloc[-3], df.iloc[-2]
    rng = safe_float(base["high"] - base["low"])
    if rng <= 0 or abs(base["close"] - base["open"]) / rng >= 0.45:
        return "NONE"
    au, ad = a["close"] > a["open"], a["close"] < a["open"]
    du, dd = departure["close"] > departure["open"], departure["close"] < departure["open"]
    if au and du: return "RBR"
    if ad and du: return "DBR"
    if au and dd: return "RBD"
    if ad and dd: return "DBD"
    return "NONE"


def detect_sweep(df, lookback=20):
    if len(df) < lookback + 2:
        return "NONE"
    prior, last = df.iloc[-lookback-1:-1], df.iloc[-1]
    ph, pl = prior["high"].max(), prior["low"].min()
    if last["low"] < pl and last["close"] > pl:
        return "SWEEP_LOW"
    if last["high"] > ph and last["close"] < ph:
        return "SWEEP_HIGH"
    return "NONE"


def detect_engulfing(df):
    if len(df) < 3:
        return "NONE"
    p, c = df.iloc[-2], df.iloc[-1]
    if p.close < p.open and c.close > c.open and c.close >= p.open and c.open <= p.close:
        return "BULLISH_ENGULFING"
    if p.close > p.open and c.close < c.open and c.close <= p.open and c.open >= p.close:
        return "BEARISH_ENGULFING"
    return "NONE"


def math_bias_snapshot(df):
    if len(df) < 30:
        return {"bias": "NEUTRAL", "up_frequency": 0.5, "down_frequency": 0.5, "momentum": 0.0}
    close = df["close"].astype(float)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(50)
    if len(ret) < 10:
        return {"bias": "NEUTRAL", "up_frequency": 0.5, "down_frequency": 0.5, "momentum": 0.0}
    up, down = int((ret > 0).sum()), int((ret < 0).sum())
    total = max(up + down, 1)
    mom = safe_float(close.iloc[-1] / max(safe_float(close.iloc[-11]), 1e-12) - 1)
    e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    bull = int(mom > 0) + int(e20 > e50) + int(up > down)
    bear = int(mom < 0) + int(e20 < e50) + int(down > up)
    bias = "BULLISH" if bull >= 2 and bull > bear else ("BEARISH" if bear >= 2 and bear > bull else "NEUTRAL")
    return {"bias": bias, "up_frequency": up / total, "down_frequency": down / total, "momentum": mom}


def feature_snapshot(q, direction, trigger, structure, zone, sweep, engulf, bias):
    return {
        "f_trend": q["trend"], "f_momentum": q["momentum"],
        "f_rsi_centered": (q["rsi"] - 50) / 50, "f_robust_z": q["z"],
        "f_efficiency_ratio": q["er"], "f_volatility_ratio": q["vol_ratio"],
        "f_atr_pct": q["atr"] / max(q["price"], 1e-9),
        "f_body_ratio": q["body_ratio"], "f_breakout": q["breakout"],
        "f_direction": 1 if direction == "BUY" else -1, "f_m1_trigger": trigger,
        "f_structure": 1 if structure == "BOS_BULLISH" else (-1 if structure == "BOS_BEARISH" else 0),
        "f_zone": 1 if zone in ("RBR", "DBR") else (-1 if zone in ("RBD", "DBD") else 0),
        "f_liquidity": 1 if sweep == "SWEEP_LOW" else (-1 if sweep == "SWEEP_HIGH" else 0),
        "f_engulfing": 1 if engulf == "BULLISH_ENGULFING" else (-1 if engulf == "BEARISH_ENGULFING" else 0),
        "f_math_bias": 1 if bias == "BULLISH" else (-1 if bias == "BEARISH" else 0)
    }


# ----------------------------- SIGNAL ENGINE --------------------------
def make_candidate(frames):
    h1, m15, m5 = (quant_snapshot(frames[k]) for k in ("H1", "M15", "M5"))
    m1 = quant_snapshot(frames["M1"])
    entry_candle_time = frames["M1"]["datetime"].iloc[-1].isoformat()
    entry = m1["price"]
    structure = detect_structure(frames["M15"])
    zone = detect_zone(frames["M15"])
    sweep = detect_sweep(frames["M5"])
    engulf = detect_engulfing(frames["M5"])
    bias = math_bias_snapshot(frames["M1"])
    physics = physics_quant_diagnostics(frames["M5"])
    adaptive = adaptive_policy(m5, physics, "BOTH")
    candidates = []

    for direction in ("BUY", "SELL"):
        sign = 1 if direction == "BUY" else -1
        score, reasons = 0, []
        if h1["trend"] == sign: score += 18; reasons.append("H1 trend searah")
        elif h1["trend"] == 0: score += 5; reasons.append("H1 netral")
        else: score -= 8; reasons.append("H1 berlawanan")
        if m15["trend"] == sign: score += 14; reasons.append("M15 trend searah")
        elif m15["trend"] == 0: score += 4
        else: score -= 7; reasons.append("M15 berlawanan")
        if m15["breakout"] == sign: score += 8; reasons.append("M15 breakout")
        if m5["momentum"] * sign > 0: score += 10; reasons.append("M5 momentum searah")
        if m5["breakout"] == sign: score += 12; reasons.append("M5 breakout")
        if (direction == "BUY" and m5["rsi"] >= 52) or (direction == "SELL" and m5["rsi"] <= 48):
            score += 7; reasons.append("RSI M5 mendukung")
        last5 = frames["M5"].iloc[-1]
        candle_sign = 1 if last5.close > last5.open else (-1 if last5.close < last5.open else 0)
        if m5["body_ratio"] >= 0.45 and candle_sign == sign:
            score += 6; reasons.append("Candle M5 tegas")
        if m5["er"] >= 0.25: score += 7; reasons.append("Efficiency ratio baik")
        elif m5["er"] < 0.10: score -= 6; reasons.append("Pasar choppy")
        if m5["vol_ratio"] >= 2.5: score -= 10; reasons.append("Volatilitas ekstrem")
        if abs(m5["z"]) >= 3.5 and m5["z"] * sign > 0:
            score -= 7; reasons.append("Harga terlalu ekstensi")
        trigger, trigger_reasons = m1_trigger(frames["M1"], direction)
        if trigger >= 2: score += 12; reasons.extend(trigger_reasons)
        else: score -= 8; reasons.append("Konfirmasi M1 belum lengkap")

        side = {
            "structure": 1 if structure == "BOS_BULLISH" else (-1 if structure == "BOS_BEARISH" else 0),
            "zone": 1 if zone in ("RBR", "DBR") else (-1 if zone in ("RBD", "DBD") else 0),
            "sweep": 1 if sweep == "SWEEP_LOW" else (-1 if sweep == "SWEEP_HIGH" else 0),
            "engulf": 1 if engulf == "BULLISH_ENGULFING" else (-1 if engulf == "BEARISH_ENGULFING" else 0),
            "bias": 1 if bias["bias"] == "BULLISH" else (-1 if bias["bias"] == "BEARISH" else 0)
        }
        for key, weight, label in [
            ("structure", 5, "M15 BOS"), ("zone", 3, "M15 zone proxy"),
            ("sweep", 4, "M5 liquidity sweep"), ("engulf", 3, "M5 engulfing"),
            ("bias", 3, "M1 math bias")
        ]:
            if side[key] == sign:
                score += weight; reasons.append(label + " searah")
            elif side[key] == -sign and key in ("structure", "zone", "engulf", "bias"):
                score -= max(2, weight - 1); reasons.append(label + " konflik")
        if h1["trend"] == -sign and m15["trend"] == -sign:
            score -= 18; reasons.append("Konflik dua timeframe tinggi")

        candidates.append({
            "direction": direction, "score": int(clamp(score, 0, 100)),
            "reasons": reasons, "trigger": trigger
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    winner, runner = candidates[0], candidates[1]
    sl_dist = clamp(max(m5["atr"] * ATR_SL_MULTIPLIER, MIN_SL_PRICE), MIN_SL_PRICE, MAX_SL_PRICE)
    tp_dist = sl_dist * TP_R_MULTIPLIER
    direction = winner["direction"]
    sl = entry - sl_dist if direction == "BUY" else entry + sl_dist
    tp = entry + tp_dist if direction == "BUY" else entry - tp_dist
    rr = abs(tp - entry) / max(abs(entry - sl), 1e-9)
    features = feature_snapshot(m5, direction, winner["trigger"], structure, zone, sweep, engulf, bias["bias"])
    return {
        "direction": direction, "score": winner["score"],
        "score_gap": winner["score"] - runner["score"], "reasons": winner["reasons"],
        "entry": entry, "sl": sl, "tp": tp, "rr": rr, "regime": m5["regime"],
        "q": m5, "features": features, "structure": structure, "zone": zone,
        "liquidity": sweep, "engulfing": engulf, "math_bias": bias["bias"],
        "math_diag": bias, "physics": physics, "adaptive": adaptive,
        "time": stamp(), "candle_time": entry_candle_time,
        "key": entry_candle_time + "_" + direction, "trigger": winner["trigger"]
    }


# ----------------------------- JOURNAL / STATS -------------------------
def read_journal():
    if not TRADE_JOURNAL.exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    try:
        df = pd.read_csv(TRADE_JOURNAL)
        for col in TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # CSV inference can turn an all-empty timestamp column into float64.
        # These columns must remain string/object so ISO timestamps can be
        # written safely during paper-trade management.
        text_columns = [
            "id", "signal_time", "entry_candle_time", "last_processed_candle",
            "close_time", "symbol", "direction", "status", "close_reason",
            "regime", "structure", "zone", "liquidity", "engulfing", "math_bias"
        ]
        for col in text_columns:
            df[col] = df[col].astype("object")

        return df[TRADE_COLUMNS]
    except Exception as exc:
        print("[WARN] Gagal membaca journal:", exc)
        return pd.DataFrame(columns=TRADE_COLUMNS)


def closed_stats():
    df = read_journal()
    if df.empty:
        return {"n": 0, "win_rate": None, "expectancy": None, "pf": None, "max_dd": None}
    closed = df[df["status"].astype(str).str.upper().eq("CLOSED")]
    r = pd.to_numeric(closed["outcome_r"], errors="coerce").dropna()
    if r.empty:
        return {"n": 0, "win_rate": None, "expectancy": None, "pf": None, "max_dd": None}
    wins, losses = r[r > 0], r[r < 0]
    gross_loss = abs(float(losses.sum()))
    equity = np.r_[0.0, r.cumsum().to_numpy()]
    dd = equity - np.maximum.accumulate(equity)
    return {
        "n": int(len(r)), "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "pf": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if len(wins) else 0.0),
        "max_dd": abs(float(dd.min()))
    }


def wilson_ci(wins, total, z=1.96):
    if total <= 0: return None, None
    p = wins / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return max(0, center-half), min(1, center+half)


def bootstrap_ci(values, reps=2000, seed=17):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 5: return None, None
    rng = np.random.default_rng(seed)
    means = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(reps)]
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def daily_realized_r():
    df = read_journal()
    if df.empty: return 0.0
    closed = df[df["status"].astype(str).str.upper().eq("CLOSED")].copy()
    if closed.empty: return 0.0
    times = pd.to_datetime(closed["close_time"], errors="coerce", utc=True)
    r = pd.to_numeric(closed["outcome_r"], errors="coerce")
    return safe_float(r.loc[times.dt.date == utc_now().date()].sum())


def research_report():
    df = read_journal()
    if df.empty:
        return {"n": 0, "status": "Belum ada trade tertutup."}
    closed = df[df["status"].astype(str).str.upper().eq("CLOSED")].copy()
    r = pd.to_numeric(closed["outcome_r"], errors="coerce").dropna()
    if len(r) == 0:
        return {"n": 0, "status": "Belum ada outcome R yang valid."}
    wins = int((r > 0).sum())
    lo, hi = wilson_ci(wins, len(r))
    elo, ehi = bootstrap_ci(r.to_numpy())
    return {
        "n": len(r), "win_rate": wins/len(r), "win_ci": (lo, hi),
        "expectancy": float(r.mean()), "expectancy_ci": (elo, ehi),
        "pf": closed_stats()["pf"], "max_dd": closed_stats()["max_dd"],
        "status": "Statistik deskriptif jurnal paper; bukan jaminan performa masa depan."
    }


# ----------------------------- OPTIONAL ML -----------------------------
class EdgeModel:
    """Optional advisory only; model probabilities are not assumed calibrated."""
    def __init__(self):
        self.model = None
        self.status = "ML off / belum terlatih"
        self.rows = 0
        self.accuracy = None
        self.brier = None
        self.mtime = None

    def train_if_changed(self):
        if not SKLEARN_OK or not TRADE_JOURNAL.exists():
            self.status = "scikit-learn atau jurnal tidak tersedia; ML nonaktif"
            return
        try:
            mtime = TRADE_JOURNAL.stat().st_mtime
            if self.mtime == mtime:
                return
            self.mtime = mtime
            df = pd.read_csv(TRADE_JOURNAL)
            if not {"status", "outcome_r"}.issubset(df.columns):
                return
            df = df[df.status.astype(str).str.upper().eq("CLOSED")].copy()
            for col in FEATURES + ["outcome_r"]:
                if col not in df: df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["outcome_r"])
            if len(df) < ML_MIN_ROWS:
                self.status = f"ML menunggu sampel: {len(df)}/{ML_MIN_ROWS}"
                return
            y = (df.outcome_r > 0).astype(int)
            if y.value_counts().min() < ML_MIN_PER_CLASS:
                self.status = "ML butuh minimal 15 sampel pada setiap kelas"
                return
            X = df[FEATURES].fillna(0)
            split = int(len(df)*0.8)
            if y.iloc[:split].nunique() < 2 or len(df)-split < 10:
                self.status = "Sampel kronologis belum memadai"
                return
            diagnostic = ExtraTreesClassifier(
                n_estimators=160, min_samples_leaf=4, max_features=0.8,
                class_weight="balanced", random_state=17, n_jobs=1
            )
            diagnostic.fit(X.iloc[:split], y.iloc[:split])
            p = diagnostic.predict_proba(X.iloc[split:])[:, list(diagnostic.classes_).index(1)]
            self.accuracy = float(accuracy_score(y.iloc[split:], p >= 0.5))
            self.brier = float(brier_score_loss(y.iloc[split:], p))
            # Refit on all closed rows for advisory inference only after recording holdout metrics.
            self.model = ExtraTreesClassifier(
                n_estimators=160, min_samples_leaf=4, max_features=0.8,
                class_weight="balanced", random_state=17, n_jobs=1
            ).fit(X, y)
            self.rows = len(df)
            self.status = (
                f"ML advisory rows={self.rows}; chronological holdout accuracy={self.accuracy:.3f}, "
                f"Brier={self.brier:.3f}. Ini diagnostik, bukan bukti edge."
            )
        except Exception as exc:
            self.status = "ML error: " + str(exc)

    def predict(self, features):
        if self.model is None:
            return None, None
        try:
            x = pd.DataFrame([[safe_float(features.get(k, 0)) for k in FEATURES]], columns=FEATURES)
            idx = list(self.model.classes_).index(1)
            p = float(self.model.predict_proba(x)[0][idx])
            votes = [float(tree.predict_proba(x)[0][idx]) for tree in self.model.estimators_]
            return p, float(np.std(votes)) if votes else None
        except Exception:
            return None, None


# ----------------------------- PAPER EXECUTION -------------------------
def open_count():
    df = read_journal()
    return int(df.status.astype(str).str.upper().eq("OPEN").sum()) if not df.empty else 0


def close_paper_trade(df, idx, candle, reason, exit_price, raw_r):
    net_r = raw_r - ESTIMATED_COST_R
    df.at[idx, "status"] = "CLOSED"
    df.at[idx, "close_time"] = candle["datetime"].isoformat()
    df.at[idx, "close_price"] = round(float(exit_price), 5)
    df.at[idx, "close_reason"] = reason
    df.at[idx, "outcome_r"] = round(float(net_r), 4)
    print(f"[PAPER CLOSE] {df.at[idx, 'direction']} | {reason} | net={net_r:+.2f}R")


def manage_open_paper_trades(frames):
    df = read_journal()
    if df.empty:
        return
    candles = frames["M1"].sort_values("datetime")
    changed = False
    now = utc_now()

    for idx, row in df[df.status.astype(str).str.upper().eq("OPEN")].iterrows():
        direction = str(row.get("direction", "")).upper()
        entry = safe_float(row.get("entry"))
        sl, tp = safe_float(row.get("sl")), safe_float(row.get("tp"))
        entry_time = pd.to_datetime(row.get("entry_candle_time"), errors="coerce", utc=True)
        last_time = pd.to_datetime(row.get("last_processed_candle", ""), errors="coerce", utc=True)
        if pd.isna(entry_time):
            # Old/invalid rows cannot safely be reconstructed; leave open and warn.
            print("[WARN] Paper trade tanpa entry_candle_time; perlu diperiksa:", row.get("id"))
            continue

        # Only process candles strictly after the entry candle and newer than last processed.
        mask = candles["datetime"] > entry_time
        if not pd.isna(last_time):
            mask &= candles["datetime"] > last_time
        new_candles = candles.loc[mask]

        for _, candle in new_candles.iterrows():
            high, low = safe_float(candle.high), safe_float(candle.low)
            hit_sl = low <= sl if direction == "BUY" else high >= sl
            hit_tp = high >= tp if direction == "BUY" else low <= tp
            df.at[idx, "last_processed_candle"] = candle.datetime.isoformat()
            changed = True
            if hit_sl or hit_tp:
                risk_dist = abs(entry-sl)
                # Conservative assumption if both levels are inside one M1 candle.
                if hit_sl:
                    close_paper_trade(df, idx, candle,
                                      "SL (conservative same-bar rule)", sl, -1.0)
                else:
                    raw_r = abs(tp-entry) / max(risk_dist, 1e-9)
                    close_paper_trade(df, idx, candle, "TP", tp, raw_r)
                break

        # Time stop: close at last available closed M1 close after max holding period.
        if str(df.at[idx, "status"]).upper() == "OPEN" and len(candles):
            latest = candles.iloc[-1]
            elapsed = (latest.datetime.to_pydatetime() - entry_time.to_pydatetime()).total_seconds()/60
            if elapsed >= MAX_HOLD_MINUTES and latest.datetime > entry_time:
                risk_dist = abs(entry-sl)
                exit_price = safe_float(latest.close)
                raw_r = ((exit_price-entry)/max(risk_dist, 1e-9)) if direction == "BUY" else (
                    (entry-exit_price)/max(risk_dist, 1e-9))
                close_paper_trade(df, idx, latest, "TIME STOP", exit_price, raw_r)
                changed = True

    if changed:
        df.to_csv(TRADE_JOURNAL, index=False)


def register_candidate(c, prob, uncertainty, decision, reason, stats):
    row = {
        "id": f"C-{int(time.time()*1000)}", "time": c["time"],
        "candle_time": c["candle_time"], "symbol": SYMBOL,
        "direction": c["direction"], "entry": round(c["entry"], 5),
        "sl": round(c["sl"], 5), "tp": round(c["tp"], 5),
        "score": c["score"], "score_gap": c["score_gap"], "rr": round(c["rr"], 2),
        "decision": decision, "reason": reason,
        "ml_probability": "" if prob is None else round(prob, 4),
        "uncertainty": "" if uncertainty is None else round(uncertainty, 4),
        "historical_win_rate": "" if stats["win_rate"] is None else round(stats["win_rate"], 4),
        "historical_expectancy_r": "" if stats["expectancy"] is None else round(stats["expectancy"], 4),
        "structure": c["structure"], "zone": c["zone"], "liquidity": c["liquidity"],
        "engulfing": c["engulfing"], "math_bias": c["math_bias"], **c["features"]
    }
    append_csv(CANDIDATE_JOURNAL, CANDIDATE_COLUMNS, row)


def register_paper_trade(c, prob):
    row = {
        "id": f"T-{int(time.time()*1000)}", "signal_time": c["time"],
        "entry_candle_time": c["candle_time"], "last_processed_candle": c["candle_time"],
        "close_time": "", "symbol": SYMBOL, "direction": c["direction"],
        "entry": round(c["entry"], 5), "sl": round(c["sl"], 5), "tp": round(c["tp"], 5),
        "risk_percent": RISK_PERCENT, "score": c["score"], "rr": round(c["rr"], 3),
        "status": "OPEN", "outcome_r": "", "close_price": "", "close_reason": "",
        "regime": c["regime"], "ml_probability": "" if prob is None else round(prob, 4),
        "ev_r": "", "structure": c["structure"], "zone": c["zone"],
        "liquidity": c["liquidity"], "engulfing": c["engulfing"],
        "math_bias": c["math_bias"], **c["features"]
    }
    append_csv(TRADE_JOURNAL, TRADE_COLUMNS, row)
    print(f"[PAPER OPEN] {c['direction']} entry={c['entry']:.3f} "
          f"SL={c['sl']:.3f} TP={c['tp']:.3f} RR={c['rr']:.2f} score={c['score']}/100")




# ---------------------- V12 EDGE RESEARCH ENGINE ------------------------
# Evaluates CLOSED paper trades. This is not a substitute for a historical
# price-level backtest or a sealed final out-of-sample test.
MIN_EDGE_SAMPLE = 50
MIN_EDGE_EXPECTANCY_R = 0.0
MIN_EDGE_PF = 1.05
CHRONOLOGICAL_HOLDOUT_FRACTION = 0.30


def _numeric_closed_trades():
    df = read_journal()
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    df = df[df["status"].astype(str).str.upper().eq("CLOSED")].copy()
    df["outcome_r"] = pd.to_numeric(df.get("outcome_r"), errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["outcome_r"])
    if "close_time" in df.columns:
        df["_sort_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)
        df = df.sort_values("_sort_time", na_position="last")
    return df.reset_index(drop=True)


def _metrics_from_r(values):
    r = np.asarray(values, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n": 0, "win_rate": None, "expectancy": None, "pf": None, "max_dd": None}
    wins, losses = r[r > 0], r[r < 0]
    gross_loss = abs(float(losses.sum()))
    equity = np.r_[0.0, np.cumsum(r)]
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "n": int(len(r)),
        "win_rate": float(np.mean(r > 0)),
        "expectancy": float(np.mean(r)),
        "pf": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if len(wins) else 0.0),
        "max_dd": abs(float(drawdown.min()))
    }


def _edge_group_key(row):
    # Keep groups interpretable and avoid over-fragmenting small samples.
    regime = str(row.get("regime", "UNKNOWN"))
    direction = str(row.get("direction", "UNKNOWN"))
    structure = str(row.get("structure", "NONE"))
    return f"{regime} | {direction} | {structure}"


def edge_research_report():
    df = _numeric_closed_trades()
    report = {
        "n": len(df),
        "overall": _metrics_from_r(df["outcome_r"].to_numpy()) if len(df) else _metrics_from_r([]),
        "groups": [],
        "holdout": None,
        "bootstrap_ci": (None, None),
        "label": "INSUFFICIENT EVIDENCE"
    }
    if len(df) < 2:
        return report

    # Chronological split is a diagnostic only: groups are not selected on
    # the holdout segment, but this is still paper-journal data, not sealed OOS.
    split = int(len(df) * (1.0 - CHRONOLOGICAL_HOLDOUT_FRACTION))
    split = max(1, min(split, len(df) - 1))
    train, holdout = df.iloc[:split], df.iloc[split:]
    report["holdout"] = _metrics_from_r(holdout["outcome_r"].to_numpy())

    try:
        report["bootstrap_ci"] = bootstrap_ci(df["outcome_r"].to_numpy(), reps=3000, seed=29)
    except Exception:
        pass

    train = train.copy()
    train["_group"] = train.apply(_edge_group_key, axis=1)
    holdout = holdout.copy()
    holdout["_group"] = holdout.apply(_edge_group_key, axis=1)
    for key, g in train.groupby("_group"):
        m = _metrics_from_r(g["outcome_r"].to_numpy())
        h = _metrics_from_r(holdout.loc[holdout["_group"] == key, "outcome_r"].to_numpy())
        if m["n"] < MIN_EDGE_SAMPLE:
            label = "SAMPLE TOO SMALL"
        elif m["expectancy"] is not None and m["expectancy"] > MIN_EDGE_EXPECTANCY_R and m["pf"] > MIN_EDGE_PF:
            if h["n"] >= 10 and h["expectancy"] is not None and h["expectancy"] > 0:
                label = "PROMISING — NEEDS INDEPENDENT VALIDATION"
            else:
                label = "CANDIDATE — HOLDOUT EVIDENCE INSUFFICIENT"
        else:
            label = "NO EDGE EVIDENCE"
        report["groups"].append({
            "key": key, "train": m, "holdout": h, "label": label
        })

    if len(df) >= MIN_EDGE_SAMPLE:
        ci = report["bootstrap_ci"]
        overall = report["overall"]
        if (overall["expectancy"] is not None and overall["expectancy"] > 0
                and overall["pf"] > MIN_EDGE_PF
                and ci[0] is not None and ci[0] > 0
                and report["holdout"]["n"] >= 20
                and report["holdout"]["expectancy"] is not None
                and report["holdout"]["expectancy"] > 0):
            report["label"] = "PROMISING — NOT YET PROVEN"
        else:
            report["label"] = "NO VALIDATED EDGE"
    return report


def print_edge_research_report():
    r = edge_research_report()
    print("\n--- V12 EDGE RESEARCH ---")
    if r["n"] == 0:
        print("Belum ada closed paper trade yang valid.")
        print("Edge status: INSUFFICIENT EVIDENCE")
        return
    m = r["overall"]
    print(f"Closed trades={m['n']} | win={m['win_rate']:.1%} | "
          f"net expectancy={m['expectancy']:+.3f}R | PF={m['pf']:.2f} | "
          f"max DD={m['max_dd']:.2f}R")
    lo, hi = r["bootstrap_ci"]
    if lo is not None:
        print(f"Bootstrap CI 95% expectancy (iid): [{lo:+.3f}, {hi:+.3f}]R")
    h = r["holdout"]
    if h and h["n"]:
        print(f"Chronological last-30% diagnostic: n={h['n']} | "
              f"EV={h['expectancy']:+.3f}R | PF={h['pf']:.2f}")
    print("EDGE STATUS:", r["label"])
    if r["groups"]:
        print("Setup groups (train / later holdout):")
        for g in sorted(r["groups"], key=lambda x: x["train"]["n"], reverse=True)[:8]:
            tm, hm = g["train"], g["holdout"]
            hv = "n/a" if hm["expectancy"] is None else f"{hm['expectancy']:+.3f}R"
            print(f" - {g['key']}: train n={tm['n']}, EV={tm['expectancy']:+.3f}R; "
                  f"holdout EV={hv}; {g['label']}")
    print("Catatan: jurnal paper bukan pengganti backtest harga historis dengan biaya dan final OOS independen.")


# ----------------------------- ADAPTIVE QUANT TRACKER ------------------
def trend_tracker(frames):
    """Descriptive multi-timeframe state tracker; probabilities are deliberately not fabricated."""
    out = {}
    for name in ("H1", "M15", "M5", "M1"):
        df = frames[name]
        q = quant_snapshot(df)
        close = df["close"].astype(float)
        ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        r = ret.tail(50)
        autocorr = safe_float(r.autocorr(lag=1)) if len(r) > 5 else 0.0
        # A descriptive strength score, not a win probability.
        trend_sign = q["trend"]
        mom_sign = 1 if q["momentum"] > 0 else (-1 if q["momentum"] < 0 else 0)
        align = 1.0 if trend_sign and trend_sign == mom_sign else (0.5 if trend_sign == 0 else 0.0)
        strength = int(round(100 * clamp(0.45*q["er"] + 0.30*min(abs(q["momentum"])/0.25, 1.0) + 0.25*align, 0, 1)))
        direction = "BULLISH" if trend_sign > 0 else ("BEARISH" if trend_sign < 0 else "NEUTRAL")
        state = "STRONG" if strength >= 70 else ("MODERATE" if strength >= 45 else "WEAK / MIXED")
        out[name] = {"direction": direction, "strength": strength, "state": state,
                     "regime": q["regime"], "er": q["er"], "momentum": q["momentum"],
                     "vol_ratio": q["vol_ratio"], "autocorr": autocorr, "price": q["price"]}
    dirs = [out[x]["direction"] for x in ("H1", "M15", "M5")]
    if dirs[0] == dirs[1] == dirs[2] and dirs[0] != "NEUTRAL":
        alignment = "ALIGNED"
    elif len(set(dirs)) > 1:
        alignment = "MIXED / PULLBACK POSSIBLE"
    else:
        alignment = "NEUTRAL"
    out["alignment"] = alignment
    return out


def quant_math_diagnostics(df):
    close = df["close"].astype(float)
    ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna().tail(50)
    if len(ret) < 10:
        return {"log_return": 0.0, "realized_vol": 0.0, "autocorr": 0.0, "up_frequency": 0.5}
    return {"log_return": safe_float(ret.iloc[-1]),
            "realized_vol": safe_float(ret.std()),
            "autocorr": safe_float(ret.autocorr(lag=1)),
            "up_frequency": safe_float((ret > 0).mean())}



# --------------------- PHYSICS-INSPIRED QUANT MATH ---------------------
def physics_quant_diagnostics(df):
    """
    Physics-inspired statistical features for research, not physical laws
    applied literally to markets. Uses only numpy/pandas already required.
    """
    close = pd.to_numeric(df["close"], errors="coerce").dropna().astype(float)
    close = close[close > 0]
    if len(close) < 40:
        return {
            "status": "INSUFFICIENT DATA", "realized_vol": 0.0,
            "vol_ratio": 0.0, "efficiency": 0.0, "momentum_short": 0.0,
            "momentum_long": 0.0, "autocorr": 0.0, "hurst_proxy": 0.5,
            "wavelet_proxy": 0.0, "change_score": 0.0, "direction_persistence": 0.5
        }

    logret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    r = logret.to_numpy(dtype=float)
    recent = r[-40:]
    long = r[-120:] if len(r) >= 120 else r
    eps = 1e-12

    # 1) Diffusion/realized-volatility proxy
    rv = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
    old = r[-120:-40] if len(r) >= 120 else r[:-40]
    old_vol = float(np.std(old, ddof=1)) if len(old) > 1 else rv
    vol_ratio = rv / max(old_vol, eps)

    # 2) Multi-scale momentum and path efficiency
    momentum_short = float(np.sum(r[-5:]))
    momentum_long = float(np.sum(r[-20:]))
    path = float(np.sum(np.abs(r[-20:])))
    efficiency = abs(momentum_long) / max(path, eps)
    direction_persistence = float(np.mean(np.sign(r[-20:]) == np.sign(np.sum(r[-20:]))))
    if momentum_long == 0:
        direction_persistence = 0.5

    # 3) Lag-one autocorrelation: descriptive only
    autocorr = float(pd.Series(recent).autocorr(lag=1)) if len(recent) > 5 else 0.0
    if not np.isfinite(autocorr):
        autocorr = 0.0

    # 4) Hurst-like scaling proxy using variance of aggregated returns.
    # This is deliberately labelled a proxy, not a formal Hurst estimate.
    scales = [1, 2, 4, 8]
    scale_points = []
    for scale in scales:
        n = (len(long) // scale) * scale
        if n < scale * 4:
            continue
        agg = long[-n:].reshape(-1, scale).sum(axis=1)
        sd = float(np.std(agg, ddof=1)) if len(agg) > 1 else 0.0
        if sd > eps:
            scale_points.append((math.log(scale), math.log(sd)))
    hurst_proxy = 0.5
    if len(scale_points) >= 2:
        slope = np.polyfit([p[0] for p in scale_points], [p[1] for p in scale_points], 1)[0]
        hurst_proxy = float(clamp(slope, 0.0, 1.0))

    # 5) Multi-scale signal decomposition proxy: fast vs slow return energy.
    fast_energy = float(np.mean(np.square(r[-5:]))) if len(r) >= 5 else 0.0
    slow_energy = float(np.mean(np.square(r[-20:]))) if len(r) >= 20 else fast_energy
    wavelet_proxy = math.sqrt(max(fast_energy, 0.0)) / max(math.sqrt(max(slow_energy, 0.0)), eps)

    # 6) Change-point proxy: difference between recent and prior means/volatility.
    prior = r[-40:-20] if len(r) >= 40 else r[:-10]
    latest = r[-20:]
    mean_shift = abs(float(np.mean(latest)) - float(np.mean(prior))) if len(prior) else 0.0
    pooled_vol = max(float(np.std(np.concatenate([prior, latest]), ddof=1)), eps)
    vol_shift = abs(float(np.std(latest, ddof=1)) - float(np.std(prior, ddof=1))) if len(prior) > 1 else 0.0
    change_score = float(clamp(0.5 * mean_shift / pooled_vol + 0.5 * vol_shift / pooled_vol, 0.0, 5.0))

    return {
        "status": "OK",
        "realized_vol": rv,
        "vol_ratio": float(clamp(vol_ratio, 0.0, 99.0)),
        "efficiency": float(clamp(efficiency, 0.0, 1.0)),
        "momentum_short": momentum_short,
        "momentum_long": momentum_long,
        "autocorr": float(clamp(autocorr, -1.0, 1.0)),
        "hurst_proxy": hurst_proxy,
        "wavelet_proxy": float(clamp(wavelet_proxy, 0.0, 99.0)),
        "change_score": change_score,
        "direction_persistence": float(clamp(direction_persistence, 0.0, 1.0))
    }


def print_physics_quant_lab(df):
    """Print tidy physics-inspired quantitative diagnostics with clear dividers."""
    d = physics_quant_diagnostics(df)
    line = "─" * 76
    print("🧪  PHYSICS-INSPIRED QUANT MATH LAB")
    print(line)
    print("[1] STOCHASTIC / DIFFUSION PROXIES")
    print("    Realized volatility : std(log-return)")
    print(f"    Realized vol       : {d['realized_vol']:.8f}")
    print(f"    Volatility ratio   : {d['vol_ratio']:.3f}x")
    print(line)
    print("[2] NONLINEAR DYNAMICS / TREND PERSISTENCE")
    print("    Efficiency          : |Σr| / Σ|r|")
    print(f"    Path efficiency    : {d['efficiency']:.3f}")
    print(f"    Direction persistence: {d['direction_persistence']:.3f}")
    print(f"    Lag-1 autocorr     : {d['autocorr']:+.3f}")
    print(f"    Hurst-like proxy   : {d['hurst_proxy']:.3f}")
    print(line)
    print("[3] MULTI-SCALE SIGNAL PROCESSING")
    print("    Fast/slow return-energy ratio; proxy only, not a formal wavelet transform.")
    print(f"    Short momentum Σr5 : {d['momentum_short']:+.6f}")
    print(f"    Long momentum Σr20 : {d['momentum_long']:+.6f}")
    print(f"    Fast/slow energy   : {d['wavelet_proxy']:.3f}x")
    print(line)
    print("[4] CHANGE-POINT / REGIME SHIFT PROXY")
    print(f"    Change score       : {d['change_score']:.3f}")
    print("    Higher values mean recent mean/volatility differs more from prior window.")
    print(line)
    print("[5] RESEARCH INTERPRETATION")
    if d["status"] != "OK":
        print("    Insufficient observations for stable diagnostics.")
    else:
        if d["change_score"] >= 1.5:
            print("    Regime-shift watch: recent distribution changed; reassess current trend.")
        else:
            print("    No strong shift flagged by this simple diagnostic.")
        if d["efficiency"] >= 0.55 and abs(d["momentum_long"]) > 0:
            print("    Price path is relatively directional over the measured window.")
        elif d["efficiency"] <= 0.25:
            print("    Price path is relatively noisy / mean-reverting over the measured window.")
        else:
            print("    Directional persistence is mixed.")
    print("    These are descriptive features, NOT future-price certainty or win probabilities.")
    print(line)


# ----------------------------- REPORT / LOOP ---------------------------
def print_open_positions():
    """Display journaled open paper positions with their original fixed levels."""
    df = read_journal()
    if df.empty or "status" not in df.columns:
        print("    No open paper position.")
        return
    opened = df[df.status.astype(str).str.upper().eq("OPEN")]
    if opened.empty:
        print("    No open paper position.")
        return
    for _, row in opened.iterrows():
        print(f"    ID: {row.get('id', 'N/A')} | {str(row.get('direction', 'N/A')).upper()} | "
              f"Entry: {safe_float(row.get('entry')):.3f} | "
              f"SL: {safe_float(row.get('sl')):.3f} | TP: {safe_float(row.get('tp')):.3f} | "
              f"Score at entry: {safe_float(row.get('score')):.0f}/100")
    print("    Note: level posisi aktif berasal dari jurnal; tidak mengikuti kandidat scan terbaru.")


# ---------------------- V17 DUAL-ENGINE ARCHITECTURE -------------------
ADAPTIVE_DIRECTION_MIN_SCORE = 62
ADAPTIVE_DIRECTION_MIN_GAP = 10
CANDIDATE_QUALITY_MIN_SCORE = MIN_SCORE
CANDIDATE_QUALITY_MIN_GAP = MIN_SCORE_GAP

def adaptive_signal_engine(frames):
    """Independent directional engine. It does not read Candidate score/direction."""
    weights = {"H1": 30, "M15": 25, "M5": 30, "M1": 15}
    side_points = {"BUY": 0.0, "SELL": 0.0}
    evidence = {"BUY": [], "SELL": []}
    snapshots = {tf: quant_snapshot(frames[tf]) for tf in ("H1", "M15", "M5", "M1")}
    for tf, q in snapshots.items():
        w = weights[tf]
        if q["trend"] == 1:
            side_points["BUY"] += w * 0.55; evidence["BUY"].append(f"{tf} trend bullish")
        elif q["trend"] == -1:
            side_points["SELL"] += w * 0.55; evidence["SELL"].append(f"{tf} trend bearish")
        if q["momentum"] > 0:
            side_points["BUY"] += w * 0.20
        elif q["momentum"] < 0:
            side_points["SELL"] += w * 0.20
        if q["breakout"] == 1:
            side_points["BUY"] += w * 0.15; evidence["BUY"].append(f"{tf} breakout bullish")
        elif q["breakout"] == -1:
            side_points["SELL"] += w * 0.15; evidence["SELL"].append(f"{tf} breakout bearish")
        if q["rsi"] >= 55:
            side_points["BUY"] += w * 0.10
        elif q["rsi"] <= 45:
            side_points["SELL"] += w * 0.10

    # Convert weighted evidence into a comparable 0-100 directional score.
    buy_score = int(round(clamp(side_points["BUY"], 0, 100)))
    sell_score = int(round(clamp(side_points["SELL"], 0, 100)))
    direction = "BUY" if buy_score >= sell_score else "SELL"
    top = max(buy_score, sell_score); gap = abs(buy_score - sell_score)
    q5 = snapshots["M5"]
    physics = physics_quant_diagnostics(frames["M5"])
    regime, mode = adaptive_regime(q5, physics)
    abstain = (regime == "HIGH_VOLATILITY" and q5["vol_ratio"] >= 3.5)
    reasons = []
    if top < ADAPTIVE_DIRECTION_MIN_SCORE: reasons.append("directional score below threshold")
    if gap < ADAPTIVE_DIRECTION_MIN_GAP: reasons.append("directional separation too small")
    if abstain: reasons.append("extreme volatility abstention")
    if regime == "TRANSITION": reasons.append("market regime transition")
    status = direction if not reasons else "WAIT"
    return {
        "direction": status, "raw_direction": direction, "buy_score": buy_score,
        "sell_score": sell_score, "score": top, "score_gap": gap,
        "regime": regime, "mode": mode, "abstain": abstain,
        "reasons": reasons, "evidence": evidence[direction],
        "snapshots": snapshots
    }

def candidate_quality_engine(candidate):
    """Candidate quality is assessed independently of Adaptive direction."""
    qualified = (candidate["score"] >= CANDIDATE_QUALITY_MIN_SCORE and
                 candidate["score_gap"] >= CANDIDATE_QUALITY_MIN_GAP and
                 candidate["rr"] >= MIN_RR and candidate["trigger"] >= 2)
    return {"qualified": bool(qualified), "direction": candidate["direction"],
            "score": candidate["score"], "score_gap": candidate["score_gap"],
            "reasons": [] if qualified else ["candidate quality/trigger/RR threshold not met"]}

def signal_fusion_engine(candidate_result, adaptive_result):
    """Fusion never deletes a qualified Candidate when Adaptive returns WAIT."""
    cdir = candidate_result["direction"]
    adir = adaptive_result["direction"]
    if not candidate_result["qualified"]:
        status = "CANDIDATE_WATCHLIST"
        detail = "Candidate belum memenuhi quality gate."
    elif adir == "WAIT":
        status = "CANDIDATE_RETAINED_ADAPTIVE_WAIT"
        detail = "Candidate berkualitas dipertahankan; Adaptive belum memberi konfirmasi arah."
    elif adir == cdir:
        status = "STRONG_BUY_CONFIRMATION" if cdir == "BUY" else "STRONG_SELL_CONFIRMATION"
        detail = "Candidate dan Adaptive searah."
    else:
        status = "DIRECTION_CONFLICT"
        detail = "Candidate dan Adaptive berlawanan; jangan eksekusi."
    return {"status": status, "detail": detail, "candidate_direction": cdir,
            "adaptive_direction": adir, "qualified": candidate_result["qualified"],
            "confirmed": bool(candidate_result["qualified"] and adir == cdir)}

def execution_readiness(candidate, fusion, prob, uncertainty):
    """Separate execution checks; paper trade requires two-engine confirmation."""
    reasons = []
    if not fusion["qualified"]: reasons.append("candidate quality gate failed")
    if not fusion["confirmed"]: reasons.append("dual-engine directional confirmation absent")
    if candidate["rr"] < MIN_RR: reasons.append(f"RR below {MIN_RR:.2f}")
    if candidate["trigger"] < 2: reasons.append("M1 trigger incomplete")
    if ML_HARD_FILTER and prob is not None and (prob < ML_MIN_PROBABILITY or
        (uncertainty is not None and uncertainty > ML_MAX_UNCERTAINTY)):
        reasons.append("ML hard filter rejected setup")
    return {"ready": not reasons, "reasons": reasons,
            "status": "PAPER_READY" if not reasons else "NOT_READY"}


def print_report(c, prob, uncertainty, model, stats, tracker=None):
    q = c["q"]
    line = "━" * 76
    print("\n" + line)
    print("🚀  JUMP STREET TRADING  |  JST APEX NEXUS V18.0")
    print("🕒  ", local_stamp(), " |  XAU/USD  |  SIGNAL + PAPER RESEARCH")
    print(line)
    print("📡  SYSTEM STATUS")
    print(f"    Data feed: Twelve Data OHLC  |  Scan interval: {SCAN_SECONDS}s")
    print(f"    Timeframes: H1 context → M15 structure → M5 primary → M1 timing")
    print(f"    Risk baseline: {RISK_PERCENT:.2f}%  |  Daily loss limit: {DAILY_LOSS_LIMIT_PERCENT:.2f}%")
    print("─" * 76)
    print("📊  MARKET SNAPSHOT")
    print(f"    Candle close : {c['entry']:.3f}       Regime : {c['regime']}")
    print(f"    RSI (14)     : {q['rsi']:.2f}          Robust Z : {q['z']:+.3f}")
    print(f"    Efficiency R.: {q['er']:.3f}          Vol ratio: {q['vol_ratio']:.3f}")
    print(f"    M5 momentum : {q['momentum']:+.4f}%       ATR : {q['atr']:.4f}")
    print(f"    EMA20 / 50  : {q['ema20']:.3f} / {q['ema50']:.3f}")
    print(f"    EMA200      : {q['ema200']:.3f}")
    print("─" * 76)
    print("📩  SIGNAL ENGINE")
    print(f"    Candidate : {c['direction']}     Score : {c['score']}/100     Score gap : {c['score_gap']}")
    print(f"    Candidate preview (not an open trade): Entry {c['entry']:.3f} | "
          f"SL {c['sl']:.3f} | TP {c['tp']:.3f} | RR {c['rr']:.2f}R")
    print(f"    Structure : {c['structure']}     Zone : {c['zone']}")
    print(f"    Liquidity : {c['liquidity']}     Engulfing : {c['engulfing']}     Math bias : {c['math_bias']}")
    print("    Score factors:")
    for reason in c["reasons"]:
        print(f"      • {reason}")
    print("─" * 76)
    print("📂  OPEN PAPER POSITION — FIXED LEVELS")
    print_open_positions()
    print("━" * 76)
    print("🧮  QUANT MATH LAB")
    print("─" * 76)
    print("[1] ROBUST STATISTICS | M5")
    print("    Log return : rₜ = ln(Pₜ / Pₜ₋₁)")
    print("    Robust Z   : (x − median) / (1.4826 × MAD)")
    print("    ER         : |Pₜ − Pₜ₋ₙ| / Σ|ΔP|")
    print(f"    Z={q['z']:+.4f} | ER={q['er']:.4f} | ATR={q['atr']:.4f} | VolRatio={q['vol_ratio']:.4f}")
    print("─" * 76)
    print("[2] CONTINUOUS TREND TRACKER | H1 → M15 → M5 → M1")
    if tracker:
        for tf in ("H1", "M15", "M5", "M1"):
            t = tracker[tf]
            print(f"    {tf:<3} | {t['direction']:<7} | strength={t['strength']:>3}/100 | {t['state']:<13} | regime={t['regime']}")
        print(f"    Alignment: {tracker['alignment']}")
        print("    Strength is a descriptive score, NOT a probability of winning.")
    print("─" * 76)
    print("[3] PROBABILITY & UNCERTAINTY")
    print("    TP-before-SL probability: N/A until a properly labeled, calibrated model is trained.")
    print("    Bayesian updates / confidence intervals require sufficient chronological observations.")
    print("─" * 76)
    print("[4] EDGE & MODEL ADAPTATION")
    print(f"    Edge research: {research_report().get('status')} | ML: {model.status}")
    print("    Model promotion requires walk-forward and independent out-of-sample validation.")
    print("━" * 76)
    print("🗂  PAPER JOURNAL & RESEARCH")
    if stats["n"]:
        lo, hi = wilson_ci(round(stats["win_rate"]*stats["n"]), stats["n"])
        outcomes = pd.to_numeric(
            read_journal().loc[lambda x: x.status.astype(str).str.upper().eq("CLOSED"), "outcome_r"],
            errors="coerce").dropna().to_numpy()
        elo, ehi = bootstrap_ci(outcomes)
        print(f"    Closed trades : {stats['n']}     Win rate : {stats['win_rate']:.2%}")
        print(f"    Expectancy    : {stats['expectancy']:+.3f}R   Profit factor : {stats['pf']:.3f}")
        print(f"    Max drawdown  : {stats['max_dd']:.3f}R")
        if lo is not None:
            print(f"    Win-rate Wilson 95% CI : {lo:.2%} – {hi:.2%}")
        if elo is not None:
            print(f"    Expectancy bootstrap 95% CI : {elo:+.3f}R – {ehi:+.3f}R")
    else:
        print("    Closed trades : 0     Win rate : N/A     Expectancy : N/A")
        print("    Profit factor : N/A   Max drawdown : N/A")
    print(f"    Edge research : {research_report().get('status')}")
    print(f"    ML engine     : {model.status}")
    if prob is not None:
        print(f"    ML advisory   : {prob:.3f}     Tree disagreement : {uncertainty:.3f}")
    print("─" * 76)
    print("🛡  DECISION POLICY")
    print(f"    Minimum score : {MIN_SCORE}/100     Minimum RR : {MIN_RR:.2f}R")
    print(f"    Estimated cost assumption : {ESTIMATED_COST_R:.2f}R per trade")
    print("    Execution mode : PAPER ONLY  |  Broker order : OFF")
    print(line)
    print(line + "\n")


def run_once(model, state):
    frames = {name: fetch_candles(name, interval, BARS[name])
              for name, interval in TIMEFRAMES.items()}
    manage_open_paper_trades(frames)
    model.train_if_changed()
    stats = closed_stats()

    # Engine 1: Candidate discovery and quality scoring.
    c = make_candidate(frames)
    candidate_result = candidate_quality_engine(c)

    # Engine 2: independent adaptive directional assessment.
    adaptive_result = adaptive_signal_engine(frames)

    # Fusion + execution readiness are separate layers.
    fusion = signal_fusion_engine(candidate_result, adaptive_result)
    prob, uncertainty = model.predict(c["features"])
    readiness = execution_readiness(c, fusion, prob, uncertainty)
    tracker = trend_tracker(frames)
    print_report(c, prob, uncertainty, model, stats, tracker)
    print_physics_quant_lab(frames["M5"])
    print_advanced_research_labs(frames["M5"])

    # V19 ML research is trained only from closed, journaled outcomes.
    ml_research_ensemble.train_if_changed()
    ml_research_report = ml_research_ensemble.report()
    lstm_trade_context.train_if_changed()
    dl_prediction = lstm_trade_context.predict(c["features"])

    # Optional second-symbol fetch; disabled by default to protect API quota.
    stat_arb_report = {"status": "DISABLED"}
    if STAT_ARB_ENABLED:
        try:
            ref_df = fetch_symbol_candles(STAT_ARB_REFERENCE_SYMBOL)
            stat_arb_report = statistical_arbitrage_research(frames["M5"], ref_df)
        except Exception as exc:
            safe_error = str(exc).replace(API_KEY, "[REDACTED]") if API_KEY else str(exc)
            stat_arb_report = {"status": "FETCH_ERROR", "error": safe_error[:300]}
    regime_report = regime_classification_lab(frames["M5"] )
    print_ml_stat_arb_report(ml_research_report, stat_arb_report, regime_report)
    print(f"Deep Learning LSTM: {dl_prediction.get('status')} | probability={dl_prediction.get('probability')} | holdout={lstm_trade_context.metrics}")

    print("\n" + "━" * 76)
    print("🧭 V19 CANDIDATE SIGNAL ENGINE")
    print(f"Candidate: {c['direction']} | Quality score: {c['score']}/100 | Gap: {c['score_gap']}")
    print(f"Quality status: {'QUALIFIED' if candidate_result['qualified'] else 'WATCHLIST'}")
    print("Candidate tetap dinilai independen dari Adaptive Engine.")
    print("─" * 76)
    print("🧠 V19 ADAPTIVE SIGNAL ENGINE — INDEPENDENT")
    print(f"Adaptive: {adaptive_result['direction']} | Raw direction: {adaptive_result['raw_direction']}")
    print(f"BUY score: {adaptive_result['buy_score']} | SELL score: {adaptive_result['sell_score']} | Gap: {adaptive_result['score_gap']}")
    print(f"Regime: {adaptive_result['regime']} | Mode: {adaptive_result['mode']}")
    print("Reasons: " + ("; ".join(adaptive_result['reasons']) if adaptive_result['reasons'] else "directional conditions passed"))
    print("─" * 76)
    print("🔀 SIGNAL FUSION ENGINE")
    print(f"Fusion status: {fusion['status']}")
    print(f"Candidate={fusion['candidate_direction']} | Adaptive={fusion['adaptive_direction']}")
    print(f"Interpretation: {fusion['detail']}")
    print("─" * 76)
    print("🛡 EXECUTION READINESS")
    print(f"Status: {readiness['status']} | Paper execution: {'ELIGIBLE' if readiness['ready'] else 'BLOCKED'}")
    if readiness["reasons"]:
        for reason in readiness["reasons"]: print(" • " + reason)
    bayes = bayesian_win_rate_summary()
    print(f"Bayesian historical win-rate summary: Beta({bayes['alpha']:.0f}, {bayes['beta']:.0f}), mean={bayes['mean']:.3f}, n={bayes['n']} (descriptive only)")

    if state.get("last_candidate_key") == c["key"]:
        print("🔄 DECISION: SKIP — candle M1 ini sudah dievaluasi.")
        return

    # Account-level risk and operational guards apply only to paper execution.
    operational_reasons = []
    daily_limit_r = DAILY_LOSS_LIMIT_PERCENT / max(RISK_PERCENT, 1e-9)
    if daily_realized_r() <= -daily_limit_r: operational_reasons.append("daily loss limit reached")
    if open_count() >= MAX_PAPER_OPEN: operational_reasons.append("paper position still open")
    today = utc_now().date().isoformat()
    if state.get("daily_date") != today:
        state["daily_date"], state["daily_signal_count"] = today, 0
    if int(state.get("daily_signal_count", 0)) >= MAX_PAPER_SIGNALS_PER_DAY:
        operational_reasons.append("daily paper signal limit reached")
    last_time = pd.to_datetime(state.get("last_signal_time", ""), errors="coerce", utc=True)
    if not pd.isna(last_time):
        elapsed = (utc_now() - last_time.to_pydatetime()).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES:
            operational_reasons.append(f"cooldown active ({elapsed:.1f}/{COOLDOWN_MINUTES} min)")

    # Log Candidate even when Adaptive is WAIT or conflicts. This is the key behavior.
    if fusion["status"] == "CANDIDATE_RETAINED_ADAPTIVE_WAIT":
        decision = "CANDIDATE_RETAINED_ADAPTIVE_WAIT"
    elif fusion["status"] == "DIRECTION_CONFLICT":
        decision = "DIRECTION_CONFLICT"
    elif not candidate_result["qualified"]:
        decision = "WATCHLIST"
    elif not readiness["ready"] or operational_reasons:
        decision = "NOT_READY"
    else:
        decision = "PAPER_CANDIDATE"

    reason_text = (fusion["detail"] + "; " + "; ".join(readiness["reasons"] + operational_reasons))
    adaptive_note = (f"adaptive={adaptive_result['direction']}; raw={adaptive_result['raw_direction']}; "
                     f"buy={adaptive_result['buy_score']}; sell={adaptive_result['sell_score']}; "
                     f"regime={adaptive_result['regime']}")
    register_candidate(c, prob, uncertainty, decision, adaptive_note + "; " + reason_text, stats)
    state["last_candidate_key"] = c["key"]
    save_state(state)

    if operational_reasons:
        print("⛔ PAPER EXECUTION BLOCKED — " + "; ".join(operational_reasons))
    elif decision == "PAPER_CANDIDATE" and PAPER_MODE:
        register_paper_trade(c, prob)
        state["last_signal_time"] = stamp()
        state["daily_signal_count"] = int(state.get("daily_signal_count", 0)) + 1
        save_state(state)
        print("📩 PAPER SIGNAL — dua engine searah; jurnal simulasi diperbarui.")
    elif decision == "CANDIDATE_RETAINED_ADAPTIVE_WAIT":
        print("🟡 CANDIDATE RETAINED — Adaptive WAIT tidak menghapus Candidate. Belum paper entry.")
    elif decision == "DIRECTION_CONFLICT":
        print("⚠️ DIRECTION CONFLICT — kandidat tetap dicatat, paper entry diblokir.")
    else:
        print("⏸️ NO PAPER ENTRY — " + decision + "; " + reason_text)




# =====================================================================
# V18 QUANTITATIVE MATHEMATICS + PHYSICS-INFORMED RESEARCH LAB
# Research diagnostics only. No method below is an automatic trade signal.
# Pure NumPy/Pandas implementation to remain Android/Pydroid friendly.
# =====================================================================

def _clean_returns(df, min_n=30):
    close = pd.to_numeric(df.get("close", pd.Series(dtype=float)), errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0].to_numpy(dtype=float)
    if len(close) < min_n + 1:
        return np.array([], dtype=float), close
    r = np.diff(np.log(close))
    return r[np.isfinite(r)], close


def _safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) != len(a):
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    v = float(np.corrcoef(a, b)[0, 1])
    return v if np.isfinite(v) else 0.0


def _ols_fit(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.column_stack([np.ones(len(x)), x])
    if len(y) <= x.shape[1] + 2:
        return None
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(1, len(y) - x.shape[1])
    sigma2 = float(np.sum(resid**2) / max(dof * (resid.shape[1] if resid.ndim > 1 else 1), 1))
    try:
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(x.shape[1], np.nan)
    return {"beta": beta, "resid": resid, "r2": 1 - float(np.sum(resid**2)) /
            max(float(np.sum((y-y.mean())**2)), 1e-12), "se": se}


def _entropy_from_values(values, bins=12):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.0
    counts, _ = np.histogram(x, bins=min(bins, max(3, int(np.sqrt(len(x))))))
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def _sample_entropy(x, m=2, tolerance=None):
    x = np.asarray(x, float)
    if len(x) < m + 8:
        return None
    sd = float(np.std(x))
    r = tolerance if tolerance is not None else 0.2 * sd
    if r <= 0:
        return None
    def count_matches(mm):
        templates = np.array([x[i:i+mm] for i in range(len(x)-mm+1)])
        count = 0
        for i in range(len(templates)):
            if i + 1 < len(templates):
                dist = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
                count += int(np.sum(dist <= r))
        return count
    a, b = count_matches(m+1), count_matches(m)
    if a <= 0 or b <= 0:
        return None
    return float(-np.log(a / b))


def _haar_dwt_energy(x, levels=3):
    a = np.asarray(x, float)
    energies = []
    for _ in range(levels):
        if len(a) < 4:
            break
        n = len(a) // 2 * 2
        even, odd = a[:n:2], a[1:n:2]
        detail = (even - odd) / np.sqrt(2.0)
        a = (even + odd) / np.sqrt(2.0)
        energies.append(float(np.mean(detail**2)))
    return energies



def _cwt_ricker_energy(x, scales=(2, 4, 8, 16)):
    """Sampled Ricker continuous-wavelet-style transform across scales."""
    x = np.asarray(x, float)
    if len(x) < 30:
        return []
    x = x - np.mean(x)
    out = []
    for scale in scales:
        radius = int(max(4, 4*scale))
        t = np.arange(-radius, radius+1, dtype=float)
        z = t / float(scale)
        wave = (1-z*z) * np.exp(-0.5*z*z)
        wave -= wave.mean()
        wave /= max(np.sqrt(np.sum(wave*wave)), 1e-12)
        conv = np.convolve(x, wave[::-1], mode="same")
        out.append({"scale": int(scale), "mean_squared_coefficient": float(np.mean(conv*conv))})
    return out


def _higuchi_fd(x, kmax=8):
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return None
    ks, lengths = [], []
    for k in range(1, min(kmax, n//5)+1):
        lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            length = np.sum(np.abs(np.diff(x[idx]))) * (n-1) / (len(idx)*k*k)
            lk.append(length)
        if lk and np.mean(lk) > 0:
            ks.append(np.log(1.0/k))
            lengths.append(np.log(np.mean(lk)))
    if len(ks) < 3:
        return None
    return float(np.polyfit(ks, lengths, 1)[0])


def _lyapunov_rosenstein_proxy(x, emb_dim=3, lag=2, horizon=5):
    """Small-sample Rosenstein-style divergence slope; highly noise-sensitive."""
    x = np.asarray(x, float)
    if len(x) < 80:
        return None
    x = (x - x.mean()) / max(x.std(), 1e-12)
    n = len(x) - (emb_dim-1)*lag - horizon
    if n < 30:
        return None
    emb = np.array([x[i:i+emb_dim*lag:lag] for i in range(n)])
    nn = np.full(n, -1, dtype=int)
    for i in range(n):
        d = np.max(np.abs(emb - emb[i]), axis=1)
        d[max(0, i-10):min(n, i+11)] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] > 1e-9:
            nn[i] = j
    curves = []
    for h in range(1, horizon+1):
        ds = []
        for i, j in enumerate(nn):
            if j >= 0 and i+h < len(x) and j+h < len(x):
                d = abs(x[i+h]-x[j+h])
                if d > 1e-12:
                    ds.append(np.log(d))
        curves.append(float(np.mean(ds)) if ds else np.nan)
    valid = np.isfinite(curves)
    if valid.sum() < 3:
        return None
    return float(np.polyfit(np.arange(1, horizon+1)[valid], np.array(curves)[valid], 1)[0])


def _kalman_local_level(x, q_ratio=0.02):
    x = np.asarray(x, float)
    if len(x) < 10:
        return None
    obs_var = max(float(np.var(np.diff(x))), 1e-12)
    q = q_ratio * obs_var
    state, variance = float(x[0]), obs_var
    filtered = []
    for z in x:
        variance += q
        gain = variance / (variance + obs_var)
        state += gain * (z - state)
        variance *= (1-gain)
        filtered.append(state)
    return {"level": float(filtered[-1]), "slope": float(filtered[-1]-filtered[-5]) if len(filtered)>=5 else 0.0,
            "variance": float(variance)}


def _ar_fit(x, p=2):
    x = np.asarray(x, float)
    if len(x) <= p + 10:
        return None
    y = x[p:]
    xx = np.column_stack([x[p-j-1:len(x)-j-1] for j in range(p)])
    fit = _ols_fit(y, xx)
    if fit is None:
        return None
    pred = float(fit["beta"][0] + np.dot(fit["beta"][1:], x[-p:][::-1]))
    return {"prediction": pred, "coefficients": fit["beta"][1:].tolist(), "r2": fit["r2"]}


def _ma_fit(x, q=2):
    x = np.asarray(x, float)
    if len(x) <= q + 10:
        return None
    # Innovations approximation: fit AR(q), then regress current residual on lags.
    ar = _ar_fit(x, max(1, q))
    if ar is None:
        return None
    p = max(1, q)
    y = x[p:]
    xx = np.column_stack([x[p-j-1:len(x)-j-1] for j in range(p)])
    fit = _ols_fit(y, xx)
    if fit is None:
        return None
    resid = fit["resid"]
    if len(resid) <= q+2:
        return None
    pred = float(np.mean(resid[-q:]))
    return {"innovation_mean": pred, "residual_std": float(np.std(resid)), "note": "MA residual approximation"}


def _arima_diff_forecast(x, p=2, d=1):
    x = np.asarray(x, float)
    z = np.diff(x, n=d) if d > 0 else x.copy()
    fit = _ar_fit(z, p)
    if fit is None:
        return None
    pred_diff = fit["prediction"]
    forecast = float(x[-1] + pred_diff) if d == 1 else float(x[-1] + pred_diff)
    return {"forecast": forecast, "differenced_prediction": pred_diff, "r2": fit["r2"], "order": f"({p},{d},0)"}


def _var_fit(x, p=1):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 30 or x.shape[1] < 2:
        return None
    y = x[p:]
    xx = np.column_stack([x[p-j-1:len(x)-j-1] for j in range(p)])
    fit = _ols_fit(y, xx)
    if fit is None:
        return None
    pred = fit["beta"][0] + x[-1] @ fit["beta"][1:1+x.shape[1], :].T if p == 1 else fit["beta"][0]
    return {"prediction": np.asarray(pred).reshape(-1).tolist(), "r2": fit["r2"], "note": "VAR OLS; exploratory"}


def _garch11(x, omega=None, alpha=0.08, beta=0.88):
    """Gaussian quasi-maximum-likelihood grid estimate for GARCH(1,1)."""
    x = np.asarray(x, float)
    if len(x) < 40:
        return None
    variance = max(float(np.var(x)), 1e-12)
    best = None
    for a in (0.03, 0.06, 0.09, 0.12, 0.15, 0.18):
        for b in (0.65, 0.72, 0.78, 0.84, 0.90, 0.94):
            if a+b >= 0.995:
                continue
            om = max(variance*(1-a-b), 1e-14)
            h = np.full(len(x), variance)
            for t in range(1, len(x)):
                h[t] = max(om+a*x[t-1]**2+b*h[t-1], 1e-14)
            loss = float(np.mean(np.log(h)+x*x/h))
            if best is None or loss < best[0]:
                best = (loss, om, a, b, h)
    if best is None:
        return None
    loss, om, a, b, h = best
    return {"next_variance": float(om+a*x[-1]**2+b*h[-1]),
            "omega": float(om), "alpha": float(a), "beta": float(b),
            "persistence": float(a+b), "gaussian_quasi_loglik_loss": loss,
            "note": "coarse grid Gaussian QMLE; sample-sensitive"}


def _markov_regime(x):
    x = np.asarray(x, float)
    if len(x) < 30:
        return None
    # Three states from rolling absolute-return quantiles: low/medium/high volatility.
    scale = np.abs(x)
    q1, q2 = np.quantile(scale, [1/3, 2/3])
    states = np.where(scale <= q1, 0, np.where(scale <= q2, 1, 2))
    counts = np.ones((3,3), dtype=float)  # Laplace smoothing
    for a,b in zip(states[:-1], states[1:]):
        counts[int(a), int(b)] += 1
    trans = counts / counts.sum(axis=1, keepdims=True)
    return {"state": int(states[-1]), "transition": trans.tolist(),
            "next_state_prob": trans[int(states[-1])].tolist()}


def _brownian_gbm_simulation(last_price, mu, sigma, steps=20, paths=200, dt=1.0):
    rng = np.random.default_rng(7)  # reproducible research diagnostic
    z = rng.standard_normal((paths, steps))
    increments = (mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*z
    paths_arr = last_price*np.exp(np.cumsum(increments, axis=1))
    terminal = paths_arr[:, -1]
    return {"terminal_q05": float(np.quantile(terminal, .05)),
            "terminal_median": float(np.median(terminal)),
            "terminal_q95": float(np.quantile(terminal, .95)),
            "paths": int(paths), "steps": int(steps)}


def _numerical_optimization(x, y, steps=300, lr=0.03, l2=1e-4):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x.reshape(-1,1)
    if len(y) != len(x) or len(y) < 20:
        return None
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = (x-mu)/np.where(sd < 1e-12, 1, sd)
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(z.shape[1])
    for _ in range(steps):
        grad = (2/len(y)) * (z.T @ (z@w-y)) + 2*l2*w
        w -= lr*grad
    pred = z@w
    return {"weights": w.tolist(), "mse": float(np.mean((pred-y)**2)),
            "constraint": "L2 regularization; unconstrained least-squares objective"}



def _projected_nonnegative_regression(x, y, steps=400, lr=0.02, l2=1e-4):
    """Projected gradient descent with intercept free and slopes constrained >= 0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(y) != len(x) or len(y) < 20:
        return None
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = (x-mu)/np.where(sd < 1e-12, 1, sd)
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(z.shape[1])
    for _ in range(steps):
        grad = (2/len(y))*(z.T@(z@w-y)) + 2*l2*w
        w -= lr*grad
        w[1:] = np.maximum(w[1:], 0.0)
    return {"weights": w.tolist(), "mse": float(np.mean((z@w-y)**2)),
            "constraint": "slope coefficients constrained >= 0 by projection"}


def _tail_risk(x, confidence=.95, threshold_quantile=.90):
    x = np.asarray(x, float)
    if len(x) < 30:
        return None
    losses = -x
    var = float(np.quantile(losses, confidence))
    es = float(np.mean(losses[losses >= var])) if np.any(losses >= var) else var
    u = float(np.quantile(losses, threshold_quantile))
    excess = losses[losses > u] - u
    gpd_shape = None
    gpd_scale = None
    gpd_method = "insufficient exceedances"
    if len(excess) >= 10 and np.std(excess) > 1e-12:
        try:
            from scipy.stats import genpareto
            gpd_shape, _, gpd_scale = genpareto.fit(excess, floc=0)
            gpd_shape, gpd_scale = float(gpd_shape), float(gpd_scale)
            gpd_method = "SciPy GPD maximum likelihood"
        except Exception:
            mean, var_ex = float(np.mean(excess)), float(np.var(excess, ddof=1))
            if var_ex > mean*mean:
                gpd_shape = 0.5*(1-mean*mean/var_ex)
                gpd_scale = mean*(1-gpd_shape)
                gpd_method = "GPD method-of-moments fallback"
    blocks = np.array([np.max(losses[i:i+10]) for i in range(0, len(losses)-9, 10)])
    gev = None
    if len(blocks) >= 5:
        try:
            from scipy.stats import genextreme
            shape, loc, scale = genextreme.fit(blocks)
            gev = {"shape": float(-shape), "location": float(loc), "scale": float(scale),
                   "method": "SciPy GEV maximum likelihood"}
        except Exception:
            sd = float(np.std(blocks, ddof=1))
            gev = {"location": float(np.mean(blocks)-0.57721*sd*0.78),
                   "scale": float(sd*0.78), "shape": 0.0,
                   "method": "Gumbel moment fallback; not full GEV MLE"}
    return {"VaR": var, "Expected_Shortfall": es, "tail_threshold": u,
            "POT_exceedances": int(len(excess)), "GPD_shape": gpd_shape,
            "GPD_scale_mom": gpd_scale, "GEV": gev}


def quantitative_mathematics_lab(df, reference_df=None):
    """Compute requested quant methods on OHLC-derived returns; research only."""
    r, close = _clean_returns(df, 35)
    if len(r) < 30:
        return {"status": "INSUFFICIENT_DATA", "n": len(r)}
    n = len(r)
    # Calculus: finite differences, cumulative integral, and gradient of rolling objective.
    dr = np.gradient(r)
    integral = _trapezoid_compat(r)
    grad_energy = float(np.gradient(np.cumsum(r*r))[-1])
    # Linear algebra: covariance eigensystem + PCA over lagged return/vol/momentum features.
    mat = np.column_stack([r[2:], r[1:-1], r[:-2]])
    cov = np.cov(mat, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    pca_ratio = float(eigval[0]/max(np.sum(eigval), 1e-15))
    # Probability: conditional empirical probabilities and Bayesian beta posterior.
    up = r > 0
    prev_up = up[:-1]
    p_up = float(np.mean(up))
    p_up_after_up = float(np.mean(up[1:][prev_up])) if np.any(prev_up) else None
    wins_proxy = int(np.sum(r > 0))
    losses_proxy = int(np.sum(r <= 0))
    bayes_mean = (wins_proxy+1)/(wins_proxy+losses_proxy+2)
    # Distribution/hypothesis: standardized moments, Jarque-Bera-style statistic.
    sd = max(float(np.std(r, ddof=1)), 1e-12)
    z = (r-r.mean())/sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    jb = float(n/6*(skew**2 + (kurt-3)**2/4))
    # Time series: AR, MA innovation approximation, ARIMA(2,1,0), VAR on return and abs return.
    ar = _ar_fit(r, 2)
    ma = _ma_fit(r, 2)
    arima = _arima_diff_forecast(r, 2, 1)
    var = _var_fit(np.column_stack([r, np.abs(r)]), 1)
    # Econometrics: OLS return on lagged return and volatility; cointegration only if reference series supplied.
    ols = _ols_fit(r[1:], np.column_stack([r[:-1], np.abs(r[:-1])]))
    coint = None
    if reference_df is not None and "close" in reference_df:
        _, c2 = _clean_returns(reference_df, 35)
        m = min(len(close), len(c2))
        if m >= 50:
            levels = np.column_stack([np.log(close[-m:]), np.log(c2[-m:])])
            reg = _ols_fit(levels[:,0], levels[:,1])
            if reg is not None:
                resid = reg["resid"]
                coint = {"hedge_ratio": float(reg["beta"][1]),
                         "residual_std": float(np.std(resid)),
                         "note": "Engle-Granger first-stage residual diagnostic only; no formal ADF critical-value test"}
    garch = _garch11(r)
    # Stochastic processes: Markov volatility states and GBM/Brownian scenario.
    markov = _markov_regime(r)
    mu, sigma = float(np.mean(r)), sd
    brownian = _brownian_gbm_simulation(float(close[-1]), mu, sigma)
    # Stochastic calculus: GBM Ito drift correction and Euler-Maruyama SDE path.
    ito_log_drift = mu - 0.5*sigma*sigma
    dt = 1/60
    euler_mean = float(close[-1] + mu*close[-1]*dt)
    # Optimization: regularized gradient descent, plus projected non-negative weights.
    opt = _numerical_optimization(mat, r[2:])
    constrained = _projected_nonnegative_regression(mat, r[2:])
    # Information theory.
    entropy = _entropy_from_values(r)
    mi = None
    if n > 20:
        a = np.digitize(r[:-1], np.histogram_bin_edges(r[:-1], bins=8))
        b = np.digitize(r[1:], np.histogram_bin_edges(r[1:], bins=8))
        joint = np.zeros((9,9), float)
        for aa,bb in zip(a,b): joint[min(int(aa), 8), min(int(bb), 8)] += 1
        p = joint/joint.sum()
        pa, pb = p.sum(axis=1), p.sum(axis=0)
        mi = float(sum(p[i,j]*np.log2(p[i,j]/(pa[i]*pb[j]))
                       for i in range(9) for j in range(9) if p[i,j] > 0 and pa[i]*pb[j] > 0))
    # Risk and EVT.
    risk = _tail_risk(r)
    return {
        "status": "OK", "n": n,
        "calculus": {"last_numerical_derivative": float(dr[-1]), "integral_sum_log_return": integral,
                     "gradient_cumulative_squared_return": grad_energy},
        "linear_algebra": {"covariance_eigenvalues": eigval.tolist(),
                           "first_pc_variance_ratio": pca_ratio, "first_pc": eigvec[:,0].tolist()},
        "probability": {"P_up": p_up, "P_up_given_previous_up": p_up_after_up,
                        "bayesian_beta_posterior_mean_directional": bayes_mean},
        "statistics": {"mean": mu, "std": sd, "skewness": skew, "kurtosis": kurt,
                       "jarque_bera_statistic": jb, "jarque_bera_pvalue_chi2_approx": float(np.exp(-jb/2))},
        "time_series": {"AR2": ar, "MA2_innovation_approx": ma, "ARIMA210": arima, "VAR1": var},
        "econometrics": {"OLS_r2": ols["r2"] if ols else None,
                         "OLS_coefficients": ols["beta"].tolist() if ols else None,
                         "cointegration": coint, "GARCH11": garch},
        "stochastic": {"Markov_volatility_states": markov, "GBM_Brownian_scenario": brownian},
        "stochastic_calculus": {"ito_log_gbm_drift": ito_log_drift,
                                "euler_maruyama_one_step_mean": euler_mean,
                                "note": "GBM illustration; not calibrated pricing model"},
        "optimization": {"gradient_descent": opt, "constrained_projection": constrained},
        "information": {"shannon_entropy_bits": entropy, "lagged_mutual_information_bits": mi},
        "risk_extreme_value": risk,
    }


def physics_informed_research_lab(df):
    """Physics-inspired diagnostics; proxies are explicitly not proof of chaos."""
    r, close = _clean_returns(df, 40)
    if len(r) < 40:
        return {"status": "INSUFFICIENT_DATA", "n": len(r)}
    # Fourier spectrum and dominant non-zero frequency.
    centered = r - np.mean(r)
    spectrum = np.abs(np.fft.rfft(centered))**2
    freqs = np.fft.rfftfreq(len(centered), d=1.0)
    if len(spectrum) > 1:
        idx = int(np.argmax(spectrum[1:])+1)
        dominant = float(freqs[idx])
        power = float(spectrum[idx]/max(np.sum(spectrum[1:]), 1e-15))
    else:
        dominant, power = 0.0, 0.0
    wavelet_energy = _haar_dwt_energy(centered, 4)
    # Scaling/Hurst estimate from variance of aggregated returns.
    scales, vars_ = [], []
    for k in (2,4,8,16):
        if len(centered) >= k*5:
            agg = np.array([np.sum(centered[i:i+k]) for i in range(0, len(centered)-k+1, k)])
            if len(agg) > 3 and np.var(agg) > 0:
                scales.append(np.log(k)); vars_.append(np.log(np.var(agg)))
    hurst = float(np.polyfit(scales, vars_, 1)[0]/2) if len(scales)>=3 else None
    # State-space local-level Kalman filter.
    state_space = _kalman_local_level(np.log(close))
    return {
        "status": "OK", "n": len(r),
        "nonlinear_dynamics": {"lyapunov_divergence_slope_proxy": _lyapunov_rosenstein_proxy(centered),
                               "note": "finite-sample divergence proxy; not a robust proof of deterministic chaos"},
        "chaos_complexity": {"sample_entropy": _sample_entropy(centered),
                             "autocorrelation_lag1": _safe_corr(centered[1:], centered[:-1]),
                             "note": "complexity diagnostics, not a chaos verdict"},
        "statistical_mechanics": {"shannon_entropy_bits": _entropy_from_values(centered),
                                  "return_energy_mean_square": float(np.mean(centered**2)),
                                  "energy_concentration": float(np.max(wavelet_energy)/max(np.sum(wavelet_energy),1e-15)) if wavelet_energy else None},
        "signal_processing": {"fft_dominant_frequency_cycles_per_bar": dominant,
                              "fft_dominant_power_share": power, "power_spectrum": spectrum.tolist()},
        "wavelet": {"haar_dwt_detail_energy_by_level": wavelet_energy,
                    "ricker_cwt_energy_by_scale": _cwt_ricker_energy(centered),
                    "note": "Haar DWT plus sampled Ricker CWT-style transform"},
        "fractal": {"hurst_scaling_estimate": hurst, "higuchi_fractal_dimension": _higuchi_fd(centered)},
        "dynamical_systems": {"kalman_local_level": state_space,
                              "note": "local-level state-space filter; simplified model"},
    }


def print_advanced_research_labs(df):
    q = quantitative_mathematics_lab(df)
    p = physics_informed_research_lab(df)
    line = "─" * 76
    print("\n" + "━" * 76)
    print("🧮 ADVANCED QUANTITATIVE MATHEMATICS LAB — RESEARCH ONLY")
    print(line)
    if q.get("status") != "OK":
        print("Quant Math: insufficient observations.")
    else:
        print(f"Observations: {q['n']} | d(log-return): {q['calculus']['last_numerical_derivative']:+.6e} | integral: {q['calculus']['integral_sum_log_return']:+.6e}")
        print(f"Linear algebra: PC1 variance share={q['linear_algebra']['first_pc_variance_ratio']:.3f} | eigenvalues={np.round(q['linear_algebra']['covariance_eigenvalues'], 8)}")
        print(f"Probability: P(up)={q['probability']['P_up']:.3f} | P(up|up)={q['probability']['P_up_given_previous_up']}")
        print(f"Statistics: skew={q['statistics']['skewness']:+.3f} | kurtosis={q['statistics']['kurtosis']:.3f} | JB={q['statistics']['jarque_bera_statistic']:.2f}")
        ts = q["time_series"]
        print(f"Time series: AR2={ts['AR2']} | MA2={ts['MA2_innovation_approx']} | ARIMA={ts['ARIMA210']}")
        ec = q["econometrics"]
        print(f"Econometrics: OLS R²={ec['OLS_r2']} | GARCH11={ec['GARCH11']} | cointegration={ec['cointegration']}")
        print(f"Stochastic: Markov={q['stochastic']['Markov_volatility_states']} | GBM scenario={q['stochastic']['GBM_Brownian_scenario']}")
        print(f"Itô/SDE: GBM log drift correction={q['stochastic_calculus']['ito_log_gbm_drift']:+.6e} | Euler step mean={q['stochastic_calculus']['euler_maruyama_one_step_mean']:.5f}")
        print(f"Optimization: gradient descent MSE={q['optimization']['gradient_descent']['mse'] if q['optimization']['gradient_descent'] else None} | constrained={q['optimization']['constrained_projection']}")
        print(f"Information: Shannon H={q['information']['shannon_entropy_bits']:.3f} bits | lag MI={q['information']['lagged_mutual_information_bits']}")
        print(f"Risk/EVT: {q['risk_extreme_value']}")
    print(line)
    print("🧪 PHYSICS-INFORMED MATHEMATICAL RESEARCH LAB")
    if p.get("status") != "OK":
        print("Physics lab: insufficient observations.")
    else:
        print(f"Nonlinear dynamics: Lyapunov proxy={p['nonlinear_dynamics']['lyapunov_divergence_slope_proxy']}")
        print(f"Chaos/complexity: SampEn={p['chaos_complexity']['sample_entropy']} | AC1={p['chaos_complexity']['autocorrelation_lag1']:+.3f}")
        print(f"Statistical mechanics: entropy={p['statistical_mechanics']['shannon_entropy_bits']:.3f} bits | energy={p['statistical_mechanics']['return_energy_mean_square']:.6e}")
        print(f"Signal processing: FFT dominant frequency={p['signal_processing']['fft_dominant_frequency_cycles_per_bar']:.4f} cycles/bar | power share={p['signal_processing']['fft_dominant_power_share']:.3f}")
        print(f"Wavelet: Haar DWT={p['wavelet']['haar_dwt_detail_energy_by_level']} | Ricker CWT-style={p['wavelet']['ricker_cwt_energy_by_scale']}")
        print(f"Fractal: Hurst scaling={p['fractal']['hurst_scaling_estimate']} | Higuchi FD={p['fractal']['higuchi_fractal_dimension']}")
        print(f"Dynamical systems: Kalman local-level={p['dynamical_systems']['kalman_local_level']}")
    print(line)
    print("CAUTION: Research diagnostics/proxies only; not validated forecasts or standalone trade filters.")
    print("━" * 76 + "\n")





# =====================================================================
# V19 MACHINE LEARNING ENSEMBLE + STATISTICAL ARBITRAGE RESEARCH
# Advisory/research only. No model directly places broker orders.
# =====================================================================

class MLResearchEnsemble:
    """Chronological holdout evaluation for RF, XGBoost, MLP and Bayesian NB."""
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.status = "NOT_TRAINED"
        self.rows = 0
        self.mtime = None
        self.feature_names = list(FEATURES)

    def _load_training_data(self):
        if not TRADE_JOURNAL.exists():
            return None, None
        df = pd.read_csv(TRADE_JOURNAL)
        if not {"status", "outcome_r"}.issubset(df.columns):
            return None, None
        df = df[df["status"].astype(str).str.upper().eq("CLOSED")].copy()
        for col in self.feature_names + ["outcome_r", "close_time", "signal_time"]:
            if col not in df:
                df[col] = np.nan if col in self.feature_names + ["outcome_r"] else ""
        df["__time"] = pd.to_datetime(
            df["close_time"].replace("", np.nan), errors="coerce", utc=True
        )
        if df["__time"].notna().any():
            df = df.sort_values("__time", kind="stable")
        for col in self.feature_names + ["outcome_r"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.feature_names + ["outcome_r"])
        if df.empty:
            return None, None
        X = df[self.feature_names].astype(float)
        y = (df["outcome_r"] > 0).astype(int)
        return X, y

    def train_if_changed(self, force=False):
        if not ML_ENSEMBLE_ENABLED:
            self.status = "DISABLED_BY_CONFIG"
            return
        if not SKLEARN_OK:
            self.status = "scikit-learn unavailable; install scikit-learn"
            return
        if not TRADE_JOURNAL.exists():
            self.status = "WAITING_FOR_CLOSED_TRADE_JOURNAL"
            return
        try:
            current_mtime = TRADE_JOURNAL.stat().st_mtime
            if not force and self.mtime == current_mtime:
                return
            self.mtime = current_mtime
            X, y = self._load_training_data()
            if X is None or len(X) < ML_ENSEMBLE_MIN_ROWS:
                self.status = f"INSUFFICIENT_ROWS ({0 if X is None else len(X)}/{ML_ENSEMBLE_MIN_ROWS})"
                return
            counts = y.value_counts()
            if len(counts) < 2 or int(counts.min()) < ML_ENSEMBLE_MIN_PER_CLASS:
                self.status = "INSUFFICIENT_CLASS_BALANCE"
                return
            split = int(len(X) * (1.0 - ML_ENSEMBLE_HOLDOUT_FRACTION))
            if split < 50 or len(X) - split < 20 or y.iloc[:split].nunique() < 2:
                self.status = "INSUFFICIENT_CHRONOLOGICAL_HOLDOUT"
                return
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            candidates = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=240, min_samples_leaf=3, max_features="sqrt",
                    class_weight="balanced_subsample", random_state=ML_ENSEMBLE_RANDOM_STATE,
                    n_jobs=1
                ),
                "BayesianGaussianNB": make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    GaussianNB(var_smoothing=1e-8)
                ),
                "NeuralNetwork_MLP": make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01,
                                  early_stopping=True, max_iter=500,
                                  random_state=ML_ENSEMBLE_RANDOM_STATE)
                ),
            }
            if LIGHTGBM_OK:
                candidates["LightGBM"] = LGBMClassifier(
                    n_estimators=180, learning_rate=0.035, num_leaves=15,
                    subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                    random_state=ML_ENSEMBLE_RANDOM_STATE, verbosity=-1, n_jobs=1
                )
            if CATBOOST_OK:
                candidates["CatBoost"] = CatBoostClassifier(
                    iterations=180, depth=5, learning_rate=0.035,
                    loss_function="Logloss", verbose=False,
                    random_seed=ML_ENSEMBLE_RANDOM_STATE, thread_count=1
                )
            if XGBOOST_OK:
                candidates["XGBoost"] = XGBClassifier(
                    n_estimators=180, max_depth=3, learning_rate=0.035,
                    subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=2.0, min_child_weight=4,
                    objective="binary:logistic", eval_metric="logloss",
                    random_state=ML_ENSEMBLE_RANDOM_STATE, n_jobs=1,
                    verbosity=0
                )
            else:
                candidates["XGBoost"] = None

            fitted, metrics = {}, {}
            for name, estimator in candidates.items():
                if estimator is None:
                    metrics[name] = {"status": "UNAVAILABLE_OPTIONAL_DEPENDENCY"}
                    continue
                try:
                    estimator.fit(X_train, y_train)
                    classes = list(estimator.classes_) if hasattr(estimator, "classes_") else list(estimator[-1].classes_)
                    positive_idx = classes.index(1)
                    p = estimator.predict_proba(X_test)[:, positive_idx]
                    pred = (p >= 0.5).astype(int)
                    m = {
                        "status": "EVALUATED",
                        "accuracy": float(accuracy_score(y_test, pred)),
                        "brier": float(brier_score_loss(y_test, p)),
                        "auc": float(roc_auc_score(y_test, p)) if y_test.nunique() == 2 else None,
                        "holdout_n": int(len(y_test)),
                    }
                    # Refit on all available historical rows only after holdout scoring.
                    estimator.fit(X, y)
                    fitted[name] = estimator
                    metrics[name] = m
                except Exception as exc:
                    metrics[name] = {"status": "MODEL_ERROR", "error": str(exc)[:240]}
            self.models, self.metrics, self.rows = fitted, metrics, len(X)
            self.status = "TRAINED_ADVISORY" if fitted else "NO_MODEL_FITTED"
        except Exception as exc:
            self.status = "TRAINING_ERROR: " + str(exc)[:300]

    def predict(self, features):
        if not self.models:
            return {"status": self.status, "probability": None, "model_probabilities": {}}
        row = pd.DataFrame(
            [[safe_float(features.get(k, 0.0)) for k in self.feature_names]],
            columns=self.feature_names
        )
        probs = {}
        for name, model in self.models.items():
            try:
                classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
                probs[name] = float(model.predict_proba(row)[0][classes.index(1)])
            except Exception:
                continue
        return {
            "status": "OK" if probs else self.status,
            "probability": float(np.mean(list(probs.values()))) if probs else None,
            "model_probabilities": probs,
            "model_dispersion": float(np.std(list(probs.values()))) if len(probs) > 1 else None,
        }

    def report(self):
        return {"status": self.status, "rows": self.rows, "metrics": self.metrics,
                "models_available": list(self.models.keys())}



class LSTMTradeContextResearch:
    """Optional LSTM on chronological closed-trade feature sequences.
    This is trade-context sequence research, NOT raw M5 candle forecasting.
    """
    def __init__(self):
        self.model = None
        self.mu = self.sigma = None
        self.status = "NOT_TRAINED"
        self.metrics = {}
        self.rows = 0
        self.mtime = None
        self.L = DL_SEQUENCE_LENGTH

    def _data(self):
        if not TRADE_JOURNAL.exists():
            return None, None
        df = pd.read_csv(TRADE_JOURNAL)
        if not {"status", "outcome_r"}.issubset(df.columns):
            return None, None
        df = df[df.status.astype(str).str.upper().eq("CLOSED")].copy()
        if "close_time" in df:
            dt = pd.to_datetime(df["close_time"].replace("", np.nan), errors="coerce", utc=True)
            if dt.notna().any():
                df["__time"] = dt
                df = df.sort_values("__time", kind="stable")
        for col in FEATURES + ["outcome_r"]:
            if col not in df:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["outcome_r"])
        if df.empty:
            return None, None
        return df[FEATURES].to_numpy(float), (df.outcome_r.to_numpy() > 0).astype(int)

    def _seq(self, x, y):
        xs, ys = [], []
        for end in range(self.L - 1, len(x)):
            xs.append(x[end-self.L+1:end+1])
            ys.append(y[end])
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32)

    def train_if_changed(self):
        if not DL_ENABLED:
            self.status = "DISABLED_BY_CONFIG"
            return
        if not TENSORFLOW_OK:
            self.status = "TENSORFLOW_UNAVAILABLE — training DL perlu Google Colab"
            return
        if not TRADE_JOURNAL.exists():
            self.status = "WAITING_FOR_CLOSED_TRADE_JOURNAL"
            return
        try:
            mt = TRADE_JOURNAL.stat().st_mtime
            if mt == self.mtime:
                return
            self.mtime = mt
            x, y = self._data()
            if x is None or len(x) < DL_MIN_ROWS:
                self.status = f"INSUFFICIENT_ROWS ({0 if x is None else len(x)}/{DL_MIN_ROWS})"
                return
            counts = np.bincount(y, minlength=2)
            if counts.min() < DL_MIN_PER_CLASS:
                self.status = f"INSUFFICIENT_CLASS_BALANCE {counts.tolist()}"
                return
            split = int(len(x) * (1 - ML_ENSEMBLE_HOLDOUT_FRACTION))
            if split <= self.L + 40 or len(x) - split < self.L + 15:
                self.status = "INSUFFICIENT_CHRONOLOGICAL_HOLDOUT"
                return
            xtr_raw, xte_raw = x[:split], x[split:]
            ytr, yte = y[:split], y[split:]
            mu, sigma = xtr_raw.mean(0), xtr_raw.std(0)
            sigma[sigma < 1e-8] = 1.0
            xtr = (xtr_raw - mu) / sigma
            xte = (np.vstack([xtr_raw[-(self.L-1):], xte_raw]) - mu) / sigma
            Xtr, Ytr = self._seq(xtr, ytr)
            yte_context = np.r_[ytr[-(self.L-1):], yte]
            Xte, Yte = self._seq(xte, yte_context)
            Xte, Yte = Xte[-len(yte):], Yte[-len(yte):]
            if len(Xtr) < 100 or len(Xte) < 15:
                self.status = "SEQUENCE_DATA_INSUFFICIENT"
                return

            tf.keras.utils.set_random_seed(DL_RANDOM_STATE)
            def build():
                m = Sequential([
                    Input(shape=(self.L, len(FEATURES))),
                    LSTM(32), Dropout(0.20), Dense(16, activation="relu"),
                    Dropout(0.10), Dense(1, activation="sigmoid")
                ])
                m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                          loss="binary_crossentropy", metrics=["accuracy"])
                return m
            probe = build()
            hist = probe.fit(
                Xtr, Ytr, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                validation_split=0.15, shuffle=False, verbose=0,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            )
            p = probe.predict(Xte, verbose=0).reshape(-1)
            pred = (p >= 0.5).astype(int)
            auc = float(roc_auc_score(Yte, p)) if len(np.unique(Yte)) == 2 else None
            self.metrics = {
                "accuracy": float(accuracy_score(Yte, pred)),
                "brier": float(brier_score_loss(Yte, p)),
                "auc": auc, "holdout_n": int(len(Yte))
            }
            # Refit on all rows only after chronological holdout metrics are recorded.
            Xa, Ya = self._seq((x-mu)/sigma, y)
            final = build()
            final.fit(Xa, Ya, epochs=max(5, len(hist.history.get("loss", []))),
                      batch_size=DL_BATCH_SIZE, shuffle=False, verbose=0)
            self.model, self.mu, self.sigma = final, mu, sigma
            self.rows = len(x)
            self.status = f"TRAINED_ADVISORY rows={self.rows}; holdout_acc={self.metrics['accuracy']:.3f}; Brier={self.metrics['brier']:.3f}"
        except Exception as exc:
            self.status = "LSTM_ERROR: " + str(exc)[:220]

    def predict(self, features):
        if self.model is None:
            return {"status": self.status, "probability": None}
        try:
            x, _ = self._data()
            if x is None or len(x) < self.L-1:
                return {"status": "INSUFFICIENT_CONTEXT", "probability": None}
            current = np.asarray([safe_float(features.get(k, 0.0)) for k in FEATURES])
            seq = np.vstack([x[-(self.L-1):], current])
            seq = (seq-self.mu)/self.sigma
            p = float(self.model.predict(seq[None, :, :], verbose=0).reshape(-1)[0])
            return {"status": "ADVISORY_ONLY", "probability": p}
        except Exception as exc:
            return {"status": "PREDICT_ERROR", "probability": None, "error": str(exc)[:150]}


def regime_classification_lab(df, n_regimes=3):
    """Unsupervised Gaussian-mixture market-state classification; descriptive only."""
    if not SKLEARN_OK or len(df) < 80:
        return {"status": "INSUFFICIENT_DATA_OR_SKLEARN"}
    close = pd.to_numeric(df["close"], errors="coerce")
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(20).std()
    momentum = ret.rolling(10).sum()
    efficiency = close.diff(10).abs() / close.diff().abs().rolling(10).sum().replace(0, np.nan)
    features = pd.DataFrame({"volatility": vol, "momentum": momentum, "efficiency": efficiency})
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < 60:
        return {"status": "INSUFFICIENT_FEATURE_ROWS", "n": len(features)}
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = GaussianMixture(n_components=min(n_regimes, max(2, len(X)//30)),
                            covariance_type="full", reg_covar=1e-5,
                            random_state=ML_ENSEMBLE_RANDOM_STATE)
    labels = model.fit_predict(X)
    latest = int(labels[-1])
    group = features.iloc[np.where(labels == latest)[0]]
    avg_vol = float(group.volatility.mean())
    avg_mom = float(group.momentum.mean())
    all_vol = float(features.volatility.median())
    if avg_vol > all_vol * 1.25:
        regime = "HIGH_VOLATILITY"
    elif avg_mom > 0:
        regime = "BULL_TREND"
    elif avg_mom < 0:
        regime = "BEAR_TREND"
    else:
        regime = "LOW_VOLATILITY_OR_RANGE"
    return {"status": "OK", "method": "GaussianMixture", "regime_id": latest,
            "regime_label": regime, "regime_probability": float(model.predict_proba(X[-1:])[0][latest]),
            "cluster_counts": {str(i): int(np.sum(labels == i)) for i in range(model.n_components)},
            "cluster_mean_volatility": avg_vol, "cluster_mean_momentum": avg_mom,
            "n": int(len(features)),
            "note": "In-sample unsupervised clustering; refit stability and chronological validation required."}


def fetch_symbol_candles(symbol, interval=STAT_ARB_INTERVAL, outputsize=STAT_ARB_OUTPUTSIZE):
    """Fetch a second Twelve Data symbol with explicit, credential-safe diagnostics."""
    if not API_KEY:
        raise RuntimeError("API key Twelve Data belum dikonfigurasi.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(outputsize),
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Koneksi Twelve Data gagal untuk {symbol}: {type(exc).__name__}: {exc}") from exc

    # Twelve Data can return a useful JSON error body with an HTTP error code.
    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code >= 400:
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("message") or payload.get("code") or payload)[:300]
        elif response.text:
            detail = response.text[:200]
        raise RuntimeError(
            f"Twelve Data HTTP {response.status_code} untuk {symbol}"
            + (f": {detail}" if detail else "")
        )

    if not isinstance(payload, dict):
        raise RuntimeError(f"Twelve Data {symbol}: respons bukan JSON object.")
    if payload.get("status") == "error" or not isinstance(payload.get("values"), list):
        detail = payload.get("message") or payload.get("code") or str(payload)
        raise RuntimeError(f"Twelve Data menolak {symbol}: {str(detail)[:300]}")

    df = pd.DataFrame(payload["values"])
    required = ["datetime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{symbol}: kolom OHLC hilang: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "volume":
            df[col] = 0.0

    df = df.dropna(subset=required)
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Exclude the latest potentially-forming bar, matching the primary feed.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    if len(df) < STAT_ARB_MIN_OBSERVATIONS:
        raise RuntimeError(
            f"{symbol}: hanya {len(df)} candle valid; perlu minimal "
            f"{STAT_ARB_MIN_OBSERVATIONS}. Periksa simbol, interval, paket/kuota API."
        )
    return df.reset_index(drop=True)


def statistical_arbitrage_research(primary_df, reference_df,
                                   entry_z=STAT_ARB_ENTRY_Z,
                                   exit_z=STAT_ARB_EXIT_Z,
                                   stop_z=STAT_ARB_STOP_Z,
                                   cost_bps_per_leg=STAT_ARB_COST_BPS_PER_LEG):
    """
    Pair/spread diagnostics. Uses log-price hedge regression and lagged positions
    for a no-lookahead illustrative spread backtest. It is not a live trading signal.
    """
    required = {"datetime", "close"}
    if primary_df is None or reference_df is None or not required.issubset(primary_df.columns) or not required.issubset(reference_df.columns):
        return {"status": "INVALID_INPUT"}
    a = primary_df[["datetime", "close"]].copy().rename(columns={"close": "asset_a"})
    b = reference_df[["datetime", "close"]].copy().rename(columns={"close": "asset_b"})
    for d in (a, b):
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
        d.iloc[:, 1] = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    merged = pd.merge(a, b, on="datetime", how="inner").dropna()
    merged = merged[(merged.asset_a > 0) & (merged.asset_b > 0)].tail(STAT_ARB_OUTPUTSIZE)
    if len(merged) < STAT_ARB_MIN_OBSERVATIONS:
        return {"status": "INSUFFICIENT_SYNCHRONIZED_DATA", "n": len(merged)}
    la, lb = np.log(merged.asset_a.to_numpy()), np.log(merged.asset_b.to_numpy())
    X = np.column_stack([np.ones(len(lb)), lb])
    beta = np.linalg.lstsq(X, la, rcond=None)[0]
    hedge_intercept, hedge_ratio = float(beta[0]), float(beta[1])
    spread = la - (hedge_intercept + hedge_ratio * lb)
    window = min(60, max(30, len(spread)//4))
    s = pd.Series(spread)
    roll_mean = s.rolling(window, min_periods=window).mean()
    roll_std = s.rolling(window, min_periods=window).std(ddof=1).replace(0, np.nan)
    z = ((s - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan)
    current_z = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None
    corr = float(np.corrcoef(la, lb)[0, 1]) if np.std(la) > 0 and np.std(lb) > 0 else None
    coint_p = adf_p = None
    if STATSMODELS_OK:
        try:
            coint_p = float(coint(la, lb)[1])
            adf_p = float(adfuller(spread, autolag="AIC")[1])
        except Exception:
            pass
    # AR(1) spread half-life estimate: Δs_t = c + phi*s_(t-1).
    ds, lag = np.diff(spread), spread[:-1]
    if len(ds) > 5 and np.var(lag) > 1e-15:
        phi = float(np.polyfit(lag, ds, 1)[0])
        half_life = float(-math.log(2) / math.log(1 + phi)) if -1 < phi < 0 else None
    else:
        phi, half_life = None, None
    # Illustrative mean-reversion backtest. Signal at t, position applied to t+1 spread change.
    z_arr = z.to_numpy()
    pos = np.zeros(len(spread))
    active = 0.0
    for i in range(len(spread)):
        zi = z_arr[i]
        if not np.isfinite(zi):
            pos[i] = active
            continue
        if active == 0:
            if zi >= entry_z: active = -1.0
            elif zi <= -entry_z: active = 1.0
        elif abs(zi) <= exit_z or abs(zi) >= stop_z:
            active = 0.0
        pos[i] = active
    dspread = np.r_[0.0, np.diff(spread)]
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    # Spread units are log-return-like; cost is charged per leg per position change.
    net = np.r_[0.0, pos[:-1] * dspread[1:]] - turnover * (2.0 * cost_bps_per_leg / 10000.0)
    valid = np.isfinite(net) & (np.arange(len(net)) >= window)
    pnl = net[valid]
    gross = pos[:-1] * dspread[1:] if len(pos) > 1 else np.array([])
    wins = int(np.sum(pnl > 0))
    losses = int(np.sum(pnl < 0))
    pf = float(np.sum(pnl[pnl > 0]) / abs(np.sum(pnl[pnl < 0]))) if np.any(pnl < 0) else None
    equity = np.cumsum(pnl) if len(pnl) else np.array([])
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:] if len(equity) else np.array([])
    max_dd = float(np.max(peak - equity)) if len(equity) else None
    return {
        "status": "OK", "n": int(len(merged)), "correlation_log_price": corr,
        "hedge_intercept": hedge_intercept, "hedge_ratio": hedge_ratio,
        "spread_zscore": current_z, "cointegration_pvalue": coint_p,
        "spread_adf_pvalue": adf_p, "ar1_phi": phi, "half_life_bars": half_life,
        "backtest": {"observations": int(len(pnl)), "wins": wins, "losses": losses,
                     "profit_factor": pf, "net_spread_return_sum": float(np.sum(pnl)) if len(pnl) else 0.0,
                     "max_drawdown_spread_units": max_dd,
                     "note": "Illustrative lagged spread backtest; log-spread units, simplified costs, not deployable PnL."},
        "interpretation": (
            "Potential stationary spread only if cointegration/ADF and stability checks support it."
            if (coint_p is not None and coint_p < 0.05 and adf_p is not None and adf_p < 0.05)
            else "No statistically confirmed stationary spread from this sample."
        ),
    }


def print_ml_stat_arb_report(ml_report, stat_arb_report, regime_report=None):
    line = "─" * 76
    print("\\n" + "━" * 76)
    print("🤖 MACHINE LEARNING RESEARCH ENSEMBLE")
    print(line)
    print(f"Status: {ml_report.get('status')} | closed samples={ml_report.get('rows', 0)}")
    if ml_report.get("metrics"):
        for name, m in ml_report["metrics"].items():
            if m.get("status") == "EVALUATED":
                print(f"{name}: accuracy={m['accuracy']:.3f} | Brier={m['brier']:.3f} | AUC={m['auc']} | holdout={m['holdout_n']}")
            else:
                print(f"{name}: {m.get('status')} {m.get('error', '')}")
    else:
        print("Models menunggu jurnal closed trade yang cukup; tidak ada model yang dipaksakan training.")
    if regime_report and regime_report.get("status") == "OK":
        print(f"Regime classification: {regime_report['regime_label']} | cluster={regime_report['regime_id']} | posterior={regime_report['regime_probability']:.3f}")
    else:
        print(f"Regime classification: {regime_report}")
    print(line)
    print("📊 STATISTICAL ARBITRAGE LAB")
    if not stat_arb_report or stat_arb_report.get("status") == "DISABLED":
        print(f"Status: DISABLED (set STAT_ARB_ENABLED=True; reference={STAT_ARB_REFERENCE_SYMBOL})")
    elif stat_arb_report.get("status") != "OK":
        print(f"Status: {stat_arb_report.get('status')} | n={stat_arb_report.get('n', 0)}")
        if stat_arb_report.get("error"):
            print(f"Detail: {stat_arb_report['error']}")
    else:
        print(f"Pair: {SYMBOL} / {STAT_ARB_REFERENCE_SYMBOL} | n={stat_arb_report['n']}")
        print(f"Correlation={stat_arb_report['correlation_log_price']} | hedge ratio={stat_arb_report['hedge_ratio']:.5f}")
        print(f"Spread Z={stat_arb_report['spread_zscore']} | cointegration p={stat_arb_report['cointegration_pvalue']} | ADF p={stat_arb_report['spread_adf_pvalue']}")
        print(f"Half-life={stat_arb_report['half_life_bars']} bars | backtest={stat_arb_report['backtest']}")
        print("Stat-arb report is research only; it does not override XAU candidate/adaptive signals.")
    print(line + "\\n")


def export_research_bundle():
    """Create portable CSV exports for Colab research; does not upload data."""
    try:
        trades = read_journal()
        trades.to_csv(ROOT / "jst_v19_trade_research_export.csv", index=False)
        candidates = pd.read_csv(CANDIDATE_JOURNAL) if CANDIDATE_JOURNAL.exists() else pd.DataFrame()
        candidates.to_csv(ROOT / "jst_v19_candidate_research_export.csv", index=False)
        print("[HYBRID EXPORT] CSV research bundle siap di folder script untuk dibuka di Colab.")
    except Exception as exc:
        print("[HYBRID EXPORT WARN]", exc)


ml_research_ensemble = MLResearchEnsemble()
lstm_trade_context = LSTMTradeContextResearch()


def main():
    if not API_KEY:
        print("\nAPI KEY BELUM DIISI.")
        print("1) Buka bagian CONFIG di atas.")
        print('2) Cari: PASTE_API_KEY_HERE = ""')
        print('3) Tempel key di antara tanda kutip, contoh: PASTE_API_KEY_HERE = "KEY_ANDA"')
        print("4) Simpan file, lalu jalankan lagi. Jangan bagikan file yang berisi key.")
        print("Alternatif: gunakan environment variable TWELVE_DATA_API_KEY.")
        return
    if RISK_PERCENT <= 0 or DAILY_LOSS_LIMIT_PERCENT <= 0:
        print("CONFIG ERROR: risk dan daily loss limit harus > 0.")
        return
    migrate_journal_if_needed()
    model, state = EdgeModel(), load_state()
    print("="*72)
    print(APP_NAME)
    print("XAU/USD | M5 primary | H1/M15 context | M1 timing | PAPER ONLY | EDGE RESEARCH")
    print(f"Environment={ENVIRONMENT_MODE} | Architecture={ARCHITECTURE} | Engine={ENGINE_MODE}")
    print("Dual independent engines: Candidate discovery + Adaptive direction + Signal Fusion.")
    print("Adaptive direction is rule-based research logic, not a trained predictive model.")
    print(f"V20 Hybrid ML/DL | TensorFlow={TENSORFLOW_OK} | LightGBM={LIGHTGBM_OK} | CatBoost={CATBOOST_OK} | XGBoost={XGBOOST_OK}")
    print("LSTM = closed-trade context sequence advisory, not raw M5 candle prediction.")
    print(f"Risk baseline={RISK_PERCENT:.2f}% | daily loss guard={DAILY_LOSS_LIMIT_PERCENT:.2f}% "
          f"| min RR={MIN_RR:.2f} | scan={SCAN_SECONDS}s")
    print("Data: Twelve Data OHLC. Tidak ada order book / COMEX footprint terhubung.")
    print("📈 Quant Math + Edge Research | Jurnal paper dipakai untuk evaluasi statistik.")
    print("Hybrid workflow: gunakan export CSV untuk training/validasi di Google Colab; hasil model tidak otomatis di-deploy.")
    print(f"Estimasi biaya={ESTIMATED_COST_R:.2f}R; ganti dengan pengukuran aktual.")
    try:
        while True:
            started = time.monotonic()
            try:
                run_once(model, state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print("\n[SCAN ERROR]", exc)
                traceback.print_exc()
            wait = max(1, SCAN_SECONDS - int(time.monotonic() - started))
            print(f"⏰  Scan berikutnya dalam ~{wait}s. Stop: tombol Stop Pydroid / Ctrl+C.\n")
            time.sleep(wait)
    except KeyboardInterrupt:
        export_research_bundle()
        print("\n⛔  JST APEX QUANT dihentikan oleh pengguna.")


if __name__ == "__main__":
    main()
