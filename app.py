import os
import io
import contextlib
import time

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="JST APEX NEXUS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("JUMP STREET TRADING")
st.subheader("JST APEX NEXUS V20.1 | Smart Auto Scanner")
st.caption("XAU/USD • Signal + Paper Research Only • No Auto-Order")

# Read Secrets before importing the engine. Keep the engine module cached across
# Streamlit reruns so its existing per-timeframe candle cache is not erased.
try:
    secret_key = str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
except Exception:
    secret_key = ""

if secret_key:
    os.environ["TWELVE_DATA_API_KEY"] = secret_key

@st.cache_resource
def load_engine():
    import jst_engine
    return jst_engine

try:
    eng = load_engine()
except Exception as exc:
    st.error(f"Engine gagal dimuat: {exc}")
    st.stop()

if not eng.API_KEY:
    st.warning("API key Twelve Data belum terbaca.")
    st.code('TWELVE_DATA_API_KEY = "API_KEY_ANDA"', language="toml")
    st.stop()

# Persistent only for the lifetime of this Streamlit browser session.
if "jst_model" not in st.session_state:
    st.session_state.jst_model = eng.EdgeModel()
if "jst_state" not in st.session_state:
    eng.migrate_journal_if_needed()
    st.session_state.jst_state = eng.load_state()
if "last_scan_output" not in st.session_state:
    st.session_state.last_scan_output = ""
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "last_scan_monotonic" not in st.session_state:
    st.session_state.last_scan_monotonic = 0.0
if "scan_error" not in st.session_state:
    st.session_state.scan_error = ""

with st.sidebar:
    st.header("System Control")
    st.write("**Symbol:**", eng.SYMBOL)
    st.write("**Primary timeframe:** M5")
    st.write("**Engine interval:**", f"{eng.SCAN_SECONDS} detik")
    st.write("**Risk baseline:**", f"{eng.RISK_PERCENT:.2f}%")
    st.write("**Mode:** PAPER ONLY")
    auto_scan = st.toggle("AUTO SCAN", value=False, help="Mengulang satu siklus scan selama sesi Streamlit ini tetap aktif.")
    scan_interval = st.selectbox(
        "Interval auto scan",
        options=[60, 120, 180, 300],
        index=0,
        format_func=lambda x: f"{x} detik",
        disabled=not auto_scan,
    )

if auto_scan:
    st.sidebar.success(f"Auto scan aktif · setiap {scan_interval} detik")
else:
    st.sidebar.info("Auto scan nonaktif · gunakan scan manual")

# The fragment timer runs only while this Streamlit app session is active.
# The engine's own _DATA_CACHE is preserved by @st.cache_resource above.
@st.fragment(run_every=scan_interval if auto_scan else None)
def live_dashboard():
    manual_scan = st.button("▶ SCAN SEKARANG", type="primary")
    now_mono = time.monotonic()
    last_mono = st.session_state.last_scan_monotonic
    due_auto = auto_scan and (last_mono == 0.0 or now_mono - last_mono >= scan_interval - 1)
    should_scan = manual_scan or due_auto

    if should_scan:
        captured = io.StringIO()
        try:
            with st.spinner("Mengambil data market dan menjalankan engine..."):
                with contextlib.redirect_stdout(captured):
                    eng.run_once(st.session_state.jst_model, st.session_state.jst_state)
            st.session_state.last_scan_output = captured.getvalue()
            st.session_state.last_scan_time = eng.local_stamp()
            st.session_state.scan_error = ""
            st.session_state.last_scan_monotonic = time.monotonic()
            st.success("Siklus pemindaian selesai.")
        except Exception as exc:
            st.session_state.last_scan_output = captured.getvalue()
            st.session_state.scan_error = str(exc)
            st.session_state.last_scan_monotonic = time.monotonic()
            st.error(f"Pemindaian gagal: {exc}")
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                st.warning("Twelve Data membatasi request (HTTP 429). Auto scan tetap mengikuti interval; jangan menekan scan manual berulang kali. Periksa kuota dan tunggu sebelum mencoba lagi.")

    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time}")
    else:
        st.caption("Belum ada scan pada sesi dashboard ini.")

    if auto_scan:
        st.info(f"AUTO SCAN aktif. Siklus berikutnya dijadwalkan sekitar setiap {scan_interval} detik selama sesi dashboard tetap aktif.")

    journal = eng.read_journal()
    if not journal.empty:
        status = journal["status"].astype(str).str.upper()
        open_trades = journal[status.eq("OPEN")]
        closed = journal[status.eq("CLOSED")]
    else:
        open_trades = pd.DataFrame()
        closed = pd.DataFrame()

    stats = eng.closed_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open paper positions", len(open_trades))
    c2.metric("Closed paper trades", stats["n"])
    c3.metric("Win rate", "N/A" if stats["win_rate"] is None else f"{stats['win_rate']:.1%}")
    c4.metric("Expectancy", "N/A" if stats["expectancy"] is None else f"{stats['expectancy']:+.3f}R")

    tab1, tab2, tab3 = st.tabs(["Signal & Scan", "Paper Journal", "System Notes"])
    with tab1:
        st.subheader("Latest scan output")
        if st.session_state.last_scan_output:
            st.code(st.session_state.last_scan_output, language="text")
        else:
            st.info("Aktifkan AUTO SCAN atau tekan SCAN SEKARANG untuk memulai.")
        if st.session_state.scan_error:
            st.error(st.session_state.scan_error)
    with tab2:
        st.subheader("Open paper positions")
        if open_trades.empty:
            st.info("Belum ada posisi paper yang terbuka.")
        else:
            st.dataframe(open_trades, width="stretch", hide_index=True)
        st.subheader("Recent closed trades")
        if closed.empty:
            st.info("Belum ada paper trade tertutup.")
        else:
            st.dataframe(closed.tail(50).iloc[::-1], width="stretch", hide_index=True)
        if eng.CANDIDATE_JOURNAL.exists():
            candidates = pd.read_csv(eng.CANDIDATE_JOURNAL)
            st.subheader("Recent candidates")
            if not candidates.empty:
                st.dataframe(candidates.tail(50).iloc[::-1], width="stretch", hide_index=True)
    with tab3:
        st.subheader("Runtime capability")
        st.write({
            "scikit_learn": eng.SKLEARN_OK,
            "TensorFlow": eng.TENSORFLOW_OK,
            "LightGBM": eng.LIGHTGBM_OK,
            "CatBoost": eng.CATBOOST_OK,
            "XGBoost": eng.XGBOOST_OK,
            "Statsmodels": eng.STATSMODELS_OK,
        })
        st.warning("Auto scan ini berjalan di dalam sesi Streamlit, bukan background worker 24/7. Jika sesi tidur, browser ditutup, atau server restart, pemindaian dapat berhenti.")
        st.info("Engine V20 memiliki cache candle per timeframe. Cache ini dipertahankan antar-rerun karena modul engine tidak di-reload setiap siklus UI. Cache mengurangi request berulang, tetapi tidak menjamin bebas HTTP 429; kuota Twelve Data tetap berlaku.")
        st.write("Paper journal dan state yang disimpan pada filesystem Streamlit Cloud tidak dijamin persisten setelah rebuild/restart. Gunakan penyimpanan eksternal sebelum mengandalkan histori jangka panjang.")

live_dashboard()

st.divider()
st.caption("Research / paper mode only. No broker connection. No automatic order execution.")
