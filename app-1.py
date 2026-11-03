import os
import io
import contextlib
import importlib
import sys

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="JST APEX NEXUS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("JUMP STREET TRADING")
st.subheader("JST APEX NEXUS V20.0 | Hybrid ML + Deep Learning")
st.caption("XAU/USD • Signal + Paper Research Only • No Auto-Order")

# Read secrets before importing the engine, because the engine reads its
# Twelve Data key at module import time.
try:
    secret_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
except Exception:
    secret_key = ""

if secret_key:
    os.environ["TWELVE_DATA_API_KEY"] = str(secret_key).strip()

def load_engine():
    # Streamlit reruns the script after Secrets changes. Reload the module so
    # its module-level API_KEY is read again from the updated environment.
    if "jst_engine" in sys.modules:
        return importlib.reload(sys.modules["jst_engine"])
    import jst_engine
    return jst_engine

try:
    eng = load_engine()
except Exception as exc:
    st.error(f"Engine gagal dimuat: {exc}")
    st.stop()

if not eng.API_KEY:
    st.warning("API key Twelve Data belum diatur.")
    st.markdown(
        "Buka **App settings → Secrets** di Streamlit Cloud, lalu masukkan:"
    )
    st.code('TWELVE_DATA_API_KEY = "API_KEY_ANDA"', language="toml")
    st.stop()

with st.sidebar:
    st.header("System Control")
    st.write("**Symbol:**", eng.SYMBOL)
    st.write("**Primary timeframe:** M5")
    st.write("**Scan interval config:**", f"{eng.SCAN_SECONDS} detik")
    st.write("**Risk baseline:**", f"{eng.RISK_PERCENT:.2f}%")
    st.write("**Mode:** PAPER ONLY")
    st.warning(
        "Versi Streamlit ini menjalankan satu siklus scan saat tombol ditekan. "
        "Ini bukan background worker 24/7."
    )
    scan_clicked = st.button("▶ SCAN SEKARANG", type="primary", use_container_width=True)
    refresh_clicked = st.button("↻ REFRESH DASHBOARD", use_container_width=True)

if scan_clicked:
    if "jst_model" not in st.session_state:
        st.session_state.jst_model = eng.EdgeModel()
    if "jst_state" not in st.session_state:
        eng.migrate_journal_if_needed()
        st.session_state.jst_state = eng.load_state()

    output = io.StringIO()
    try:
        with st.spinner("Mengambil data market dan menjalankan engine..."):
            with contextlib.redirect_stdout(output):
                eng.run_once(st.session_state.jst_model, st.session_state.jst_state)
        st.session_state["last_scan_output"] = output.getvalue()
        st.session_state["last_scan_time"] = eng.local_stamp()
        st.success("Satu siklus pemindaian selesai.")
    except Exception as exc:
        st.session_state["last_scan_output"] = output.getvalue() + f"\n[SCAN ERROR] {exc}"
        st.error(f"Pemindaian gagal: {exc}")

if refresh_clicked:
    st.rerun()

last_scan = st.session_state.get("last_scan_time", "Belum ada scan pada sesi dashboard ini.")
st.caption(f"Last scan: {last_scan}")

# Summary cards from the paper journal.
journal = eng.read_journal()
if not journal.empty:
    open_trades = journal[journal["status"].astype(str).str.upper().eq("OPEN")]
    closed = journal[journal["status"].astype(str).str.upper().eq("CLOSED")]
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
    scan_output = st.session_state.get("last_scan_output")
    if scan_output:
        st.code(scan_output, language="text")
    else:
        st.info("Tekan SCAN SEKARANG untuk menjalankan satu siklus analisis.")
    st.caption(
        "Output mempertahankan laporan engine asli. Kandidat, Adaptive, Fusion, "
        "Execution Readiness, Quant Math, Physics, ML/DL dan riset lain ditampilkan "
        "sesuai ketersediaan data."
    )

with tab2:
    st.subheader("Open paper positions")
    if open_trades.empty:
        st.info("Belum ada posisi paper yang terbuka.")
    else:
        st.dataframe(open_trades, use_container_width=True, hide_index=True)
    st.subheader("Recent closed trades")
    if closed.empty:
        st.info("Belum ada paper trade tertutup.")
    else:
        st.dataframe(closed.tail(50).iloc[::-1], use_container_width=True, hide_index=True)

    if eng.CANDIDATE_JOURNAL.exists():
        candidates = pd.read_csv(eng.CANDIDATE_JOURNAL)
        st.subheader("Recent candidates")
        if not candidates.empty:
            st.dataframe(candidates.tail(50).iloc[::-1], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Runtime capability")
    st.write({
        "scikit_learn": eng.SKLEARN_OK,
        "TensorFlow": eng.TENSORFLOW_OK,
        "LightGBM": eng.LIGHTGBM_OK,
        "CatBoost": eng.CATBOOST_OK,
        "XGBoost": eng.XGBOOST_OK,
        "Statmodels": eng.STATSMODELS_OK,
    })
    st.warning(
        "Streamlit Community Cloud bukan worker nonstop. Pemindaian di versi awal "
        "ini berjalan ketika pengguna menekan tombol dan server sedang aktif. "
        "Untuk scanning 24/7, nanti pindahkan worker Python ke VPS/cloud worker."
    )
    st.write(
        "LSTM V20 menggunakan TensorFlow secara opsional. Jika TensorFlow tidak "
        "tersedia, bagian LSTM akan melaporkan status unavailable; engine inti "
        "tetap dapat berjalan."
    )
    st.write(
        "Jurnal CSV dan state pada filesystem Streamlit Cloud dapat hilang saat "
        "instance dibangun ulang atau tidur. Untuk penggunaan berkelanjutan, "
        "tambahkan penyimpanan persisten eksternal sebelum mengandalkan jurnal."
    )

st.divider()
st.caption("Research / paper mode only. No broker connection. No automatic order execution.")
