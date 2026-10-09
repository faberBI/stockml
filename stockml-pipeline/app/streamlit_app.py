"""Frontend Streamlit: NON carica il modello, parla solo con l'API."""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

def _api_url() -> str:
    if os.getenv("API_URL"):
        return os.environ["API_URL"]
    try:  # Streamlit Community Cloud -> Settings > Secrets
        return st.secrets["API_URL"]
    except Exception:  # noqa: BLE001
        return "http://localhost:8000"


API_URL = _api_url().rstrip("/")

st.set_page_config(page_title="StockML", page_icon="📈", layout="wide")
st.title("📈 StockML – previsione rendimento a 1 giorno")


@st.cache_data(ttl=300)
def get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


try:
    health = get("/health")
    st.sidebar.success(f"API online · v{health['version']}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"API non raggiungibile su {API_URL}: {e}")
    st.stop()

tab_pred, tab_lb = st.tabs(["🔮 Previsione", "🏁 Leaderboard"])

with tab_pred:
    ticker = st.text_input("Ticker Yahoo Finance", "SPY").strip().upper()
    if st.button("Prevedi", type="primary") and ticker:
        try:
            p = get(f"/predict/{ticker}")
        except requests.HTTPError as e:
            st.error(e.response.json().get("detail", str(e)))
            st.stop()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ultima chiusura", f"{p['last_close']:.2f}", help=p["as_of"])
        c2.metric("Close previsto", f"{p['pred_next_close']:.2f}",
                  f"{p['pred_log_return'] * 100:+.3f}%")
        c3.metric("Direzione", p["pred_direction"])
        c4.metric("Modello", p["model"], help=f"trained {p['trained_at']}")
        h = pd.DataFrame(p["history"])
        fig = go.Figure(go.Scatter(x=h["date"], y=h["close"], name="close"))
        fig.add_trace(go.Scatter(x=[h["date"].iloc[-1]], y=[p["pred_next_close"]],
                                 mode="markers", marker=dict(size=12), name="previsione t+1"))
        st.plotly_chart(fig, use_container_width=True)

with tab_lb:
    lb = get("/leaderboard")
    m = lb["meta"]
    st.caption(f"Ultimo training {m['trained_at']} · dati {m['data_start']} → {m['data_end']}")
    df = pd.DataFrame(lb["leaderboard"]).drop(columns="is_baseline")
    st.dataframe(df.style.highlight_min(subset=["rmse", "mae"])
                 .highlight_max(subset=["hit_rate", "ic"]), use_container_width=True)
