"""FastAPI: espone leaderboard e previsioni del modello migliore."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .. import __version__, config
from ..data import download_prices
from ..features import FEATURES, build_features

app = FastAPI(title="StockML API", version=__version__)


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "Modello non ancora allenato: esegui `stockml-train`")
    return joblib.load(config.MODEL_PATH)


def fetch_prices(ticker: str, period: str = "1y") -> pd.DataFrame:  # punto di mock nei test
    return download_prices([ticker], period)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "model_loaded": config.MODEL_PATH.exists()}


@app.get("/leaderboard")
def leaderboard():
    if not config.LEADERBOARD_JSON.exists():
        raise HTTPException(404, "Leaderboard non disponibile")
    return json.loads(config.LEADERBOARD_JSON.read_text())


@app.get("/model")
def model_info():
    return load_bundle()["meta"]


@app.get("/predict/{ticker}")
def predict(ticker: str, history_days: int = Query(120, ge=10, le=1000)):
    bundle = load_bundle()
    try:
        prices = fetch_prices(ticker.upper())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"Dati non disponibili per {ticker}: {e}") from e
    feats = build_features(prices, dropna_target=False)
    if feats.empty:
        raise HTTPException(422, "Storico insufficiente per calcolare le feature")
    last = feats.iloc[[-1]]
    pred = float(bundle["model"].predict(last[FEATURES])[0])
    close = float(last["close"].iloc[0])
    hist = prices.tail(history_days)
    return {
        "ticker": ticker.upper(),
        "as_of": str(last["date"].iloc[0].date()),
        "model": bundle["meta"]["best_model"],
        "trained_at": bundle["meta"]["trained_at"],
        "pred_log_return": pred,
        "pred_direction": "UP" if pred > 0 else "DOWN",
        "last_close": close,
        "pred_next_close": close * float(np.exp(pred)),
        "history": {"date": hist["date"].dt.strftime("%Y-%m-%d").tolist(),
                    "close": hist["close"].round(4).tolist()},
    }
