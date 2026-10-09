"""Feature engineering scale-free (rendimenti) -> modello 'pooled' su più ticker."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21",
    "vol_5", "vol_21", "vol_ratio",
    "rsi_14", "dist_sma_20", "dist_sma_50",
    "hl_range", "volume_z",
]
TARGET = "target"  # log-return del giorno successivo


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    lr = np.log(g["close"]).diff()
    for n in (1, 5, 10, 21):
        g[f"ret_{n}"] = np.log(g["close"]).diff(n)
    g["vol_5"] = lr.rolling(5).std()
    g["vol_21"] = lr.rolling(21).std()
    g["vol_ratio"] = g["vol_5"] / g["vol_21"]
    g["rsi_14"] = _rsi(g["close"]) / 100
    g["dist_sma_20"] = g["close"] / g["close"].rolling(20).mean() - 1
    g["dist_sma_50"] = g["close"] / g["close"].rolling(50).mean() - 1
    g["hl_range"] = (g["high"] - g["low"]) / g["close"]
    lv = np.log1p(g["volume"].astype(float))
    g["volume_z"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    g[TARGET] = lr.shift(-1)  # y_t = rendimento t -> t+1 (nessun leakage nelle feature)
    return g


def build_features(prices: pd.DataFrame, dropna_target: bool = True) -> pd.DataFrame:
    df = pd.concat([_per_ticker(g) for _, g in prices.groupby("ticker")], ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    if dropna_target:
        df = df.dropna(subset=[TARGET])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)
