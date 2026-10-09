"""Download dati da Yahoo Finance."""
from __future__ import annotations

import pandas as pd


def download_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Ritorna un DataFrame long: date, ticker, open, high, low, close, volume."""
    import yfinance as yf  # import lazy: test e API non dipendono dalla rete

    raw = yf.download(tickers, period=period, auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    frames = []
    for t in tickers:
        df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        df = df.dropna(how="all").rename(columns=str.lower).reset_index()
        df = df.rename(columns={"Date": "date", "index": "date"})
        df["ticker"] = t
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise RuntimeError(f"Nessun dato scaricato per {tickers}")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
