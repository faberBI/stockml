"""Training: walk-forward CV su date -> metriche -> leaderboard -> salvataggio del migliore."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config
from .data import download_prices
from .features import FEATURES, TARGET, build_features
from .leaderboard import write_leaderboard
from .models import get_models

log = logging.getLogger("stockml")


def date_splits(dates: pd.Series, n_splits: int, gap_days: int = 1):
    """Walk-forward su date uniche: tutti i ticker dello stesso giorno stanno nello stesso fold.
    Il gap evita che il target (t+1) del train si sovrapponga al test."""
    udates = np.sort(dates.unique())
    fold = len(udates) // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = udates[fold * k - 1 - gap_days]
        test_start, test_end = udates[fold * k], udates[min(fold * (k + 1), len(udates)) - 1]
        tr = np.where(dates <= train_end)[0]
        te = np.where((dates >= test_start) & (dates <= test_end))[0]
        yield tr, te


def metrics(y, p) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "hit_rate": float(np.mean(np.sign(y) == np.sign(p))),       # direzione corretta
        "ic": ic if np.isfinite(ic := float(pd.Series(p).corr(pd.Series(y), method="spearman")))
        else 0.0,  # information coefficient (0 se previsione costante, es. baseline)
    }


def evaluate(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    candidates = get_models()
    rows = []
    for name, model in {**candidates, "naive_zero (baseline)": None}.items():
        fold_scores = []
        for tr, te in date_splits(df["date"], n_splits):
            if model is None:
                pred = np.zeros(len(te))
            else:
                pred = clone(model).fit(X[tr], y[tr]).predict(X[te])
            fold_scores.append(metrics(y[te], pred))
        s = pd.DataFrame(fold_scores)
        rows.append({"model": name, **s.mean().to_dict(),
                     "rmse_std": float(s["rmse"].std()), "is_baseline": model is None})
        log.info("%-24s rmse=%.6f hit=%.3f", name, rows[-1]["rmse"], rows[-1]["hit_rate"])
    lb = pd.DataFrame(rows).sort_values(config.PRIMARY_METRIC).reset_index(drop=True)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    return lb


def run(prices: pd.DataFrame | None = None) -> dict:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices = download_prices(config.TICKERS, config.PERIOD)
    df = build_features(prices)
    log.info("Dataset: %d righe, %d ticker, %s -> %s", len(df), df["ticker"].nunique(),
             df["date"].min().date(), df["date"].max().date())

    lb = evaluate(df, config.N_SPLITS)
    best_name = lb.loc[~lb["is_baseline"], "model"].iloc[0]
    best = clone(get_models()[best_name]).fit(df[FEATURES], df[TARGET])  # refit su tutto

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "trained_at": run_ts,
        "best_model": best_name,
        "tickers": config.TICKERS,
        "features": FEATURES,
        "n_rows": int(len(df)),
        "data_start": str(df["date"].min().date()),
        "data_end": str(df["date"].max().date()),
        "primary_metric": config.PRIMARY_METRIC,
        "beats_baseline": bool(
            lb.loc[lb.model == best_name, "rmse"].iloc[0]
            < lb.loc[lb.is_baseline, "rmse"].iloc[0]),
    }
    joblib.dump({"model": best, "meta": meta}, config.MODEL_PATH)
    config.LEADERBOARD_JSON.write_text(
        json.dumps({"meta": meta, "leaderboard": lb.to_dict(orient="records")}, indent=2))

    hist = lb.assign(trained_at=run_ts)
    hist.to_csv(config.HISTORY_CSV, mode="a", index=False,
                header=not config.HISTORY_CSV.exists())
    write_leaderboard(lb, meta)
    log.info("Vincitore: %s (salvato in %s)", best_name, config.MODEL_PATH)
    return meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Allena i 5 modelli e aggiorna la leaderboard")
    ap.add_argument("--tickers", help="es. SPY,QQQ,AAPL")
    ap.add_argument("--period", help="es. 5y, 10y, max")
    a = ap.parse_args()
    if a.tickers:
        config.TICKERS = a.tickers.split(",")
    if a.period:
        config.PERIOD = a.period
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
