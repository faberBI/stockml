"""Rende la leaderboard visibile su GitHub: LEADERBOARD.md + blocco nel README."""
from __future__ import annotations

import re

import pandas as pd

from . import config

START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"


def to_markdown(lb: pd.DataFrame, meta: dict) -> str:
    t = lb.copy()
    t["model"] = [("🏆 " if m == meta["best_model"] else "") + m for m in t["model"]]
    t = t[["rank", "model", "rmse", "rmse_std", "mae", "hit_rate", "ic"]]
    table = t.to_markdown(index=False, floatfmt=".5f")
    flag = "✅ batte" if meta["beats_baseline"] else "⚠️ NON batte"
    return (
        f"**Ultimo training:** `{meta['trained_at']}` · **Ticker:** {', '.join(meta['tickers'])} · "
        f"**Dati:** {meta['data_start']} → {meta['data_end']} ({meta['n_rows']} righe)\n\n"
        f"Metrica di selezione: **{meta['primary_metric'].upper()}** (walk-forward CV, "
        f"{config.N_SPLITS} fold). Il vincitore {flag} la baseline naive.\n\n{table}\n"
    )


def write_leaderboard(lb: pd.DataFrame, meta: dict) -> None:
    body = to_markdown(lb, meta)
    config.LEADERBOARD_MD.write_text(
        "# 🏁 Model Leaderboard\n\n" + body +
        "\nStorico completo: [`artifacts/leaderboard_history.csv`]"
        "(artifacts/leaderboard_history.csv)\n")
    if config.README.exists():
        txt = config.README.read_text()
        if START in txt:
            txt = re.sub(f"{START}.*?{END}", lambda _: f"{START}\n{body}\n{END}", txt, flags=re.S)
            config.README.write_text(txt)
