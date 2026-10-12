# 📈 StockML Pipeline

![CI](https://github.com/faberBI/stockml/actions/workflows/ci.yml/badge.svg)
![Weekly training](https://github.com/faberBI/stockml/actions/workflows/weekly-train.yml/badge.svg)

Pipeline end-to-end: **Yahoo Finance → 5 modelli ML → leaderboard → modello migliore → FastAPI → Streamlit**,
con retraining settimanale e CI/CD su GitHub Actions.

## 🏁 Leaderboard (aggiornata automaticamente ogni settimana)

<!-- LEADERBOARD:START -->
**Ultimo training:** `2026-10-12T12:50:00Z` · **Ticker:** SPY, QQQ, AAPL, MSFT, NVDA · **Dati:** 2021-12-20 → 2026-10-08 (6025 righe)

Metrica di selezione: **RMSE** (walk-forward CV, 5 fold). Il vincitore ⚠️ NON batte la baseline naive.

|   rank | model                  |    rmse |   rmse_std |     mae |   hit_rate |       ic |
|-------:|:-----------------------|--------:|-----------:|--------:|-----------:|---------:|
|      1 | naive_zero (baseline)  | 0.01882 |    0.00215 | 0.01246 |    0.00080 |  0.00000 |
|      2 | 🏆 knn                  | 0.01887 |    0.00220 | 0.01256 |    0.50760 |  0.01603 |
|      3 | random_forest          | 0.01888 |    0.00227 | 0.01257 |    0.52140 |  0.01271 |
|      4 | ridge                  | 0.01909 |    0.00266 | 0.01263 |    0.51060 |  0.02765 |
|      5 | mlp                    | 0.01917 |    0.00227 | 0.01281 |    0.49180 | -0.01700 |
|      6 | hist_gradient_boosting | 0.01946 |    0.00245 | 0.01307 |    0.49800 |  0.01333 |

<!-- LEADERBOARD:END -->

## Architettura

```
 ┌──────────── GitHub Actions ─────────────┐
 │ weekly-train.yml (cron lunedì 05:00 UTC)│
 │   yfinance → features → 5 modelli       │
 │   walk-forward CV → leaderboard         │
 │   commit: artifacts/ + README + LB.md   │
 └───────────────┬─────────────────────────┘
                 │ workflow_run
 ┌───────────────▼─────────────────────────┐
 │ cd.yml: CI gate → docker build → GHCR   │
 │         → deploy hook (Render/Railway)  │
 └───────────────┬─────────────────────────┘
                 ▼
   FastAPI (best_model.joblib)  ◄── HTTP ──  Streamlit app
```

## Problema ML
Previsione del **log-return a 1 giorno** con un modello *pooled* su più ticker (feature scale-free:
momentum, volatilità, RSI, distanza dalle medie, range, volume z-score).
Validazione **walk-forward per date** con gap di 1 giorno (niente leakage). Vince il modello con RMSE
più basso; in classifica c'è anche una baseline `naive_zero` per capire se il modello aggiunge valore.

| Modello | Perché |
|---|---|
| Ridge | baseline lineare regolarizzata |
| Random Forest | non-linearità, robusto |
| HistGradientBoosting | boosting veloce stile LightGBM |
| KNN | approccio per analogia |
| MLP | rete neurale piccola |

## Avvio locale
```bash
make install      # pip install -e ".[api,app,dev]"
make train        # scarica dati, allena, scrive artifacts/ e leaderboard
make api          # http://localhost:8000/docs
make app          # http://localhost:8501
# oppure tutto in container
make up
```

## Endpoint API
| Metodo | Path | Descrizione |
|---|---|---|
| GET | `/health` | stato + modello caricato |
| GET | `/leaderboard` | classifica completa (JSON) |
| GET | `/model` | metadati del modello in produzione |
| GET | `/predict/{ticker}` | previsione t+1 + storico prezzi |

## Configurazione
Variabili d'ambiente (o *Repository variables* in GitHub): `STOCKML_TICKERS`, `STOCKML_PERIOD`,
`STOCKML_N_SPLITS`. Secrets opzionali per il deploy: `DEPLOY_HOOK_API`, `DEPLOY_HOOK_APP`.

> ⚠️ Progetto didattico, non è una consulenza finanziaria.
