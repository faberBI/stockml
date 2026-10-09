# 📈 StockML Pipeline

![CI](https://github.com/faberBI/stockml/actions/workflows/ci.yml/badge.svg)
![Weekly training](https://github.com/faberBI/stockml/actions/workflows/weekly-train.yml/badge.svg)

Pipeline end-to-end: **Yahoo Finance → 5 modelli ML → leaderboard → modello migliore → FastAPI → Streamlit**,
con retraining settimanale e CI/CD su GitHub Actions.

## 🏁 Leaderboard (aggiornata automaticamente ogni settimana)

<!-- LEADERBOARD:START -->
_Nessun training ancora eseguito: lancia il workflow **Weekly training** dal tab Actions._
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
