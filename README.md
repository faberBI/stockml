# 📈 StockML Pipeline

![CI](https://github.com/faberBI/stockml/actions/workflows/ci.yml/badge.svg)
![Weekly training](https://github.com/faberBI/stockml/actions/workflows/weekly-train.yml/badge.svg)

Pipeline end-to-end: **Yahoo Finance → >50 modelli ML → leaderboard → modello migliore → FastAPI → Streamlit**,
con retraining settimanale e CI/CD su GitHub Actions.

## 🏁 Leaderboard (aggiornata automaticamente ogni settimana)

<!-- LEADERBOARD:START -->
**Ultimo training:** `2026-10-19T10:34:19Z` · **Ticker:** SPY, QQQ, AAPL, MSFT, NVDA · **Dati:** 2021-12-28 → 2026-10-15 (6025 righe)

Metrica di selezione: **RMSE** (walk-forward CV, 5 fold). Il vincitore ⚠️ NON batte la baseline naive.

|   rank | model                 |    rmse |   rmse_std |     mae |   hit_rate |       ic |
|-------:|:----------------------|--------:|-----------:|--------:|-----------:|---------:|
|      1 | naive_zero (baseline) | 0.01873 |    0.00210 | 0.01240 |    0.00080 |  0.00000 |
|      2 | 🏆 knn_k200            | 0.01878 |    0.00224 | 0.01246 |    0.51440 |  0.01787 |
|      3 | ard                   | 0.01879 |    0.00224 | 0.01243 |    0.53460 |  0.00000 |
|      4 | enet_a0.1_l10.8       | 0.01879 |    0.00224 | 0.01243 |    0.53460 |  0.00000 |
|      5 | extratrees_d4         | 0.01879 |    0.00227 | 0.01247 |    0.52520 |  0.00822 |
|      6 | rf_d4                 | 0.01880 |    0.00228 | 0.01250 |    0.52400 |  0.01172 |
|      7 | knn_k100              | 0.01880 |    0.00220 | 0.01251 |    0.51080 |  0.02256 |
|      8 | knn_k100_uniform      | 0.01880 |    0.00221 | 0.01251 |    0.50940 |  0.02155 |
|      9 | extratrees_d8         | 0.01880 |    0.00227 | 0.01249 |    0.51180 |  0.00439 |
|     10 | lasso_a0.05           | 0.01881 |    0.00226 | 0.01244 |    0.53460 |  0.01392 |
|     11 | sgd_huber             | 0.01882 |    0.00224 | 0.01244 |    0.51120 |  0.03811 |
|     12 | ridge_a1000           | 0.01884 |    0.00232 | 0.01246 |    0.50780 |  0.01549 |
|     13 | rf_d8                 | 0.01885 |    0.00228 | 0.01257 |    0.50920 |  0.01041 |
|     14 | bayesian_ridge        | 0.01886 |    0.00235 | 0.01247 |    0.50920 |  0.01876 |
|     15 | bagging_tree          | 0.01886 |    0.00228 | 0.01255 |    0.51640 |  0.01863 |
|     16 | svr_rbf_c0.1          | 0.01886 |    0.00212 | 0.01251 |    0.51060 |  0.02039 |
|     17 | enet_a0.1_l10.2       | 0.01887 |    0.00236 | 0.01246 |    0.53280 |  0.01600 |
|     18 | adaboost              | 0.01889 |    0.00230 | 0.01253 |    0.47500 | -0.00191 |
|     19 | pls_2                 | 0.01890 |    0.00240 | 0.01249 |    0.49820 |  0.01155 |
|     20 | tree_d3               | 0.01891 |    0.00223 | 0.01262 |    0.52900 |  0.01743 |
|     21 | catboost_d4_huber     | 0.01892 |    0.00228 | 0.01261 |    0.49900 |  0.00784 |
|     22 | knn_k50               | 0.01892 |    0.00219 | 0.01261 |    0.51020 |  0.01039 |
|     23 | ridge_a100            | 0.01895 |    0.00250 | 0.01252 |    0.51240 |  0.03148 |
|     24 | lasso_a0.01           | 0.01895 |    0.00251 | 0.01251 |    0.51700 |  0.01703 |
|     25 | enet_a0.01_l10.8      | 0.01896 |    0.00254 | 0.01252 |    0.51400 |  0.01877 |
|     26 | linear_svr            | 0.01898 |    0.00249 | 0.01251 |    0.51640 |  0.04320 |
|     27 | catboost_d4           | 0.01900 |    0.00231 | 0.01267 |    0.51100 |  0.00852 |
|     28 | enet_a0.01_l10.2      | 0.01902 |    0.00263 | 0.01256 |    0.51440 |  0.03004 |
|     29 | huber                 | 0.01904 |    0.00263 | 0.01254 |    0.51860 |  0.04180 |
|     30 | ridge_a10             | 0.01905 |    0.00267 | 0.01258 |    0.51000 |  0.03411 |
|     31 | xgb_d2                | 0.01905 |    0.00238 | 0.01268 |    0.51120 |  0.00076 |
|     32 | knn_k25               | 0.01905 |    0.00223 | 0.01279 |    0.50520 |  0.01207 |
|     33 | lasso_a0.001          | 0.01905 |    0.00268 | 0.01258 |    0.51440 |  0.03191 |
|     34 | pls_5                 | 0.01906 |    0.00267 | 0.01262 |    0.50420 |  0.02606 |
|     35 | ridge_a1              | 0.01907 |    0.00271 | 0.01260 |    0.50920 |  0.03394 |
|     36 | ridge_a0.1            | 0.01907 |    0.00272 | 0.01260 |    0.50960 |  0.03393 |
|     37 | ols                   | 0.01907 |    0.00272 | 0.01260 |    0.50940 |  0.03392 |
|     38 | catboost_d6           | 0.01907 |    0.00232 | 0.01277 |    0.49860 | -0.00078 |
|     39 | mlp_small             | 0.01913 |    0.00239 | 0.01277 |    0.50240 |  0.01912 |
|     40 | gbr_huber             | 0.01915 |    0.00239 | 0.01271 |    0.50680 |  0.00087 |
|     41 | hgb_d3_lr0.03         | 0.01918 |    0.00243 | 0.01285 |    0.50000 | -0.00529 |
|     42 | mlp_large             | 0.01918 |    0.00256 | 0.01289 |    0.51160 |  0.05657 |
|     43 | lasso_poly2           | 0.01920 |    0.00277 | 0.01267 |    0.49820 | -0.00283 |
|     44 | hgb_d2_lr0.05         | 0.01921 |    0.00242 | 0.01285 |    0.50500 | -0.00118 |
|     45 | hgb_d5_lr0.02         | 0.01924 |    0.00239 | 0.01288 |    0.50140 | -0.01120 |
|     46 | xgb_d3                | 0.01924 |    0.00234 | 0.01282 |    0.51060 |  0.01327 |
|     47 | mlp_medium            | 0.01927 |    0.00259 | 0.01290 |    0.50000 | -0.00988 |
|     48 | tree_d6               | 0.01927 |    0.00234 | 0.01304 |    0.51620 |  0.00274 |
|     49 | lgbm_l7               | 0.01928 |    0.00242 | 0.01289 |    0.51160 |  0.01649 |
|     50 | svr_rbf_c1            | 0.01935 |    0.00225 | 0.01292 |    0.50980 |  0.01281 |
|     51 | xgb_d4                | 0.01941 |    0.00232 | 0.01294 |    0.51200 |  0.01704 |
|     52 | ridge_poly2           | 0.01952 |    0.00297 | 0.01292 |    0.49580 |  0.00405 |
|     53 | lgbm_l15              | 0.01958 |    0.00240 | 0.01318 |    0.49900 |  0.01257 |
|     54 | lgbm_l31              | 0.01979 |    0.00231 | 0.01341 |    0.49980 |  0.00653 |

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
