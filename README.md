# 📈 StockML Pipeline

![CI](https://github.com/faberBI/stockml/actions/workflows/ci.yml/badge.svg)
![Weekly training](https://github.com/faberBI/stockml/actions/workflows/weekly-train.yml/badge.svg)

Pipeline end-to-end: **Yahoo Finance → >50 modelli ML → leaderboard → modello migliore → FastAPI → Streamlit**,
con retraining settimanale e CI/CD su GitHub Actions.

## 🏁 Leaderboard (aggiornata automaticamente ogni settimana)

<!-- LEADERBOARD:START -->
**Ultimo training:** `2026-10-26T10:48:42Z` · **Ticker:** SPY, QQQ, AAPL, MSFT, NVDA · **Dati:** 2022-01-04 → 2026-10-22 (6025 righe)

Metrica di selezione: **RMSE** (walk-forward CV, 5 fold). Il vincitore ⚠️ NON batte la baseline naive.

|   rank | model                 |    rmse |   rmse_std |     mae |   hit_rate |       ic |
|-------:|:----------------------|--------:|-----------:|--------:|-----------:|---------:|
|      1 | naive_zero (baseline) | 0.01868 |    0.00201 | 0.01236 |    0.00080 |  0.00000 |
|      2 | 🏆 enet_a0.1_l10.8     | 0.01873 |    0.00212 | 0.01238 |    0.53580 |  0.00000 |
|      3 | ard                   | 0.01873 |    0.00212 | 0.01238 |    0.53580 |  0.00000 |
|      4 | knn_k200              | 0.01873 |    0.00216 | 0.01242 |    0.51220 |  0.01727 |
|      5 | extratrees_d4         | 0.01874 |    0.00218 | 0.01242 |    0.52120 |  0.00902 |
|      6 | extratrees_d8         | 0.01875 |    0.00218 | 0.01244 |    0.51200 |  0.00561 |
|      7 | knn_k100              | 0.01876 |    0.00213 | 0.01247 |    0.50840 |  0.02225 |
|      8 | knn_k100_uniform      | 0.01876 |    0.00213 | 0.01247 |    0.50840 |  0.02155 |
|      9 | lasso_a0.05           | 0.01876 |    0.00218 | 0.01240 |    0.53580 |  0.01243 |
|     10 | rf_d4                 | 0.01877 |    0.00225 | 0.01248 |    0.51800 |  0.01431 |
|     11 | sgd_huber             | 0.01877 |    0.00216 | 0.01241 |    0.50740 |  0.03825 |
|     12 | adaboost              | 0.01879 |    0.00217 | 0.01246 |    0.47640 | -0.00816 |
|     13 | svr_rbf_c0.1          | 0.01880 |    0.00199 | 0.01247 |    0.51480 |  0.02977 |
|     14 | ridge_a1000           | 0.01881 |    0.00226 | 0.01243 |    0.50360 |  0.00753 |
|     15 | rf_d8                 | 0.01883 |    0.00225 | 0.01255 |    0.50860 |  0.01486 |
|     16 | mlp_large             | 0.01883 |    0.00228 | 0.01255 |    0.50960 |  0.02695 |
|     17 | bayesian_ridge        | 0.01884 |    0.00232 | 0.01245 |    0.50280 |  0.01064 |
|     18 | bagging_tree          | 0.01885 |    0.00226 | 0.01256 |    0.50820 |  0.01311 |
|     19 | enet_a0.1_l10.2       | 0.01885 |    0.00232 | 0.01244 |    0.52160 |  0.01217 |
|     20 | knn_k50               | 0.01887 |    0.00212 | 0.01258 |    0.50660 |  0.01000 |
|     21 | pls_2                 | 0.01889 |    0.00236 | 0.01248 |    0.49840 |  0.00316 |
|     22 | catboost_d4_huber     | 0.01892 |    0.00227 | 0.01263 |    0.50860 |  0.01126 |
|     23 | linear_svr            | 0.01892 |    0.00239 | 0.01248 |    0.51300 |  0.04241 |
|     24 | catboost_d4           | 0.01893 |    0.00228 | 0.01263 |    0.50480 |  0.00616 |
|     25 | ridge_a100            | 0.01894 |    0.00250 | 0.01253 |    0.50800 |  0.02564 |
|     26 | tree_d3               | 0.01895 |    0.00216 | 0.01261 |    0.50000 |  0.00253 |
|     27 | lasso_a0.01           | 0.01896 |    0.00253 | 0.01251 |    0.51300 |  0.01334 |
|     28 | xgb_d2                | 0.01897 |    0.00232 | 0.01265 |    0.50360 | -0.00263 |
|     29 | enet_a0.01_l10.8      | 0.01898 |    0.00256 | 0.01253 |    0.50980 |  0.01507 |
|     30 | knn_k25               | 0.01900 |    0.00215 | 0.01275 |    0.50620 |  0.01387 |
|     31 | huber                 | 0.01903 |    0.00262 | 0.01254 |    0.51620 |  0.04010 |
|     32 | enet_a0.01_l10.2      | 0.01903 |    0.00266 | 0.01258 |    0.51300 |  0.02632 |
|     33 | pls_5                 | 0.01904 |    0.00265 | 0.01263 |    0.50300 |  0.02225 |
|     34 | ridge_a10             | 0.01906 |    0.00270 | 0.01260 |    0.50800 |  0.02984 |
|     35 | catboost_d6           | 0.01906 |    0.00230 | 0.01277 |    0.50480 | -0.00386 |
|     36 | lasso_a0.001          | 0.01907 |    0.00272 | 0.01260 |    0.51180 |  0.02786 |
|     37 | ridge_a1              | 0.01908 |    0.00274 | 0.01262 |    0.50880 |  0.03019 |
|     38 | gbr_huber             | 0.01908 |    0.00233 | 0.01271 |    0.50680 |  0.00034 |
|     39 | ridge_a0.1            | 0.01908 |    0.00275 | 0.01262 |    0.50920 |  0.03029 |
|     40 | ols                   | 0.01908 |    0.00275 | 0.01262 |    0.50920 |  0.03029 |
|     41 | mlp_small             | 0.01911 |    0.00240 | 0.01283 |    0.49420 |  0.01014 |
|     42 | lasso_poly2           | 0.01916 |    0.00271 | 0.01265 |    0.50100 |  0.00466 |
|     43 | mlp_medium            | 0.01917 |    0.00228 | 0.01283 |    0.49380 |  0.01074 |
|     44 | xgb_d3                | 0.01917 |    0.00232 | 0.01283 |    0.50420 |  0.01488 |
|     45 | hgb_d3_lr0.03         | 0.01919 |    0.00243 | 0.01287 |    0.49440 | -0.00969 |
|     46 | hgb_d2_lr0.05         | 0.01922 |    0.00248 | 0.01287 |    0.50160 | -0.01050 |
|     47 | hgb_d5_lr0.02         | 0.01922 |    0.00241 | 0.01288 |    0.49840 | -0.00527 |
|     48 | tree_d6               | 0.01922 |    0.00220 | 0.01297 |    0.50540 | -0.00221 |
|     49 | lgbm_l7               | 0.01926 |    0.00244 | 0.01287 |    0.50640 |  0.02117 |
|     50 | svr_rbf_c1            | 0.01933 |    0.00220 | 0.01291 |    0.51260 |  0.02307 |
|     51 | xgb_d4                | 0.01936 |    0.00239 | 0.01295 |    0.50280 |  0.01069 |
|     52 | ridge_poly2           | 0.01948 |    0.00292 | 0.01289 |    0.49840 |  0.00723 |
|     53 | lgbm_l15              | 0.01963 |    0.00248 | 0.01322 |    0.50120 |  0.00919 |
|     54 | lgbm_l31              | 0.01984 |    0.00237 | 0.01345 |    0.50460 |  0.00439 |

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
