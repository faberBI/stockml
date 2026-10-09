# 🏁 Model Leaderboard

**Ultimo training:** `2026-10-09T18:52:21Z` · **Ticker:** ^GSPC · **Dati:** 2016-12-19 → 2026-10-08 (2464 righe)

Metrica di selezione: **RMSE** (walk-forward CV, 5 fold). Il vincitore ⚠️ NON batte la baseline naive.

|   rank | model                  |    rmse |   rmse_std |     mae |   hit_rate |       ic |
|-------:|:-----------------------|--------:|-----------:|--------:|-----------:|---------:|
|      1 | naive_zero (baseline)  | 0.01193 |    0.00307 | 0.00801 |    0.00000 |  0.00000 |
|      2 | 🏆 knn                  | 0.01197 |    0.00310 | 0.00803 |    0.51171 | -0.04635 |
|      3 | random_forest          | 0.01203 |    0.00310 | 0.00812 |    0.49317 | -0.04891 |
|      4 | mlp                    | 0.01219 |    0.00320 | 0.00820 |    0.48098 | -0.03881 |
|      5 | ridge                  | 0.01224 |    0.00308 | 0.00847 |    0.49171 |  0.01428 |
|      6 | hist_gradient_boosting | 0.01286 |    0.00360 | 0.00859 |    0.50829 | -0.00310 |

Storico completo: [`artifacts/leaderboard_history.csv`](artifacts/leaderboard_history.csv)
