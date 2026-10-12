# 🏁 Model Leaderboard

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

Storico completo: [`artifacts/leaderboard_history.csv`](artifacts/leaderboard_history.csv)
