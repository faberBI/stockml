import numpy as np
import pandas as pd
import pytest


def synthetic_prices(tickers=("AAA", "BBB"), n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    frames = []
    for t in tickers:
        r = rng.normal(0.0003, 0.012, n)
        close = 100 * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": rng.integers(1e6, 5e6, n)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def prices():
    return synthetic_prices()


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Redirige artifacts/README/LEADERBOARD in una dir temporanea."""
    from stockml import config
    (tmp_path / "artifacts").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text("# x\n<!-- LEADERBOARD:START -->\nold\n<!-- LEADERBOARD:END -->\n")
    for k, v in {
        "ARTIFACTS_DIR": tmp_path / "artifacts",
        "MODEL_PATH": tmp_path / "artifacts/best_model.joblib",
        "LEADERBOARD_JSON": tmp_path / "artifacts/leaderboard.json",
        "HISTORY_CSV": tmp_path / "artifacts/leaderboard_history.csv",
        "LEADERBOARD_MD": tmp_path / "LEADERBOARD.md",
        "README": readme, "N_SPLITS": 3, "TICKERS": ["AAA", "BBB"],
    }.items():
        monkeypatch.setattr(config, k, v)
    return tmp_path
