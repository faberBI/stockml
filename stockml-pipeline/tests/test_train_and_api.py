import json

from fastapi.testclient import TestClient

from stockml import config
from stockml.train import run


def test_pipeline_end_to_end(prices, tmp_project, monkeypatch):
    meta = run(prices)
    assert config.MODEL_PATH.exists()
    lb = json.loads(config.LEADERBOARD_JSON.read_text())["leaderboard"]
    assert len(lb) == 6  # 5 modelli + baseline
    assert meta["best_model"] in {r["model"] for r in lb if not r["is_baseline"]}
    assert "🏆" in config.README.read_text()

    from stockml.api import main
    main.load_bundle.cache_clear()
    monkeypatch.setattr(main, "fetch_prices", lambda t, period="1y": prices[prices.ticker == "AAA"])
    c = TestClient(main.app)
    assert c.get("/health").json()["model_loaded"]
    assert c.get("/leaderboard").status_code == 200
    r = c.get("/predict/AAA").json()
    assert r["pred_direction"] in {"UP", "DOWN"} and r["last_close"] > 0
