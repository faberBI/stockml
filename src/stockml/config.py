"""Configurazione centralizzata (override via variabili d'ambiente)."""
import os
from pathlib import Path

ROOT = Path(os.getenv("STOCKML_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACTS_DIR = Path(os.getenv("STOCKML_ARTIFACTS", ROOT / "artifacts"))

TICKERS = os.getenv("STOCKML_TICKERS", "SPY,QQQ,AAPL,MSFT,NVDA").split(",")
PERIOD = os.getenv("STOCKML_PERIOD", "5y")       # storico scaricato da Yahoo
N_SPLITS = int(os.getenv("STOCKML_N_SPLITS", "5")) # fold walk-forward
PRIMARY_METRIC = "rmse"                           # metrica che decide il vincitore (minore = meglio)
RANDOM_STATE = 42

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
LEADERBOARD_JSON = ARTIFACTS_DIR / "leaderboard.json"
HISTORY_CSV = ARTIFACTS_DIR / "leaderboard_history.csv"
LEADERBOARD_MD = ROOT / "LEADERBOARD.md"
README = ROOT / "README.md"
