from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
STRATEGIES_DIR = ROOT_DIR / "strategies"
EXPORTS_DIR = ROOT_DIR / "exports"
LOGS_DIR = ROOT_DIR / "logs"

SYMBOL = "XAUUSD"
TRADING_STYLE = "Hybrid"
ENTRY_TIMEFRAME = "M5"
CONFIRMATION_TIMEFRAME = "M15"
TREND_TIMEFRAME = "H1"
RISK_PER_TRADE = 0.01
M1_HISTORY_BARS = 20000
DATA_SOURCE = "DUKASCOPY"  # MT5 or DUKASCOPY
DUKASCOPY_SYMBOL = "XAUUSD"
DUKASCOPY_PRICE_SCALE = 1000
DUKASCOPY_YEARS = 5

RESULTS_CSV = RESULTS_DIR / "strategy_results.csv"
BEST_STRATEGY_JSON = RESULTS_DIR / "best_strategy.json"
EA_EXPORT_PATH = EXPORTS_DIR / "EA_XAU_HYBRID.mq5"
SERVER_TIME_REPORT = LOGS_DIR / "server_time_offset.json"
BROKER_TIME_OFFSET_REPORT = LOGS_DIR / "broker_time_offset.json"
DUKASCOPY_DIR = DATA_DIR / "dukascopy"
DUKASCOPY_RAW_DIR = DUKASCOPY_DIR / "raw_ticks"
DUKASCOPY_TICK_DIR = DUKASCOPY_DIR / "ticks_csv_gz"
DUKASCOPY_BARS_DIR = DUKASCOPY_DIR / "bars_csv_gz"

SESSIONS_UTC = {
    "asia": (0, 8),
    "london": (7, 16),
    "new_york": (12, 21),
    "london_new_york_overlap": (12, 16),
}

load_dotenv(ROOT_DIR / ".env")


def ensure_directories() -> None:
    for directory in (
        DATA_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        STRATEGIES_DIR,
        EXPORTS_DIR,
        LOGS_DIR,
        DUKASCOPY_RAW_DIR,
        DUKASCOPY_TICK_DIR,
        DUKASCOPY_BARS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
