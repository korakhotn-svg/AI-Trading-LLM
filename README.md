# AI GOLD RESEARCH

Portable XAUUSD research factory handoff repo.

## Restore

1. Clone repo.
2. Create venv with Python 3.12.
3. Install dependencies from `requirements.txt`.
4. Copy `.env.example` to `.env` and add secrets locally.
5. Rebuild/download large market data using scripts, or restore from external storage.

## Important

Large data is not stored in git:
- `data/dukascopy/raw_ticks/`
- `data/dukascopy/ticks_csv_gz/`
- `data/dukascopy/bars_csv_gz/`
- MT5 `.tst` cache files
- secret `.env`

Use Codex for code/execution only. Use ChatGPT for strategy interpretation.
