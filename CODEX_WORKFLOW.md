# AI GOLD RESEARCH Workflow

## Roles

Codex:
- Create and modify code.
- Read result files.
- Analyze performance.
- Adjust strategy and backtest logic.

Python on VPS:
- Generate/load strategy specs.
- Run tick backtests.
- Run Optuna evolution loops.
- Write result files for Codex to review.

## Data

Current Dukascopy dataset:
- Symbol: XAUUSD
- Year: 2025
- Tick storage: `data/dukascopy/ticks_csv_gz/XAUUSD/2025`
- Format: daily `csv.gz`

## Commands

Build bars from stored ticks:

```powershell
.\venv\Scripts\python.exe tick_to_bars.py --symbol XAUUSD --start 2025-01-01 --end 2025-12-31
```

Run one tick backtest:

```powershell
.\venv\Scripts\python.exe research_runner.py --symbol XAUUSD --start 2025-01-01 --end 2025-12-31
```

Run evolution:

```powershell
.\venv\Scripts\python.exe evolution_runner.py --symbol XAUUSD --start 2025-01-01 --end 2025-12-31 --trials 20
```

Generate an LLM-style strategy from latest reports:

```powershell
.\venv\Scripts\python.exe llm_strategy_runner.py --output xau_llm_strategy_v1.json
.\venv\Scripts\python.exe walk_forward_runner.py --symbol XAUUSD --strategy-file strategies\xau_llm_strategy_v1.json
```

## Private Chat Strategy

The linked ChatGPT conversation is private and cannot be read by Codex directly.
Paste the strategy rules into a file under `strategies/`, or paste them in chat,
then Codex can convert them into a `ResearchStrategy` JSON/spec.
