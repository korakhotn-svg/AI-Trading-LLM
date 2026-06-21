from __future__ import annotations

import argparse

from config import STRATEGIES_DIR, ensure_directories
from llm_strategy_generator import generate_llm_strategy, latest_performance_context, save_llm_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an LLM-style XAUUSD strategy JSON from latest reports.")
    parser.add_argument("--output", default="xau_llm_strategy_v1.json")
    parser.add_argument("--model", default="gpt-5-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    context = latest_performance_context()
    strategy = generate_llm_strategy(context, model=args.model)
    output = STRATEGIES_DIR / args.output
    save_llm_strategy(strategy, output)
    print(f"strategy={output}")
    print(strategy)


if __name__ == "__main__":
    main()
