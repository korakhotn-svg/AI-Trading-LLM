from __future__ import annotations


IMPORTS = [
    "openai",
    "MetaTrader5",
    "pandas",
    "numpy",
    "matplotlib",
    "ta",
    "xgboost",
    "sklearn",
    "backtesting",
    "optuna",
    "scipy",
    "dotenv",
]


def main() -> int:
    failures = []
    for module_name in IMPORTS:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "version n/a")
            print(f"OK {module_name} {version}")
        except Exception as exc:
            failures.append((module_name, exc))
            print(f"FAIL {module_name}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
