@echo off
cd /d "C:\workspace\ai-crypto-research-bot"
".\.venv\Scripts\python.exe" -m scripts.continuous_controller --parallel 1 --artifacts-root experiments/artifacts --prices-csv "examples\sample_prices_for_cli.csv" %*
