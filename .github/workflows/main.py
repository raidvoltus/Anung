name: Stock Analysis Daily

on:
  schedule:
    - cron: '0 2 * * *'  # setiap hari jam 09:00 WIB (02:00 UTC)
  workflow_dispatch:     # bisa dijalankan manual dari GitHub

jobs:
  run-stock-analysis:
    runs-on: ubuntu-latest

    env:
      TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
      TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}

    steps:
      - name: Checkout Repo
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install yfinance lightgbm tensorflow scikit-learn pandas numpy ta joblib requests matplotlib

      - name: Run Analysis Script
        run: python your_script.py
