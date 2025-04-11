name: Stock Analysis & Commit Generated Files

on:
  schedule:
    - cron: '0 2 * * *'   # setiap hari jam 02:00 UTC (09:00 WIB)
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest

    env:
      TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
      TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Analysis Script
        run: |
          python main.py

      - name: Configure Git
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Set Remote URL using GH_PAT
        run: |
          git remote set-url origin https://$GH_PAT@github.com/${{ github.repository }}.git
        env:
          GH_PAT: ${{ secrets.GH_PAT }}

      - name: Commit and Push Changes
        run: |
          git add -A
          # Jika tidak ada perubahan, commit akan gagal sehingga tambahkan || true
          git commit -m "Update generated files by analysis script" || true
          git push origin HEAD:${{ github.ref_name }}
        env:
          GH_PAT: ${{ secrets.GH_PAT }}
