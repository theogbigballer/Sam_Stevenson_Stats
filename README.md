# Big 12 Player Stats (Men's D1)

Compares current NCAA Division I per-game stats for Big 12 players.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Data source: NCAA.com individual stats pages.
- Turnovers and plus/minus are not published as individual stats on NCAA.com, so they are shown as N/A.
