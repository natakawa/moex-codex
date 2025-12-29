# moex-portfolio

Python project for building and defending a MOEX-based portfolio:

- Data source: `moexapi` (MOEX ISS).
- All pulled data is saved to CSV (`data/raw`), so you can recompute manually.
- Portfolio construction: constrained near-max Sharpe with robustness hooks (rolling windows, turnover penalty).

## Quick start

Install:

```bash
python3 -m pip install -e .
```

Run the pipeline (downloads ~5 years daily candles for the configured universe):

```bash
moexpf fetch
moexpf build
moexpf analyze
moexpf optimize
moexpf robustness
moexpf backtest
moexpf report
```

All outputs go into `data/`.

## Configure universe / constraints

Edit `config/universe.yml` and `config/strategy.yml`.

## Web

```bash
moexpf web
```

## TODO / rubric checklist

See `docs/TODO.md`.

## Web View

To view the portfolio dashboard:

```bash
moexpf web
```

Or manually:

```bash
streamlit run src/moex_portfolio/web.py
```
