# Project TODO (rubric-aligned)

This checklist matches the rubric shown in your slides and the “clarity requirement” (every number traceable to method + assumption).

## 1) Correct portfolio analytics (frontier, weights, metrics, consistency) — 35%
- [x] Store all MOEX pulls to CSV (`data/raw/*`) with request metadata (`*.meta.json`).
- [x] Build a common-calendar price panel and price-return panel (`data/processed/*`).
- [x] Compute baseline asset metrics (return/vol/Sharpe/MDD/VaR/ES) and save to CSV (`data/analytics/asset_metrics.csv`).
- [x] Compute and save portfolio weights (constrained max Sharpe) to CSV (`data/analytics/weights_max_sharpe.csv`).
- [ ] Compute **efficient frontier** (GMV + curve under constraints) and export to CSV (`data/analytics/frontier.csv`).
- [ ] Export “chosen portfolio point on frontier” (return/vol/Sharpe) to CSV (`data/analytics/portfolio_point.csv`).
- [ ] Add consistency checks (Σ PSD, weights sum=1, constraint feasibility) and surface them in `report.md`.

## 2) Risk management quality (tail risk, drawdown, concentration, governance) — 30%
- [x] Drawdown (MDD) and tail risk (VaR/ES 95% historical) for assets.
- [ ] Portfolio-level MDD/VaR/ES (not only asset-level).
- [ ] **Concentration metrics**: max weight, top-2 share, HHI, effective-N.
- [ ] **Risk contributions** (MC/RC) from covariance matrix.
- [ ] Governance & monitoring: limits, breach tracking, rebalance schedule.
- [ ] Optional: turnover + transaction cost sensitivity (in backtest).

## 3) Factor-model reasoning (CAPM + multi-factor, interpretation, robustness) — 20%
- [x] CAPM alpha/beta/R² (vs IMOEX) to CSV.
- [ ] **Multi-factor regression** (configurable factor set), export coefficients/R².
- [x] Robustness via rolling weights CSV.
- [ ] Summarize robustness trade-off vs max Sharpe in report.

## 4) Mandate fit & investor communication (constraints, narrative, trade-offs) — 10%
- [x] Constraints configurable via YAML and Streamlit controls.
- [ ] “Cost of constraints”: compare unconstrained vs constrained (or ε-optimal) outcomes.
- [ ] Plain-language narrative of exposures and why they match Mandate 2.

## 5) Presentation quality (structure, transparency, answers) — 5%
- [x] Streamlit dashboard with audit/download hooks.
- [ ] Reproducible **2-slide + appendix** artifact (generated markdown) referencing CSVs:
  - Slide 1: weights + mandate compliance + key metrics
  - Slide 2: frontier position + factor exposures + governance summary
  - Appendix: data/annualization + µ/Σ method + risk metric definitions + factor outputs

