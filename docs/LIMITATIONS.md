# Known limitations (read before interpreting results)

- **Historical full-sample selection bias:** Previous V6 rewards and factor selection used the full available input data even when subsequent windows were called "strict." Those archived window metrics are not fully independent OOS results; public code now restricts candidate search and reward fitting to a development block and reserves a final chronological block.
- **Adaptive research risk:** A time holdout in a dataset the researcher previously examined is not a fresh blind holdout; repeated use of holdout performance for tuning compromises independence.
- **Research labels vs fills:** Original close-to-close target does not align with same-close execution. Public backtest uses the next-open-to-following-open proxy, ignoring pre-open order placement constraints, gaps and liquidity.
- **A-share tradability:** Pure equal-weight long-short cross-sectional spreads do not model borrowing, short-sale eligibility, limits, T+1 selling restrictions, halts or cost impacts. Simulation statistics are not executable returns.
- **Trading cost approximation:** Costs are flat per day/period rather than applied per actual turnover, volume and venue. Net return estimates are illustrative only.
- **Survivorship/corporate actions:** No point-in-time index constituent controls, delisting/adjustment database, or complete dividend/split handling. Real universe selection must be done externally.
- **Panel IC robustness:** Small cross-sections, correlated stocks, temporal dependence and regime shifts need appropriate uncertainty estimates, bootstrap or heteroskedasticity/autocorrelation-robust inference.
- **Formula generation:** LSTM/Transformer can produce syntactically invalid expressions. Quick demo may fall back to random formulas; no claim is made that every run finds valid alpha.
- **Predictor leakage:** The original predictor computed medians on all chronological rows before splitting. Public code uses only training-period medians and split by date; repeated model selection on development AUC remains a limitation.
- **VeighNa adapter:** Monitor-only by default. Symbol mapping, timezone, position state, live order handling and fills are unvalidated. Enabling live orders is at user risk and *not recommended* based on this prototype.
- **Data/rights/security:** Original private archive contained a Tushare credential; rotate it. No raw Tushare data / experiment weights or restricted employer research data is included here. Publication requires permission from the rights holder.
