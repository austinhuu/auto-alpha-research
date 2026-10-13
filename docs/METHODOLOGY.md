# Methodology and evaluation protocol

## Inputs

OHLCV + amount, per (date, symbol); vwap = amount / volume (the physical units must be harmonized for real market data). Returns and every time-series formula are computed within symbol, and the final formula values are z-scored **cross-sectionally per date**. The `zscore` expression operator is a *causal expanding* transform in this release (the prior full-series z-score was not causal).

## Formula generation

The RandomFactorGenerator produces explicit expressions from a restricted operator set. LSTM / Transformer language models learn expression token sequences from randomly generated formulas and optionally receive policy-gradient updates. The scalar REINFORCE reward is based on development sample ICIR, Rank ICIR, diversity, decay, predictor AUC and penalties. Frequent reuse of development AUC can lead to adaptive overfitting. In the `--quick` neural smoke test, the reinforcement-learning stage is intentionally disabled; random fallback is clearly labeled when used.

The formula evaluator restricts code to allowed arithmetic, field variables and functions using an AST validator. Untrusted formula CSVs must not be treated as general Python programs.

## Targets and available information

- Research target `future_return`: `close[t+1] / close[t] - 1` (general `period` configured in `config.py`). Used for factor IC and neural-predictor label assessment; this is a *close-to-close* direction target.
- Illustrative more executable return proxy `tradable_return`: `open[t+2] / open[t+1] - 1` for default `period=1`. At time `t` the factor uses information through the close. An idealized portfolio enters at the next open (t+1) and exits the following open (t+2). This is still a model, not proof of achievable fills.
- `alpha_signals.csv` excludes both targets to avoid outputting future outcomes as live signals.

## Chronological workflow

1. Load the entire chronological panel and construct labels for offline evaluation.
2. Reserve the last 20% of dates as a final holdout. Purge `period + 1` dates immediately before the holdout from *development* to avoid labeling outcomes across the boundary.
3. Train formula-generator rewards, pick candidates and determine factor signs **only on development dates**.
4. Within development, rolling validation performs per-window training-period selection; drop last `period + 1` training rows to avoid using labels crossing to the window's next test block. Compute historical formula features across the available development period to retain lookback warm-up context; only past values can influence a given date's value.
**Caution:** The formula generator is fit on the whole development block; therefore earlier rolling windows are *model-selection diagnostics*, not a fully nested or truly untouched OOS test of the complete search procedure. The final frozen block is a separate chronological check for this run.

5. Freeze chosen formula IDs and IC signs, calculate their values over the whole history to preserve causal warm-up context, then evaluate **only on final holdout dates**. Holdout metrics should be reviewed after research choices are finalized, not used repeatedly for further tuning.
6. Compare both development and holdout illustrative high-minus-low equity curves (do **not** confuse these with a tradable broker P&L or randomized train/test splits).

## Model / metric definitions

- IC = cross-sectional Pearson correlation of a formula and future close-to-close return by date; Rank IC uses Spearman.
- ICIR = mean daily IC / standard deviation of daily IC, for available dates. Small number of trading days/stock count yields unstable estimates.
- Predictor uses a chronological date-based split, training-only median imputation, scaling and binary cross-entropy. AUC measures ranking, not net portfolio profitability. Predictor evaluations repeated during RL are *development diagnostics*, not independent holdout assessments.
- Correlation filter removes redundant factors using development-period pairs.
- Backtest creates a daily equal-weight long-minus-short spread proxy. Cost is a **simplified constant per-period proxy** for two legs; actual turnover / margin / shorting limits / A-share T+1 are not modeled.

## Reproducibility

`--seed` seeds Python, NumPy and PyTorch. Neural generation and CPU/GPU kernels can still vary across devices or versions. GitHub CI validates the synthetic random baseline and unit tests only. It does not demonstrate a historical market edge.
