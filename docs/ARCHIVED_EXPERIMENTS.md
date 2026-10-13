# Archived V6 experiment context — NOT VALIDATED OOS PERFORMANCE

The following values are *read directly from the original uploaded archive's* `artifacts/model_comparison.csv`. They are included solely so the development history of the portfolio is not lost. The original pipeline used development/full-sample-derived formula rewards and selection. Therefore the `backtest_results.csv` metrics do **not** establish fully independent OOS performance and must not be used as quantified "live-like" track records on a resume.

| Original V6 experiment | Reported cumulative return | Reported Sharpe | Reported max drawdown | Predictor held-out-internal AUC |
|---|---:|---:|---:|---:|
| LSTM formula generator | +14.21% | 0.224 | −23.27% | 0.4961 |
| Transformer formula generator | +208.34% | 2.469 | −20.60% | 0.4963 |

**Interpretation:** The original Transformer run had a much higher recorded development-stage trading metric, yet both predictor internal-test AUC values were close to 0.50. This inconsistency makes validation, timing and trading assumptions especially important to audit. It does not prove genuine strategy profitability. The refined release was not run here on licensed real market data and **no replacement real-market profit figure is claimed**.

Historical experiment outputs are intentionally excluded from the repository because of bulk size, derived market data, and potential local path information. Users with rights to use those data can reproduce research privately and commit only verified, appropriately licensed summaries.
