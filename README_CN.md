# Auto Alpha Research：自动化 Alpha 因子发现与验证

本项目是在原 V1–V6 Auto Alpha 代码基础上整理的 **GitHub 公开研究版**，重点是自动生成表达式、利用数据评分、组合筛选和分析因子稳定性，而非宣称实盘可盈利。

## 研究流程

1. **V1 随机公式生成**：通过 OHLCV 和时间序列算子构造语法表达式。
2. **V2 LSTM / Transformer**：学习公式 token 序列并生成新候选因子。
3. **V3 REINFORCE**：根据历史因子评价奖励，更新神经网络公式生成器。
4. **V4 多目标评分**：ICIR、Rank ICIR、相关性、多样性、复杂度及滚动 IC 衰减。
5. **V5 时间序列验证**：Walk-forward 窗口下对因子进行评估。
6. **V6 综合框架**：Predictor AUC 诊断、训练窗口内选因子、单独时间区间的冻结测试、CSV 信号适配 VeighNa。

**注意：** 原始 V6 中曾存在候选生成利用全区间数据、`zscore` 未来信息问题、明文数据接口密钥、Predictor 预处理问题和不现实的回测成交时点。公开版已为新的运行流程修复或隔离相关问题；以前保存的历史收益没有经新代码真实行情复现，**不能表述为经过严格独立样本外验证的收益**。

## 一键复现流程（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python run_auto_alpha.py --demo --mode random --quick
python -m pytest -q
```

`--demo` 生成虚拟合成行情，只用于证明程序链路可以运行，不是真实 A 股市场效果。运行产生的 CSV 和指标自动写入 `artifacts/`（已被 Git 忽略）。

若已有自己**获授权**的市场数据：

```powershell
python run_auto_alpha.py --data ".\local_data\my_market_data.csv" --mode transformer --experiment v6_transformer
python run_auto_alpha.py --data ".\local_data\my_market_data.csv" --mode lstm --experiment v6_lstm
python run_model_comparison.py
```

快速模式下，神经网络仅训练 1 轮，**不会运行 REINFORCE**；如果无法生成合法公式，程序会明确显示 `DEMO FALLBACK`，用随机公式继续完成管线测试。该结果绝不代表 LSTM / Transformer 学到了有效 Alpha。

CSV 字段：`datetime,symbol,open,high,low,close,volume,amount`；支持 `YYYY-MM-DD` 与 `YYYYMMDD`。

## 实验输出与方法细节

- 表达式：`generated_factors.csv`
- 因子评分：`factor_scores.csv`、`selected_factors_v6.csv`
- 诊断：`factor_diversity_report.csv`、`factor_decay_report.csv`、`predictor_report.csv`
- 开发期时序验证：`strict_walk_forward_*.csv`
- 开发期策略示例：`backtest_results.csv`、`equity_curve.csv`
- 冻结区间示例：`holdout_backtest_results.csv`、`holdout_equity_curve.csv`
- 无未来标签的接口：`alpha_signals.csv`、`holdout_alpha_signals.csv`

详细假设： [Methodology](docs/METHODOLOGY.md) · [Limitations](docs/LIMITATIONS.md) · [Archived experiments](docs/ARCHIVED_EXPERIMENTS.md) · [GitHub 上传说明](GITHUB_UPLOAD.md)

**严谨性说明：** 生成和训练过程中保留最后 20% 的日历日期不参与拟合、选因子或定符号。但这些输入数据如已在历史 V1–V6 实验中反复查看，不能把新版本时间留出自动等同于“完全未被研究者接触”的真正盲测。示例策略假设收盘后产生信号、下一交易日开盘执行，仍未模拟 A 股做空限制、涨跌停、调仓换手及真实订单成交。

**公开前务必失效原项目包中的 Tushare Token，并确认实习代码公开授权。**
