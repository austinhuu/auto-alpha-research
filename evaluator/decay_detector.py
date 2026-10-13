import numpy as np
import pandas as pd
from scipy.stats import linregress


def _safe_corr(x, y):
    valid = x.notna() & y.notna()

    if valid.sum() < 5:
        return np.nan

    x = x[valid]
    y = y[valid]

    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan

    return x.corr(y)


def calculate_daily_ic_series(factor_matrix, factor_id):
    rows = []

    for dt, g in factor_matrix.groupby("datetime"):
        ic = _safe_corr(g[factor_id], g["future_return"])

        rows.append(
            {
                "datetime": dt,
                "factor_id": factor_id,
                "daily_ic": ic,
            }
        )

    return pd.DataFrame(rows)


def calculate_decay_metrics(
    factor_matrix,
    factor_cols,
    rolling_window=20,
    recent_window=40,
):
    """
    输出每个因子的衰减指标：

    mean_ic：全样本平均 IC
    recent_ic：最近一段时间平均 IC
    ic_slope：rolling IC 的趋势斜率
    decay_penalty：衰减惩罚，越大代表衰减越明显
    """
    rows = []

    for factor_id in factor_cols:
        ic_df = calculate_daily_ic_series(factor_matrix, factor_id)
        ic_df = ic_df.dropna(subset=["daily_ic"]).copy()

        if len(ic_df) < max(rolling_window, 10):
            rows.append(
                {
                    "factor_id": factor_id,
                    "mean_ic": np.nan,
                    "recent_ic": np.nan,
                    "ic_slope": np.nan,
                    "decay_penalty": 0.0,
                }
            )
            continue

        ic_df["rolling_ic"] = ic_df["daily_ic"].rolling(
            rolling_window,
            min_periods=max(5, rolling_window // 2),
        ).mean()

        valid_rolling = ic_df.dropna(subset=["rolling_ic"])

        if len(valid_rolling) >= 5:
            x = np.arange(len(valid_rolling))
            y = valid_rolling["rolling_ic"].values

            slope = linregress(x, y).slope
        else:
            slope = 0.0

        mean_ic = ic_df["daily_ic"].mean()
        recent_ic = ic_df["daily_ic"].tail(recent_window).mean()

        # 衰减逻辑：
        # 1. 最近 IC 明显低于整体 IC
        # 2. rolling IC 趋势斜率为负
        mean_abs = abs(mean_ic) if pd.notna(mean_ic) else 0.0
        recent_abs = abs(recent_ic) if pd.notna(recent_ic) else 0.0

        drop_ratio = max(0.0, mean_abs - recent_abs) / (mean_abs + 1e-12)
        slope_penalty = max(0.0, -slope)

        decay_penalty = drop_ratio + slope_penalty

        rows.append(
            {
                "factor_id": factor_id,
                "mean_ic": mean_ic,
                "recent_ic": recent_ic,
                "ic_slope": slope,
                "decay_penalty": decay_penalty,
            }
        )

    return pd.DataFrame(rows)