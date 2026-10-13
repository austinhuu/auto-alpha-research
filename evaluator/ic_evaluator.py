import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE_COLUMNS = {
    "datetime",
    "symbol",
    "future_return",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
}


def get_factor_columns(df):
    """
    自动识别因子列。

    兼容：
    alpha_0001
    rl_alpha_0001

    同时也兼容以后其他非基础字段的因子列。
    """
    factor_cols = []

    for col in df.columns:
        if col in BASE_COLUMNS:
            continue

        if col.startswith("alpha_") or col.startswith("rl_alpha_"):
            factor_cols.append(col)

    return factor_cols


def _safe_corr(x, y, method="pearson"):
    valid = x.notna() & y.notna()

    if valid.sum() < 5:
        return np.nan

    x = x[valid]
    y = y[valid]

    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan

    if method == "pearson":
        return x.corr(y)

    if method == "spearman":
        return spearmanr(x, y).correlation

    raise ValueError(f"Unknown method: {method}")


def evaluate_factor_ic(factor_matrix):
    """
    按日期计算每个因子的截面 IC / Rank IC，然后汇总。

    返回：
    factor_id
    mean_ic
    mean_rank_ic
    ic_std
    icir
    rank_icir
    abs_icir
    """
    factor_cols = get_factor_columns(factor_matrix)

    rows = []

    if len(factor_cols) == 0:
        print("警告：evaluate_factor_ic 没有识别到任何因子列。")
        print("当前 factor_matrix 字段：")
        print(factor_matrix.columns.tolist())

        return pd.DataFrame(
            columns=[
                "factor_id",
                "mean_ic",
                "mean_rank_ic",
                "ic_std",
                "icir",
                "rank_icir",
                "abs_icir",
            ]
        )

    for factor in factor_cols:
        daily_ic = []
        daily_rank_ic = []

        for _, g in factor_matrix.groupby("datetime"):
            ic = _safe_corr(
                g[factor],
                g["future_return"],
                method="pearson",
            )

            rank_ic = _safe_corr(
                g[factor],
                g["future_return"],
                method="spearman",
            )

            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)

        daily_ic = pd.Series(daily_ic).dropna()
        daily_rank_ic = pd.Series(daily_rank_ic).dropna()

        if len(daily_ic) == 0:
            mean_ic = np.nan
            ic_std = np.nan
            icir = np.nan
        else:
            mean_ic = daily_ic.mean()
            ic_std = daily_ic.std()
            icir = mean_ic / (ic_std + 1e-12)

        if len(daily_rank_ic) == 0:
            mean_rank_ic = np.nan
            rank_icir = np.nan
        else:
            mean_rank_ic = daily_rank_ic.mean()
            rank_icir = mean_rank_ic / (daily_rank_ic.std() + 1e-12)

        rows.append(
            {
                "factor_id": factor,
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
                "ic_std": ic_std,
                "icir": icir,
                "rank_icir": rank_icir,
                "abs_icir": abs(icir) if pd.notna(icir) else np.nan,
            }
        )

    score_df = pd.DataFrame(rows)

    score_df = score_df.sort_values(
        "abs_icir",
        ascending=False,
    ).reset_index(drop=True)

    return score_df