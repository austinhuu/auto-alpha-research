import numpy as np
import pandas as pd


def _pick_first_existing_column(df, candidate_cols, default_value=0.0):
    """
    从多个可能的列名里选择第一个存在的列。
    如果都不存在，则返回一个默认值 Series。
    """
    for col in candidate_cols:
        if col in df.columns:
            return df[col]

    return pd.Series(default_value, index=df.index)


def calculate_factor_stability_score(
    score_df,
    decay_report,
    oos_report,
    icir_weight=0.40,
    oos_weight=0.40,
    decay_weight=0.20,
):
    """
    综合稳定性评分：

    stability_score =
        icir_weight * abs(in_sample_icir)
      + oos_weight * abs(oos_icir)
      - decay_weight * decay_penalty

    这个版本兼容 pandas merge 后出现的列名冲突：
    decay_penalty / decay_penalty_x / decay_penalty_y / decay_penalty_decay
    """

    df = score_df.copy()

    # 避免重复 merge 时产生复杂后缀。
    # 如果 score_df 里已经有 decay_penalty，就不再重复 merge。
    has_decay_col = any(
        col in df.columns
        for col in [
            "decay_penalty",
            "decay_penalty_x",
            "decay_penalty_y",
            "decay_penalty_decay",
        ]
    )

    if not has_decay_col and decay_report is not None and len(decay_report) > 0:
        keep_cols = [
            col for col in [
                "factor_id",
                "decay_penalty",
                "recent_ic",
                "ic_slope",
            ]
            if col in decay_report.columns
        ]

        df = df.merge(
            decay_report[keep_cols],
            on="factor_id",
            how="left",
        )

    # 合并样本外 walk-forward 结果
    if oos_report is not None and len(oos_report) > 0:
        keep_cols = [
            col for col in [
                "factor_id",
                "oos_mean_ic",
                "oos_icir",
                "oos_mean_rank_ic",
                "oos_rank_icir",
                "n_windows",
            ]
            if col in oos_report.columns
        ]

        df = df.merge(
            oos_report[keep_cols],
            on="factor_id",
            how="left",
        )

    # 兼容不同 merge 后缀
    df["decay_penalty_for_stability"] = _pick_first_existing_column(
        df,
        [
            "decay_penalty",
            "decay_penalty_y",
            "decay_penalty_x",
            "decay_penalty_decay",
        ],
        default_value=0.0,
    ).fillna(0.0)

    df["oos_icir_for_stability"] = _pick_first_existing_column(
        df,
        [
            "oos_icir",
            "oos_icir_y",
            "oos_icir_x",
        ],
        default_value=0.0,
    ).fillna(0.0)

    df["icir_for_stability"] = _pick_first_existing_column(
        df,
        [
            "icir",
            "icir_x",
            "icir_y",
        ],
        default_value=0.0,
    ).fillna(0.0)

    df["stability_score"] = (
        icir_weight * df["icir_for_stability"].abs()
        + oos_weight * df["oos_icir_for_stability"].abs()
        - decay_weight * df["decay_penalty_for_stability"]
    )

    df = df.sort_values(
        "stability_score",
        ascending=False,
    ).reset_index(drop=True)

    return df