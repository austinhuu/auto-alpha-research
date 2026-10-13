import numpy as np
import pandas as pd


def calculate_factor_correlation_matrix(factor_matrix, factor_cols):
    factor_data = factor_matrix[factor_cols].replace([np.inf, -np.inf], np.nan)
    corr_matrix = factor_data.corr().abs()
    return corr_matrix


def calculate_diversity_scores(factor_matrix, factor_cols):
    """
    diversity_score = 1 - mean_abs_corr_with_other_factors

    分数越高，说明这个因子和其他因子越不一样。
    """
    if len(factor_cols) == 0:
        return {}

    if len(factor_cols) == 1:
        return {factor_cols[0]: 1.0}

    corr_matrix = calculate_factor_correlation_matrix(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
    )

    diversity_scores = {}

    for col in factor_cols:
        others = corr_matrix[col].drop(index=col, errors="ignore").dropna()

        if len(others) == 0:
            diversity_scores[col] = 1.0
        else:
            mean_corr = float(others.mean())
            diversity_scores[col] = 1.0 - mean_corr

    return diversity_scores


def build_diversity_report(factor_matrix, factor_cols):
    diversity_scores = calculate_diversity_scores(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
    )

    rows = []

    for factor_id, diversity_score in diversity_scores.items():
        rows.append(
            {
                "factor_id": factor_id,
                "diversity_score": diversity_score,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "diversity_score",
        ascending=False,
    ).reset_index(drop=True)