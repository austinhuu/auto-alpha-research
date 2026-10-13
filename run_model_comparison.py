from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

EXPERIMENTS = {
    "LSTM Generator": ARTIFACTS_ROOT / "v6_lstm",
    "Transformer Generator": ARTIFACTS_ROOT / "v6_transformer",
}


def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def read_first_existing_csv(paths):
    """
    依次读取多个候选文件，返回第一个存在的 CSV。
    避免 DataFrame 直接用 or 导致：
    ValueError: The truth value of a DataFrame is ambiguous
    """
    for path in paths:
        df = read_csv_if_exists(path)
        if df is not None:
            return df
    return None


def summarize_experiment(model_name, exp_dir):
    backtest = read_csv_if_exists(exp_dir / "backtest_results.csv")

    selected = read_first_existing_csv(
        [
            exp_dir / "selected_factors_v6.csv",
            exp_dir / "selected_factors_v5.csv",
            exp_dir / "selected_factors.csv",
        ]
    )

    scores = read_first_existing_csv(
        [
            exp_dir / "factor_scores_v5.csv",
            exp_dir / "factor_scores_v4.csv",
            exp_dir / "factor_scores.csv",
        ]
    )

    wf = read_first_existing_csv(
        [
            exp_dir / "strict_walk_forward_summary.csv",
            exp_dir / "walk_forward_summary.csv",
        ]
    )

    history = read_csv_if_exists(exp_dir / "rl_training_history.csv")
    predictor = read_csv_if_exists(exp_dir / "predictor_report.csv")
    diversity = read_csv_if_exists(exp_dir / "factor_diversity_report.csv")
    decay = read_csv_if_exists(exp_dir / "factor_decay_report.csv")

    row = {
        "model": model_name,
        "experiment_dir": str(exp_dir),
    }

    if backtest is not None and len(backtest) > 0:
        bt = backtest.iloc[0].to_dict()
        row.update(
            {
                "total_return": bt.get("total_return"),
                "annual_return": bt.get("annual_return"),
                "annual_volatility": bt.get("annual_volatility"),
                "sharpe": bt.get("sharpe"),
                "sortino": bt.get("sortino"),
                "max_drawdown": bt.get("max_drawdown"),
                "calmar": bt.get("calmar"),
            }
        )

    if selected is not None and len(selected) > 0:
        row["selected_factor_count"] = len(selected)

        if "abs_icir" in selected.columns:
            row["selected_mean_abs_icir"] = selected["abs_icir"].mean()

        if "stability_score" in selected.columns:
            row["selected_mean_stability_score"] = selected["stability_score"].mean()

        if "diversity_score" in selected.columns:
            row["selected_mean_diversity_score"] = selected["diversity_score"].mean()

        if "decay_penalty" in selected.columns:
            row["selected_mean_decay_penalty"] = selected["decay_penalty"].mean()

    if scores is not None and len(scores) > 0:
        if "abs_icir" in scores.columns:
            row["all_factor_mean_abs_icir"] = scores["abs_icir"].mean()

        if "diversity_score" in scores.columns:
            row["all_factor_mean_diversity"] = scores["diversity_score"].mean()

        if "decay_penalty" in scores.columns:
            row["all_factor_mean_decay_penalty"] = scores["decay_penalty"].mean()

        if "stability_score" in scores.columns:
            row["all_factor_mean_stability_score"] = scores["stability_score"].mean()

    if wf is not None and len(wf) > 0:
        if "abs_oos_icir" in wf.columns:
            row["mean_abs_oos_icir"] = wf["abs_oos_icir"].mean()

        if "oos_mean_ic" in wf.columns:
            row["mean_oos_ic"] = wf["oos_mean_ic"].mean()

        if "oos_icir" in wf.columns:
            row["mean_oos_icir"] = wf["oos_icir"].mean()

    if history is not None and len(history) > 0:
        if "avg_reward" in history.columns:
            row["final_avg_reward"] = history["avg_reward"].iloc[-1]

        if "max_reward" in history.columns:
            row["best_max_reward"] = history["max_reward"].max()

        if "predictor_auc" in history.columns:
            row["final_rl_predictor_auc"] = history["predictor_auc"].iloc[-1]

        if "predictor_test_auc" in history.columns:
            row["final_rl_predictor_test_auc"] = history["predictor_test_auc"].iloc[-1]

    if predictor is not None and len(predictor) > 0:
        p = predictor.iloc[0].to_dict()

        row["predictor_auc"] = p.get("predictor_auc")
        row["predictor_train_auc"] = p.get("predictor_train_auc")
        row["predictor_test_auc"] = p.get("predictor_test_auc")
        row["predictor_loss"] = p.get("predictor_loss")
        row["predictor_n_samples"] = p.get("n_samples")
        row["predictor_n_features"] = p.get("n_features")

    if diversity is not None and len(diversity) > 0:
        if "diversity_score" in diversity.columns:
            row["diversity_report_mean"] = diversity["diversity_score"].mean()

    if decay is not None and len(decay) > 0:
        if "decay_penalty" in decay.columns:
            row["decay_report_mean"] = decay["decay_penalty"].mean()

    return row


def main():
    rows = []

    for model_name, exp_dir in EXPERIMENTS.items():
        if not exp_dir.exists():
            print(f"实验目录不存在，跳过: {exp_dir}")
            continue

        row = summarize_experiment(model_name, exp_dir)
        rows.append(row)

    if not rows:
        print("没有找到可汇总的实验结果。")
        return

    comparison_df = pd.DataFrame(rows)

    output_path = ARTIFACTS_ROOT / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("模型对比结果已保存：")
    print(output_path)
    print(comparison_df)


if __name__ == "__main__":
    main()