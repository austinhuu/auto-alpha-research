import numpy as np
import pandas as pd

from operators.expression_evaluator import calculate_factor_matrix
from evaluator.ic_evaluator import evaluate_factor_ic
from factor_filter.correlation_filter import select_low_corr_factors


class StrictWalkForwardValidator:
    """
    更严格的 Walk-forward 验证器。

    普通 Walk-forward：
        先固定一批候选因子，然后在多个测试窗口上评价。

    Strict Walk-forward：
        每个窗口内部：
        1. train_dates 上计算因子表现；
        2. train_dates 上筛选因子；
        3. 只把该窗口筛出的因子拿到 test_dates 上评价；
        4. 汇总每个因子的样本外表现。

    注意：
        这个版本不会在每个窗口重新训练 LSTM/Transformer 生成器。
        它的严格性只体现在“每个窗口只使用训练窗口筛选因子，再到测试窗口验证”。
        生成器在整个 development 区间预训练/使用 RL 奖励，早期窗口并非完整系统的
        独立样本外验证；必须依赖最终 frozen holdout 进一步诊断。
    """

    def __init__(
        self,
        data,
        factor_df,
        windows,
        artifacts_dir,
        top_k=10,
        max_corr=0.65,
        purge_days=2,
    ):
        self.data = data
        self.factor_df = factor_df
        self.windows = windows
        self.artifacts_dir = artifacts_dir
        self.top_k = top_k
        self.max_corr = max_corr
        self.purge_days = purge_days

    def run(self):
        # Compute once on the available development history to preserve trailing windows.
        all_factor_matrix = calculate_factor_matrix(self.data, self.factor_df)
        all_test_score_rows = []
        all_selected_rows = []

        print("\n开始 Strict Walk-forward 样本外验证...")

        for window in self.windows:
            window_id = window["window_id"]

            train_data = self.data[
                self.data["datetime"].isin(window["train_dates"])
            ].copy()

            test_data = self.data[
                self.data["datetime"].isin(window["test_dates"])
            ].copy()

            print(
                f"[Strict Walk-forward] Window {window_id}: "
                f"train {window['train_start']} -> {window['train_end']}, "
                f"test {window['test_start']} -> {window['test_end']}"
            )

            if len(train_data) == 0 or len(test_data) == 0:
                print(f"Window {window_id} 数据不足，跳过。")
                continue

            usable_dates = list(window["train_dates"])
            if len(usable_dates) <= self.purge_days:
                continue
            usable_dates = usable_dates[:-self.purge_days]
            train_factor_matrix = all_factor_matrix[
                all_factor_matrix['datetime'].isin(usable_dates)
            ].copy()

            train_score_df = evaluate_factor_ic(train_factor_matrix)

            if len(train_score_df) == 0:
                print(f"Window {window_id} 训练窗口没有有效因子，跳过。")
                continue

            selected_train_df = select_low_corr_factors(
                factor_matrix=train_factor_matrix,
                score_df=train_score_df,
                top_k=self.top_k,
                max_corr=self.max_corr,
            )

            if len(selected_train_df) == 0:
                print(f"Window {window_id} 没有筛选出训练期因子，跳过。")
                continue

            selected_factor_ids = selected_train_df["factor_id"].tolist()

            selected_factor_df = self.factor_df[
                self.factor_df["factor_id"].isin(selected_factor_ids)
            ].copy()

            selected_train_df = selected_train_df.merge(
                self.factor_df[[c for c in ("factor_id", "expression", "complexity")
                                if c in self.factor_df.columns]],
                on="factor_id",
                how="left",
            )

            selected_train_df["window_id"] = window_id
            selected_train_df["train_start"] = window["train_start"]
            selected_train_df["train_end"] = window["train_end"]
            selected_train_df["test_start"] = window["test_start"]
            selected_train_df["test_end"] = window["test_end"]

            all_selected_rows.append(selected_train_df)

            test_factor_matrix = all_factor_matrix[
                all_factor_matrix['datetime'].isin(window["test_dates"])
            ][['datetime', 'symbol', 'future_return'] + selected_factor_ids].copy()

            test_score_df = evaluate_factor_ic(test_factor_matrix)

            if len(test_score_df) == 0:
                print(f"Window {window_id} 测试窗口没有有效评分。")
                continue

            test_score_df = test_score_df.merge(
                selected_factor_df[[c for c in ("factor_id", "expression", "complexity")
                                    if c in selected_factor_df.columns]],
                on="factor_id",
                how="left",
            )

            test_score_df["window_id"] = window_id
            test_score_df["train_start"] = window["train_start"]
            test_score_df["train_end"] = window["train_end"]
            test_score_df["test_start"] = window["test_start"]
            test_score_df["test_end"] = window["test_end"]

            all_test_score_rows.append(test_score_df)

            selected_train_df.to_csv(
                self.artifacts_dir / f"strict_walk_forward_window_{window_id:03d}_selected_train_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )

            test_score_df.to_csv(
                self.artifacts_dir / f"strict_walk_forward_window_{window_id:03d}_test_scores.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if len(all_test_score_rows) == 0:
            print("Strict Walk-forward 没有生成有效测试结果。")
            return pd.DataFrame()

        detail_df = pd.concat(all_test_score_rows, ignore_index=True)

        detail_df.to_csv(
            self.artifacts_dir / "strict_walk_forward_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if len(all_selected_rows) > 0:
            selected_detail_df = pd.concat(all_selected_rows, ignore_index=True)
            selected_detail_df.to_csv(
                self.artifacts_dir / "strict_walk_forward_selected_detail.csv",
                index=False,
                encoding="utf-8-sig",
            )

        summary_df = self._summarize_oos_ic(detail_df)

        summary_df.to_csv(
            self.artifacts_dir / "strict_walk_forward_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Strict Walk-forward 验证完成。")

        return summary_df

    def _summarize_oos_ic(self, detail_df):
        rows = []

        for factor_id, g in detail_df.groupby("factor_id"):
            ic_values = g["mean_ic"].dropna()
            rank_ic_values = g["mean_rank_ic"].dropna()

            if len(ic_values) == 0:
                oos_mean_ic = np.nan
                oos_icir = np.nan
            else:
                oos_mean_ic = ic_values.mean()
                oos_icir = oos_mean_ic / (ic_values.std() + 1e-12)

            if len(rank_ic_values) == 0:
                oos_mean_rank_ic = np.nan
                oos_rank_icir = np.nan
            else:
                oos_mean_rank_ic = rank_ic_values.mean()
                oos_rank_icir = oos_mean_rank_ic / (rank_ic_values.std() + 1e-12)

            row = {
                "factor_id": factor_id,
                "oos_mean_ic": oos_mean_ic,
                "oos_icir": oos_icir,
                "oos_mean_rank_ic": oos_mean_rank_ic,
                "oos_rank_icir": oos_rank_icir,
                "n_windows": len(g),
            }

            if "expression" in g.columns:
                row["expression"] = g["expression"].dropna().iloc[0] if g["expression"].notna().any() else ""

            if "complexity" in g.columns:
                row["complexity"] = g["complexity"].dropna().iloc[0] if g["complexity"].notna().any() else np.nan

            rows.append(row)

        summary_df = pd.DataFrame(rows)

        if len(summary_df) > 0:
            summary_df["abs_oos_icir"] = summary_df["oos_icir"].abs()
            summary_df = summary_df.sort_values(
                "abs_oos_icir",
                ascending=False,
            ).reset_index(drop=True)

        return summary_df