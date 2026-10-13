import numpy as np
import pandas as pd

from operators.expression_evaluator import calculate_factor_matrix
from evaluator.ic_evaluator import evaluate_factor_ic


class WalkForwardValidator:
    def __init__(
        self,
        data,
        factor_df,
        windows,
        artifacts_dir,
    ):
        self.data = data
        self.factor_df = factor_df
        self.windows = windows
        self.artifacts_dir = artifacts_dir

    def run(self):
        all_matrix = calculate_factor_matrix(self.data, self.factor_df)
        all_rows = []

        print("\n开始 Walk-forward 样本外验证...")

        for window in self.windows:
            window_id = window["window_id"]

            test_data = self.data[
                self.data["datetime"].isin(window["test_dates"])
            ].copy()

            print(
                f"[Walk-forward] Window {window_id}: "
                f"test {window['test_start']} -> {window['test_end']}"
            )

            factor_matrix = all_matrix[
                all_matrix['datetime'].isin(window['test_dates'])
            ].copy()

            score_df = evaluate_factor_ic(factor_matrix)

            score_df["window_id"] = window_id
            score_df["test_start"] = window["test_start"]
            score_df["test_end"] = window["test_end"]

            all_rows.append(score_df)

            score_df.to_csv(
                self.artifacts_dir / f"walk_forward_window_{window_id:03d}_scores.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if len(all_rows) == 0:
            return pd.DataFrame()

        detail_df = pd.concat(all_rows, ignore_index=True)

        summary_df = self._summarize_oos_ic(detail_df)

        detail_df.to_csv(
            self.artifacts_dir / "walk_forward_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_df.to_csv(
            self.artifacts_dir / "walk_forward_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Walk-forward 验证完成。")

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

            rows.append(
                {
                    "factor_id": factor_id,
                    "oos_mean_ic": oos_mean_ic,
                    "oos_icir": oos_icir,
                    "oos_mean_rank_ic": oos_mean_rank_ic,
                    "oos_rank_icir": oos_rank_icir,
                    "n_windows": len(g),
                }
            )

        summary_df = pd.DataFrame(rows)

        if len(summary_df) > 0:
            summary_df["abs_oos_icir"] = summary_df["oos_icir"].abs()
            summary_df = summary_df.sort_values(
                "abs_oos_icir",
                ascending=False,
            ).reset_index(drop=True)

        return summary_df