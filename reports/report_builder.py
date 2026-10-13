from pathlib import Path

from plot_utils import (
    plot_equity_curve,
    plot_reward_curve,
    plot_factor_icir_bar,
    plot_factor_stability_bar,
    plot_predictor_report,
    plot_walk_forward_oos_icir,
    plot_model_comparison,
    plot_model_comparison_predictor_auc,
    plot_model_comparison_total_return,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def build_experiment_report(exp_name):
    exp_dir = ARTIFACTS_ROOT / exp_name

    if not exp_dir.exists():
        print(f"实验目录不存在: {exp_dir}")
        return

    plot_equity_curve(exp_dir)
    plot_reward_curve(exp_dir)
    plot_factor_icir_bar(exp_dir)
    plot_factor_stability_bar(exp_dir)
    plot_predictor_report(exp_dir)
    plot_walk_forward_oos_icir(exp_dir)

    print(f"实验图表已生成: {exp_dir}")


def build_all_reports():
    for exp_name in ["v6_lstm", "v6_transformer"]:
        build_experiment_report(exp_name)

    plot_model_comparison(ARTIFACTS_ROOT)
    plot_model_comparison_predictor_auc(ARTIFACTS_ROOT)
    plot_model_comparison_total_return(ARTIFACTS_ROOT)

    print("全部报告图表生成完成。")


if __name__ == "__main__":
    build_all_reports()