from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(exp_dir):
    exp_dir = Path(exp_dir)
    path = exp_dir / "equity_curve.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "datetime" not in df.columns or "equity" not in df.columns:
        return

    df["datetime"] = pd.to_datetime(df["datetime"])

    plt.figure(figsize=(10, 5))
    plt.plot(df["datetime"], df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(exp_dir / "equity_curve.png", dpi=150)
    plt.close()


def plot_reward_curve(exp_dir):
    exp_dir = Path(exp_dir)
    path = exp_dir / "rl_training_history.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "epoch" not in df.columns:
        return

    plt.figure(figsize=(10, 5))

    if "avg_reward" in df.columns:
        plt.plot(df["epoch"], df["avg_reward"], label="avg_reward")

    if "max_reward" in df.columns:
        plt.plot(df["epoch"], df["max_reward"], label="max_reward")

    if "predictor_auc" in df.columns:
        plt.plot(df["epoch"], df["predictor_auc"], label="predictor_auc")

    plt.title("Policy Gradient Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "reward_curve.png", dpi=150)
    plt.close()


def plot_factor_icir_bar(exp_dir, top_n=10):
    exp_dir = Path(exp_dir)

    path_candidates = [
        exp_dir / "selected_factors_v6.csv",
        exp_dir / "selected_factors_v5.csv",
        exp_dir / "selected_factors.csv",
    ]

    path = None

    for p in path_candidates:
        if p.exists():
            path = p
            break

    if path is None:
        return

    df = pd.read_csv(path)

    if "factor_id" not in df.columns or "abs_icir" not in df.columns:
        return

    df = df.sort_values("abs_icir", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(df["factor_id"], df["abs_icir"])
    plt.title("Top Selected Factors by abs(ICIR)")
    plt.xlabel("Factor")
    plt.ylabel("abs(ICIR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(exp_dir / "factor_icir_bar.png", dpi=150)
    plt.close()


def plot_factor_stability_bar(exp_dir, top_n=10):
    exp_dir = Path(exp_dir)

    path_candidates = [
        exp_dir / "selected_factors_v6.csv",
        exp_dir / "selected_factors_v5.csv",
        exp_dir / "factor_stability_report.csv",
    ]

    path = None

    for p in path_candidates:
        if p.exists():
            path = p
            break

    if path is None:
        return

    df = pd.read_csv(path)

    if "factor_id" not in df.columns or "stability_score" not in df.columns:
        return

    df = df.sort_values("stability_score", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(df["factor_id"], df["stability_score"])
    plt.title("Top Factors by Stability Score")
    plt.xlabel("Factor")
    plt.ylabel("Stability Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(exp_dir / "factor_stability_bar.png", dpi=150)
    plt.close()


def plot_predictor_report(exp_dir):
    exp_dir = Path(exp_dir)
    path = exp_dir / "predictor_report.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    required = ["predictor_train_auc", "predictor_test_auc"]

    if not all(c in df.columns for c in required):
        return

    row = df.iloc[0]

    labels = ["Train AUC", "Test AUC"]
    values = [
        row.get("predictor_train_auc"),
        row.get("predictor_test_auc"),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Predictor Network AUC")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(exp_dir / "predictor_auc.png", dpi=150)
    plt.close()


def plot_model_comparison(artifacts_root):
    artifacts_root = Path(artifacts_root)
    path = artifacts_root / "model_comparison.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "model" not in df.columns or "sharpe" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["sharpe"])
    plt.title("Model Comparison by Sharpe")
    plt.xlabel("Model")
    plt.ylabel("Sharpe")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(artifacts_root / "model_comparison_sharpe.png", dpi=150)
    plt.close()


def plot_model_comparison_predictor_auc(artifacts_root):
    artifacts_root = Path(artifacts_root)
    path = artifacts_root / "model_comparison.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "model" not in df.columns:
        return

    auc_col = None

    for col in ["predictor_auc", "predictor_test_auc"]:
        if col in df.columns:
            auc_col = col
            break

    if auc_col is None:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[auc_col])
    plt.title("Model Comparison by Predictor AUC")
    plt.xlabel("Model")
    plt.ylabel("Predictor AUC")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(artifacts_root / "model_comparison_predictor_auc.png", dpi=150)
    plt.close()


def plot_model_comparison_total_return(artifacts_root):
    artifacts_root = Path(artifacts_root)
    path = artifacts_root / "model_comparison.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "model" not in df.columns or "total_return" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["total_return"])
    plt.title("Model Comparison by Total Return")
    plt.xlabel("Model")
    plt.ylabel("Total Return")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(artifacts_root / "model_comparison_total_return.png", dpi=150)
    plt.close()


def plot_walk_forward_oos_icir(exp_dir):
    exp_dir = Path(exp_dir)

    path_candidates = [
        exp_dir / "strict_walk_forward_summary.csv",
        exp_dir / "walk_forward_summary.csv",
    ]

    path = None

    for p in path_candidates:
        if p.exists():
            path = p
            break

    if path is None:
        return

    df = pd.read_csv(path)

    if "factor_id" not in df.columns or "abs_oos_icir" not in df.columns:
        return

    df = df.sort_values("abs_oos_icir", ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(df["factor_id"], df["abs_oos_icir"])
    plt.title("Top Factors by Out-of-sample abs(ICIR)")
    plt.xlabel("Factor")
    plt.ylabel("abs(OOS ICIR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(exp_dir / "walk_forward_oos_icir.png", dpi=150)
    plt.close()