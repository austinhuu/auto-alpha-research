import argparse
import random
from pathlib import Path
import numpy as np
import torch

import pandas as pd

import config as cfg

from data.data_loader import load_market_data
from data.label_builder import build_forward_return_label
from tools.generate_sample_data import generate_sample_data

from generator.random_generator import RandomFactorGenerator
from generator.neural_generator import NeuralFactorGenerator

from operators.expression_evaluator import calculate_factor_matrix

from evaluator.ic_evaluator import evaluate_factor_ic
from evaluator.decay_detector import calculate_decay_metrics
from evaluator.predictor_network import evaluate_factor_predictor

from factor_filter.correlation_filter import select_low_corr_factors
from factor_filter.diversity_filter import (
    calculate_factor_correlation_matrix,
    build_diversity_report,
)

from backtesting.portfolio_backtester import run_long_short_backtest
from veighna_adapter.signal_exporter import export_alpha_signals

from rl_trainer.policy_gradient_trainer import PolicyGradientTrainer

from validation.time_split import build_walk_forward_windows, split_development_holdout
from validation.walk_forward_validator import WalkForwardValidator
from validation.factor_stability import calculate_factor_stability_score
from validation.strict_walk_forward_validator import StrictWalkForwardValidator


def get_cfg(name, default=None):
    return getattr(cfg, name, default)


def build_generated_factor_dataframe(factors):
    generated_df = pd.DataFrame(
        [
            {
                "factor_id": f"alpha_{i + 1:04d}",
                "expression": expr,
                "complexity": expr.count("("),
            }
            for i, expr in enumerate(factors)
        ]
    )

    return generated_df


def generate_factors_by_mode(data):
    if cfg.GENERATOR_MODE == "random":
        print("当前生成模式：random")
        print("使用 V1 随机公式因子生成器...")

        random_generator = RandomFactorGenerator(random_seed=cfg.RANDOM_SEED)
        factors = random_generator.generate_many(n=cfg.N_RANDOM_FACTORS)

        print(f"随机生成因子数量：{len(factors)}")
        return factors

    elif cfg.GENERATOR_MODE in ["lstm", "transformer"]:
        print(f"当前生成模式：{cfg.GENERATOR_MODE}")
        print("进入 V2/V3/V4/V5/V6：神经网络公式生成器流程")

        print("\n[3.1] 使用随机生成器构造预训练表达式样本...")

        random_generator = RandomFactorGenerator(random_seed=cfg.RANDOM_SEED)

        pretrain_expressions = random_generator.generate_many(
            n=cfg.N_PRETRAIN_EXPRESSIONS,
        )

        print(f"预训练表达式数量：{len(pretrain_expressions)}")
        print("预训练样本示例：")

        for expr in pretrain_expressions[:5]:
            print("  ", expr)

        print("\n[3.2] 初始化神经网络公式生成器...")

        neural_generator = NeuralFactorGenerator(
            model_type=cfg.GENERATOR_MODE,
            max_len=cfg.MAX_EXPR_LEN,
            embed_size=cfg.GENERATOR_EMBED_SIZE,
            hidden_size=cfg.GENERATOR_HIDDEN_SIZE,
            num_layers=cfg.GENERATOR_NUM_LAYERS,
            dropout=cfg.GENERATOR_DROPOUT,
            lr=cfg.GENERATOR_LR,
            batch_size=cfg.GENERATOR_BATCH_SIZE,
            epochs=cfg.GENERATOR_EPOCHS,
        )

        print("\n[3.3] 开始 V2 预训练：让模型学习公式语法...")
        neural_generator.fit(pretrain_expressions)

        if cfg.USE_POLICY_GRADIENT:
            print("\n[3.4] 开始 V6 策略梯度训练：加入 Predictor Network AUC...")

            pg_trainer = PolicyGradientTrainer(
                neural_generator=neural_generator,
                market_data=data,
                artifacts_dir=cfg.ARTIFACTS_DIR,
                reward_metric=cfg.REWARD_METRIC,
                rl_epochs=cfg.RL_EPOCHS,
                generate_per_epoch=cfg.RL_GENERATE_PER_EPOCH,
                rl_lr=cfg.RL_LR,
                reward_clip=cfg.REWARD_CLIP,
                corr_penalty_weight=cfg.CORR_PENALTY_WEIGHT,
                complexity_penalty_weight=cfg.COMPLEXITY_PENALTY_WEIGHT,
                min_valid_factors_per_epoch=cfg.MIN_VALID_FACTORS_PER_EPOCH,
                temperature=1.0,

                icir_weight=cfg.ICIR_WEIGHT,
                rank_icir_weight=cfg.RANK_ICIR_WEIGHT,
                diversity_weight=cfg.DIVERSITY_WEIGHT,
                decay_penalty_weight=cfg.DECAY_PENALTY_WEIGHT,
                rolling_ic_window=cfg.ROLLING_IC_WINDOW,
                recent_ic_window=cfg.RECENT_IC_WINDOW,

                use_predictor_network=get_cfg("USE_PREDICTOR_NETWORK", True),
                predictor_hidden_size=get_cfg("PREDICTOR_HIDDEN_SIZE", 64),
                predictor_epochs=max(5, get_cfg("PREDICTOR_EPOCHS", 20) // 2),
                predictor_lr=get_cfg("PREDICTOR_LR", 1e-3),
                predictor_batch_size=get_cfg("PREDICTOR_BATCH_SIZE", 128),
                predictor_test_ratio=get_cfg("PREDICTOR_TEST_RATIO", 0.30),
                predictor_auc_weight=get_cfg("PREDICTOR_AUC_WEIGHT", 0.20),
                min_predictor_auc=get_cfg("MIN_PREDICTOR_AUC", 0.50),
            )

            pg_trainer.train()

        print("\n[3.5] 使用训练后的神经网络生成最终 Alpha 因子...")

        factors = neural_generator.generate_many(
            n=cfg.N_NEURAL_FACTORS,
            temperature=0.8,
        )

        print(f"最终神经网络生成因子数量：{len(factors)}")

        if len(factors) == 0 and get_cfg('QUICK_DEMO_FALLBACK', False):
            print('DEMO FALLBACK: neural generator emitted no valid formulas after one epoch; '
                  'using random formulas for end-to-end smoke test (NOT neural performance).')
            factors = random_generator.generate_many(n=cfg.N_NEURAL_FACTORS)

        if len(factors) == 0:
            raise RuntimeError(
                "神经网络没有生成任何合法因子。"
                "可以降低训练难度，或者先把 GENERATOR_MODE 改成 random。"
            )

        print("最终生成因子示例：")

        for expr in factors[:5]:
            print("  ", expr)

        return factors

    else:
        raise ValueError(
            f"未知 GENERATOR_MODE: {cfg.GENERATOR_MODE}，"
            "请在 config.py 中设置为 random、lstm 或 transformer。"
        )


def main():
    parser = argparse.ArgumentParser(description='Auto Alpha V6 research pipeline')
    parser.add_argument('--demo', action='store_true', help='Generate reproducible SYNTHETIC input data')
    parser.add_argument('--data', type=Path, help='CSV file with OHLCV market data')
    parser.add_argument('--mode', choices=['random', 'lstm', 'transformer'], default='random')
    parser.add_argument('--experiment', help='Artifact directory name, default demo_random or v6_MODE')
    parser.add_argument('--quick', action='store_true', help='Small CPU smoke-test settings')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.demo and args.data is None:
        parser.error('Specify --demo or --data PATH')
    if args.demo and args.data is not None:
        parser.error('Use only one of --demo and --data')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg.RANDOM_SEED = args.seed
    cfg.QUICK_DEMO_FALLBACK = args.quick
    cfg.GENERATOR_MODE = args.mode
    cfg.EXPERIMENT_NAME = args.experiment or ('demo_' + args.mode if args.demo else 'v6_' + args.mode)
    cfg.ARTIFACTS_DIR = cfg.PROJECT_ROOT / 'artifacts' / cfg.EXPERIMENT_NAME
    if args.demo:
        cfg.SAMPLE_DATA_PATH = cfg.PROJECT_ROOT / 'sample_data' / 'demo_market_data.csv'
        generate_sample_data(cfg.SAMPLE_DATA_PATH, n_symbols=12 if args.quick else 15,
                             n_days=110 if args.quick else 190, seed=args.seed)
    else:
        cfg.SAMPLE_DATA_PATH = args.data.resolve()
    if args.quick:
        cfg.N_RANDOM_FACTORS = 8
        cfg.N_PRETRAIN_EXPRESSIONS = 40
        cfg.N_NEURAL_FACTORS = 8
        cfg.GENERATOR_EPOCHS = 1
        cfg.RL_EPOCHS = 1
        cfg.RL_GENERATE_PER_EPOCH = 4
        cfg.MIN_VALID_FACTORS_PER_EPOCH = 2
        cfg.PREDICTOR_EPOCHS = 2
        cfg.WALK_FORWARD_TRAIN_DAYS = 48
        cfg.WALK_FORWARD_TEST_DAYS = 12
        cfg.WALK_FORWARD_STEP_DAYS = 12
        cfg.ROLLING_IC_WINDOW = 8
        cfg.RECENT_IC_WINDOW = 12
        cfg.TOP_K_FACTORS = 4
        if args.mode != 'random':
            cfg.USE_POLICY_GRADIENT = False  # generator-only smoke test, not REINFORCE proof
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Auto Alpha Project V6：生成-预测网络、样本外验证与 VeighNa 接入最终版启动")
    print("=" * 90)

    print("\n[1/9] 加载市场数据...")

    full_data = load_market_data(cfg.SAMPLE_DATA_PATH)
    full_data = build_forward_return_label(full_data, period=cfg.FORWARD_RETURN_PERIOD)
    data, holdout_data, development_dates, holdout_dates = split_development_holdout(
        full_data, forward_period=cfg.FORWARD_RETURN_PERIOD,
    )
    print(f'Development through {development_dates[-1]}; untouched holdout begins {holdout_dates[0]}')
    print('WARNING: All research choices and factor reward fitting use development dates only.')

    print(
        f"数据加载完成：{data.shape[0]} 行，"
        f"股票数量：{data['symbol'].nunique()}，"
        f"日期数量：{data['datetime'].nunique()}"
    )

    print("数据字段：")
    print(data.columns.tolist())

    print("\n[2/9] 构造未来收益率标签...")

    # Labels were constructed on the full chronological data before splitting,
    # then train/holdout periods separated with a conservative purge.

    print(f"预测周期：未来 {cfg.FORWARD_RETURN_PERIOD} 期收益率")
    print("标签字段：future_return")

    print("\n[3/9] 生成公式化 Alpha 因子...")

    factors = generate_factors_by_mode(data)

    generated_df = build_generated_factor_dataframe(factors)

    generated_df.to_csv(
        cfg.ARTIFACTS_DIR / "generated_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n生成因子已保存：generated_factors.csv")
    print(generated_df.head(10))

    print("\n[4/9] 计算因子值矩阵...")

    factor_matrix = calculate_factor_matrix(
        data=data,
        factor_df=generated_df,
    )

    print(f"因子矩阵形状：{factor_matrix.shape}")

    factor_matrix.to_csv(
        cfg.ARTIFACTS_DIR / "factor_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("因子矩阵已保存：factor_matrix.csv")

    print("\n[5/9] 计算 IC / Rank IC / ICIR...")

    score_df = evaluate_factor_ic(factor_matrix)

    score_df = score_df.merge(
        generated_df[["factor_id", "expression", "complexity"]],
        on="factor_id",
        how="left",
    )

    score_df.to_csv(
        cfg.ARTIFACTS_DIR / "factor_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("因子评分已保存：factor_scores.csv")
    print(score_df.head(10))

    print("\n[5.1/9] 生成因子相关性、多样性与衰减报告...")

    factor_cols = [
        c for c in factor_matrix.columns
        if c.startswith("alpha_")
    ]

    corr_matrix = calculate_factor_correlation_matrix(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
    )

    corr_matrix.to_csv(
        cfg.ARTIFACTS_DIR / "factor_corr_matrix.csv",
        encoding="utf-8-sig",
    )

    diversity_report = build_diversity_report(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
    )

    diversity_report.to_csv(
        cfg.ARTIFACTS_DIR / "factor_diversity_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decay_report = calculate_decay_metrics(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
        rolling_window=cfg.ROLLING_IC_WINDOW,
        recent_window=cfg.RECENT_IC_WINDOW,
    )

    decay_report.to_csv(
        cfg.ARTIFACTS_DIR / "factor_decay_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    score_df = score_df.merge(
        diversity_report,
        on="factor_id",
        how="left",
    )

    score_df = score_df.merge(
        decay_report,
        on="factor_id",
        how="left",
        suffixes=("", "_decay"),
    )

    score_df.to_csv(
        cfg.ARTIFACTS_DIR / "factor_scores_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("V4 综合因子评分已保存：factor_scores_v4.csv")

    print("\n[5.2/9] 运行 Walk-forward 样本外验证...")

    oos_report = pd.DataFrame()

    if get_cfg("USE_WALK_FORWARD_VALIDATION", True):
        windows = build_walk_forward_windows(
            data=data,
            train_days=cfg.WALK_FORWARD_TRAIN_DAYS,
            test_days=cfg.WALK_FORWARD_TEST_DAYS,
            step_days=cfg.WALK_FORWARD_STEP_DAYS,
        )

        print(f"Walk-forward 窗口数量：{len(windows)}")

        if len(windows) > 0:
            use_strict = get_cfg("USE_STRICT_WALK_FORWARD_VALIDATION", False)

            if use_strict:
                print("启用 Strict Walk-forward：每个窗口内先训练期筛选，再测试期评价。")

                strict_validator = StrictWalkForwardValidator(
                    data=data,
                    factor_df=generated_df,
                    windows=windows,
                    artifacts_dir=cfg.ARTIFACTS_DIR,
                    top_k=cfg.TOP_K_FACTORS,
                    max_corr=cfg.MAX_FACTOR_CORR,
                )

                oos_report = strict_validator.run()

            else:
                print("启用普通 Walk-forward：固定候选因子集合进行滚动样本外评价。")

                wf_validator = WalkForwardValidator(
                    data=data,
                    factor_df=generated_df,
                    windows=windows,
                    artifacts_dir=cfg.ARTIFACTS_DIR,
                )

                oos_report = wf_validator.run()

            if len(oos_report) > 0:
                stability_df = calculate_factor_stability_score(
                    score_df=score_df,
                    decay_report=decay_report,
                    oos_report=oos_report,
                    icir_weight=cfg.STABILITY_WEIGHT_ICIR,
                    oos_weight=cfg.STABILITY_WEIGHT_OOS,
                    decay_weight=cfg.STABILITY_WEIGHT_DECAY,
                )

                stability_df.to_csv(
                    cfg.ARTIFACTS_DIR / "factor_stability_report.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                score_df = score_df.merge(
                    oos_report,
                    on="factor_id",
                    how="left",
                )

                score_df = score_df.merge(
                    stability_df[["factor_id", "stability_score"]],
                    on="factor_id",
                    how="left",
                )

                score_df.to_csv(
                    cfg.ARTIFACTS_DIR / "factor_scores_v5.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                print("Walk-forward 明细与稳定性报告已保存。")

            else:
                print("Walk-forward 没有生成有效样本外结果，跳过稳定性评分。")
                score_df.to_csv(
                    cfg.ARTIFACTS_DIR / "factor_scores_v5.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        else:
            print("数据长度不足，无法构造 Walk-forward 窗口，跳过 V5 验证。")
            score_df.to_csv(
                cfg.ARTIFACTS_DIR / "factor_scores_v5.csv",
                index=False,
                encoding="utf-8-sig",
            )
    else:
        print("USE_WALK_FORWARD_VALIDATION=False，跳过 Walk-forward 验证。")
        score_df.to_csv(
            cfg.ARTIFACTS_DIR / "factor_scores_v5.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n[5.3/9] 运行 Predictor Network 最终预测能力评估...")

    predictor_report = {}

    if get_cfg("USE_PREDICTOR_NETWORK", True):
        predictor_report = evaluate_factor_predictor(
            factor_matrix=factor_matrix,
            factor_cols=factor_cols,
            hidden_size=get_cfg("PREDICTOR_HIDDEN_SIZE", 64),
            epochs=get_cfg("PREDICTOR_EPOCHS", 20),
            lr=get_cfg("PREDICTOR_LR", 1e-3),
            batch_size=get_cfg("PREDICTOR_BATCH_SIZE", 128),
            test_ratio=get_cfg("PREDICTOR_TEST_RATIO", 0.30),
        )

        predictor_report_df = pd.DataFrame([predictor_report])

        predictor_report_df.to_csv(
            cfg.ARTIFACTS_DIR / "predictor_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("Predictor Network 预测报告已保存：predictor_report.csv")
        print(predictor_report_df)

    else:
        print("USE_PREDICTOR_NETWORK=False，跳过 Predictor Network。")

    print("\n[6/9] 按稳定性、ICIR、多样性、相关性和衰减情况筛选因子...")

    if "stability_score" in score_df.columns:
        score_df = score_df.sort_values(
            "stability_score",
            ascending=False,
        ).reset_index(drop=True)
        print("已启用 stability_score 作为优先筛选依据。")
    else:
        score_df = score_df.sort_values(
            "abs_icir",
            ascending=False,
        ).reset_index(drop=True)
        print("未检测到 stability_score，退回使用 abs_icir 排序。")

    selected_df = select_low_corr_factors(
        factor_matrix=factor_matrix,
        score_df=score_df,
        top_k=cfg.TOP_K_FACTORS,
        max_corr=cfg.MAX_FACTOR_CORR,
    )

    selected_df.to_csv(
        cfg.ARTIFACTS_DIR / "selected_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_df.to_csv(
        cfg.ARTIFACTS_DIR / "selected_factors_v6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"最终保留 {len(selected_df)} 个因子")
    print("筛选结果已保存：selected_factors_v6.csv")
    print(selected_df)

    if len(selected_df) == 0:
        raise RuntimeError(
            "没有筛选出任何有效因子。"
            "你可以尝试增加 N_NEURAL_FACTORS，或者放宽 MAX_FACTOR_CORR。"
        )

    print("\n[7/9] 导出 alpha_signals.csv，方便后续接入 VeighNa...")

    signal_df = export_alpha_signals(
        factor_matrix=factor_matrix,
        selected_factors=selected_df,
        output_path=cfg.ARTIFACTS_DIR / "alpha_signals.csv",
    )

    print("信号文件已保存：alpha_signals.csv (labels are excluded from CSV)")
    print(signal_df.head(10))

    print("\n[8/9] 运行简单多空组合回测...")

    backtest_result, equity_curve = run_long_short_backtest(
        signal_df=signal_df,
        long_quantile=cfg.LONG_QUANTILE,
        short_quantile=cfg.SHORT_QUANTILE,
        commission_rate=cfg.COMMISSION_RATE,
        slippage_rate=cfg.SLIPPAGE_RATE,
    )

    pd.DataFrame([backtest_result]).to_csv(
        cfg.ARTIFACTS_DIR / "backtest_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equity_curve.to_csv(
        cfg.ARTIFACTS_DIR / "equity_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("回测结果已保存：backtest_results.csv")
    print("净值曲线已保存：equity_curve.csv (DEVELOPMENT DIAGNOSTIC)")

    print("\n回测绩效：")

    for key, value in backtest_result.items():
        print(f"{key}: {value}")

    print("\n[8.1/9] Evaluate frozen development-selected factors on a held-out time block...")
    frozen_factor_df = generated_df[
        generated_df['factor_id'].isin(selected_df['factor_id'])
    ].copy()
    # Compute with full historical context, but NEVER use holdout labels to train,
    # generate formulas, tune hyperparameters, or select factor directions.
    frozen_full_matrix = calculate_factor_matrix(full_data, frozen_factor_df)
    frozen_holdout_matrix = frozen_full_matrix[
        frozen_full_matrix['datetime'].isin(holdout_dates)
    ].copy()
    holdout_score_df = evaluate_factor_ic(frozen_holdout_matrix)
    holdout_score_df.to_csv(cfg.ARTIFACTS_DIR / 'holdout_factor_scores.csv', index=False)
    holdout_signal_df = export_alpha_signals(
        factor_matrix=frozen_holdout_matrix,
        selected_factors=selected_df,
        output_path=cfg.ARTIFACTS_DIR / 'holdout_alpha_signals.csv',
    )
    try:
        holdout_result, holdout_equity = run_long_short_backtest(
            signal_df=holdout_signal_df,
            long_quantile=cfg.LONG_QUANTILE,
            short_quantile=cfg.SHORT_QUANTILE,
            commission_rate=cfg.COMMISSION_RATE,
            slippage_rate=cfg.SLIPPAGE_RATE,
        )
    except ValueError as error:
        print(f'HOLDOUT backtest skipped: {error}')
    else:
        pd.DataFrame([holdout_result]).to_csv(
            cfg.ARTIFACTS_DIR / 'holdout_backtest_results.csv', index=False)
        holdout_equity.to_csv(cfg.ARTIFACTS_DIR / 'holdout_equity_curve.csv', index=False)
        print('Holdout illustrative performance:', holdout_result)

    print("\n[9/9] V6 输出检查...")

    output_files = [
        "generated_factors.csv",
        "factor_matrix.csv",
        "factor_scores.csv",
        "factor_scores_v4.csv",
        "factor_scores_v5.csv",
        "factor_corr_matrix.csv",
        "factor_diversity_report.csv",
        "factor_decay_report.csv",
        "walk_forward_detail.csv",
        "walk_forward_summary.csv",
        "strict_walk_forward_detail.csv",
        "strict_walk_forward_summary.csv",
        "factor_stability_report.csv",
        "predictor_report.csv",
        "selected_factors_v6.csv",
        "alpha_signals.csv",
        "backtest_results.csv",
        "equity_curve.csv",
        "rl_training_history.csv",
    ]

    for filename in output_files:
        path = cfg.ARTIFACTS_DIR / filename
        if path.exists():
            print(f"[OK] {filename}")
        else:
            print(f"[SKIP] {filename}")

    print("\n" + "=" * 90)
    print("Auto Alpha Project V6 全部完成")
    print("=" * 90)


if __name__ == "__main__":
    main()