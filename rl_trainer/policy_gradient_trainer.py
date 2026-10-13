import numpy as np
import pandas as pd
import torch

from operators.expression_evaluator import calculate_factor_matrix
from evaluator.ic_evaluator import evaluate_factor_ic
from evaluator.decay_detector import calculate_decay_metrics
from evaluator.predictor_network import evaluate_factor_predictor
from factor_filter.diversity_filter import calculate_diversity_scores


class PolicyGradientTrainer:
    """
    V6 策略梯度训练器。

    在 V4 reward 基础上，加入 Predictor Network AUC：

    reward =
        icir_weight * abs(ICIR)
      + rank_icir_weight * abs(RankICIR)
      + diversity_weight * diversity_score
      + predictor_auc_weight * max(0, predictor_auc - 0.5)
      - corr_penalty_weight * corr_penalty
      - decay_penalty_weight * decay_penalty
      - complexity_penalty_weight * complexity_penalty

    这样项目就形成了更严格的“生成-预测”网络：
    Generator 负责生成公式因子；
    Predictor 负责评估这些因子对未来收益方向的预测能力；
    Policy Gradient 用综合 reward 更新 Generator。
    """

    def __init__(
        self,
        neural_generator,
        market_data,
        artifacts_dir,
        reward_metric="icir",
        rl_epochs=10,
        generate_per_epoch=64,
        rl_lr=5e-5,
        reward_clip=5.0,
        corr_penalty_weight=0.1,
        complexity_penalty_weight=0.003,
        min_valid_factors_per_epoch=5,
        temperature=1.0,
        icir_weight=0.45,
        rank_icir_weight=0.25,
        diversity_weight=0.20,
        decay_penalty_weight=0.10,
        rolling_ic_window=20,
        recent_ic_window=40,
        use_predictor_network=True,
        predictor_hidden_size=64,
        predictor_epochs=10,
        predictor_lr=1e-3,
        predictor_batch_size=128,
        predictor_test_ratio=0.30,
        predictor_auc_weight=0.20,
        min_predictor_auc=0.50,
    ):
        self.neural_generator = neural_generator
        self.market_data = market_data
        self.artifacts_dir = artifacts_dir

        self.reward_metric = reward_metric
        self.rl_epochs = rl_epochs
        self.generate_per_epoch = generate_per_epoch
        self.rl_lr = rl_lr
        self.reward_clip = reward_clip

        self.corr_penalty_weight = corr_penalty_weight
        self.complexity_penalty_weight = complexity_penalty_weight
        self.min_valid_factors_per_epoch = min_valid_factors_per_epoch
        self.temperature = temperature

        self.icir_weight = icir_weight
        self.rank_icir_weight = rank_icir_weight
        self.diversity_weight = diversity_weight
        self.decay_penalty_weight = decay_penalty_weight
        self.rolling_ic_window = rolling_ic_window
        self.recent_ic_window = recent_ic_window

        self.use_predictor_network = use_predictor_network
        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_epochs = predictor_epochs
        self.predictor_lr = predictor_lr
        self.predictor_batch_size = predictor_batch_size
        self.predictor_test_ratio = predictor_test_ratio
        self.predictor_auc_weight = predictor_auc_weight
        self.min_predictor_auc = min_predictor_auc

        self.optimizer = self.neural_generator.get_optimizer(lr=rl_lr)
        self.history = []

    def train(self):
        print("\n开始 V6 策略梯度训练：加入多样性、衰减惩罚与 Predictor Network AUC...")

        for epoch in range(1, self.rl_epochs + 1):
            print(f"\n[Policy Gradient V6] Epoch {epoch}/{self.rl_epochs}")

            sampled = self._sample_expressions()

            if len(sampled) == 0:
                print("本轮没有采样到合法表达式，跳过。")
                continue

            expressions = [x["expression"] for x in sampled]
            factor_df = self._build_factor_df(expressions)

            factor_matrix = calculate_factor_matrix(
                data=self.market_data,
                factor_df=factor_df,
            )

            score_df = evaluate_factor_ic(factor_matrix)

            reward_map, reward_detail_df, predictor_report = self._build_reward_map(
                score_df=score_df,
                factor_df=factor_df,
                factor_matrix=factor_matrix,
            )

            valid_items = []

            for item in sampled:
                factor_id = item["factor_id"]

                if factor_id in reward_map:
                    valid_items.append(
                        {
                            "factor_id": factor_id,
                            "expression": item["expression"],
                            "log_prob": item["log_prob"],
                            "reward": reward_map[factor_id],
                        }
                    )

            if len(valid_items) < self.min_valid_factors_per_epoch:
                print(
                    f"有效因子数量不足：{len(valid_items)}，"
                    f"低于阈值 {self.min_valid_factors_per_epoch}，跳过本轮更新。"
                )

                self._save_epoch_outputs(
                    epoch=epoch,
                    sampled=sampled,
                    score_df=score_df,
                    reward_detail_df=reward_detail_df,
                    predictor_report=predictor_report,
                )
                continue

            loss = self._policy_gradient_update(valid_items)

            rewards = [x["reward"] for x in valid_items]
            avg_reward = float(np.mean(rewards))
            max_reward = float(np.max(rewards))
            min_reward = float(np.min(rewards))

            predictor_auc = predictor_report.get("predictor_auc", np.nan)
            predictor_test_auc = predictor_report.get("predictor_test_auc", np.nan)

            print(
                f"有效因子数={len(valid_items)}, "
                f"avg_reward={avg_reward:.6f}, "
                f"max_reward={max_reward:.6f}, "
                f"min_reward={min_reward:.6f}, "
                f"predictor_auc={predictor_auc}, "
                f"loss={loss:.6f}"
            )

            self.history.append(
                {
                    "epoch": epoch,
                    "valid_factors": len(valid_items),
                    "avg_reward": avg_reward,
                    "max_reward": max_reward,
                    "min_reward": min_reward,
                    "loss": loss,
                    "predictor_auc": predictor_auc,
                    "predictor_test_auc": predictor_test_auc,
                }
            )

            self._save_epoch_outputs(
                epoch=epoch,
                sampled=sampled,
                score_df=score_df,
                reward_detail_df=reward_detail_df,
                predictor_report=predictor_report,
            )

        self._save_training_history()
        print("\nV6 策略梯度训练完成。")

    def _sample_expressions(self):
        sampled = []
        seen = set()

        max_trials = self.generate_per_epoch * 10
        trials = 0

        while len(sampled) < self.generate_per_epoch and trials < max_trials:
            expr, log_prob = self.neural_generator.sample_with_log_prob(
                temperature=self.temperature
            )

            trials += 1

            if not self.neural_generator._is_basic_valid(expr):
                continue

            if expr in seen:
                continue

            seen.add(expr)

            sampled.append(
                {
                    "factor_id": f"rl_alpha_{len(sampled) + 1:04d}",
                    "expression": expr,
                    "log_prob": log_prob,
                }
            )

        print(f"本轮采样合法表达式数量：{len(sampled)}")
        return sampled

    def _build_factor_df(self, expressions):
        rows = []

        for i, expr in enumerate(expressions):
            rows.append(
                {
                    "factor_id": f"rl_alpha_{i + 1:04d}",
                    "expression": expr,
                    "complexity": expr.count("("),
                }
            )

        return pd.DataFrame(rows)

    def _build_reward_map(self, score_df, factor_df, factor_matrix):
        reward_map = {}
        reward_rows = []

        factor_cols = [
            c for c in factor_matrix.columns
            if c.startswith("rl_alpha_")
        ]

        if len(factor_cols) == 0:
            return reward_map, pd.DataFrame(), {}

        corr_penalty_map = self._calculate_corr_penalty(
            factor_matrix=factor_matrix,
            factor_cols=factor_cols,
        )

        diversity_score_map = calculate_diversity_scores(
            factor_matrix=factor_matrix,
            factor_cols=factor_cols,
        )

        decay_df = calculate_decay_metrics(
            factor_matrix=factor_matrix,
            factor_cols=factor_cols,
            rolling_window=self.rolling_ic_window,
            recent_window=self.recent_ic_window,
        )

        decay_map = {
            row["factor_id"]: row["decay_penalty"]
            for _, row in decay_df.iterrows()
        }

        recent_ic_map = {
            row["factor_id"]: row.get("recent_ic", np.nan)
            for _, row in decay_df.iterrows()
        }

        ic_slope_map = {
            row["factor_id"]: row.get("ic_slope", np.nan)
            for _, row in decay_df.iterrows()
        }

        complexity_map = {
            row["factor_id"]: row["complexity"]
            for _, row in factor_df.iterrows()
        }

        expression_map = {
            row["factor_id"]: row["expression"]
            for _, row in factor_df.iterrows()
        }

        predictor_report = {
            "predictor_auc": np.nan,
            "predictor_train_auc": np.nan,
            "predictor_test_auc": np.nan,
            "predictor_loss": np.nan,
            "n_samples": 0,
            "n_features": len(factor_cols),
        }

        if self.use_predictor_network:
            predictor_report = evaluate_factor_predictor(
                factor_matrix=factor_matrix,
                factor_cols=factor_cols,
                hidden_size=self.predictor_hidden_size,
                epochs=self.predictor_epochs,
                lr=self.predictor_lr,
                batch_size=self.predictor_batch_size,
                test_ratio=self.predictor_test_ratio,
                device=self.neural_generator.device,
            )

        predictor_auc = predictor_report.get("predictor_auc", np.nan)

        if pd.isna(predictor_auc):
            predictor_auc_score = 0.0
        else:
            predictor_auc_score = max(
                0.0,
                float(predictor_auc) - self.min_predictor_auc,
            )

        for _, row in score_df.iterrows():
            factor_id = row["factor_id"]

            icir = row.get("icir", np.nan)
            rank_icir = row.get("rank_icir", np.nan)
            mean_ic = row.get("mean_ic", np.nan)
            mean_rank_ic = row.get("mean_rank_ic", np.nan)

            if pd.isna(icir):
                continue

            icir_score = abs(float(icir))

            if pd.isna(rank_icir):
                rank_icir_score = 0.0
            else:
                rank_icir_score = abs(float(rank_icir))

            corr_penalty = corr_penalty_map.get(factor_id, 0.0)
            diversity_score = diversity_score_map.get(factor_id, 0.0)
            decay_penalty = decay_map.get(factor_id, 0.0)
            complexity_penalty = complexity_map.get(factor_id, 0.0)

            reward = (
                self.icir_weight * icir_score
                + self.rank_icir_weight * rank_icir_score
                + self.diversity_weight * diversity_score
                + self.predictor_auc_weight * predictor_auc_score
                - self.corr_penalty_weight * corr_penalty
                - self.decay_penalty_weight * decay_penalty
                - self.complexity_penalty_weight * complexity_penalty
            )

            reward = max(0.0, reward)
            reward = float(np.clip(reward, 0.0, self.reward_clip))

            reward_map[factor_id] = reward

            reward_rows.append(
                {
                    "factor_id": factor_id,
                    "expression": expression_map.get(factor_id, ""),
                    "mean_ic": mean_ic,
                    "mean_rank_ic": mean_rank_ic,
                    "icir": icir,
                    "rank_icir": rank_icir,
                    "icir_score": icir_score,
                    "rank_icir_score": rank_icir_score,
                    "corr_penalty": corr_penalty,
                    "diversity_score": diversity_score,
                    "decay_penalty": decay_penalty,
                    "recent_ic": recent_ic_map.get(factor_id, np.nan),
                    "ic_slope": ic_slope_map.get(factor_id, np.nan),
                    "complexity_penalty": complexity_penalty,
                    "predictor_auc": predictor_auc,
                    "predictor_auc_score": predictor_auc_score,
                    "reward": reward,
                }
            )

        reward_detail_df = pd.DataFrame(reward_rows)

        if len(reward_detail_df) > 0:
            reward_detail_df = reward_detail_df.sort_values(
                "reward",
                ascending=False,
            ).reset_index(drop=True)

        return reward_map, reward_detail_df, predictor_report

    def _calculate_corr_penalty(self, factor_matrix, factor_cols):
        penalty = {}

        if len(factor_cols) == 0:
            return penalty

        if len(factor_cols) == 1:
            return {factor_cols[0]: 0.0}

        factor_data = factor_matrix[factor_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        corr = factor_data.corr().abs()

        for col in factor_cols:
            others = corr[col].drop(index=col, errors="ignore").dropna()

            if len(others) == 0:
                penalty[col] = 0.0
            else:
                penalty[col] = float(others.mean())

        return penalty

    def _policy_gradient_update(self, valid_items):
        rewards = torch.tensor(
            [x["reward"] for x in valid_items],
            dtype=torch.float32,
            device=self.neural_generator.device,
        )

        if rewards.std().item() < 1e-8:
            rewards = rewards - rewards.mean()
        else:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        losses = []

        for item, reward in zip(valid_items, rewards):
            log_prob = item["log_prob"]
            losses.append(-reward.detach() * log_prob)

        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.neural_generator.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return float(loss.item())

    def _save_epoch_outputs(
        self,
        epoch,
        sampled,
        score_df,
        reward_detail_df,
        predictor_report,
    ):
        sampled_rows = []

        reward_map = {}

        if reward_detail_df is not None and len(reward_detail_df) > 0:
            reward_map = {
                row["factor_id"]: row["reward"]
                for _, row in reward_detail_df.iterrows()
            }

        for item in sampled:
            factor_id = item["factor_id"]

            sampled_rows.append(
                {
                    "epoch": epoch,
                    "factor_id": factor_id,
                    "expression": item["expression"],
                    "reward": reward_map.get(factor_id, np.nan),
                }
            )

        sampled_df = pd.DataFrame(sampled_rows)

        sampled_df.to_csv(
            self.artifacts_dir / f"rl_epoch_{epoch:03d}_sampled_factors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        score_df.to_csv(
            self.artifacts_dir / f"rl_epoch_{epoch:03d}_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if reward_detail_df is not None and len(reward_detail_df) > 0:
            reward_detail_df.to_csv(
                self.artifacts_dir / f"rl_epoch_{epoch:03d}_reward_detail.csv",
                index=False,
                encoding="utf-8-sig",
            )

        predictor_report_df = pd.DataFrame([predictor_report])
        predictor_report_df.to_csv(
            self.artifacts_dir / f"rl_epoch_{epoch:03d}_predictor_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def _save_training_history(self):
        if len(self.history) == 0:
            return

        history_df = pd.DataFrame(self.history)

        history_df.to_csv(
            self.artifacts_dir / "rl_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )