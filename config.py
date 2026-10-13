from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "demo_market_data.csv"

# 实验名称：
# 跑 LSTM 时用：v6_lstm
# 跑 Transformer 时用：v6_transformer
EXPERIMENT_NAME = "demo_random"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / EXPERIMENT_NAME


# ============================================================
# Basic data and backtest config
# ============================================================

FORWARD_RETURN_PERIOD = 1

N_RANDOM_FACTORS = 60
TOP_K_FACTORS = 10

# 最终因子相关性过滤阈值
MAX_FACTOR_CORR = 0.65

LONG_QUANTILE = 0.20
SHORT_QUANTILE = 0.20

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005

RANDOM_SEED = 42


# ============================================================
# V2 Neural Generator config
# ============================================================

# 可选：
# "random"       使用 V1 随机生成器
# "lstm"         使用 LSTM 生成器
# "transformer"  使用 Transformer 生成器
GENERATOR_MODE = "random"

# 随机生成器先构造一批合法公式，作为 LSTM / Transformer 的预训练语料
N_PRETRAIN_EXPRESSIONS = 1000

# 神经网络最终生成的候选因子数量
N_NEURAL_FACTORS = 200

GENERATOR_EPOCHS = 20
GENERATOR_BATCH_SIZE = 64
GENERATOR_LR = 1e-3

MAX_EXPR_LEN = 80
GENERATOR_HIDDEN_SIZE = 128
GENERATOR_EMBED_SIZE = 64
GENERATOR_NUM_LAYERS = 2
GENERATOR_DROPOUT = 0.1


# ============================================================
# V3 Policy Gradient config
# ============================================================

USE_POLICY_GRADIENT = True

RL_EPOCHS = 10
RL_BATCH_SIZE = 32
RL_GENERATE_PER_EPOCH = 64

RL_LR = 5e-5

REWARD_METRIC = "icir"
# 可选："icir", "rank_icir", "mean_ic"

REWARD_CLIP = 5.0

# V6 中建议不要太大，否则 reward 容易被惩罚项压低
CORR_PENALTY_WEIGHT = 0.10
COMPLEXITY_PENALTY_WEIGHT = 0.003

MIN_VALID_FACTORS_PER_EPOCH = 5


# ============================================================
# V4 Diversity loss and decay detection config
# ============================================================

USE_DIVERSITY_LOSS = True
USE_DECAY_DETECTION = True

# V4 综合 reward 权重
ICIR_WEIGHT = 0.45
RANK_ICIR_WEIGHT = 0.25
DIVERSITY_WEIGHT = 0.20
DECAY_PENALTY_WEIGHT = 0.10

# Rolling IC 衰减检测参数
ROLLING_IC_WINDOW = 20
RECENT_IC_WINDOW = 40

# 备用参数，目前主要使用 MAX_FACTOR_CORR
MAX_SELECTED_FACTOR_CORR = 0.65

MIN_ABS_ICIR = 0.03
MIN_RECENT_ABS_IC = 0.005
MAX_DECAY_PENALTY = 1.0


# ============================================================
# V5 Walk-forward validation config
# ============================================================

USE_WALK_FORWARD_VALIDATION = True

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20

WALK_FORWARD_TRAIN_DAYS = 100
WALK_FORWARD_TEST_DAYS = 20
WALK_FORWARD_STEP_DAYS = 20

MIN_OOS_ABS_IC = 0.005
MIN_OOS_ICIR = 0.02

STABILITY_WEIGHT_ICIR = 0.40
STABILITY_WEIGHT_OOS = 0.40
STABILITY_WEIGHT_DECAY = 0.20

GENERATE_REPORT_PLOTS = True


# ============================================================
# V6 Strict Walk-forward config
# ============================================================

# True:
#   每个窗口先在 train_dates 上筛选因子，
#   再到 test_dates 上做样本外评价，更严谨。
#
# False:
#   使用普通 Walk-forward，即固定候选因子集合，
#   在多个测试窗口上评价。
USE_STRICT_WALK_FORWARD_VALIDATION = True


# ============================================================
# V6 Generator-Predictor Network config
# ============================================================

USE_PREDICTOR_NETWORK = True

PREDICTOR_HIDDEN_SIZE = 64
PREDICTOR_EPOCHS = 20
PREDICTOR_LR = 1e-3
PREDICTOR_BATCH_SIZE = 128
PREDICTOR_TEST_RATIO = 0.30

# Predictor AUC 加入策略梯度 reward 的权重
PREDICTOR_AUC_WEIGHT = 0.20

# AUC 超过 0.5 的部分才作为正向奖励
MIN_PREDICTOR_AUC = 0.50


# ============================================================
# VeighNa signal strategy config
# ============================================================

# 离线系统最终导出的信号文件一般是：
# artifacts / EXPERIMENT_NAME / alpha_signals.csv
#
# VeighNa 策略里可以直接填这个绝对路径。
DEFAULT_SIGNAL_FILE = ARTIFACTS_DIR / "alpha_signals.csv"

# 策略默认下单手数，用于 veighna_adapter/alpha_signal_strategy.py
DEFAULT_FIXED_SIZE = 100

# True 表示按日期匹配信号；
# False 表示按完整 datetime 匹配信号。
USE_DATE_ONLY_SIGNAL_MATCH = True


# ============================================================
# Report and model comparison config
# ============================================================

# 跑完 LSTM 和 Transformer 后，run_model_comparison.py 会默认读取这两个目录
LSTM_EXPERIMENT_NAME = "v6_lstm"
TRANSFORMER_EXPERIMENT_NAME = "demo_random"

MODEL_COMPARISON_OUTPUT = PROJECT_ROOT / "artifacts" / "model_comparison.csv"