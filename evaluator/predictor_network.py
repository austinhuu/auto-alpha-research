import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata


def binary_auc_score(y_true, y_score):
    """
    不依赖 sklearn 的 AUC 计算。
    y_true: 0/1
    y_score: 预测概率
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0:
        return np.nan

    pos = y_true == 1
    neg = y_true == 0

    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = rankdata(y_score, method="average")

    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


class FactorPredictorMLP(nn.Module):
    """
    预测网络：
    输入：因子值
    输出：未来收益为正的概率
    """

    def __init__(self, input_size, hidden_size=64, dropout=0.1):
        super().__init__()

        hidden_size_2 = max(8, hidden_size // 2)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _prepare_predictor_data(factor_matrix, factor_cols, max_nan_ratio=0.80):
    """
    准备 Predictor Network 的训练数据。

    关键改动：
    1. 不再要求所有因子同时非空；
    2. 先删除缺失率太高或常数型因子；
    3. 对剩余因子的 NaN 用中位数填充；
    4. 返回 usable_factor_cols，避免 n_samples=0。
    """
    if len(factor_cols) == 0:
        return None

    required_cols = ["datetime", "symbol", "future_return"]
    available_factor_cols = [c for c in factor_cols if c in factor_matrix.columns]

    if len(available_factor_cols) == 0:
        return None

    df = factor_matrix[required_cols + available_factor_cols].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["future_return"])

    if len(df) == 0:
        return None

    df["target"] = (df["future_return"] > 0).astype(int)

    if df["target"].nunique() < 2:
        return None

    usable_factor_cols = []

    for col in available_factor_cols:
        nan_ratio = df[col].isna().mean()
        nunique = df[col].nunique(dropna=True)

        if nan_ratio <= max_nan_ratio and nunique > 1:
            usable_factor_cols.append(col)

    if len(usable_factor_cols) == 0:
        return None

    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep NaNs until after the chronological split; compute imputers on train only.
    X = df[usable_factor_cols].astype(float).values
    y = df["target"].astype(float).values

    return df, X, y, usable_factor_cols


def _empty_predictor_result(n_samples=0, n_features=0):
    return {
        "predictor_auc": np.nan,
        "predictor_train_auc": np.nan,
        "predictor_test_auc": np.nan,
        "predictor_loss": np.nan,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def evaluate_factor_predictor(
    factor_matrix,
    factor_cols,
    hidden_size=64,
    epochs=20,
    lr=1e-3,
    batch_size=128,
    test_ratio=0.30,
    device=None,
):
    """
    训练一个 MLP Predictor，评估因子集合对 future_return 方向的预测能力。

    返回：
    predictor_auc
    predictor_train_auc
    predictor_test_auc
    predictor_loss
    n_samples
    n_features
    """

    if len(factor_cols) == 0:
        return _empty_predictor_result(n_samples=0, n_features=0)

    prepared = _prepare_predictor_data(
        factor_matrix=factor_matrix,
        factor_cols=factor_cols,
        max_nan_ratio=0.80,
    )

    if prepared is None:
        return _empty_predictor_result(
            n_samples=0,
            n_features=len(factor_cols),
        )

    df, X, y, usable_factor_cols = prepared

    n = len(X)
    n_features = len(usable_factor_cols)

    if n_features == 0:
        return _empty_predictor_result(n_samples=n, n_features=0)

    # Split by full dates, never through the middle of a daily cross section.
    chronological_dates = df['datetime'].drop_duplicates().sort_values().tolist()
    cutoff_index = int(len(chronological_dates) * (1 - test_ratio))
    if cutoff_index <= 0 or cutoff_index >= len(chronological_dates):
        return _empty_predictor_result(n_samples=n, n_features=n_features)
    cutoff_date = chronological_dates[cutoff_index]
    split = int((df['datetime'] < cutoff_date).sum())

    if split <= 10 or n - split <= 10:
        return _empty_predictor_result(
            n_samples=n,
            n_features=n_features,
        )

    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    X_test = np.where(np.isfinite(X_test), X_test, np.nan)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return _empty_predictor_result(
            n_samples=n,
            n_features=n_features,
        )

    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-12

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = FactorPredictorMLP(
        input_size=n_features,
        hidden_size=hidden_size,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    last_loss = np.nan

    for _ in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        last_loss = total_loss / max(n_batches, 1)

    model.eval()

    with torch.no_grad():
        train_logits = model(
            torch.tensor(X_train, dtype=torch.float32, device=device)
        )
        test_logits = model(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        )

        train_prob = torch.sigmoid(train_logits).cpu().numpy()
        test_prob = torch.sigmoid(test_logits).cpu().numpy()

    train_auc = binary_auc_score(y_train, train_prob)
    test_auc = binary_auc_score(y_test, test_prob)

    return {
        "predictor_auc": test_auc,
        "predictor_train_auc": train_auc,
        "predictor_test_auc": test_auc,
        "predictor_loss": last_loss,
        "n_samples": n,
        "n_features": n_features,
    }