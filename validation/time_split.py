import pandas as pd


def split_by_time_ratio(
    data,
    train_ratio=0.6,
    valid_ratio=0.2,
    test_ratio=0.2,
):
    """
    按日期切分训练集、验证集、测试集。
    """
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1")

    dates = sorted(data["datetime"].dropna().unique())
    n = len(dates)

    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]

    train_data = data[data["datetime"].isin(train_dates)].copy()
    valid_data = data[data["datetime"].isin(valid_dates)].copy()
    test_data = data[data["datetime"].isin(test_dates)].copy()

    return train_data, valid_data, test_data


def build_walk_forward_windows(
    data,
    train_days=100,
    test_days=20,
    step_days=20,
):
    """
    构造 walk-forward 窗口。

    每个窗口：
    train_dates: 用于生成/筛选因子
    test_dates: 用于样本外测试
    """
    dates = sorted(data["datetime"].dropna().unique())
    windows = []

    start = 0

    while start + train_days + test_days <= len(dates):
        train_dates = dates[start:start + train_days]
        test_dates = dates[start + train_days:start + train_days + test_days]

        windows.append(
            {
                "window_id": len(windows) + 1,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            }
        )

        start += step_days

    return windows

def split_development_holdout(data, forward_period=1, development_ratio=0.8):
    """Reserve an unmodified final chronological block, purge boundary outcome look-ahead."""
    dates = sorted(data['datetime'].dropna().unique())
    if len(dates) < 45:
        raise ValueError('At least 45 dates are required')
    boundary = int(len(dates) * development_ratio)
    purge = forward_period + 1  # trades from t+1 open to t+period+1 open
    if boundary <= purge:
        raise ValueError('Insufficient development dates')
    dev_dates = dates[:boundary - purge]
    holdout_dates = dates[boundary:]
    return (data[data['datetime'].isin(dev_dates)].copy(),
            data[data['datetime'].isin(holdout_dates)].copy(), dev_dates, holdout_dates)
