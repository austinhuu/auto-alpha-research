import numpy as np
import pandas as pd

def calculate_backtest_metrics(equity_curve):
    r = equity_curve['daily_return'].fillna(0.0).astype(float)
    equity = equity_curve['equity'].ffill().astype(float)
    if len(equity) == 0:
        raise ValueError('Empty equity curve')
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = (1.0 + total_return) ** (252.0 / len(equity)) - 1.0 if 1 + total_return > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float(np.sqrt(252) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else np.nan
    downside = np.minimum(r, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252))
    sortino = float(r.mean() * 252 / downside_deviation) if downside_deviation > 0 else np.nan
    dd = equity / np.maximum(1.0, equity.cummax()) - 1
    max_dd = float(dd.min())
    calmar = float(annual_return / abs(max_dd)) if max_dd < 0 and np.isfinite(annual_return) else np.nan
    return {'total_return': total_return, 'annual_return': annual_return,
            'annual_volatility': vol, 'sharpe': sharpe, 'sortino': sortino,
            'max_drawdown': max_dd, 'calmar': calmar}
