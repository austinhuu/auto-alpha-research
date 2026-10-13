"""Illustrative cross-sectional long-short spread; not an order-book simulator."""
import pandas as pd
from backtesting.metrics import calculate_backtest_metrics

def run_long_short_backtest(signal_df, long_quantile=0.2, short_quantile=0.2,
                            commission_rate=0.001, slippage_rate=0.0005,
                            return_column='tradable_return'):
    if return_column not in signal_df.columns:
        raise ValueError(f'Missing {return_column}: calculate execution-aligned labels first')
    rows = []
    # Simplified per-period cost, not a turnover-aware fill model.
    estimated_round_trip_cost = 2 * (commission_rate + slippage_rate)
    for dt, group in signal_df.groupby('datetime'):
        group = group.dropna(subset=['alpha_score', return_column]).copy()
        if len(group) < 5:
            continue
        high = group['alpha_score'].quantile(1 - long_quantile)
        low = group['alpha_score'].quantile(short_quantile)
        long = group[group['alpha_score'] >= high]
        short = group[group['alpha_score'] <= low]
        long_return = float(long[return_column].mean()) if len(long) else 0.0
        short_return = float(short[return_column].mean()) if len(short) else 0.0
        net = long_return - short_return - estimated_round_trip_cost
        rows.append({'datetime': dt, 'long_return': long_return,
                     'short_return': short_return, 'daily_return': net,
                     'n_long': len(long), 'n_short': len(short)})
    if not rows:
        raise ValueError('Insufficient valid cross-sections for backtest')
    equity = pd.DataFrame(rows).sort_values('datetime').reset_index(drop=True)
    equity['equity'] = (1 + equity['daily_return']).cumprod()
    return calculate_backtest_metrics(equity), equity
