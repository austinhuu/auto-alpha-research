"""Labels are computed offline for evaluation only and never exported to the VeighNa CSV."""
import pandas as pd

def build_forward_return_label(df, period=1):
    if period < 1:
        raise ValueError('period must be >= 1')
    df = df.sort_values(['symbol', 'datetime']).copy()
    by_symbol = df.groupby('symbol', sort=False)
    # Historical research target: close(t) -> close(t + period).
    df['future_return'] = by_symbol['close'].shift(-period) / df['close'] - 1
    # More executable timing proxy: signals known after close(t),
    # enter at open(t+1) and leave at open(t+1+period).
    df['tradable_return'] = (
        by_symbol['open'].shift(-(period + 1)) / by_symbol['open'].shift(-1) - 1
    )
    return df
