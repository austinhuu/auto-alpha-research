from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']

def load_market_data(path):
    path = Path(path)
    df = pd.read_csv(path, dtype={'symbol': str})
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required CSV columns: {missing}')
    date_text = df['datetime'].astype(str).str.strip()
    if date_text.str.fullmatch(r'\d{8}').all():
        df['datetime'] = pd.to_datetime(date_text, format='%Y%m%d', errors='raise')
    else:
        df['datetime'] = pd.to_datetime(date_text, errors='raise')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.duplicated(['datetime', 'symbol']).any():
        raise ValueError('Duplicate datetime/symbol rows; fix the CSV before running.')
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['returns'] = df.groupby('symbol')['close'].pct_change(fill_method=None)
    return df
