import numpy as np
import pandas as pd
def safe_div(a,b): return a / b.replace(0, np.nan)
def ts_mean(s,w): return s.rolling(w, min_periods=max(2,w//2)).mean()
def ts_std(s,w): return s.rolling(w, min_periods=max(2,w//2)).std()
def ts_min(s,w): return s.rolling(w, min_periods=max(2,w//2)).min()
def ts_max(s,w): return s.rolling(w, min_periods=max(2,w//2)).max()
def ts_rank(s,w):
    def last_rank(x): return pd.Series(x).rank(pct=True).iloc[-1] if len(x)>0 else np.nan
    return s.rolling(w, min_periods=max(2,w//2)).apply(last_rank, raw=False)
def delta(s,w): return s - s.shift(w)
def delay(s,w): return s.shift(w)
def zscore(s):
    # Expanding (causal) mean/std: no full-period look-ahead in an alpha formula.
    mean = s.expanding(min_periods=5).mean()
    std = s.expanding(min_periods=5).std().replace(0, np.nan)
    return (s - mean) / std
def signed_log(s): return np.sign(s)*np.log1p(np.abs(s))
