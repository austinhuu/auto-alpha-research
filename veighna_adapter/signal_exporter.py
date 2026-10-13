import pandas as pd

def export_alpha_signals(factor_matrix, selected_factors, output_path, include_labels=False):
    cols = selected_factors['factor_id'].tolist()
    if not cols:
        raise ValueError('No selected factors; cannot export a signal CSV.')
    fields = ['datetime', 'symbol', 'future_return'] + cols
    if 'tradable_return' in factor_matrix.columns:
        fields.insert(3, 'tradable_return')
    df = factor_matrix[fields].copy()
    signs = {row['factor_id']: (1 if row['mean_ic'] >= 0 else -1)
             for _, row in selected_factors.iterrows()}
    df['alpha_score'] = sum(df[c] * signs[c] for c in cols) / len(cols)
    rank_pct = df.groupby('datetime')['alpha_score'].rank(pct=True)
    df['signal'] = 0
    df.loc[rank_pct >= 0.8, 'signal'] = 1
    df.loc[rank_pct <= 0.2, 'signal'] = -1
    public_cols = ['datetime', 'symbol', 'alpha_score', 'signal']
    df[public_cols + (['future_return', 'tradable_return'] if include_labels and 'tradable_return' in df else [])].to_csv(
        output_path, index=False, encoding='utf-8-sig')
    return df[public_cols + [c for c in ('future_return', 'tradable_return') if c in df.columns]]
