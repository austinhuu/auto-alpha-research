"""Evaluate formula expressions with a minimal AST allowlist, no Python attribute access."""
import ast
import warnings
import numpy as np
import pandas as pd
from operators.basic_ops import safe_div, ts_mean, ts_std, ts_min, ts_max, ts_rank, delta, delay, zscore, signed_log

_FIELDS = {'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'returns'}
_FUNCS = {'safe_div': safe_div, 'ts_mean': ts_mean, 'ts_std': ts_std, 'ts_min': ts_min,
          'ts_max': ts_max, 'ts_rank': ts_rank, 'delta': delta, 'delay': delay,
          'zscore': zscore, 'signed_log': signed_log, 'abs': abs}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)

def validate_formula(expression):
    """Reject calls/attributes/imports/subscripts and excessive complexity up front."""
    if not isinstance(expression, str) or len(expression) > 250:
        raise ValueError('Formula must be a string of at most 250 characters')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 160:
        raise ValueError('Expression too complex')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _OPERATORS): raise ValueError('Operator not allowed')
            continue
        if isinstance(node, _OPERATORS): continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)): raise ValueError('Unary operator not allowed')
            continue
        if isinstance(node, (ast.UAdd, ast.USub)): continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
                raise ValueError('Function not allowed')
            if node.func.id in {'ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_rank', 'delta', 'delay'}:
                if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant) or type(node.args[1].value) is not int or not 1 <= node.args[1].value <= 256:
                    raise ValueError('Invalid time-series window')
            continue
        if isinstance(node, ast.Name):
            if node.id not in _FIELDS | set(_FUNCS): raise ValueError(f'Unknown variable: {node.id}')
            continue
        if isinstance(node, ast.Constant):
            if type(node.value) not in (float, int) or abs(node.value) > 1e9:
                raise ValueError('Invalid numeric constant')
            continue
        raise ValueError(f'Disallowed expression element: {type(node).__name__}')
    return compile(tree, '<factor>', 'eval')

def _evaluate_expression_for_symbol(g, expression):
    environment = {k: g[k] for k in _FIELDS}
    environment.update(_FUNCS)
    try:
        code = validate_formula(expression)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = eval(code, {'__builtins__': {}}, environment)
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=g.index)
        return pd.to_numeric(value, errors='coerce')
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError, KeyError):
        return pd.Series(np.nan, index=g.index)

def calculate_factor_matrix(data, factor_df):
    result = data[['datetime', 'symbol', 'future_return']].copy()
    if 'tradable_return' in data.columns:
        result['tradable_return'] = data['tradable_return']
    pieces = []
    for _, row in factor_df.iterrows():
        vals = [_evaluate_expression_for_symbol(g, row['expression'])
                for _, g in data.groupby('symbol', sort=False)]
        pieces.append(pd.concat(vals).sort_index().rename(row['factor_id']))
    if pieces:
        result = pd.concat([result] + pieces, axis=1)
    factor_cols = [c for c in result.columns if c.startswith(('alpha_', 'rl_alpha_'))]
    for col in factor_cols:
        cleaned = result[col].replace([np.inf, -np.inf], np.nan)
        result[col] = cleaned.groupby(result['datetime']).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )
    return result
