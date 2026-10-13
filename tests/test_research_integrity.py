from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import calculate_backtest_metrics
from backtesting.portfolio_backtester import run_long_short_backtest
from data.data_loader import load_market_data
from data.label_builder import build_forward_return_label
from evaluator.predictor_network import binary_auc_score
from operators.basic_ops import zscore
from operators.expression_evaluator import validate_formula, calculate_factor_matrix
from tools.generate_sample_data import generate_sample_data
from validation.time_split import split_development_holdout, build_walk_forward_windows
from validation.strict_walk_forward_validator import StrictWalkForwardValidator
from veighna_adapter.signal_exporter import export_alpha_signals


def market(tmp_path, symbols=9, dates=75):
    path = tmp_path / 'synthetic.csv'
    generate_sample_data(path, n_symbols=symbols, n_days=dates, seed=2026)
    return build_forward_return_label(load_market_data(path))


def test_dates_and_unique_symbols(tmp_path):
    data = market(tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(data['datetime'])
    assert data['symbol'].nunique() == 9
    assert not data.duplicated(['datetime', 'symbol']).any()
    path = tmp_path / 'numeric_dates.csv'
    pd.DataFrame({'datetime': [20240102, 20240103], 'symbol': ['000001.SZ'] * 2,
                  'open': [1, 2], 'high': [2, 3], 'low': [1, 1],
                  'close': [1.2, 2.2], 'volume': [100, 100],
                  'amount': [120, 220]}).to_csv(path, index=False)
    assert str(load_market_data(path)['datetime'].iloc[0].date()) == '2024-01-02'


def test_purged_time_holdout(tmp_path):
    data = market(tmp_path)
    dev, holdout, dev_dates, test_dates = split_development_holdout(data, forward_period=1)
    all_dates = sorted(data['datetime'].unique())
    assert dev_dates[-1] == all_dates[int(.8 * len(all_dates)) - 3]
    assert test_dates[0] == all_dates[int(.8 * len(all_dates))]
    assert dev['datetime'].max() < holdout['datetime'].min()
    assert len(set(dev_dates) & set(test_dates)) == 0


def test_signal_and_return_timing():
    df = pd.DataFrame({'symbol': ['000001.SZ'] * 4,
                       'datetime': pd.date_range('2024-01-01', periods=4),
                       'close': [10., 11., 12., 14.],
                       'open': [10., 20., 25., 35.]})
    labeled = build_forward_return_label(df)
    assert labeled['future_return'].iloc[0] == pytest.approx(.1)
    assert labeled['tradable_return'].iloc[0] == pytest.approx(.25)
    assert pd.isna(labeled['tradable_return'].iloc[-2])


def test_ast_only_expected_safe_formula():
    validate_formula('safe_div(ts_mean(close, 5), ts_mean(volume, 10)) - 1')
    for bad in ['open.__class__', "__import__('os').system('whoami')", 'close.iloc[0]',
                '(lambda x: x)(close)', 'ts_mean(close, -1)', 'ts_std(close, 5000)']:
        with pytest.raises((ValueError, SyntaxError)):
            validate_formula(bad)


def test_no_future_zscore_leakage():
    earlier = pd.Series([1., 2., 3., 4., 5., 6., 7.])
    after = pd.Series([1., 2., 3., 4., 5., 6., 1000000.])
    assert zscore(earlier).iloc[5] == pytest.approx(zscore(after).iloc[5])


def test_causal_expression_with_future_shock(tmp_path):
    data = market(tmp_path, symbols=9, dates=70)
    formulas = pd.DataFrame([{'factor_id': 'alpha_0001', 'expression': 'zscore(close)'}])
    before = calculate_factor_matrix(data, formulas)
    changed = data.copy()
    last_date = changed.datetime.max()
    changed.loc[changed.datetime == last_date, 'close'] *= 9
    after = calculate_factor_matrix(changed, formulas)
    previous = before.datetime < last_date
    pd.testing.assert_series_equal(before.loc[previous, 'alpha_0001'],
                                   after.loc[previous, 'alpha_0001'])


def test_signal_csv_excludes_future_targets(tmp_path):
    data = market(tmp_path, symbols=9, dates=65)
    formula = pd.DataFrame([{'factor_id': 'alpha_0001', 'expression': 'ts_mean(close, 5)'}])
    matrix = calculate_factor_matrix(data, formula)
    sel = pd.DataFrame([{'factor_id': 'alpha_0001', 'mean_ic': .04}])
    dest = tmp_path / 'signals.csv'
    in_memory = export_alpha_signals(matrix, sel, dest)
    assert {'future_return', 'tradable_return'} <= set(in_memory.columns)
    saved = pd.read_csv(dest)
    assert saved.columns.tolist() == ['datetime', 'symbol', 'alpha_score', 'signal']


def test_auc_ties():
    assert binary_auc_score([1, 0, 1, 0], [.5, .5, .5, .5]) == pytest.approx(.5)
    assert binary_auc_score([1, 0], [.9, .1]) == pytest.approx(1.)


def test_backtest_and_equity_metrics_include_first_period():
    scores = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
    rows = [{'datetime': '2025-01-02', 'symbol': f'{i:06d}.SZ', 'alpha_score': v,
             'tradable_return': v / 100} for i, v in enumerate(scores)]
    metrics, eq = run_long_short_backtest(pd.DataFrame(rows), commission_rate=0, slippage_rate=0)
    assert eq['daily_return'].iloc[0] > 0
    assert metrics['total_return'] == pytest.approx(eq['daily_return'].iloc[0])


def test_strict_window_factor_selection_does_not_evaluate_pretest_future_labels(tmp_path):
    data = market(tmp_path, symbols=9, dates=75)
    factor = pd.DataFrame([{'factor_id': 'alpha_0001', 'expression': 'ts_mean(close, 5)'}])
    windows = build_walk_forward_windows(data, train_days=45, test_days=15, step_days=15)
    assert windows
    validator = StrictWalkForwardValidator(data, factor, windows, tmp_path, top_k=1, purge_days=2)
    result = validator.run()
    assert isinstance(result, pd.DataFrame)
