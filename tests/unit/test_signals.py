import numpy as np
import pandas as pd
import pytest
from src import signals

@pytest.fixture
def sample_features():
    idx = pd.date_range('2020-01-01', periods=30)
    return {
        'AAPL': pd.DataFrame({
            'momentum_20d': np.random.randn(30),
            'macd_signal': np.random.randn(30),
            'rsi_14': np.random.uniform(30, 70, 30),
            'williams_r': np.random.uniform(-100, 0, 30),
            'term_structure_slope': np.random.randn(30),
            'vix_percentile': np.random.rand(30),
            'semiconductor_pmi': np.random.uniform(40, 60, 30),
            'realized_volatility': np.abs(np.random.randn(30)),
            'momentum_rank': np.random.rand(30),
            'nvda_amd_spread': np.random.randn(30),
            'returns': np.random.randn(30),
            'price': np.random.uniform(100, 200, 30),
        }, index=idx)
    }

@pytest.fixture
def sample_returns():
    idx = pd.date_range('2020-01-01', periods=30)
    return pd.DataFrame({'AAPL': np.random.randn(30)}, index=idx)

@pytest.fixture
def sample_regime():
    idx = pd.date_range('2020-01-01', periods=30)
    return {'AAPL': pd.Series(['High Vol']*10 + ['Medium Vol']*10 + ['Low Vol']*10, index=idx)}

def test_momentum_signal(sample_features):
    f = sample_features['AAPL']
    out = signals.momentum_signal(f)
    assert isinstance(out, pd.Series)
    assert out.shape == f.shape[:1]

def test_meanrev_signal(sample_features):
    f = sample_features['AAPL']
    out = signals.meanrev_signal(f)
    assert isinstance(out, pd.Series)
    assert out.shape == f.shape[:1]

def test_macro_signal(sample_features):
    f = sample_features['AAPL']
    out = signals.macro_signal(f)
    assert isinstance(out, pd.Series)
    assert out.shape == f.shape[:1]

def test_vol_breakout_signal(sample_features):
    f = sample_features['AAPL']
    out = signals.vol_breakout_signal(f)
    assert isinstance(out, pd.Series)
    assert out.shape == f.shape[:1]

def test_cross_sectional_signal(sample_features):
    f = sample_features['AAPL']
    out = signals.cross_sectional_signal(f)
    assert isinstance(out, pd.Series)
    assert out.shape == f.shape[:1]

def test_generate_signals_core(sample_returns, sample_features, sample_regime):
    out = signals.generate_signals_core(sample_returns, sample_features, sample_regime)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == sample_returns.shape
    assert not out.isnull().any().any()

def test_signal_generator_class(sample_returns, sample_features, sample_regime):
    sg = signals.SignalGenerator()
    market_data = {
        'returns': sample_returns,
        'features': sample_features,
        'regime_series_dict': sample_regime
    }
    out = sg.generate_signals(market_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == sample_returns.shape
    assert not out.isnull().any().any()

def test_diagnose_signals(sample_returns, sample_features, sample_regime):
    sg = signals.SignalGenerator()
    market_data = {
        'returns': sample_returns,
        'features': sample_features,
        'regime_series_dict': sample_regime
    }
    signals_df = sg.generate_signals(market_data)
    diag = sg.diagnose_signals(signals_df, sample_returns, sample_regime['AAPL'])
    assert isinstance(diag, dict)
    assert 'signal_stats' in diag
    assert 'correlations' in diag 