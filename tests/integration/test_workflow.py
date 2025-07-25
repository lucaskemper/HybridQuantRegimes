# tests/unit/test_risk.py
import pytest
import numpy as np
import pandas as pd
from src.risk import RiskManager, RiskConfig
from src.signals import SignalGenerator
from src.regime import get_regime_series_for_signals

def test_risk_metrics(sample_market_data):
    config = RiskConfig()
    risk_manager = RiskManager(config)
    metrics = risk_manager.calculate_metrics(sample_market_data['returns'])
    
    assert 'volatility' in metrics
    assert 'var' in metrics
    assert 'sharpe' in metrics
    assert metrics['volatility'].shape == (3,)  # One per asset

def test_signals_regime_integration():
    # Create synthetic returns/features for two tickers
    dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
    tickers = ["AAPL", "NVDA"]
    returns = pd.DataFrame(np.random.normal(0, 0.01, size=(len(dates), len(tickers))), index=dates, columns=tickers)
    features = {}
    for t in tickers:
        df = pd.DataFrame({
            'returns': returns[t],
            'momentum_20d': returns[t].rolling(20, min_periods=1).mean(),
            'macd_signal': np.random.normal(0, 1, len(dates)),
            'ma_20d': returns[t].rolling(20, min_periods=1).mean(),
            'price': 100 + np.cumsum(returns[t]),
            'rsi_14': np.clip(50 + np.random.normal(0, 10, len(dates)), 0, 100),
            'williams_r': np.random.uniform(-100, 0, len(dates)),
            'vix_percentile': np.random.uniform(0, 1, len(dates)),
            'yield_curve_slope': np.random.normal(0, 1, len(dates)),
            'semiconductor_pmi': np.random.uniform(45, 60, len(dates)),
            'realized_volatility': returns[t].rolling(20, min_periods=1).std(),
            'momentum_rank': np.random.uniform(0, 1, len(dates)),
            'nvda_amd_spread': np.random.normal(0, 1, len(dates)),
        }, index=dates)
        features[t] = df
    market_data = {'returns': returns, 'features': features}
    # Get regime labels
    regime_dict = get_regime_series_for_signals({t: returns[t] for t in tickers})
    # Generate signals
    sg = SignalGenerator()
    signals = sg.generate_signals(market_data, regime_dict)
    # Assert output shape and not all zero
    assert signals.shape == returns.shape
    assert (signals.abs().sum().sum() > 0)