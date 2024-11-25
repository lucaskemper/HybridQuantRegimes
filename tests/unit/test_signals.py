# tests/unit/test_signals.py
import pytest
import numpy as np
import pandas as pd
from src.signals import SignalGenerator

def test_signal_generation(sample_market_data):
    signal_gen = SignalGenerator()
    signals = signal_gen.generate_signals(sample_market_data)
    
    # Basic shape and type checks
    assert signals is not None
    assert isinstance(signals, pd.DataFrame)
    assert signals.shape == sample_market_data['returns'].shape
    
    # Check signal bounds
    assert not signals.isna().any().any(), "Should not contain any NaN values"
    assert (signals >= -1).all().all(), "All signals should be >= -1"
    assert (signals <= 1).all().all(), "All signals should be <= 1"
    
    # Check specific signal properties
    assert signals.iloc[0].all() == 0, "First signals should be neutral (0)"
    assert not (signals == 0).all().all(), "Should contain non-zero signals"
    
    # Print debug information if test fails
    if not ((signals >= -1) & (signals <= 1)).all().all():
        print("\nSignal Statistics:")
        print(f"Min values: {signals.min()}")
        print(f"Max values: {signals.max()}")
        print(f"NaN count: {signals.isna().sum()}")

def test_signal_consistency(sample_market_data):
    """Test if signals are consistent with market data"""
    signal_gen = SignalGenerator()
    signals = signal_gen.generate_signals(sample_market_data)
    
    # Check if signals change with market movement
    returns = sample_market_data['returns']
    for ticker in returns.columns:
        # Get correlation between returns and signals
        correlation = returns[ticker].corr(signals[ticker])
        assert not np.isnan(correlation), f"Correlation should not be NaN for {ticker}"