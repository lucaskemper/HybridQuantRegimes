# tests/unit/test_risk.py
import pytest
from src.risk import RiskManager, RiskConfig

def test_risk_metrics(sample_market_data):
    config = RiskConfig()
    risk_manager = RiskManager(config)
    metrics = risk_manager.calculate_metrics(sample_market_data['returns'])
    
    assert 'volatility' in metrics
    assert 'var' in metrics
    assert 'sharpe' in metrics
    assert metrics['volatility'].shape == (3,)  # One per asset