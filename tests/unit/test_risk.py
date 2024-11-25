# tests/unit/test_risk.py
import pytest
import pandas as pd
import numpy as np
from src.risk import RiskManager, RiskConfig

@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), 3)),
        columns=['NVDA', 'AMD', 'INTC'],
        index=dates
    )
    return returns

def test_risk_metrics_calculation(sample_returns):
    """Test risk metrics calculation"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)
    
    # Test core metrics exist
    assert 'portfolio_volatility' in metrics
    assert 'var_95' in metrics
    assert 'var_99' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'expected_shortfall_95' in metrics
    assert 'rolling_volatility' in metrics
    assert 'correlation' in metrics
    
    # Test metric values are reasonable
    assert 0 < metrics['portfolio_volatility'] < 1
    assert metrics['var_95'] < 0  # VaR should be negative
    assert metrics['var_99'] < metrics['var_95']  # 99% VaR should be more extreme
    assert -1 <= metrics['max_drawdown'] <= 0
    assert isinstance(metrics['sharpe_ratio'], float)

def test_stress_testing(sample_returns):
    """Test stress testing scenarios"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)
    
    # Test metric values under normal conditions
    assert 0 < metrics['portfolio_volatility'] < 1
    assert metrics['var_95'] < 0
    assert -1 <= metrics['max_drawdown'] <= 0

def test_rolling_metrics(sample_returns):
    """Test rolling metric calculations"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)
    
    # Test rolling volatility
    assert 'rolling_volatility' in metrics
    rolling_vol = metrics['rolling_volatility']
    assert isinstance(rolling_vol, pd.DataFrame)
    assert len(rolling_vol) == len(sample_returns)
    assert '21d' in rolling_vol.columns
    assert '63d' in rolling_vol.columns

def test_config_impact(sample_returns):
    """Test impact of different configurations"""
    config1 = RiskConfig(confidence_level=0.95)
    config2 = RiskConfig(confidence_level=0.99)
    
    risk_manager1 = RiskManager(config1)
    risk_manager2 = RiskManager(config2)
    
    metrics1 = risk_manager1.calculate_metrics(sample_returns)
    metrics2 = risk_manager2.calculate_metrics(sample_returns)
    
    assert metrics1['var_95'] != metrics2['var_99']