import numpy as np
import pandas as pd
import pytest
from src.risk import RiskConfig, RiskManager

def synthetic_returns(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range('2020-01-01', periods=n)
    return pd.Series(np.random.normal(0, 0.01, n), index=idx)

def synthetic_returns_df(n=100, assets=3, seed=42):
    np.random.seed(seed)
    idx = pd.date_range('2020-01-01', periods=n)
    data = np.random.normal(0, 0.01, (n, assets))
    cols = [f'Asset{i+1}' for i in range(assets)]
    return pd.DataFrame(data, index=idx, columns=cols)

def test_risk_config_properties():
    config = RiskConfig(confidence_level=0.95, max_drawdown_limit=0.2, volatility_target=0.15)
    config.weights = [0.5, 0.5]
    assert config.confidence_level == 0.95
    assert config.max_drawdown_limit == 0.2
    assert config.volatility_target == 0.15
    assert np.isclose(sum(config.weights), 1.0)
    with pytest.raises(ValueError):
        config.weights = [0.7, 0.7]
    with pytest.raises(TypeError):
        config.weights = 'not_a_list'
    with pytest.raises(ValueError):
        config.weights = [-0.5, 1.5]

def test_risk_manager_metrics():
    config = RiskConfig()
    rm = RiskManager(config)
    rets = synthetic_returns()
    metrics = rm.calculate_metrics(rets)
    assert isinstance(metrics, dict)
    assert 'mean_return' in metrics
    assert 'std_return' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    # Check bootstrapped metrics
    metrics_bs = rm.calculate_metrics(rets, n_bootstraps=10)
    assert 'sharpe_ci' in metrics_bs
    assert 'var_95_ci' in metrics_bs
    assert 'max_drawdown_ci' in metrics_bs

def test_risk_manager_drawdown_and_vol():
    config = RiskConfig()
    rm = RiskManager(config)
    rets = synthetic_returns()
    dd = rm._calculate_max_drawdown(rets)
    vol = rm._calculate_volatility(rets)
    assert isinstance(dd, float)
    assert isinstance(vol, float)

def test_risk_manager_risk_report():
    config = RiskConfig()
    rm = RiskManager(config)
    rets = synthetic_returns()
    report = rm.generate_risk_report(rets, n_bootstraps=2)
    assert isinstance(report, dict)
    assert 'risk_metrics' in report
    assert 'risk_assessment' in report
    assert 'recommendations' in report
    assert 'summary' in report

def test_risk_manager_decompose_risk():
    config = RiskConfig()
    rm = RiskManager(config, weights=[0.4, 0.3, 0.3])
    rets_df = synthetic_returns_df(assets=3)
    out = rm._decompose_risk(rets_df)
    assert 'risk_contribution' in out
    assert 'diversification_score' in out
    assert 'concentration_score' in out
    assert isinstance(out['risk_contribution'], pd.Series)

def test_risk_manager_bootstrap_metric():
    config = RiskConfig()
    rm = RiskManager(config)
    rets = synthetic_returns()
    ci = rm.bootstrap_metric(np.mean, rets, n=10)
    assert isinstance(ci, (tuple, list, np.ndarray))
    assert len(ci) == 2

def test_risk_manager_adjust_signal_by_risk_forecast():
    config = RiskConfig()
    rm = RiskManager(config)
    signal = pd.Series(np.random.randn(10))
    adjusted = rm.adjust_signal_by_risk_forecast(signal, forecasted_vol=0.2)
    assert isinstance(adjusted, pd.Series)
    assert adjusted.shape == signal.shape 