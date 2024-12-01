import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.risk import RiskConfig, RiskManager


@pytest.fixture
def sample_returns():
    """Create sample returns for testing with different market regimes"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")

    # Create returns with regime shifts
    n_days = len(dates)
    regime_1 = np.random.normal(0.001, 0.02, (n_days // 3, 3))  # Low vol
    regime_2 = np.random.normal(0.0, 0.04, (n_days // 3, 3))  # Medium vol
    regime_3 = np.random.normal(
        -0.002, 0.06, (n_days - 2 * (n_days // 3), 3)
    )  # High vol

    returns_data = np.vstack([regime_1, regime_2, regime_3])
    returns = pd.DataFrame(
        returns_data,
        columns=["NVDA", "AMD", "INTC"],
        index=dates,
    )
    return returns


@pytest.fixture
def risk_manager():
    """Create risk manager instance with test config"""
    config = RiskConfig(
        confidence_level=0.95,
        max_drawdown_limit=0.20,
        volatility_target=0.15,
        var_calculation_method="historical",
        regime_detection_method="volatility",
        correlation_regime=True,
    )
    return RiskManager(config)


def test_core_risk_metrics(risk_manager, sample_returns):
    """Test core risk metric calculations"""
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test volatility
    assert 0 < metrics["portfolio_volatility"] < 1
    assert metrics["portfolio_volatility"] == pytest.approx(
        np.std(sample_returns.mean(axis=1)) * np.sqrt(252), rel=1e-2
    )

    # Test VaR
    assert metrics["var_95"] < 0
    assert metrics["expected_shortfall_95"] <= metrics["var_95"]


def test_regime_detection(risk_manager, sample_returns):
    """Test market regime detection functionality"""
    metrics = risk_manager.calculate_metrics(sample_returns)
    regime = metrics["market_regime"]

    # Test regime structure
    assert isinstance(regime, dict)
    assert all(
        key in regime
        for key in ["volatility_regime", "skewness_regime", "tail_regime", "confidence"]
    )

    # Test regime values
    assert regime["volatility_regime"] in ["high", "normal", "low"]
    assert regime["skewness_regime"] in ["positive", "neutral", "negative"]
    assert regime["tail_regime"] in ["fat", "normal", "thin"]
    assert 0 <= regime["confidence"] <= 1.0


def test_risk_decomposition(risk_manager, sample_returns):
    """Test risk decomposition analysis"""
    metrics = risk_manager.calculate_metrics(sample_returns)
    decomp = metrics["risk_decomposition"]

    # Test structure
    assert isinstance(decomp, dict)
    assert "risk_contribution" in decomp
    assert "diversification_score" in decomp

    # Test risk contributions
    risk_contrib = decomp["risk_contribution"]
    assert isinstance(risk_contrib, pd.Series)
    assert len(risk_contrib) == len(sample_returns.columns)
    assert all(0 <= x <= 1 for x in risk_contrib)


def test_error_handling(risk_manager):
    """Test error handling and edge cases"""
    # Test empty DataFrame
    with pytest.raises(ValueError):
        risk_manager.calculate_metrics(pd.DataFrame())

    # Test invalid data types
    with pytest.raises(TypeError):
        risk_manager.calculate_metrics(np.array([1, 2, 3]))

    # Test single observation
    single_day = pd.DataFrame({"NVDA": [0.01]}, index=[pd.Timestamp("2023-01-01")])
    metrics = risk_manager.calculate_metrics(single_day)
    assert isinstance(metrics, dict)


def test_risk_config_validation():
    """Test RiskConfig validation"""
    # Test valid configuration
    config = RiskConfig(
        confidence_level=0.95, max_drawdown_limit=0.15, volatility_target=0.10
    )
    assert config.confidence_level == 0.95

    # Test invalid configurations
    with pytest.raises(ValueError):
        RiskConfig(confidence_level=1.5)
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown_limit=-0.1)
    with pytest.raises(ValueError):
        RiskConfig(volatility_target=-0.1)


def test_conditional_metrics(risk_manager, sample_returns):
    """Test conditional risk metrics"""
    portfolio_returns = sample_returns.mean(axis=1)
    metrics = risk_manager._calculate_conditional_metrics(portfolio_returns)

    # Test structure
    assert isinstance(metrics, dict)
    assert any(key.endswith("_volatility") for key in metrics)
    assert any(key.endswith("_var") for key in metrics)
    assert any(key.endswith("_es") for key in metrics)


def test_extreme_value_analysis(risk_manager, sample_returns):
    """Test EVT analysis"""
    portfolio_returns = sample_returns.mean(axis=1)
    evt_metrics = risk_manager._extreme_value_analysis(portfolio_returns)

    # Test structure
    assert all(
        key in evt_metrics
        for key in ["tail_index", "scale", "threshold", "exceedance_rate"]
    )
    assert 0 <= evt_metrics["exceedance_rate"] <= 1


def test_parametric_var_calculation(risk_manager, sample_returns):
    """Test parametric VaR calculation method"""
    risk_manager.config.var_calculation_method = "parametric"
    metrics = risk_manager.calculate_metrics(sample_returns)

    assert metrics["var_95"] < 0
    assert isinstance(metrics["var_95"], float)
    assert np.isfinite(metrics["var_95"])


def test_dynamic_correlation_calculation(risk_manager, sample_returns):
    """Test dynamic correlation calculation with different regimes"""
    # Test high volatility regime
    high_vol_returns = sample_returns.copy()
    high_vol_returns.iloc[-21:] *= 2  # Increase volatility in recent period
    corr_matrix_high = risk_manager._calculate_dynamic_correlation(high_vol_returns)
    assert isinstance(corr_matrix_high, pd.DataFrame)
    assert corr_matrix_high.shape == (
        len(sample_returns.columns),
        len(sample_returns.columns),
    )

    # Test normal volatility regime
    corr_matrix_normal = risk_manager._calculate_dynamic_correlation(sample_returns)
    assert isinstance(corr_matrix_normal, pd.DataFrame)
    assert corr_matrix_normal.shape == (
        len(sample_returns.columns),
        len(sample_returns.columns),
    )


def test_weights_validation(risk_manager, sample_returns):
    """Test weights validation and handling"""
    # Test with valid weights
    valid_weights = [0.4, 0.3, 0.3]
    risk_manager.config.weights = valid_weights
    metrics = risk_manager.calculate_metrics(sample_returns)
    assert "portfolio_volatility" in metrics

    # Test with invalid weights
    with pytest.raises(ValueError):
        risk_manager.config.weights = [0.5, 0.6]  # Wrong length
        risk_manager.calculate_metrics(sample_returns)

    with pytest.raises(ValueError):
        risk_manager.config.weights = [-0.1, 0.6, 0.5]  # Negative weight
        risk_manager.calculate_metrics(sample_returns)


def test_market_regime_detection(risk_manager, sample_returns):
    """Test market regime detection with different volatility levels"""
    # Test high volatility regime
    high_vol_returns = sample_returns.copy()
    high_vol_returns.iloc[-21:] *= 2  # Increase recent volatility
    high_vol_regime = risk_manager._detect_market_regime(high_vol_returns)
    assert isinstance(high_vol_regime, dict)
    assert "volatility_regime" in high_vol_regime
    assert high_vol_regime["volatility_regime"] in ["high", "normal", "low"]

    # Test low volatility regime
    low_vol_returns = sample_returns.copy()
    low_vol_returns.iloc[-21:] *= 0.2  # Decrease recent volatility
    low_vol_regime = risk_manager._detect_market_regime(low_vol_returns)
    assert isinstance(low_vol_regime, dict)
    assert "volatility_regime" in low_vol_regime
    assert low_vol_regime["volatility_regime"] in ["high", "normal", "low"]
