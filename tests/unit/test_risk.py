# tests/unit/test_risk.py
import numpy as np
import pandas as pd
import pytest

from src.risk import RiskConfig, RiskManager


@pytest.fixture
def sample_returns():
    """Create sample returns for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), 3)),
        columns=["NVDA", "AMD", "INTC"],
        index=dates,
    )
    return returns


def test_risk_metrics_calculation(sample_returns):
    """Test risk metrics calculation"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test core metrics exist
    assert "portfolio_volatility" in metrics
    assert "var_95" in metrics
    assert "expected_shortfall_95" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert "rolling_volatility" in metrics
    assert "correlation" in metrics

    # Test metric values are reasonable
    assert 0 < metrics["portfolio_volatility"] < 1
    assert metrics["var_95"] < 0  # VaR should be negative
    assert (
        metrics["expected_shortfall_95"] <= metrics["var_95"]
    )  # ES should be more extreme than VaR
    assert -1 <= metrics["max_drawdown"] <= 0
    assert isinstance(metrics["sharpe_ratio"], float)


def test_stress_testing(sample_returns):
    """Test stress testing scenarios"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    stress_test = metrics["stress_test"]
    assert "worst_month" in stress_test
    assert "worst_quarter" in stress_test
    assert "recovery_time" in stress_test
    assert "max_consecutive_loss" in stress_test

    # Test stress test values are reasonable
    assert stress_test["worst_month"] <= 0
    assert stress_test["worst_quarter"] <= 0
    assert stress_test["recovery_time"] >= 0
    assert isinstance(stress_test["max_consecutive_loss"], int)
    assert stress_test["max_consecutive_loss"] >= 0


def test_rolling_metrics(sample_returns):
    """Test rolling metric calculations"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test rolling volatility
    rolling_vol = metrics["rolling_volatility"]
    assert isinstance(rolling_vol, pd.DataFrame)
    assert len(rolling_vol) == len(sample_returns)
    assert "21d" in rolling_vol.columns
    assert "63d" in rolling_vol.columns

    # Test rolling values are reasonable
    # Exclude NaN values from the comparison
    assert (rolling_vol["21d"].dropna() >= 0).all()
    assert (rolling_vol["63d"].dropna() >= 0).all()

    # Test that we have some non-NaN values
    assert not rolling_vol["21d"].isna().all()
    assert not rolling_vol["63d"].isna().all()


def test_position_scaling(sample_returns):
    """Test position scaling calculations"""
    risk_config = RiskConfig(volatility_target=0.15, max_drawdown_limit=0.20)
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test risk-adjusted position
    assert "risk_adjusted_position" in metrics
    position = metrics["risk_adjusted_position"]
    assert 0.2 <= position <= 1.0  # Should be between minimum (20%) and maximum (100%)


def test_drawdown_calculations(sample_returns):
    """Test drawdown-related calculations"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test drawdown metrics
    assert "drawdown_ratio" in metrics
    assert metrics["drawdown_ratio"] >= 0
    assert metrics["max_drawdown"] <= 0

    # Test recovery time calculation
    stress_test = metrics["stress_test"]
    assert "recovery_time" in stress_test
    assert stress_test["recovery_time"] >= 0


def test_volatility_ratio(sample_returns):
    """Test volatility ratio calculations"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    metrics = risk_manager.calculate_metrics(sample_returns)

    # Test volatility ratio
    assert "volatility_ratio" in metrics
    vol_ratio = metrics["volatility_ratio"]
    assert vol_ratio > 0
    assert isinstance(vol_ratio, float)


def test_edge_cases():
    """Test edge cases and error handling"""
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)

    # Test with completely empty DataFrame
    empty_returns = pd.DataFrame()
    with pytest.raises(
        ValueError, match="Returns DataFrame must have at least one column"
    ):
        risk_manager.calculate_metrics(empty_returns)

    # Test with DataFrame that has index but no columns
    no_columns = pd.DataFrame(index=[pd.Timestamp("2023-01-01")])
    with pytest.raises(
        ValueError, match="Returns DataFrame must have at least one column"
    ):
        risk_manager.calculate_metrics(no_columns)

    # Test with DataFrame that has columns but no rows
    no_rows = pd.DataFrame(columns=["NVDA", "AMD", "INTC"])
    with pytest.raises(
        ValueError, match="Returns DataFrame must have at least one row"
    ):
        risk_manager.calculate_metrics(no_rows)

    # Test with single day of returns
    single_day = pd.DataFrame(
        {"NVDA": [0.01], "AMD": [0.02], "INTC": [0.03]},
        index=[pd.Timestamp("2023-01-01")],
    )
    metrics = risk_manager.calculate_metrics(single_day)
    assert metrics["volatility_ratio"] == 1.0
    assert metrics["drawdown_ratio"] == 0.0
