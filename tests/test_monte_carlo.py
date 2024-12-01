# tests/unit/test_monte_carlo.py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.monte_carlo import MonteCarlo, SimConfig


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    n_assets = 3

    # Generate returns directly instead of prices
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (len(dates), n_assets)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )

    return {"returns": returns}


@pytest.fixture
def config():
    """Create simulation configuration"""
    return SimConfig(
        n_sims=1000,
        n_days=252,
        risk_free_rate=0.05,
        confidence_levels=(0.05, 0.25, 0.5, 0.75, 0.95),
    )


def test_sim_config():
    """Test SimConfig initialization and validation"""
    # Test valid config
    valid_config = SimConfig(n_sims=1000, n_days=252)
    assert valid_config.n_sims == 1000
    assert valid_config.n_days == 252

    # Test invalid configs
    with pytest.raises(ValueError):
        SimConfig(n_sims=-1)
    with pytest.raises(ValueError):
        SimConfig(n_days=0)
    with pytest.raises(ValueError):
        SimConfig(confidence_levels=(-0.1, 0.5))
    with pytest.raises(ValueError):
        SimConfig(distribution="invalid")


def test_monte_carlo_basic(sample_market_data, config):
    """Test basic Monte Carlo simulation functionality"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)

    # Check results structure
    assert isinstance(results, dict)
    assert "expected_return" in results
    assert "simulation_volatility" in results
    assert "confidence_intervals" in results

    # Check dimensions
    n_assets = len(sample_market_data["returns"].columns)
    assert results["paths"].shape == (config.n_sims, n_assets, config.n_days)


def test_monte_carlo_validation(sample_market_data, config):
    """Test Monte Carlo validation methods"""
    mc = MonteCarlo(config)

    # Test input validation
    with pytest.raises(ValueError):
        mc.simulate({"returns": pd.DataFrame()})

    with pytest.raises(TypeError):
        mc.simulate(None)

    # Test valid simulation
    results = mc.simulate(sample_market_data)
    assert mc._validate_results(results)


def test_monte_carlo_distributions(sample_market_data):
    """Test different distribution configurations"""
    base_config = SimConfig(n_sims=1000, n_days=252)

    # Test normal distribution
    normal_config = SimConfig(n_sims=1000, n_days=252, distribution="normal")
    mc_normal = MonteCarlo(normal_config)
    normal_results = mc_normal.simulate(sample_market_data)

    # Test Student's t distribution
    t_config = SimConfig(n_sims=1000, n_days=252, distribution="t")
    mc_t = MonteCarlo(t_config)
    t_results = mc_t.simulate(sample_market_data)

    # Results should be different but valid
    assert isinstance(normal_results, dict)
    assert isinstance(t_results, dict)


def test_monte_carlo_confidence_intervals(sample_market_data, config):
    """Test confidence interval calculations"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)

    intervals = results["confidence_intervals"]

    # Check interval properties
    for level in config.confidence_levels:
        assert level in intervals
        interval = intervals[level]
        assert isinstance(interval, (float, np.ndarray))


def test_monte_carlo_reproducibility(sample_market_data, config):
    """Test simulation reproducibility with same seed"""
    mc = MonteCarlo(config)

    np.random.seed(42)
    results1 = mc.simulate(sample_market_data)

    np.random.seed(42)
    results2 = mc.simulate(sample_market_data)

    # Key metrics should be identical
    assert np.allclose(results1["expected_return"], results2["expected_return"])
    assert np.allclose(
        results1["simulation_volatility"], results2["simulation_volatility"]
    )


def test_forecast_volatility(sample_market_data, config):
    """Test GARCH volatility forecasting"""
    mc = MonteCarlo(config)
    returns = sample_market_data["returns"]

    forecasted_vol = mc._forecast_volatility(returns)

    assert isinstance(forecasted_vol, np.ndarray)
    assert len(forecasted_vol) == len(returns.columns)
    assert np.all(forecasted_vol > 0)  # Volatility should be positive


def test_calculate_statistics(sample_market_data, config):
    """Test statistics calculation"""
    mc = MonteCarlo(config)
    portfolio_values = np.random.normal(100, 10, 1000)

    stats = mc._calculate_statistics(portfolio_values)

    # Check required statistics
    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "skew" in stats
    assert "kurtosis" in stats
    assert "sharpe_ratio" in stats
    assert "bias" in stats
    assert "variance" in stats
    assert "cross_val_score" in stats


def test_bias_variance_calculation(config):
    """Test bias-variance decomposition"""
    mc = MonteCarlo(config)
    portfolio_values = np.random.normal(100, 10, 1000)

    bias_variance = mc._calculate_bias_variance(portfolio_values)

    assert "bias" in bias_variance
    assert "variance" in bias_variance
    assert "cross_val_score" in bias_variance


def test_simulation_validation(sample_market_data, config):
    """Test simulation validation checks"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    validation = results["validation"]

    assert isinstance(validation, dict)
    assert "positive_values" in validation
    assert "correlation_preservation" in validation
    assert "reasonable_returns" in validation
    assert "volatility_alignment" in validation


def test_distribution_types(sample_market_data, config):
    """Test different distribution types"""
    mc = MonteCarlo(config)

    # Test normal distribution
    mc.distribution = "normal"
    normal_results = mc.simulate(sample_market_data)

    # Test Student's t distribution
    mc.distribution = "t"
    t_results = mc.simulate(sample_market_data)

    # Results should be different but valid
    assert isinstance(normal_results, dict)
    assert isinstance(t_results, dict)

    # Invalid distribution should raise error
    mc.distribution = "invalid"
    with pytest.raises(ValueError):
        mc.simulate(sample_market_data)
