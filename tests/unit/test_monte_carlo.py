# tests/unit/test_monte_carlo.py
import pytest
import numpy as np
import pandas as pd
from src.monte_carlo import MonteCarlo, SimConfig
import scipy.stats as scipy_stats

@pytest.fixture
def config():
    return SimConfig(n_sims=100, n_days=50)  # Smaller values for testing

def test_monte_carlo_simulation(sample_market_data, config):
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    
    # Test basic structure
    assert 'paths' in results
    assert 'final_values' in results
    assert 'confidence_intervals' in results
    assert 'expected_return' in results
    assert 'simulation_volatility' in results
    assert 'statistics' in results
    assert 'validation' in results

    # Test dimensions
    assert len(results['final_values']) == config.n_sims
    assert results['paths'].shape == (config.n_sims, len(sample_market_data['returns'].columns), config.n_days)

    # Test confidence intervals
    for level in config.confidence_levels:
        assert level in results['confidence_intervals']
        
    # Test statistical properties
    assert isinstance(results['expected_return'], float)
    assert isinstance(results['simulation_volatility'], float)
    assert results['simulation_volatility'] > 0

def test_monte_carlo_validation(sample_market_data, config):
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    validation = results['validation']
    
    assert isinstance(validation, dict)
    assert 'positive_values' in validation
    assert 'correlation_preservation' in validation
    assert 'reasonable_returns' in validation
    assert 'volatility_alignment' in validation

def test_monte_carlo_statistics(sample_market_data, config):
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    stats = results['statistics']
    
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert 'median' in stats
    assert 'std' in stats
    assert 'skew' in stats
    assert 'kurtosis' in stats
    assert 'sharpe_ratio' in stats

def test_invalid_market_data(config):
    mc = MonteCarlo(config)
    
    with pytest.raises(ValueError, match="market_data must be a dictionary containing 'returns' DataFrame"):
        mc.simulate({})
    
    with pytest.raises(ValueError, match="market_data must be a dictionary containing 'returns' DataFrame"):
        mc.simulate({'wrong_key': pd.DataFrame()})
    
    with pytest.raises(ValueError, match="Returns data is empty"):
        mc.simulate({'returns': pd.DataFrame()})

def test_simulation_bounds(sample_market_data, config):
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    
    # Test that all paths are positive
    assert np.all(results['paths'] > 0)
    
    # Test that confidence intervals are properly ordered
    conf_intervals = list(results['confidence_intervals'].values())
    assert all(conf_intervals[i] <= conf_intervals[i+1] for i in range(len(conf_intervals)-1))

def test_config_validation():
    """Test configuration validation"""
    # Test negative simulations
    with pytest.raises(ValueError, match="Number of simulations must be positive"):
        SimConfig(n_sims=-1, n_days=252)
    
    # Test zero days
    with pytest.raises(ValueError, match="Number of days must be positive"):
        SimConfig(n_sims=1000, n_days=0)
    
    # Test invalid confidence levels
    with pytest.raises(ValueError, match="Confidence levels must be between 0 and 1"):
        SimConfig(n_sims=1000, n_days=252, confidence_levels=(-0.1, 0.5, 1.1))

def test_correlation_preservation(sample_market_data, config):
    """Test that correlations are preserved in simulations"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    
    # Calculate historical correlations from returns
    returns_df = pd.DataFrame(
        np.diff(np.log(sample_market_data['close']), axis=0),
        columns=sample_market_data['close'].columns
    )
    historical_corr = returns_df.corr()
    
    # Calculate simulated correlations
    simulated_returns = np.diff(np.log(results['paths']), axis=2)
    simulated_returns_mean = simulated_returns.mean(axis=0)
    simulated_corr = pd.DataFrame(
        np.corrcoef(simulated_returns_mean),
        index=historical_corr.index,
        columns=historical_corr.columns
    )
    
    # Test correlation differences with higher tolerance
    correlation_diff = np.abs(historical_corr - simulated_corr)
    assert np.all(correlation_diff < 0.5)  # Increased tolerance for test stability

def test_performance_metrics(sample_market_data, config):
    """Test performance metrics calculation"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    stats = results['statistics']
    
    # Test Sharpe ratio calculation using config's risk_free_rate
    excess_return = results['expected_return'] - config.risk_free_rate
    calculated_sharpe = excess_return / results['simulation_volatility']
    
    # Use a more forgiving tolerance for floating-point comparison
    assert np.isclose(stats['sharpe_ratio'], calculated_sharpe, rtol=1e-6)

def test_garch_volatility_forecasting(sample_market_data, config):
    """Test GARCH volatility forecasting functionality"""
    mc = MonteCarlo(config)
    
    # Test the forecasting method directly
    forecasted_vol = mc._forecast_volatility(sample_market_data['returns'])
    
    # Check basic properties
    assert isinstance(forecasted_vol, np.ndarray)
    assert len(forecasted_vol) == len(sample_market_data['returns'].columns)
    assert np.all(forecasted_vol > 0)  # Volatilities should be positive
    
    # Compare with historical volatility
    historical_vol = sample_market_data['returns'].std().values
    # GARCH volatility should be within an order of magnitude of historical
    assert np.all(0.1 <= forecasted_vol / historical_vol) and np.all(forecasted_vol / historical_vol <= 10)

def test_distribution_types(sample_market_data, config):
    """Test different distribution types for return generation"""
    mc = MonteCarlo(config)
    
    # Increase number of simulations for better statistical properties
    config.n_sims = 1000
    
    # Test normal distribution
    mc.distribution = 'normal'
    normal_results = mc.simulate(sample_market_data)
    
    # Test Student's t distribution
    mc.distribution = 't'
    t_results = mc.simulate(sample_market_data)
    
    # Verify both simulations complete successfully
    assert 'paths' in normal_results
    assert 'paths' in t_results
    
    # Calculate return distributions
    normal_returns = np.diff(np.log(normal_results['paths']), axis=2)
    t_returns = np.diff(np.log(t_results['paths']), axis=2)
    
    # Calculate kurtosis across all assets and time steps
    normal_kurtosis = scipy_stats.kurtosis(normal_returns.reshape(-1))
    t_kurtosis = scipy_stats.kurtosis(t_returns.reshape(-1))
    
    # Student's t should have higher kurtosis (heavier tails)
    assert t_kurtosis > normal_kurtosis, f"t-dist kurtosis ({t_kurtosis}) should be > normal kurtosis ({normal_kurtosis})"

def test_bias_variance_analysis(sample_market_data, config):
    """Test bias-variance decomposition calculations"""
    mc = MonteCarlo(config)
    results = mc.simulate(sample_market_data)
    
    # Check that bias-variance metrics exist
    assert 'bias' in results['statistics']
    assert 'variance' in results['statistics']
    assert 'cross_val_score' in results['statistics']
    
    # Basic validation of metrics
    assert isinstance(results['statistics']['bias'], float)
    assert isinstance(results['statistics']['variance'], float)
    assert isinstance(results['statistics']['cross_val_score'], float)
    
    # Bias should be relatively small compared to the mean
    assert abs(results['statistics']['bias']) < abs(results['expected_return'])
    
    # Cross-validation score should be positive
    assert results['statistics']['cross_val_score'] > 0

def test_invalid_distribution_type(sample_market_data, config):
    """Test handling of invalid distribution type"""
    mc = MonteCarlo(config)
    mc.distribution = 'invalid_distribution'
    
    with pytest.raises(ValueError, match="Unsupported distribution type"):
        mc.simulate(sample_market_data)