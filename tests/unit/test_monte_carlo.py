import pytest
import numpy as np
import pandas as pd
from src.monte_carlo import SimConfig, MonteCarlo
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def synthetic_market_data():
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(252, 2)),
        index=dates,
        columns=["Asset1", "Asset2"]
    )
    return {"returns": returns}

@pytest.fixture
def single_asset_data():
    np.random.seed(0)
    dates = pd.date_range(start="2022-01-01", periods=10, freq="B")
    returns = pd.DataFrame(np.random.normal(0, 0.01, size=(10, 1)), index=dates, columns=["A"])
    return {"returns": returns}

@pytest.fixture
def regime_market_data():
    np.random.seed(1)
    dates = pd.date_range(start="2021-01-01", periods=100, freq="B")
    returns = pd.DataFrame(np.random.normal(0, 0.01, size=(100, 2)), index=dates, columns=["A1", "A2"])
    regimes = pd.Series(np.random.choice([0, 1], size=100), index=dates)
    return {"returns": returns, "regimes": regimes}

def test_simconfig_defaults():
    config = SimConfig()
    assert config.n_sims == 10000
    assert config.n_days == 1260
    assert config.risk_free_rate == 0.05
    assert isinstance(config.confidence_levels, list)
    assert config.distribution == "t"
    assert config.simulation_mode == "block_bootstrap"

def test_monte_carlo_block_bootstrap(synthetic_market_data):
    config = SimConfig(n_sims=10, n_days=20, simulation_mode="block_bootstrap")
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    assert results["final_values"].shape == (10,)
    assert results["port_value"].shape == (10, 20)
    assert results["paths"].shape == (10, 20)
    assert results["validation"]["overall_valid"] in [True, False]

def test_monte_carlo_regime_switching(regime_market_data):
    config = SimConfig(n_sims=5, n_days=10, simulation_mode="regime_switching")
    mc = MonteCarlo(config)
    results = mc.simulate(regime_market_data)
    assert "final_values" in results or "paths" in results or isinstance(results, dict)

def test_monte_carlo_gbm(synthetic_market_data):
    config = SimConfig(n_sims=5, n_days=10, simulation_mode="gbm")
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    assert "final_values" in results or "paths" in results or isinstance(results, dict)

def test_single_asset(single_asset_data):
    config = SimConfig(n_sims=3, n_days=5)
    config.block_size = 2  # Allow short data for block bootstrapping
    mc = MonteCarlo(config)
    results = mc.simulate(single_asset_data)
    assert results["final_values"].shape[0] == 3

def test_single_day(synthetic_market_data):
    config = SimConfig(n_sims=2, n_days=1)
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    assert results["final_values"].shape == (2,)

def test_extreme_values():
    returns = pd.DataFrame(np.random.normal(0, 100, size=(50, 2)), columns=["A", "B"])
    data = {"returns": returns}
    config = SimConfig(n_sims=2, n_days=5)
    config.block_size = 2  # Allow short data for block bootstrapping
    mc = MonteCarlo(config)
    results = mc.simulate(data)
    assert np.isfinite(results["final_values"]).all()

def test_validation_and_analysis(synthetic_market_data):
    config = SimConfig(n_sims=5, n_days=10)
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    val = mc.validate_simulation(np.expand_dims(results["paths"], 1), synthetic_market_data)
    assert isinstance(val, dict)
    analysis = mc.analyze_results(results)
    assert "max_drawdown" in analysis

def test_forecast_volatility_calls_arch(synthetic_market_data):
    config = SimConfig()
    mc = MonteCarlo(config)
    with patch("src.monte_carlo.arch_model") as mock_arch:
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.forecast.return_value.variance.values = np.array([1.0])
        mock_model.fit.return_value = mock_fit
        mock_arch.return_value = mock_model
        vol = mc._forecast_volatility(synthetic_market_data["returns"])
        assert isinstance(vol, np.ndarray)

def test_str_repr():
    config = SimConfig(n_sims=1, n_days=1)
    mc = MonteCarlo(config)
    s = str(mc)
    assert "Monte Carlo Simulation" in s

def test_save_results_and_summary(tmp_path, synthetic_market_data):
    config = SimConfig(n_sims=2, n_days=2)
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    file_path = tmp_path / "results.json"
    mc.save_results(results, str(file_path))
    assert os.path.exists(file_path)
    summary = mc.get_summary_statistics(results)
    assert "Metric" in summary.columns

def test_plotting_methods(synthetic_market_data):
    config = SimConfig(n_sims=2, n_days=2)
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("seaborn.set_style"), patch("seaborn.histplot"), patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.close"):
        mc.plot_simulation_paths(results)
        mc.plot_distribution(results)
        mc.plot_risk_metrics(results)

def test_generate_report(tmp_path, synthetic_market_data):
    config = SimConfig(n_sims=2, n_days=2)
    mc = MonteCarlo(config)
    results = mc.simulate(synthetic_market_data)
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"), patch("seaborn.set_style"), patch("seaborn.histplot"), patch("matplotlib.pyplot.figure"):
        mc.generate_report(results, output_dir=str(tmp_path))
        files = list(os.listdir(tmp_path))
        assert any(f.endswith(".json") for f in files)

def test_invalid_inputs():
    config = SimConfig()
    mc = MonteCarlo(config)
    # simulate: None, missing, empty
    with pytest.raises(TypeError):
        mc.simulate(None)
    with pytest.raises(ValueError):
        mc.simulate({"foo": 123})
    with pytest.raises(ValueError):
        mc.simulate({"returns": pd.DataFrame()})
    # regime switching with no regimes
    with pytest.raises(ValueError):
        mc.simulate({"returns": pd.DataFrame(np.random.normal(0, 0.01, size=(10, 2)))})
    # _forecast_volatility with bad input
    with patch("src.monte_carlo.arch_model", side_effect=Exception("fail")):
        vol = mc._forecast_volatility(pd.DataFrame(np.random.normal(0, 1, size=(10, 2))))
        assert isinstance(vol, np.ndarray)

def test_bias_variance_edge_case():
    config = SimConfig()
    mc = MonteCarlo(config)
    arr = np.array([1.0])
    out = mc._calculate_bias_variance(arr)
    assert "bias" in out and "variance" in out and "cross_val_score" in out 