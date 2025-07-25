import pytest
import numpy as np
import pandas as pd
from src.regime import RegimeConfig, MarketRegimeDetector, get_regime_series_for_signals

@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
    returns = pd.Series(np.random.normal(0, 0.01, size=len(dates)), index=dates)
    return returns

@pytest.fixture
def synthetic_returns_dict():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
    tickers = ["AAPL", "NVDA"]
    return {t: pd.Series(np.random.normal(0, 0.01, size=len(dates)), index=dates) for t in tickers}

def test_regime_config_defaults():
    config = RegimeConfig()
    assert config.n_regimes == 3
    assert config.window_size == 21
    assert isinstance(config.features, list)
    assert config.min_size == 21
    assert config.smoothing_window == 5
    assert not config.use_deep_learning

def test_regime_config_invalid_values():
    with pytest.raises(ValueError, match="Number of regimes must be at least 2"):
        RegimeConfig(n_regimes=1)
    with pytest.raises(ValueError, match="Window size must be at least 10"):
        RegimeConfig(window_size=5)
    with pytest.raises(ValueError, match="Alert threshold must be between 0 and 1"):
        RegimeConfig(alert_threshold=1.5)
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        RegimeConfig(min_confidence=0)

def test_market_regime_detector_fit_predict(synthetic_returns):
    config = RegimeConfig(n_regimes=3, window_size=21, smoothing_window=0)
    detector = MarketRegimeDetector(config)
    detector.fit(synthetic_returns)
    regimes = detector.predict(synthetic_returns)
    assert isinstance(regimes, pd.Series)
    assert regimes.shape == synthetic_returns.shape
    assert set(regimes.unique()).issubset(set(detector.regime_labels))

def test_market_regime_detector_predict_proba(synthetic_returns):
    config = RegimeConfig(n_regimes=3, window_size=21, smoothing_window=0)
    detector = MarketRegimeDetector(config)
    detector.fit(synthetic_returns)
    proba = detector.predict_proba(synthetic_returns)
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape[0] == synthetic_returns.shape[0]
    assert proba.shape[1] == config.n_regimes
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)

def test_market_regime_detector_predict_before_fit(synthetic_returns):
    config = RegimeConfig(n_regimes=3, window_size=21)
    detector = MarketRegimeDetector(config)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict(synthetic_returns)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict_proba(synthetic_returns)

def test_market_regime_detector_invalid_feature(synthetic_returns):
    config = RegimeConfig(features=["returns", "notafeature"])
    detector = MarketRegimeDetector(config)
    with pytest.raises(ValueError, match="Unknown feature: notafeature"):
        detector.fit(synthetic_returns)

def test_market_regime_detector_update_real_time_validation(synthetic_returns):
    config = RegimeConfig(n_regimes=3, window_size=21)
    detector = MarketRegimeDetector(config)
    # Not fitted
    with pytest.raises(RuntimeError, match="Real-time update failed: Model must be fitted before real-time updates"):
        detector.update_real_time(synthetic_returns)
    # Fit, then pass too-short data
    detector.fit(synthetic_returns)
    short_returns = synthetic_returns.iloc[:5]
    with pytest.raises(RuntimeError, match="Real-time update failed: Input data must have at least 21 observations"):
        detector.update_real_time(short_returns)
    # Pass non-Series
    with pytest.raises(RuntimeError, match="Real-time update failed: new_returns must be a pandas Series"):
        detector.update_real_time([1,2,3])
    # Pass data with NaN
    returns_with_nan = synthetic_returns.copy()
    returns_with_nan.iloc[0] = np.nan
    with pytest.raises(RuntimeError, match="Input data contains missing values"):
        detector.update_real_time(returns_with_nan)
    # Pass data with duplicate index
    returns_dup = synthetic_returns.copy()
    idx = returns_dup.index
    new_idx = idx.tolist()
    new_idx[1] = new_idx[0]  # Make the first two indices the same
    returns_dup.index = new_idx
    with pytest.raises(RuntimeError, match="Input data contains duplicate timestamps"):
        detector.update_real_time(returns_dup)

def test_market_regime_detector_get_confidence_metrics_errors(synthetic_returns):
    config = RegimeConfig(n_regimes=3, window_size=21)
    detector = MarketRegimeDetector(config)
    # Not fitted
    with pytest.raises(ValueError, match="Model must be fitted and have current regime"):
        detector.get_confidence_metrics()
    # Fit but no update_real_time called (so _current_regime is None)
    detector.fit(synthetic_returns)
    with pytest.raises(ValueError, match="Model must be fitted and have current regime"):
        detector.get_confidence_metrics()

def test_market_regime_detector_get_transition_matrix_error():
    config = RegimeConfig(n_regimes=3, window_size=21)
    detector = MarketRegimeDetector(config)
    with pytest.raises(ValueError, match="Model must be fitted before getting transition matrix"):
        detector.get_transition_matrix()

def test_get_regime_series_for_signals(synthetic_returns_dict):
    regime_dict = get_regime_series_for_signals(synthetic_returns_dict)
    assert isinstance(regime_dict, dict)
    assert set(regime_dict.keys()) == set(synthetic_returns_dict.keys())
    for k, v in regime_dict.items():
        assert isinstance(v, pd.Series)
        assert v.shape == synthetic_returns_dict[k].shape 