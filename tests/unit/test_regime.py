# tests/unit/test_regime.py
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.regime import MarketRegimeDetector, RegimeConfig


@pytest.fixture
def sample_regime_config():
    """Create a sample regime configuration for testing"""
    return RegimeConfig(
        n_regimes=3,
        n_iter=100,
        window_size=21,
        features=["returns", "volatility", "momentum"],
        min_regime_size=21,
        smoothing_window=5,
    )


@pytest.fixture
def mock_market_returns():
    """Create mock market returns for testing"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-06-01", freq="B")

    # Generate returns with different volatility regimes
    n_samples = len(dates)
    returns = []

    # Low volatility regime
    returns.extend(np.random.normal(0.0001, 0.001, n_samples // 3))
    # Medium volatility regime
    returns.extend(np.random.normal(0.0002, 0.002, n_samples // 3))
    # High volatility regime
    returns.extend(np.random.normal(0.0003, 0.003, n_samples - 2 * (n_samples // 3)))

    return pd.Series(returns, index=dates)


class TestRegimeConfig:
    """Test suite for RegimeConfig"""

    def test_regime_config_initialization(self):
        """Test RegimeConfig initialization with default values"""
        config = RegimeConfig()
        assert config.n_regimes == 3
        assert config.n_iter == 1000
        assert config.window_size == 21
        assert config.features == ["returns", "volatility", "momentum"]

    def test_regime_config_validation(self):
        """Test RegimeConfig validation"""
        # Test invalid n_regimes
        with pytest.raises(ValueError):
            RegimeConfig(n_regimes=1)

        # Test invalid window_size
        with pytest.raises(ValueError):
            RegimeConfig(window_size=5)

        # Test valid configuration
        config = RegimeConfig(n_regimes=3, window_size=21)
        assert config is not None


class TestMarketRegimeDetector:
    """Test suite for MarketRegimeDetector"""

    def test_detector_initialization(self, sample_regime_config):
        """Test MarketRegimeDetector initialization"""
        detector = MarketRegimeDetector(sample_regime_config)
        assert detector.config == sample_regime_config
        assert detector.regime_labels == ["Low Vol", "Medium Vol", "High Vol"]

    def test_feature_calculation(self, sample_regime_config, mock_market_returns):
        """Test feature calculation"""
        detector = MarketRegimeDetector(sample_regime_config)
        features = detector._calculate_features(mock_market_returns)

        # Check all expected features are present
        expected_features = [
            "returns",
            "volatility",
            "realized_vol",
            "momentum",
            "momentum_vol",
            "skewness",
            "kurtosis",
        ]
        assert all(f in features for f in expected_features)

        # Check feature shapes
        assert all(len(features[f]) == len(mock_market_returns) for f in features)

    def test_feature_preparation(self, sample_regime_config, mock_market_returns):
        """Test feature preparation and preprocessing"""
        detector = MarketRegimeDetector(sample_regime_config)
        X = detector._prepare_features(mock_market_returns)

        # Check shape
        assert X.shape[0] == len(mock_market_returns)
        assert X.shape[1] == len(sample_regime_config.features)

        # Check for NaN values
        assert not np.isnan(X).any()

        # Check scaling
        assert np.abs(X.mean(axis=0)).max() < 1e-10  # Close to 0
        assert np.abs(X.std(axis=0) - 1.0).max() < 1e-10  # Close to 1

    def test_hmm_initialization(self, sample_regime_config, mock_market_returns):
        """Test HMM parameter initialization"""
        detector = MarketRegimeDetector(sample_regime_config)
        X = detector._prepare_features(mock_market_returns)

        # Test initialization
        detector._initialize_hmm_params(X)

        # Verify shapes and properties
        assert detector.model.means_.shape == (
            sample_regime_config.n_regimes,
            X.shape[1],
        )
        assert detector.model.covars_.shape == (
            sample_regime_config.n_regimes,
            X.shape[1],
            X.shape[1],
        )
        assert detector.model.transmat_.shape == (
            sample_regime_config.n_regimes,
            sample_regime_config.n_regimes,
        )
        assert np.allclose(detector.model.transmat_.sum(axis=1), 1.0)

    def test_regime_prediction(self, sample_regime_config, mock_market_returns):
        """Test regime prediction"""
        detector = MarketRegimeDetector(sample_regime_config)
        regimes = detector.fit_predict(mock_market_returns)

        # Check basic properties
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(mock_market_returns)
        assert all(regime in detector.regime_labels for regime in regimes.unique())

        # Check regime transitions using a different method
        regime_changes = (regimes != regimes.shift()).sum()
        assert regime_changes >= 0
        assert regime_changes < len(regimes)

    def test_regime_smoothing(self, sample_regime_config):
        """Test regime smoothing"""
        detector = MarketRegimeDetector(sample_regime_config)

        # Create a series with artificial regime changes
        regimes = pd.Series(
            ["Low Vol"] * 10 + ["High Vol"] * 1 + ["Low Vol"] * 10,
            index=pd.date_range("2024-01-01", periods=21, freq="B"),
        )

        smoothed = detector._smooth_regimes(regimes)

        # The single 'High Vol' regime should be smoothed out
        assert "High Vol" not in smoothed.unique()

    def test_regime_statistics(self, sample_regime_config, mock_market_returns):
        """Test regime statistics calculation"""
        detector = MarketRegimeDetector(sample_regime_config)
        regimes = detector.fit_predict(mock_market_returns)
        stats = detector.get_regime_stats(mock_market_returns, regimes)

        # Check statistics for each regime
        for regime in detector.regime_labels:
            assert regime in stats
            regime_stats = stats[regime]
            assert all(
                key in regime_stats
                for key in ["mean_return", "volatility", "frequency", "sharpe"]
            )

    def test_model_validation(self, sample_regime_config, mock_market_returns):
        """Test model validation metrics"""
        detector = MarketRegimeDetector(sample_regime_config)

        # First ensure the model fits successfully
        regimes = detector.fit_predict(mock_market_returns)
        assert detector._is_fitted, "Model should be fitted after fit_predict"

        # Now test validation
        validation = detector.validate_model(mock_market_returns, regimes)

        # Check validation metrics
        assert isinstance(validation, dict)
        assert all(
            key in validation
            for key in [
                "information_ratio",
                "regime_persistence",
                "log_likelihood",
                "aic",
                "bic",
                "avg_regime_duration",
            ]
        )

    def test_transition_matrix(self, sample_regime_config, mock_market_returns):
        """Test transition probability matrix calculation"""
        detector = MarketRegimeDetector(sample_regime_config)

        # First ensure the model fits successfully
        detector.fit_predict(mock_market_returns)
        assert detector._is_fitted, "Model should be fitted after fit_predict"

        # Now test transition matrix
        trans_mat = detector.get_transition_matrix()

        # Check matrix properties
        assert isinstance(trans_mat, pd.DataFrame)
        assert trans_mat.shape == (
            sample_regime_config.n_regimes,
            sample_regime_config.n_regimes,
        )
        assert np.allclose(trans_mat.sum(axis=1), 1.0)  # Rows sum to 1
        assert (trans_mat >= 0).all().all()  # All probabilities non-negative
