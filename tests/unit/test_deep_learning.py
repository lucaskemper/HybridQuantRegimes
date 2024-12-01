import numpy as np
import pandas as pd
import pytest

from src.deep_learning import DeepLearningConfig, LSTMRegimeDetector
from src.regime import MarketRegimeDetector, RegimeConfig


@pytest.fixture
def sample_returns():
    """Generate sample return data"""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.normal(0, 0.02, 500), index=dates)
    return returns


@pytest.fixture
def deep_learning_config():
    """Create test configuration with smaller network"""
    return DeepLearningConfig(
        sequence_length=21,
        hidden_dims=[32, 16],  # Smaller for testing
        epochs=2,  # Fewer epochs for testing
        batch_size=32,
    )


@pytest.fixture
def regime_config(deep_learning_config):
    """Create regime config with deep learning enabled"""
    return RegimeConfig(
        use_deep_learning=True,
        deep_learning_config=deep_learning_config,
        n_regimes=3,
        window_size=21,
    )


class TestLSTMRegimeDetector:
    def test_initialization(self, deep_learning_config):
        """Test model initialization"""
        detector = LSTMRegimeDetector(deep_learning_config)
        assert detector.model is not None
        assert not detector._is_fitted

    def test_feature_preparation(self, deep_learning_config, sample_returns):
        """Test feature engineering"""
        detector = LSTMRegimeDetector(deep_learning_config)
        features = detector._prepare_features(sample_returns)

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 7  # Number of features
        assert not np.isnan(features).any()  # No NaN values

    def test_sequence_creation(self, deep_learning_config, sample_returns):
        """Test sequence preparation"""
        detector = LSTMRegimeDetector(deep_learning_config)
        features = detector._prepare_features(sample_returns)
        X, y = detector._create_sequences(features)

        assert len(X.shape) == 3  # (samples, sequence_length, features)
        assert X.shape[1] == deep_learning_config.sequence_length
        assert y.shape[1] == deep_learning_config.n_regimes  # One-hot encoded

    def test_fit_predict(self, deep_learning_config, sample_returns):
        """Test full training and prediction pipeline"""
        detector = LSTMRegimeDetector(deep_learning_config)

        # Test fitting
        history = detector.fit(sample_returns)
        assert detector._is_fitted
        assert "loss" in history.history
        assert "val_loss" in history.history

        # Test prediction
        predictions = detector.predict(sample_returns)
        assert isinstance(predictions, pd.Series)
        assert (
            len(predictions)
            == len(sample_returns) - deep_learning_config.sequence_length
        )
        assert predictions.isin([0, 1, 2]).all()  # Check valid regime values


class TestMarketRegimeDetector:
    def test_combined_predictions(self, regime_config, sample_returns):
        """Test HMM and LSTM combination"""
        detector = MarketRegimeDetector(regime_config)

        # Get combined predictions
        regimes = detector.fit_predict(sample_returns)

        assert isinstance(regimes, pd.Series)
        assert regimes.isin([0, 1, 2]).all()

        # Test regime statistics
        stats = detector.get_regime_stats(sample_returns, regimes)
        assert all(regime in stats for regime in detector.regime_labels)
        assert all("volatility" in stats[regime] for regime in detector.regime_labels)

    def test_fallback_behavior(self, regime_config, sample_returns):
        """Test fallback to HMM when LSTM fails"""
        detector = MarketRegimeDetector(regime_config)

        # Simulate LSTM failure by providing invalid data
        invalid_returns = pd.Series([np.nan] * 100)
        regimes = detector.fit_predict(invalid_returns)

        # Should still get valid predictions from HMM
        assert isinstance(regimes, pd.Series)
        assert regimes.isin([0, 1, 2]).all()

    def test_model_validation(self, regime_config, sample_returns):
        """Test validation metrics"""
        detector = MarketRegimeDetector(regime_config)
        regimes = detector.fit_predict(sample_returns)

        validation = detector.validate_model(sample_returns, regimes)

        assert "information_ratio" in validation
        assert "regime_persistence" in validation
        assert "log_likelihood" in validation
        assert "aic" in validation
        assert "bic" in validation
        assert "avg_regime_duration" in validation
