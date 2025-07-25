import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.deep_learning import DeepLearningConfig, LSTMRegimeDetector, TransformerRegimeDetector, BayesianLSTMRegimeForecaster

@pytest.fixture
def synthetic_returns():
    np.random.seed(123)
    dates = pd.date_range(start="2022-01-01", periods=50, freq="B")
    returns = pd.Series(np.random.normal(0, 0.01, size=len(dates)), index=dates)
    return returns

@pytest.fixture
def synthetic_regimes():
    # For 50 points, sequence_length=21, so 50-21=29 labels
    return np.random.randint(0, 4, size=29)

def test_deep_learning_config_defaults():
    config = DeepLearningConfig()
    assert config.sequence_length == 21
    assert isinstance(config.hidden_dims, list)
    assert config.n_regimes == 4
    assert config.batch_size == 64
    assert config.use_attention
    assert config.bidirectional
    assert config.residual_connections
    assert config.batch_normalization
    assert config.l2_regularization == 0.001
    assert config.gradient_clipping == 1.0
    assert config.learning_rate_schedule["type"] == "cosine_annealing"

def test_lstm_regime_detector_fit_predict(synthetic_returns, synthetic_regimes):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = LSTMRegimeDetector(config)
    with patch.object(detector.model, 'fit', return_value=None) as mock_fit, \
         patch.object(detector.model, 'predict', return_value=np.eye(4)[synthetic_regimes]) as mock_predict:
        detector.fit(synthetic_returns, np.concatenate([synthetic_regimes, [0]*21]))
        assert detector._is_fitted
        result = detector.predict(synthetic_returns)
        assert isinstance(result, pd.Series)
        proba = detector.predict_proba(synthetic_returns)
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape[1] == config.n_regimes
        with patch.object(detector.model, 'predict', return_value=np.array([[0.1,0.2,0.3,0.4]])):
            latest = detector.predict_latest(synthetic_returns)
            assert isinstance(latest, np.ndarray)
            assert latest.shape == (4,)

def test_lstm_regime_detector_predict_before_fit(synthetic_returns):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = LSTMRegimeDetector(config)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict(synthetic_returns)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict_proba(synthetic_returns)

def test_lstm_regime_detector_predict_latest_too_short():
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = LSTMRegimeDetector(config)
    detector._is_fitted = True
    short_returns = pd.Series(np.random.normal(0, 0.01, size=10))
    with pytest.raises(ValueError, match="Need at least 21 points for real-time LSTM prediction."):
        detector.predict_latest(short_returns)

def test_lstm_regime_detector_save_load(tmp_path, synthetic_returns, synthetic_regimes):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = LSTMRegimeDetector(config)
    with patch.object(detector.model, 'save', return_value=None) as mock_save, \
         patch('joblib.dump', return_value=None) as mock_dump:
        detector.save(str(tmp_path))
        assert mock_save.called
        assert mock_dump.call_count == 2
    with patch('tensorflow.keras.models.load_model', return_value=MagicMock()) as mock_load_model, \
         patch('joblib.load', side_effect=[MagicMock(), config]) as mock_load:
        detector.load(str(tmp_path))
        assert detector._is_fitted

# --- TransformerRegimeDetector tests ---
def test_transformer_regime_detector_fit_predict(synthetic_returns, synthetic_regimes):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = TransformerRegimeDetector(config)
    with patch.object(detector.model, 'fit', return_value=None) as mock_fit, \
         patch.object(detector.model, 'predict', return_value=np.eye(4)[synthetic_regimes]) as mock_predict:
        detector.fit(synthetic_returns, np.concatenate([synthetic_regimes, [0]*21]))
        detector._is_fitted = True
        result = detector.predict(synthetic_returns)
        assert isinstance(result, pd.Series)
        proba = detector.predict_proba(synthetic_returns)
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape[1] == config.n_regimes
        with patch.object(detector.model, 'predict', return_value=np.array([[0.1,0.2,0.3,0.4]])):
            latest = detector.predict_latest(synthetic_returns)
            assert isinstance(latest, np.ndarray)
            assert latest.shape == (4,)

def test_transformer_regime_detector_predict_before_fit(synthetic_returns):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = TransformerRegimeDetector(config)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict(synthetic_returns)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        detector.predict_proba(synthetic_returns)

def test_transformer_regime_detector_predict_latest_too_short():
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    detector = TransformerRegimeDetector(config)
    detector._is_fitted = True
    short_returns = pd.Series(np.random.normal(0, 0.01, size=10))
    with pytest.raises(ValueError, match="Need at least 21 points for real-time Transformer prediction."):
        detector.predict_latest(short_returns)

# --- BayesianLSTMRegimeForecaster tests ---
def test_bayesian_lstm_forecaster_fit_predict(synthetic_returns):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    forecaster = BayesianLSTMRegimeForecaster(config)
    y = np.random.normal(0, 1, size=29)  # 50-21=29
    with patch.object(forecaster.model, 'fit', return_value=None) as mock_fit, \
         patch.object(forecaster.model, '__call__', return_value=np.random.normal(0, 1, size=(100, 29, 2))) as mock_call:
        forecaster.fit(synthetic_returns, y)
        forecaster._is_fitted = True
        preds = forecaster.predict(synthetic_returns, n_samples=100)
        assert 'mean' in preds and 'std' in preds and 'all_samples' in preds
        assert preds['mean'].shape == (29,)
        assert preds['std'].shape == (29,)
        # predict_latest
        with patch.object(forecaster.model, '__call__', return_value=np.random.normal(0, 1, size=(1,2))):
            latest = forecaster.predict_latest(synthetic_returns, n_samples=10)
            assert 'mean' in latest and 'std' in latest and 'all_samples' in latest

def test_bayesian_lstm_forecaster_predict_before_fit(synthetic_returns):
    config = DeepLearningConfig(sequence_length=21, n_regimes=4)
    forecaster = BayesianLSTMRegimeForecaster(config)
    with pytest.raises(Exception):
        # Should error because model is not fitted and __call__ is not patched
        forecaster.predict(synthetic_returns, n_samples=10) 