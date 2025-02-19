from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from deep_learning import DeepLearningConfig, LSTMRegimeDetector


@dataclass
class RegimeConfig:
    n_regimes: int = 3
    n_iter: int = 1000
    random_state: int = 42
    window_size: int = 21
    features: List[str] = field(
        default_factory=lambda: ["returns", "volatility", "momentum"]
    )
    min_regime_size: int = 21
    smoothing_window: int = 5
    use_deep_learning: bool = False
    deep_learning_config: Optional[DeepLearningConfig] = None

    def __post_init__(self):
        if self.features is None:
            self.features = ["returns", "volatility", "momentum"]
        if self.n_regimes < 2:
            raise ValueError("Number of regimes must be at least 2")
        if self.window_size < 10:
            raise ValueError("Window size must be at least 10")
        if self.use_deep_learning and self.deep_learning_config is None:
            self.deep_learning_config = DeepLearningConfig(
                n_regimes=self.n_regimes, sequence_length=self.window_size
            )


class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig):
        """Initialize regime detector with configuration"""
        self.config = config
        self.regime_labels = ["Low Vol", "Medium Vol", "High Vol"][: config.n_regimes]

        # Initialize HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=config.n_regimes,
            covariance_type="full",
            n_iter=config.n_iter,
            random_state=config.random_state,
            init_params="",
        )

        # Initialize LSTM if enabled
        self.lstm_model = (
            LSTMRegimeDetector(config.deep_learning_config)
            if config.use_deep_learning
            else None
        )

        self._is_fitted = False
        self.scaler = StandardScaler()
        self._last_fit_size = None

    def _calculate_features(self, returns: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate all possible features for regime detection"""
        features = {}

        # Basic returns feature
        features["returns"] = returns.values

        # Volatility features
        features["volatility"] = returns.ewm(span=self.config.window_size).std().values
        features["realized_vol"] = returns.rolling(self.config.window_size).std().values

        # Momentum features
        features["momentum"] = returns.rolling(self.config.window_size).mean().values
        features["momentum_vol"] = (
            (returns - returns.rolling(self.config.window_size).mean())
            / returns.rolling(self.config.window_size).std()
        ).values

        # Skewness and kurtosis
        features["skewness"] = returns.rolling(self.config.window_size).skew().values
        features["kurtosis"] = returns.rolling(self.config.window_size).kurt().values

        return features

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for HMM with enhanced preprocessing"""
        features = []
        all_features = self._calculate_features(returns)

        for feature_name in self.config.features:
            if feature_name not in all_features:
                raise ValueError(f"Unknown feature: {feature_name}")

            feature_data = all_features[feature_name]

            # Winsorize extreme values
            feature_data = np.clip(
                feature_data,
                np.nanpercentile(feature_data, 1),
                np.nanpercentile(feature_data, 99),
            )

            features.append(feature_data.reshape(-1, 1))

        # Combine features
        X = np.hstack(features)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Scale features
        X = self.scaler.fit_transform(X)

        return X

    def _initialize_hmm_params(self, X: np.ndarray):
        """Initialize HMM parameters with improved initialization"""
        n_samples, n_features = X.shape
        self.hmm_model.n_features = n_features  # Set n_features explicitly

        # Initialize means using KMeans
        kmeans = KMeans(
            n_clusters=self.config.n_regimes,
            random_state=self.config.random_state,
            n_init=10,
        )
        clusters = kmeans.fit_predict(X)

        # Initialize means
        self.hmm_model.means_ = kmeans.cluster_centers_

        # Initialize covariance matrices with proper positive-definite matrices
        covars = []
        for i in range(self.config.n_regimes):
            cluster_data = X[clusters == i]
            if len(cluster_data) > 1:
                cov = np.cov(cluster_data.T) + np.eye(n_features) * 1e-3
            else:
                cov = np.eye(n_features)
            covars.append(cov)
        self.hmm_model.covars_ = np.array(covars)

        # Initialize transition matrix with slight preference for staying in same state
        transmat = np.ones((self.config.n_regimes, self.config.n_regimes))
        np.fill_diagonal(transmat, 2)  # Preference for staying in same state
        self.hmm_model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)

        # Initialize starting probabilities
        self.hmm_model.startprob_ = (
            np.ones(self.config.n_regimes) / self.config.n_regimes
        )

    def _smooth_regimes(self, regimes: pd.Series) -> pd.Series:
        """Apply smoothing to reduce regime switching"""
        smoothed = regimes.copy()
        window = self.config.smoothing_window

        for i in range(window, len(regimes) - window):
            window_regimes = regimes.iloc[i - window : i + window + 1]
            mode_regime = window_regimes.mode().iloc[0]
            smoothed.iloc[i] = mode_regime

        return smoothed

    def fit(self, returns: pd.Series) -> None:
        """Fit the HMM model to the data"""
        X = self._prepare_features(returns)
        self._initialize_hmm_params(X)
        
        try:
            self.hmm_model.fit(X)
            self._is_fitted = True
            self._last_fit_size = len(returns)
        except Exception as e:
            self._is_fitted = False
            raise RuntimeError(f"HMM fitting failed: {str(e)}") from e

    def predict(self, returns: pd.Series) -> pd.Series:
        """Predict regimes for the given returns"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self._prepare_features(returns)
        raw_regimes = self.hmm_model.predict(X)
        
        # Create Series with proper index
        regimes = pd.Series(raw_regimes, index=returns.index)
        
        # Apply smoothing if configured
        if self.config.smoothing_window > 0:
            regimes = self._smooth_regimes(regimes)
            
        return regimes

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Enhanced fit and predict with both HMM and LSTM"""
        # First fit and get HMM predictions
        self.fit(returns)
        hmm_regimes = self.predict(returns)

        if self.config.use_deep_learning and self.lstm_model is not None:
            try:
                # Train LSTM with HMM predictions as initial regimes
                self.lstm_model.fit(returns, hmm_regimes)

                # Get LSTM predictions
                lstm_regimes = self.lstm_model.predict(returns)

                # Combine predictions
                combined_regimes = self._combine_predictions(hmm_regimes, lstm_regimes)
                return combined_regimes

            except Exception as e:
                print(f"Deep learning prediction failed: {str(e)}")
                print("Falling back to HMM predictions")
                return hmm_regimes

        return hmm_regimes

    def _combine_predictions(
        self, hmm_regimes: pd.Series, lstm_regimes: pd.Series
    ) -> pd.Series:
        """Combine HMM and LSTM predictions"""
        # Ensure indices align
        common_idx = hmm_regimes.index.intersection(lstm_regimes.index)
        hmm = hmm_regimes.loc[common_idx]
        lstm = lstm_regimes.loc[common_idx]

        # Simple ensemble: take the most conservative regime
        # (higher regime number = more volatile)
        combined = pd.Series(np.maximum(hmm.values, lstm.values), index=common_idx)

        return combined

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return transition probability matrix"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting transition matrix")

        trans_mat = pd.DataFrame(
            self.hmm_model.transmat_,
            index=self.regime_labels,
            columns=self.regime_labels,
        )
        return trans_mat

    def get_regime_stats(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """Calculate regime statistics

        Parameters:
            returns (pd.Series): Portfolio returns series
            regimes (pd.Series): Regime classifications

        Returns:
            Dict: Statistics for each regime
        """
        # Ensure alignment of indices
        common_idx = returns.index.intersection(regimes.index)
        aligned_returns = returns.loc[common_idx]
        aligned_regimes = regimes.loc[common_idx]

        stats = {}
        for regime in self.regime_labels:
            # Create boolean mask and ensure index alignment
            regime_mask = aligned_regimes == regime
            regime_returns = aligned_returns[regime_mask]

            if len(regime_returns) > 0:
                stats[regime] = {
                    "mean_return": regime_returns.mean() * 252,
                    "volatility": regime_returns.std() * np.sqrt(252),
                    "frequency": len(regime_returns) / len(aligned_returns),
                    "sharpe": (regime_returns.mean() * 252)
                    / (regime_returns.std() * np.sqrt(252)),
                }
            else:
                stats[regime] = {
                    "mean_return": np.nan,
                    "volatility": np.nan,
                    "frequency": 0.0,
                    "sharpe": np.nan,
                }

        return stats

    def validate_model(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """Validate HMM model performance"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before validation")

        # Ensure alignment
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]

        # 1. Information Ratio by Regime
        ir_by_regime = {}
        for regime in self.regime_labels:
            regime_returns = returns[regimes == regime]
            if len(regime_returns) > 0:
                ir = (regime_returns.mean() * 252) / (
                    regime_returns.std() * np.sqrt(252)
                )
                ir_by_regime[regime] = ir
            else:
                ir_by_regime[regime] = np.nan

        # 2. Regime Persistence
        transitions = 0
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != regimes.iloc[i - 1]:
                transitions += 1
        persistence = 1 - (transitions / len(regimes))

        # 3. Log Likelihood
        X = self._prepare_features(returns)
        log_likelihood = self.hmm_model.score(X)

        # 4. AIC and BIC
        n_params = (self.config.n_regimes * self.config.n_regimes - 1) + (
            self.config.n_regimes * 2
        )  # transition probs + means/vars
        n_samples = len(returns)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params

        # 5. Regime Stability
        regime_runs = []
        current_regime = regimes.iloc[0]
        current_run = 1

        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_run += 1
            else:
                regime_runs.append(current_run)
                current_run = 1
                current_regime = regimes.iloc[i]

        regime_runs.append(current_run)
        avg_regime_duration = np.mean(regime_runs)

        return {
            "information_ratio": ir_by_regime,
            "regime_persistence": persistence,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "avg_regime_duration": avg_regime_duration,
        }
