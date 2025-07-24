from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.deep_learning import DeepLearningConfig, LSTMRegimeDetector
from src.features import calculate_enhanced_features
from src.statistical_validation import StatisticalValidator

# Configure logging
logger = logging.getLogger(__name__)

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
    
    # New real-time configuration options
    alert_threshold: float = 0.7  # Probability threshold for transition alerts
    min_confidence: float = 0.6  # Minimum confidence required for regime assignment
    update_frequency: str = '1D'  # Pandas frequency string for updates
    history_size: int = 100  # Number of historical regime entries to maintain
    
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
        if self.alert_threshold <= 0 or self.alert_threshold >= 1:
            raise ValueError("Alert threshold must be between 0 and 1")
        if self.min_confidence <= 0 or self.min_confidence >= 1:
            raise ValueError("Minimum confidence must be between 0 and 1")

@dataclass
class PerformanceMetrics:
    """Track performance metrics for regime detection"""
    update_times: List[float] = field(default_factory=list)
    prediction_times: List[float] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    total_updates: int = 0
    successful_updates: int = 0

class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig):
        """Initialize regime detector with configuration"""
        self.config = config
        if config.n_regimes <= 3:
            self.regime_labels = ["Low Vol", "Medium Vol", "High Vol"][: config.n_regimes]
        else:
            self.regime_labels = [f"Regime {i}" for i in range(config.n_regimes)]

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
        
        # New attributes for real-time tracking
        self._current_regime = None
        self._regime_history = pd.DataFrame(columns=['regime', 'confidence', 'transition_probability'])
        self._last_update_time = None
        self._alert_threshold = config.alert_threshold
        
        # Performance monitoring
        self._performance = PerformanceMetrics()
        
        logger.info(f"Initialized MarketRegimeDetector with {config.n_regimes} regimes")
        
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        if self._performance.update_times:
            avg_update_time = np.mean(self._performance.update_times[-100:])  # Last 100 updates
            success_rate = (self._performance.successful_updates / 
                          max(1, self._performance.total_updates)) * 100
            
            logger.info(
                f"Performance Metrics - "
                f"Avg Update Time: {avg_update_time:.3f}s, "
                f"Success Rate: {success_rate:.1f}%, "
                f"Error Count: {self._performance.error_count}"
            )
            
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics
        
        Returns:
            Dict: Performance statistics
        """
        metrics = {
            'avg_update_time': (np.mean(self._performance.update_times[-100:])
                              if self._performance.update_times else 0),
            'avg_prediction_time': (np.mean(self._performance.prediction_times[-100:])
                                  if self._performance.prediction_times else 0),
            'error_rate': (self._performance.error_count / 
                         max(1, self._performance.total_updates)),
            'success_rate': (self._performance.successful_updates / 
                           max(1, self._performance.total_updates)),
            'total_updates': self._performance.total_updates,
            'last_error': self._performance.last_error
        }
        return metrics

    def _prepare_features(self, returns: pd.Series, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for HMM with enhanced preprocessing (centralized)"""
        # If returns is already a DataFrame with all features, use it directly
        if isinstance(returns, pd.DataFrame):
            features_df = returns
        else:
            features_df = calculate_enhanced_features(returns)
        # Select only the features specified in config
        selected_features = self.config.features
        features = []
        for feature_name in selected_features:
            if feature_name not in features_df.columns:
                raise ValueError(f"Unknown feature: {feature_name}")
            feature_data = features_df[feature_name].values
            # Winsorize extreme values
            feature_data = np.clip(
                feature_data,
                np.nanpercentile(feature_data, 1),
                np.nanpercentile(feature_data, 99),
            )
            features.append(feature_data.reshape(-1, 1))
        X = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        # During prediction, select only the features used during fit
        if not fit_scaler and hasattr(self, '_used_feature_names'):
            used_indices = [selected_features.index(name) for name in self._used_feature_names]
            X = X[:, used_indices]
        # Only transform (never fit) with the scaler here
        if hasattr(self.scaler, 'scale_'):
            X = self.scaler.transform(X)
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
                cov = np.cov(cluster_data.T) + np.eye(n_features) * 1e-2
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
        X = self._prepare_features(returns, fit_scaler=True)
        # Remove constant or near-constant features
        variances = X.var(axis=0)
        nonconstant = variances > 1e-8
        if not np.all(nonconstant):
            X = X[:, nonconstant]
            self._used_feature_names = [name for i, name in enumerate(self.config.features) if nonconstant[i]]
        else:
            self._used_feature_names = self.config.features.copy()
        # Fit the scaler on the reduced X
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self._initialize_hmm_params(X)
        
        try:
            self.hmm_model.fit(X)
            self._is_fitted = True
            self._last_fit_size = len(returns)
        except Exception as e:
            self._is_fitted = False
            raise RuntimeError(f"HMM fitting failed: {str(e)}") from e

    def predict(self, returns: pd.Series, output_labels: bool = True) -> pd.Series:
        """Predict regimes for the given returns. If output_labels is False, return integer regime values."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self._prepare_features(returns, fit_scaler=False)
        raw_regimes = self.hmm_model.predict(X)

        if output_labels:
            # Map integer regimes to string labels
            regime_mapping = {i: label for i, label in enumerate(self.regime_labels)}
            regimes = pd.Series([regime_mapping[r] for r in raw_regimes], index=returns.index)
            # Apply smoothing if configured
            if self.config.smoothing_window > 0:
                regimes = self._smooth_regimes(regimes)
            return regimes
        else:
            regimes = pd.Series(raw_regimes, index=returns.index)
            # Apply smoothing if configured
            if self.config.smoothing_window > 0:
                regimes = self._smooth_regimes(regimes)
            return regimes

    def _sanitize_returns(self, returns: pd.Series) -> pd.Series:
        """Ensure returns are numeric, float, and have no NaNs or infs."""
        returns = pd.to_numeric(returns, errors='coerce')
        returns = returns.astype(float)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        if returns.isnull().any():
            n_missing = returns.isnull().sum()
            print(f"Warning: {n_missing} NaN or inf values in returns. Filling with 0.")
            returns = returns.fillna(0)
        return returns

    def fit_predict(self, features: pd.DataFrame, returns: pd.Series, output_labels: bool = True) -> pd.Series:
        """Enhanced fit and predict with both HMM (features) and LSTM (returns). If output_labels is False, return integer regime values."""
        # First fit and get HMM predictions
        self.fit(features)
        hmm_regimes_labels = self.predict(features, output_labels=output_labels)
        hmm_regimes_int = self.predict(features, output_labels=False)

        if self.config.use_deep_learning and self.lstm_model is not None:
            try:
                # Sanitize returns for LSTM
                returns_clean = self._sanitize_returns(returns)
                # Train LSTM with HMM predictions as initial regimes (must be integer)
                self.lstm_model.fit(returns_clean, hmm_regimes_int)

                # Get regime probabilities, not just predictions
                hmm_probs = self.predict_proba(features)
                lstm_probs = self.lstm_model.predict_proba(returns_clean)

                # Combine predictions
                combined_probs = self._combine_predictions_bayesian(hmm_probs, lstm_probs)
                combined_regimes = combined_probs.idxmax(axis=1)
                return combined_regimes

            except Exception as e:
                print(f"Deep learning prediction failed: {str(e)}")
                print("Falling back to HMM predictions")
                return hmm_regimes_labels

        return hmm_regimes_labels

    def _combine_predictions_bayesian(self, hmm_probs: pd.DataFrame, lstm_probs: pd.DataFrame) -> pd.Series:
        """Bayesian Model Averaging for regime probabilities"""
        window_size = min(50, len(hmm_probs))
        if window_size < 10:
            weights = np.array([0.5, 0.5])
        else:
            recent_hmm = hmm_probs.iloc[-window_size:]
            recent_lstm = lstm_probs.iloc[-window_size:]
            hmm_entropy = -np.sum(recent_hmm * np.log(recent_hmm + 1e-8), axis=1).mean()
            lstm_entropy = -np.sum(recent_lstm * np.log(recent_lstm + 1e-8), axis=1).mean()
            hmm_weight = 1 / (hmm_entropy + 1e-8)
            lstm_weight = 1 / (lstm_entropy + 1e-8)
            total_weight = hmm_weight + lstm_weight
            weights = np.array([hmm_weight/total_weight, lstm_weight/total_weight])
        combined_probs = weights[0] * hmm_probs + weights[1] * lstm_probs
        return combined_probs.idxmax(axis=1)

    def _bayesian_model_averaging(self, hmm_probs: pd.DataFrame, lstm_probs: pd.DataFrame) -> pd.DataFrame:
        """Proper Bayesian Model Averaging with evidence weighting"""
        hmm_log_evidence = np.sum(np.log(hmm_probs.max(axis=1) + 1e-8))
        lstm_log_evidence = np.sum(np.log(lstm_probs.max(axis=1) + 1e-8))
        evidence_array = np.array([hmm_log_evidence, lstm_log_evidence])
        weights = np.exp(evidence_array) / np.sum(np.exp(evidence_array))
        return weights[0] * hmm_probs + weights[1] * lstm_probs

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

    def validate_model_statistically(self, features: pd.DataFrame, regimes: pd.Series) -> Dict:
        """Enhanced statistical validation"""
        validator = StatisticalValidator()
        # Cross-validation
        macro_data = None
        if hasattr(self, 'macro_data') and self.macro_data is not None:
            macro_data = self.macro_data
        elif hasattr(self, 'config') and hasattr(self.config, 'macro_data'):
            macro_data = self.config.macro_data
        cv_results = validator.cross_validate_regime_detection(features, self, macro_data)
        # Add to existing validation
        existing_validation = self.validate_model(features, regimes)
        existing_validation.update({
            'cross_validation': cv_results,
            'statistical_significance': cv_results['stability_significant']
        })
        return existing_validation

    def predict_proba(self, returns: pd.Series) -> pd.DataFrame:
        """Enhanced probability prediction with Bayesian averaging"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        start_time = time.time()
        try:
            X = self._prepare_features(returns)
            hmm_probs = self.hmm_model.predict_proba(X)
            hmm_probs_df = pd.DataFrame(hmm_probs, index=returns.index, columns=self.regime_labels)
            if self.config.use_deep_learning and self.lstm_model is not None:
                try:
                    lstm_probs_df = self.lstm_model.predict_proba(returns)
                    # Bayesian model averaging
                    probs_df = self._bayesian_model_averaging(hmm_probs_df, lstm_probs_df)
                except Exception as e:
                    logger.warning(f"LSTM prediction failed, using HMM only: {str(e)}")
                    probs_df = hmm_probs_df
            else:
                probs_df = hmm_probs_df
            prediction_time = time.time() - start_time
            self._performance.prediction_times.append(prediction_time)
            return probs_df
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def update_real_time(self, new_returns: pd.Series) -> Dict:
        """Update regime detection in real-time with new data"""
        start_time = time.time()
        self._performance.total_updates += 1
        
        try:
            # Input validation
            if not isinstance(new_returns, pd.Series):
                raise ValueError("new_returns must be a pandas Series")
            
            if not self._is_fitted:
                raise ValueError("Model must be fitted before real-time updates")
                
            if len(new_returns) < self.config.window_size:
                raise ValueError(f"Input data must have at least {self.config.window_size} observations")
                
            # Check for data quality
            if new_returns.isnull().any():
                raise RuntimeError("Input data contains missing values")
                
            if new_returns.index.duplicated().any():
                raise RuntimeError("Input data contains duplicate timestamps")
                
            # Ensure proper time ordering
            if self._last_update_time is not None:
                if new_returns.index[-1] <= self._last_update_time:
                    raise ValueError("New data must be more recent than last update")
            
            # Predict regime probabilities
            probs = self.predict_proba(new_returns)
            current_probs = probs.iloc[-1]
            
            # Get current regime and confidence
            current_regime = self.regime_labels[current_probs.argmax()]
            confidence = current_probs.max()
            
            # Check confidence threshold
            if confidence < self.config.min_confidence:
                current_regime = "Uncertain"
                logger.warning(f"Low confidence regime detection: {confidence:.2f}")
                
            # Calculate transition probabilities if we have a previous regime and it's valid
            transition_alert = None
            if self._current_regime is not None and self._current_regime != "Uncertain" and self._current_regime in self.regime_labels:
                trans_mat = self.get_transition_matrix()
                current_idx = self.regime_labels.index(self._current_regime)
                
                # Get transition probabilities from current regime
                transition_probs = trans_mat.iloc[current_idx]
                
                # Check for potential transitions
                for next_regime, prob in transition_probs.items():
                    if (prob > self.config.alert_threshold and 
                        next_regime != self._current_regime):
                        transition_alert = {
                            'from_regime': self._current_regime,
                            'to_regime': next_regime,
                            'probability': prob,
                            'confidence': confidence,
                            'alert_threshold': self.config.alert_threshold
                        }
                        logger.info(
                            f"Regime transition alert: {self._current_regime} -> {next_regime} "
                            f"(prob: {prob:.2f}, conf: {confidence:.2f})"
                        )
            
            # Update tracking attributes
            self._current_regime = current_regime
            self._last_update_time = pd.Timestamp.now()
            
            # Add to history (with size limit)
            history_entry = pd.DataFrame({
                'regime': [current_regime],
                'confidence': [confidence],
                'transition_probability': [transition_alert['probability'] if transition_alert else None]
            }, index=[self._last_update_time])
            
            self._regime_history = pd.concat([self._regime_history, history_entry])
            
            # Maintain history size limit
            if len(self._regime_history) > self.config.history_size:
                self._regime_history = self._regime_history.iloc[-self.config.history_size:]
            
            # Update performance metrics
            update_time = time.time() - start_time
            self._performance.update_times.append(update_time)
            self._performance.successful_updates += 1
            
            # Log performance periodically
            if self._performance.total_updates % 100 == 0:
                self._log_performance_metrics()
            
            return {
                'current_regime': current_regime,
                'confidence': confidence,
                'transition_alert': transition_alert,
                'probabilities': current_probs.to_dict(),
                'timestamp': self._last_update_time,
                'data_quality': {
                    'sample_size': len(new_returns),
                    'missing_values': 0,
                    'duplicates': 0
                },
                'performance': {
                    'update_time': update_time,
                    'success_rate': (self._performance.successful_updates / 
                                   self._performance.total_updates)
                }
            }
            
        except Exception as e:
            self._performance.error_count += 1
            self._performance.last_error = str(e)
            logger.error(f"Real-time update failed: {str(e)}")
            raise RuntimeError(f"Real-time update failed: {str(e)}") from e

    def get_regime_history(self, lookback_periods: Optional[int] = None) -> pd.DataFrame:
        """Get historical regime data with optional lookback window
        
        Parameters:
            lookback_periods (int, optional): Number of periods to look back
            
        Returns:
            pd.DataFrame: Historical regime data
        """
        if lookback_periods is not None:
            return self._regime_history.iloc[-lookback_periods:]
        return self._regime_history

    def get_confidence_metrics(self) -> Dict:
        """Calculate confidence metrics for current regime detection
        
        Returns:
            Dict: Various confidence metrics
        """
        if not self._is_fitted or self._current_regime is None:
            raise ValueError("Model must be fitted and have current regime")
            
        # Get recent history
        recent_history = self._regime_history.iloc[-self.config.window_size:]
        
        # Calculate regime stability
        regime_changes = (recent_history['regime'] != recent_history['regime'].shift()).sum()
        stability = 1 - (regime_changes / len(recent_history))
        
        # Calculate average confidence
        avg_confidence = recent_history['confidence'].mean()
        
        # Calculate regime persistence
        current_regime_mask = recent_history['regime'] == self._current_regime
        persistence = current_regime_mask.sum() / len(recent_history)
        
        # Get transition matrix for current regime
        trans_mat = self.get_transition_matrix()
        current_idx = self.regime_labels.index(self._current_regime)
        stay_prob = trans_mat.iloc[current_idx, current_idx]
        
        return {
            'stability': stability,
            'average_confidence': avg_confidence,
            'persistence': persistence,
            'transition_stability': stay_prob,
            'sample_size': len(recent_history)
        }

    def cleanup(self):
        """Clean up resources and reset state
        
        This method should be called when the detector is no longer needed
        or needs to be reset.
        """
        # Clear history and performance metrics
        self._regime_history = pd.DataFrame(columns=['regime', 'confidence', 'transition_probability'])
        self._performance = PerformanceMetrics()
        
        # Reset state
        self._current_regime = None
        self._last_update_time = None
        self._is_fitted = False
        
        # Clear model parameters
        if hasattr(self.hmm_model, 'means_'):
            del self.hmm_model.means_
        if hasattr(self.hmm_model, 'covars_'):
            del self.hmm_model.covars_
        if hasattr(self.hmm_model, 'transmat_'):
            del self.hmm_model.transmat_
        
        # Clear LSTM model if present
        if self.lstm_model is not None:
            self.lstm_model = None
            
        logger.info("Cleaned up MarketRegimeDetector resources")
        
    def trim_history(self, max_age_days: Optional[int] = None):
        """Trim regime history to manage memory
        
        Parameters:
            max_age_days (int, optional): Maximum age of history entries in days
        """
        if max_age_days is not None:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
            self._regime_history = self._regime_history[self._regime_history.index > cutoff_date]
        
        # Ensure we don't exceed history size limit
        if len(self._regime_history) > self.config.history_size:
            self._regime_history = self._regime_history.iloc[-self.config.history_size:]
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction
            
    def reset_performance_metrics(self):
        """Reset performance tracking"""
        self._performance = PerformanceMetrics()
        logger.info("Reset performance metrics")
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics
        
        Returns:
            Dict[str, float]: Memory usage in MB for different components
        """
        import sys
        
        memory_usage = {
            'history': sys.getsizeof(self._regime_history) / (1024 * 1024),
            'performance_metrics': sys.getsizeof(self._performance) / (1024 * 1024)
        }
        
        # Add HMM model size if fitted
        if self._is_fitted:
            hmm_size = (
                sys.getsizeof(getattr(self.hmm_model, 'means_', 0)) +
                sys.getsizeof(getattr(self.hmm_model, 'covars_', 0)) +
                sys.getsizeof(getattr(self.hmm_model, 'transmat_', 0))
            ) / (1024 * 1024)
            memory_usage['hmm_model'] = hmm_size
        
        # Add LSTM model size if present
        if self.lstm_model is not None:
            try:
                lstm_size = sys.getsizeof(self.lstm_model) / (1024 * 1024)
                memory_usage['lstm_model'] = lstm_size
            except:
                memory_usage['lstm_model'] = 0
        
        memory_usage['total'] = sum(memory_usage.values())
        return memory_usage

    def compare_to_benchmark(self, features: pd.DataFrame, benchmark: str = 'hmm') -> Dict:
        """Compare Bayesian ensemble to HMM and random baselines using the full feature set"""
        from src.regime import MarketRegimeDetector
        # Fit HMM baseline
        detector = MarketRegimeDetector(self.config)
        detector.fit(features)
        hmm_regimes = detector.predict(features)
        # Use the 'returns' column for all comparisons
        ensemble_returns = features['returns']
        if benchmark == 'hmm':
            hmm_regimes_int = detector.hmm_model.predict(detector._prepare_features(features, fit_scaler=False))
            baseline_returns = features['returns']
        elif benchmark == 'random':
            np.random.seed(self.config.random_state)
            random_regimes = np.random.choice(np.unique(hmm_regimes), size=len(features))
            baseline_returns = features['returns']
        elif benchmark == 'moving_average':
            if 'returns' in features.columns:
                ma = features['returns'].rolling(window=10, min_periods=1).mean()
                baseline_returns = ma
            else:
                raise ValueError("Moving average benchmark requires a 'returns' column in features")
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        # Hypothesis test (pass only 1D Series)
        test_result = StatisticalValidator().test_model_performance_hypothesis(ensemble_returns, baseline_returns)
        return {
            'ensemble_mean_return': ensemble_returns.mean(),
            'baseline_mean_return': baseline_returns.mean(),
            'test_result': test_result
        }