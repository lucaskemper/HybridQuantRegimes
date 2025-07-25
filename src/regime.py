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
import traceback

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
    min_size: int = 21
    smoothing_window: int = 5
    use_deep_learning: bool = False
    deep_learning_config: Optional[DeepLearningConfig] = None
    
    # New real-time configuration options
    alert_threshold: float = 0.7  # Probability threshold for transition alerts
    min_confidence: float = 0.6  # Minimum confidence required for regime assignment
    update_frequency: str = '1D'  # Pandas frequency string for updates
    history_size: int = 100  # Number of historical regime entries to maintain
    min_regime_size: int = 21  # Added to support config
    
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
            n_iter=2000,
            random_state=config.random_state,
            init_params="",
            min_covar=1e-6,
            verbose=True,
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
        # Diagnostics for input X
        print("[HMM FIT] Fitting HMM on input with shape:", X.shape)
        print("[HMM FIT] NaNs in input:", np.isnan(X).sum())
        print("[HMM FIT] Mean/std of features:", np.mean(X), np.std(X))
        print("[HMM FIT] Features columns:", self.config.features)
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
            # DIAGNOSTIC: Immediately test predict_proba after fit
            proba_test = self.hmm_model.predict_proba(X)
            print('[HMM FIT] predict_proba after fit (head):', proba_test[:5])
            print('[HMM FIT] Any NaNs in predict_proba after fit:', np.isnan(proba_test).any())
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

    def _get_aligned_trimmed_returns(self, returns: pd.Series) -> pd.Series:
        """Trim returns to match LSTM sequence length for valid fusion."""
        if self.config.use_deep_learning and self.lstm_model is not None:
            seq_len = self.lstm_model.sequence_length  # Always use the LSTM model's sequence_length
            # Only use the last len(returns) - seq_len rows
            if len(returns) > seq_len:
                return returns.iloc[seq_len:]
            else:
                return returns.iloc[0:0]  # Empty
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

                # Align and trim returns for both models
                trimmed_returns = self._get_aligned_trimmed_returns(returns_clean)
                if trimmed_returns.empty:
                    raise ValueError("Not enough data after trimming for LSTM sequence length.")
                # Always require features to be a DataFrame for hybrid fusion
                trimmed_features = features.loc[trimmed_returns.index]
                # Define generic regime labels for fusion
                n_regimes = len(self.regime_labels)
                generic_labels = [f"regime_{i}" for i in range(n_regimes)]
                # Get LSTM probabilities (index will match trimmed_returns)
                lstm_probs = self.lstm_model.predict_proba(trimmed_returns)
                # Ensure both LSTM and HMM columns are set to generic_labels for fusion
                lstm_probs.columns = generic_labels
                hmm_probs_full = self.predict_proba(trimmed_features)
                # Ensure both LSTM and HMM columns are set to generic_labels for fusion
                hmm_probs_full.columns = generic_labels
                # Check that regime labels match exactly
                if list(lstm_probs.columns) != generic_labels:
                    print(f"[FUSION ERROR] LSTM columns: {lstm_probs.columns}, generic_labels: {generic_labels}")
                    raise ValueError(f"LSTM regime columns do not match generic regime labels!")
                if list(hmm_probs_full.columns) != generic_labels:
                    print(f"[FUSION ERROR] HMM columns: {hmm_probs_full.columns}, generic_labels: {generic_labels}")
                    raise ValueError(f"HMM regime columns do not match generic regime labels!")
                # Robust intersection of indices
                valid_idx = hmm_probs_full.index.intersection(lstm_probs.index)
                hmm_probs = hmm_probs_full.loc[valid_idx]
                lstm_probs = lstm_probs.loc[valid_idx]
                # Filter out any rows where either is all-NaN
                valid_mask = (~hmm_probs.isnull().any(axis=1)) & (~lstm_probs.isnull().any(axis=1))
                hmm_probs = hmm_probs[valid_mask]
                lstm_probs = lstm_probs[valid_mask]
                if len(hmm_probs) == 0 or len(lstm_probs) == 0:
                    print('[HYBRID FUSION DIAGNOSTIC] No valid rows after robust intersection and filtering!')
                    print('hmm_probs_full index head:', hmm_probs_full.index[:10])
                    print('lstm_probs index head:', lstm_probs.index[:10])
                    print('hmm_probs_full columns:', hmm_probs_full.columns)
                    print('lstm_probs columns:', lstm_probs.columns)
                    print('hmm_probs_full isnull sum:', hmm_probs_full.isnull().sum())
                    print('lstm_probs isnull sum:', lstm_probs.isnull().sum())
                    raise ValueError("No valid rows after robust intersection and filtering for hybrid regime fusion.")
                if list(hmm_probs.columns) != generic_labels or list(lstm_probs.columns) != generic_labels:
                    print(f"[FUSION ERROR] Columns mismatch after relabeling. hmm_probs: {hmm_probs.columns}, lstm_probs: {lstm_probs.columns}, generic_labels: {generic_labels}")
                    raise ValueError("Regime columns do not match generic labels after relabeling!")
                # Combine predictions
                regimes = self._combine_predictions_bayesian(hmm_probs, lstm_probs, output_labels=output_labels)
                return regimes

            except Exception as e:
                print(f"Deep learning prediction failed: {str(e)}")
                traceback.print_exc()
                print("Falling back to HMM predictions")
                return hmm_regimes_labels

        return hmm_regimes_labels

    def _validate_fusion_inputs(self, hmm_probs, lstm_probs):
        """Validate inputs before fusion"""
        # Check for completely invalid inputs
        if hmm_probs.empty or lstm_probs.empty:
            raise ValueError("Empty probability matrices")
        if hmm_probs.isnull().all().all():
            raise ValueError("HMM probabilities are all NaN")
        if lstm_probs.isnull().all().all():
            raise ValueError("LSTM probabilities are all NaN")
        # Check column compatibility
        if not set(hmm_probs.columns).intersection(set(lstm_probs.columns)):
            raise ValueError(f"No common columns: HMM {hmm_probs.columns}, LSTM {lstm_probs.columns}")
        return True

    def _calculate_prediction_weights(self, hmm_probs, lstm_probs):
        """Calculate weights based on prediction confidence and consistency"""
        # Calculate confidence (max probability)
        hmm_confidence = hmm_probs.max(axis=1).mean()
        lstm_confidence = lstm_probs.max(axis=1).mean()
        # Calculate consistency (inverse of entropy, but capped)
        hmm_entropy = -(hmm_probs * np.log(hmm_probs + 1e-8)).sum(axis=1).mean()
        lstm_entropy = -(lstm_probs * np.log(lstm_probs + 1e-8)).sum(axis=1).mean()
        # Cap entropy values to avoid extreme weights
        hmm_entropy = np.clip(hmm_entropy, 0.1, 2.0)
        lstm_entropy = np.clip(lstm_entropy, 0.1, 2.0)
        # Combine confidence and consistency
        hmm_score = hmm_confidence / hmm_entropy
        lstm_score = lstm_confidence / lstm_entropy
        total_score = hmm_score + lstm_score
        if total_score <= 0:
            return np.array([0.5, 0.5])
        return np.array([hmm_score/total_score, lstm_score/total_score])

    def _combine_predictions_bayesian(self, hmm_probs: pd.DataFrame, lstm_probs: pd.DataFrame, output_labels: bool = True) -> pd.Series:
        import traceback
        # Store original index for final result
        original_index = hmm_probs.index.copy()
        # Align on common indices
        common_idx = hmm_probs.index.intersection(lstm_probs.index)
        if len(common_idx) == 0:
            raise ValueError("No common indices between HMM and LSTM predictions")
        hmm_aligned = hmm_probs.loc[common_idx]
        lstm_aligned = lstm_probs.loc[common_idx]
        # Remove invalid rows (keep indices aligned)
        # Fix: Only keep rows where BOTH HMM and LSTM have all non-NaN and sum > 1e-5
        valid_mask = (
            ~hmm_aligned.isnull().any(axis=1) &
            ~lstm_aligned.isnull().any(axis=1) &
            (lstm_aligned.sum(axis=1) > 1e-5) &
            (hmm_aligned.sum(axis=1) > 1e-5)
        )
        num_valid = valid_mask.sum()
        if num_valid == 0:
            print(f"[Hybrid Fusion] No valid rows after filtering: HMM shape {hmm_aligned.shape}, LSTM shape {lstm_aligned.shape}")
            print(f"[Hybrid Fusion] HMM NaN rows: {hmm_aligned.isnull().all(axis=1).sum()}, LSTM NaN rows: {lstm_aligned.isnull().all(axis=1).sum()}")
            print(f"[Hybrid Fusion] HMM sum==0 rows: {(hmm_aligned.sum(axis=1) <= 1e-5).sum()}, LSTM sum==0 rows: {(lstm_aligned.sum(axis=1) <= 1e-5).sum()}")
            raise ValueError("No valid rows after filtering for hybrid regime fusion. Check for NaNs or misaligned indices in HMM/LSTM outputs.")
        hmm_valid = hmm_aligned[valid_mask]
        lstm_valid = lstm_aligned[valid_mask]
        # Validate inputs
        self._validate_fusion_inputs(hmm_valid, lstm_valid)
        # Ensure both columns are generic_labels before fusion
        generic_labels = list(hmm_valid.columns)
        lstm_valid.columns = generic_labels
        hmm_valid.columns = generic_labels
        # Defensive: check columns match
        assert set(lstm_valid.columns) == set(hmm_valid.columns), "Post-mapping column mismatch"
        # Find common columns
        common_cols = lstm_valid.columns.intersection(hmm_valid.columns)
        lstm_valid = lstm_valid[common_cols]
        hmm_valid = hmm_valid[common_cols]
        # Normalize probabilities
        lstm_valid = lstm_valid.div(lstm_valid.sum(axis=1), axis=0).fillna(0)
        hmm_valid = hmm_valid.div(hmm_valid.sum(axis=1), axis=0).fillna(0)
        # Calculate robust weights
        weights = self._calculate_prediction_weights(hmm_valid, lstm_valid)
        combined_probs = weights[0] * hmm_valid + weights[1] * lstm_valid
        # Defensive: ensure no NaNs
        if combined_probs.isnull().any().any():
            raise ValueError("NaNs in combined_probs after fusion")
        regimes_blend = combined_probs.idxmax(axis=1)
        # Start with HMM predictions for all indices
        hmm_only_regimes = hmm_probs.idxmax(axis=1)
        final_regimes = hmm_only_regimes.copy()
        # Only update where we have valid fusion results
        final_regimes.loc[regimes_blend.index] = regimes_blend
        # No reindexing needed!
        if output_labels:
            return final_regimes
        else:
            reverse_mapping = {label: i for i, label in enumerate(self.regime_labels)}
            return final_regimes.map(reverse_mapping)

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
            # DIAGNOSTIC: Print features for HMM
            print('[HMM PREDICT_PROBA] Features columns:', self.config.features)
            print('[HMM PREDICT_PROBA] Features shape:', X.shape)
            print('[HMM PREDICT_PROBA] Any NaNs in features:', np.isnan(X).any())
            # DIAGNOSTIC: Print scaler output
            if hasattr(self.scaler, 'scale_'):
                X_scaled = self.scaler.transform(X)
                print('[HMM PREDICT_PROBA] Scaled features (head):', X_scaled[:5])
                print('[HMM PREDICT_PROBA] Scaled features (tail):', X_scaled[-5:])
                print('[HMM PREDICT_PROBA] Any NaNs in scaled features:', np.isnan(X_scaled).any())
            print(f"[HMM PREDICT_PROBA] Feature shape: {X.shape}")
            print(f"[HMM PREDICT_PROBA] Model fitted: {self._is_fitted}")
            hmm_probs = self.hmm_model.predict_proba(X)
            print('[HMM PREDICT_PROBA] predict_proba (head):', hmm_probs[:5])
            print('[HMM PREDICT_PROBA] Any NaNs in predict_proba:', np.isnan(hmm_probs).any())
            print(f"[HMM PREDICT_PROBA] Raw HMM output shape: {hmm_probs.shape}")
            print(f"[HMM PREDICT_PROBA] Regime labels: {self.regime_labels}")
            # Always use self.regime_labels for columns
            columns = self.regime_labels
            # DEBUG: Check dimensions and fix index mismatch
            print(f"DEBUG: returns.index length: {len(returns.index)}")
            print(f"DEBUG: hmm_probs shape: {hmm_probs.shape}")
            print(f"DEBUG: hmm_probs contains NaN: {np.isnan(hmm_probs).any()}")
            if hmm_probs.shape[0] != len(returns.index):
                print(f"WARNING: Shape mismatch! Using last {hmm_probs.shape[0]} indices")
                safe_index = returns.iloc[-hmm_probs.shape[0]:].index if len(returns.index) >= hmm_probs.shape[0] else returns.index
            else:
                safe_index = returns.index
            # Always use generic regime column names for fusion
            generic_labels = [f"regime_{i}" for i in range(len(self.regime_labels))]
            if hmm_probs.shape[1] == 0 or len(generic_labels) == 0:
                # Defensive: create a DataFrame of zeros if HMM returns no probabilities
                print("[FORCE FIX] HMM returned no probabilities, creating zero-prob DataFrame.")
                hmm_probs_df = pd.DataFrame(0, index=safe_index, columns=generic_labels)
            else:
                hmm_probs_df = pd.DataFrame(hmm_probs, index=safe_index, columns=generic_labels)
            print(f"[HMM PREDICT_PROBA] HMM probs DataFrame columns (generic): {hmm_probs_df.columns}")
            print(f"[HMM PREDICT_PROBA] HMM probs index: {hmm_probs_df.index}")
            print(f"[HMM PREDICT_PROBA] HMM probs_df shape: {hmm_probs_df.shape}")
            if hmm_probs_df.shape[1] == 0:
                raise ValueError(f"HMM regime probability output has zero columns! This usually means the HMM was fitted with n_regimes=1 or failed to fit. Check your config and input data. Regime labels: {self.regime_labels}")
            if hmm_probs_df.isnull().all().all():
                raise ValueError(f"All values in HMM regime probability output are NaN! Check HMM fit and regime labels: {self.regime_labels}")
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

    def get_ensemble_probabilities(self, features: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Return the hybrid (ensemble) regime probabilities for the LSTM-valid index if deep learning is enabled, else HMM probabilities for the full index."""
        # If not fitted, fit first
        if not self._is_fitted:
            self.fit(features)
        # If deep learning is enabled and LSTM is present
        if self.config.use_deep_learning and self.lstm_model is not None:
            print("[DIAGNOSTIC] get_ensemble_probabilities: Using hybrid (HMM+LSTM) path")
            # Sanitize returns for LSTM
            returns_clean = self._sanitize_returns(returns)
            hmm_regimes_int = self.predict(features, output_labels=False)
            self.lstm_model.fit(returns_clean, hmm_regimes_int)
            trimmed_returns = self._get_aligned_trimmed_returns(returns_clean)
            if trimmed_returns.empty:
                raise ValueError("Not enough data after trimming for LSTM sequence length.")
            # Always require features to be a DataFrame for hybrid fusion
            trimmed_features = features.loc[trimmed_returns.index]
            lstm_probs = self.lstm_model.predict_proba(trimmed_returns)
            hmm_probs_full = self.predict_proba(trimmed_features)
            # Robust intersection of indices
            valid_idx = hmm_probs_full.index.intersection(lstm_probs.index)
            hmm_probs = hmm_probs_full.loc[valid_idx]
            lstm_probs = lstm_probs.loc[valid_idx]
            # Filter out any rows where either is all-NaN
            valid_mask = (~hmm_probs.isnull().any(axis=1)) & (~lstm_probs.isnull().any(axis=1))
            hmm_probs = hmm_probs[valid_mask]
            lstm_probs = lstm_probs[valid_mask]
            print(f"[DIAGNOSTIC] HMM probs shape after filtering: {hmm_probs.shape}")
            print(f"[DIAGNOSTIC] LSTM probs shape after filtering: {lstm_probs.shape}")
            if len(hmm_probs) == 0 or len(lstm_probs) == 0:
                print('[HYBRID FUSION DIAGNOSTIC] No valid rows after robust intersection and filtering!')
                print('hmm_probs_full index head:', hmm_probs_full.index[:10])
                print('lstm_probs index head:', lstm_probs.index[:10])
                print('hmm_probs_full columns:', hmm_probs_full.columns)
                print('lstm_probs columns:', lstm_probs.columns)
                print('hmm_probs_full isnull sum:', hmm_probs_full.isnull().sum())
                print('lstm_probs isnull sum:', lstm_probs.isnull().sum())
                raise ValueError("No valid rows after robust intersection and filtering for hybrid regime fusion.")
            hmm_probs = hmm_probs.loc[:, [col for col in self.regime_labels if col in hmm_probs.columns]]
            # Map LSTM regime columns to HMM regime labels if needed
            lstm_labels = list(lstm_probs.columns)
            hmm_labels = list(self.regime_labels)
            if all(l.startswith('regime_') for l in lstm_labels) and len(lstm_labels) == len(hmm_labels):
                mapping = {lstm_label: hmm_label for lstm_label, hmm_label in zip(lstm_labels, hmm_labels)}
                lstm_probs = lstm_probs.rename(columns=mapping)
            # Find common columns
            common_cols = lstm_probs.columns.intersection(hmm_probs.columns)
            lstm_probs = lstm_probs[common_cols]
            hmm_probs = hmm_probs[common_cols]
            # Normalize
            lstm_probs = lstm_probs.div(lstm_probs.sum(axis=1), axis=0).fillna(0)
            hmm_probs = hmm_probs.div(hmm_probs.sum(axis=1), axis=0).fillna(0)
            # Calculate weights
            weights = self._calculate_prediction_weights(hmm_probs, lstm_probs)
            combined_probs = weights[0] * hmm_probs + weights[1] * lstm_probs
            print(f"[DIAGNOSTIC] Returning combined_probs shape: {combined_probs.shape}, columns: {combined_probs.columns}")
            return combined_probs
        else:
            print("[DIAGNOSTIC] get_ensemble_probabilities: Using HMM-only path")
            hmm_probs = self.predict_proba(returns)
            print(f"[DIAGNOSTIC] Returning hmm_probs shape: {hmm_probs.shape}, columns: {hmm_probs.columns}")
            return hmm_probs

def get_regime_series_for_signals(returns_dict: Dict[str, pd.Series], config: Optional[RegimeConfig] = None) -> Dict[str, pd.Series]:
    """
    Given a dict of returns per ticker, returns a dict of regime label Series per ticker.
    Uses MarketRegimeDetector from this module.
    """
    if config is None:
        config = RegimeConfig()
    regime_dict = {}
    for ticker, ret in returns_dict.items():
        detector = MarketRegimeDetector(config)
        detector.fit(ret)
        regime_dict[ticker] = detector.predict(ret)
    return regime_dict