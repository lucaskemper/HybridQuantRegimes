from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import itertools
from typing import Callable

from .deep_learning import DeepLearningConfig, LSTMRegimeDetector, TransformerRegimeDetector

# Configure logging
logger = logging.getLogger(__name__)

# --- Fix pomegranate import ---
try:
    from pomegranate import HiddenMarkovModel, NormalDistribution
except ImportError:
    HiddenMarkovModel = None
    NormalDistribution = None

@dataclass
class RegimeConfig:
    n_regimes: int = 4  # Updated to 4 regimes for semiconductor business cycles
    n_iter: int = 1000
    random_state: int = 42
    window_size: int = 15  # Faster regime change detection
    features: List[str] = field(
        default_factory=lambda: [
            "returns", "volatility", "momentum", "rsi", 
            "volume", "vix_level", "yield_spread"
        ]
    )
    min_regime_size: int = 21
    smoothing_window: int = 3  # Reduced for more responsive detection
    use_deep_learning: bool = True
    use_transformer: bool = True  # New transformer model
    deep_learning_config: Optional[DeepLearningConfig] = None
    
    # Enhanced ensemble configuration
    ensemble_method: str = "dynamic_weighted_confidence"
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "hmm": 0.25,
        "lstm": 0.40,
        "transformer": 0.35
    })
    
    # Regime transition constraints
    transition_constraints: Dict[str, Any] = field(default_factory=lambda: {
        "min_regime_duration": 5,
        "transition_probability_floor": 0.05,
        "self_transition_bias": 0.1
    })
    
    # New real-time configuration options
    alert_threshold: float = 0.7  # Probability threshold for transition alerts
    min_confidence: float = 0.6  # Minimum confidence required for regime assignment
    update_frequency: str = '1D'  # Pandas frequency string for updates
    history_size: int = 100  # Number of historical regime entries to maintain
    
    labeling_metric: str = 'risk_adjusted_return'  # Updated to risk-adjusted return
    max_flips: int = 3  # For transition penalty logic
    transition_window: int = 10  # For transition penalty logic
    
    # Dynamic regime assignment
    dynamic_assignment_method: Optional[str] = None  # Options: None, 'rolling_quantile', 'rolling_kmeans'
    dynamic_assignment_window: int = 63  # Window for dynamic assignment
    
    # Alternative model selection
    model_type: str = 'hmm'  # Options: 'hmm', 'bayesian_hmm', 'markov_switching', 'random_forest'
    
    # Semiconductor-specific features
    include_semiconductor_features: bool = True
    
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
            )  # type: ignore
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
        """Initialize regime detector with configuration and model selection"""
        self.config = config
        self.regime_labels = ["Low_Vol_Growth", "High_Vol_Growth", "Correction", "Crisis"][: config.n_regimes]
        self.model_type = config.model_type
        
        # Model selection
        if self.model_type == 'hmm':
            from hmmlearn import hmm
            self.hmm_model = hmm.GaussianHMM(
                n_components=config.n_regimes,
                covariance_type="full",
                n_iter=config.n_iter,
                random_state=config.random_state,
                init_params="",
            )
            self.alt_model = None
        elif self.model_type == 'bayesian_hmm':
            if HiddenMarkovModel is None or NormalDistribution is None:
                raise ImportError("pomegranate is not installed. Please install it to use bayesian_hmm.")
            self.hmm_model = None
            self.alt_model = HiddenMarkovModel.from_samples(
                NormalDistribution, n_components=config.n_regimes, n_jobs=1
            )
        elif self.model_type == 'markov_switching':
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            self.hmm_model = None
            self.alt_model = None  # Will be fit in fit()
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.hmm_model = None
            self.alt_model = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Initialize LSTM if enabled
        self.lstm_model = None
        if config.use_deep_learning and config.deep_learning_config is not None:
            self.lstm_model = LSTMRegimeDetector(config.deep_learning_config)

        # Initialize Transformer if enabled
        self.transformer_model = None
        if config.use_transformer and config.deep_learning_config is not None:
            self.transformer_model = TransformerRegimeDetector(config.deep_learning_config)

        self._is_fitted = False
        self.scaler = RobustScaler()  # Changed to RobustScaler for better outlier handling
        self._last_fit_size = None
        
        # New attributes for real-time tracking
        self._current_regime = None
        self._regime_history = pd.DataFrame(columns=['regime', 'confidence', 'transition_probability'])
        self._last_update_time = None
        self._alert_threshold = config.alert_threshold
        
        # Performance monitoring
        self._performance = PerformanceMetrics()
        
        logger.info(f"Initialized MarketRegimeDetector with {config.n_regimes} regimes and model_type {self.model_type}")
        
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

    def _calculate_enhanced_features(self, returns: pd.Series, macro_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Calculate enhanced features for regime detection including semiconductor-specific features"""
        features = {}

        # Basic returns feature
        features["returns"] = returns.values
        features["log_returns"] = np.log(returns + 1).values

        # Volatility features
        features["volatility"] = returns.ewm(span=self.config.window_size).std().values
        features["realized_volatility"] = returns.rolling(self.config.window_size).std().values

        # Momentum features
        roll_mean = returns.rolling(self.config.window_size).mean()
        if hasattr(roll_mean, 'to_numpy') and not isinstance(roll_mean, np.ndarray):
            features["momentum"] = roll_mean.to_numpy()
        else:
            features["momentum"] = np.asarray(roll_mean)
        
        # Multiple momentum periods
        for period in [5, 20, 60]:
            features[f"momentum_{period}d"] = returns.rolling(period).mean().values
            features[f"roc_{period}d"] = (returns / returns.shift(period) - 1).values

        # Technical indicators
        features["rsi_14"] = self._calculate_rsi(returns, 14).values
        features["rsi_30"] = self._calculate_rsi(returns, 30).values
        features["macd_signal"] = self._calculate_macd(returns).values
        features["bollinger_position"] = self._calculate_bollinger_position(returns).values
        features["williams_r"] = self._calculate_williams_r(returns).values

        # Volume-based features (using volatility as proxy)
        features["volume_ratio"] = (returns.rolling(5).std() / returns.rolling(21).std()).values
        features["volume_sma_ratio"] = (returns.rolling(5).std() / returns.rolling(21).std()).values
        features["on_balance_volume"] = self._calculate_obv(returns).values

        # Macro features
        if macro_data:
            if 'VIX' in macro_data:
                features["vix_level"] = macro_data['VIX'].values
                features["vix_change"] = macro_data['VIX'].pct_change().values
            if 'TNX' in macro_data and 'TYX' in macro_data:
                features["yield_spread"] = (macro_data['TYX'] - macro_data['TNX']).values
                features["term_structure_slope"] = (macro_data['TYX'] - macro_data['TNX']).values
            if 'DXY' in macro_data:
                features["dollar_strength"] = macro_data['DXY'].pct_change().values
        else:
            # Fallback to proxies
            features["vix_level"] = returns.rolling(21).std().values * np.sqrt(252)
            features["vix_change"] = returns.rolling(21).std().pct_change().values
            features["yield_spread"] = returns.rolling(63).mean().values
            features["term_structure_slope"] = returns.rolling(63).mean().values
            features["dollar_strength"] = returns.rolling(21).std().values

        # Semiconductor-specific features
        if self.config.include_semiconductor_features:
            features["semiconductor_pmi"] = self._calculate_semiconductor_pmi(returns).values
            features["memory_vs_logic_spread"] = self._calculate_memory_logic_spread(returns).values
            features["equipment_vs_design_ratio"] = self._calculate_equipment_design_ratio(returns).values

        # Skewness and kurtosis
        features["skewness"] = returns.rolling(self.config.window_size).skew().values
        features["kurtosis"] = returns.rolling(self.config.window_size).kurt().values

        return features

    def _calculate_rsi(self, returns: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, returns: pd.Series) -> pd.Series:
        """Calculate MACD signal line"""
        ema12 = returns.ewm(span=12).mean()
        ema26 = returns.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return signal

    def _calculate_bollinger_position(self, returns: pd.Series) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = returns.rolling(20).mean()
        std = returns.rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (returns - lower) / (upper - lower)
        return position

    def _calculate_williams_r(self, returns: pd.Series) -> pd.Series:
        """Calculate Williams %R"""
        high = returns.rolling(14).max()
        low = returns.rolling(14).min()
        williams_r = ((high - returns) / (high - low)) * -100
        return williams_r

    def _calculate_obv(self, returns: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (using returns as proxy)"""
        obv = pd.Series(index=returns.index, dtype=float)
        obv.iloc[0] = 0
        for i in range(1, len(returns)):
            if returns.iloc[i] > returns.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + abs(returns.iloc[i])
            elif returns.iloc[i] < returns.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - abs(returns.iloc[i])
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def _calculate_semiconductor_pmi(self, returns: pd.Series) -> pd.Series:
        """Calculate semiconductor PMI proxy using momentum and volatility"""
        momentum = returns.rolling(20).mean()
        volatility = returns.rolling(20).std()
        pmi = (momentum / volatility) * 50 + 50  # Scale to 0-100 range
        return pmi

    def _calculate_memory_logic_spread(self, returns: pd.Series) -> pd.Series:
        """Calculate memory vs logic spread proxy"""
        short_momentum = returns.rolling(5).mean()
        long_momentum = returns.rolling(20).mean()
        spread = short_momentum - long_momentum
        return spread

    def _calculate_equipment_design_ratio(self, returns: pd.Series) -> pd.Series:
        """Calculate equipment vs design ratio proxy"""
        equipment_proxy = returns.rolling(10).std()  # Higher volatility for equipment
        design_proxy = returns.rolling(30).std()     # Lower volatility for design
        ratio = equipment_proxy / (design_proxy + 1e-6)
        return ratio

    def _prepare_features(self, returns: pd.Series, macro_data: Optional[Dict] = None) -> np.ndarray:
        """Prepare features for HMM with enhanced preprocessing"""
        features = []
        all_features = self._calculate_enhanced_features(returns, macro_data)

        # Select only the features specified in config, ensuring we don't exceed 15
        selected_features = self.config.features[:15]  # Limit to 15 features
        
        for feature_name in selected_features:
            if feature_name not in all_features:
                # Skip missing features instead of raising error
                continue

            feature_data = all_features[feature_name]

            # Winsorize extreme values
            feature_data = np.clip(
                feature_data,
                np.nanpercentile(feature_data, 1),
                np.nanpercentile(feature_data, 99),
            )

            features.append(feature_data.reshape(-1, 1))

        # Combine features
        if features:
            X = np.hstack(features)
        else:
            # Fallback to basic features if none are available
            X = np.column_stack([
                returns.values.reshape(-1, 1),
                returns.rolling(21).std().values.reshape(-1, 1),
                returns.rolling(21).mean().values.reshape(-1, 1)
            ])

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Scale features
        X = self.scaler.fit_transform(X)

        return X

    def _initialize_hmm_params(self, X: np.ndarray):
        """Initialize HMM parameters with improved initialization"""
        n_samples, n_features = X.shape
        n_components = self.config.n_regimes
        
        # Improved initialization using K-means clustering
        kmeans = KMeans(n_clusters=n_components, random_state=self.config.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Initialize means from cluster centers
        means = kmeans.cluster_centers_
        
        # Initialize covariances from cluster data
        covariances = []
        for i in range(n_components):
            cluster_data = X[cluster_labels == i]
            if len(cluster_data) > 1:
                cov = np.cov(cluster_data.T)
                # Ensure positive definiteness
                cov = cov + np.eye(n_features) * 1e-6
            else:
                cov = np.eye(n_features)
            covariances.append(cov)
        
        # Initialize transition matrix with slight bias toward staying in same state
        transition_matrix = np.eye(n_components) * 0.8 + np.ones((n_components, n_components)) * 0.2 / n_components
        
        # Initialize start probabilities uniformly
        start_probs = np.ones(n_components) / n_components
        
        # Set HMM parameters
        self.hmm_model.means_ = means
        self.hmm_model.covars_ = np.array(covariances)
        self.hmm_model.transmat_ = transition_matrix
        self.hmm_model.startprob_ = start_probs
        
        # Set initialization parameters to prevent re-initialization
        self.hmm_model.init_params = ""

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
        """Fit the regime detection model with improved error handling"""
        try:
            # Prepare features with macro data if available
            macro_data = None
            if hasattr(self, 'data') and self.data is not None:
                macro_data = self.data.get('macro')
            
            X = self._prepare_features(returns, macro_data)
            
            # Remove any remaining NaN or infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values
            X = np.clip(X, -10, 10)
            
            if self.hmm_model is not None:
                # Initialize HMM parameters
                self._initialize_hmm_params(X)
                
                # Fit HMM with correct parameters
                try:
                    self.hmm_model.fit(X)
                    self._is_fitted = True
                except Exception as e:
                    logger.warning(f"HMM fitting failed: {e}. Trying with different initialization.")
                    # Try with different initialization
                    self._initialize_hmm_params(X)
                    self.hmm_model.fit(X)
                    self._is_fitted = True
                    
            elif self.alt_model is not None:
                # Fit alternative model
                if self.model_type == 'bayesian_hmm':
                    self.alt_model.fit(X)
                elif self.model_type == 'random_forest':
                    # For random forest, we need labels - use K-means for initial clustering
                    kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
                    labels = kmeans.fit_predict(X)
                    self.alt_model.fit(X, labels)
                self._is_fitted = True
                
            # Fit deep learning models if enabled
            if self.config.use_deep_learning and self.lstm_model is not None:
                try:
                    # Get initial regimes from HMM for LSTM training
                    if self.hmm_model is not None and self._is_fitted:
                        initial_regimes = self.hmm_model.predict(X)
                        self.lstm_model.fit(returns, initial_regimes)
                    else:
                        # Use K-means for initial regimes if HMM failed
                        kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
                        initial_regimes = kmeans.fit_predict(X)
                        self.lstm_model.fit(returns, initial_regimes)
                except Exception as e:
                    logger.warning(f"LSTM training failed: {e}")
                    
            # Fit transformer model if enabled
            if self.config.use_transformer and self.transformer_model is not None:
                try:
                    # Get initial regimes for transformer training
                    if self.hmm_model is not None and self._is_fitted:
                        initial_regimes = self.hmm_model.predict(X)
                        self.transformer_model.fit(returns, initial_regimes)
                    else:
                        # Use K-means for initial regimes if HMM failed
                        kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
                        initial_regimes = kmeans.fit_predict(X)
                        self.transformer_model.fit(returns, initial_regimes)
                except Exception as e:
                    logger.warning(f"Transformer training failed: {e}")
                    
            self._last_fit_size = len(returns)
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            self._is_fitted = False
            raise

    def predict(self, returns: pd.Series) -> pd.Series:
        """Predict regimes for the given returns using the selected model"""
        X = self._prepare_features(returns)
        if self.model_type == 'hmm':
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            if self.hmm_model is not None and hasattr(self.hmm_model, 'predict'):
                raw_regimes = self.hmm_model.predict(X)
                regimes = pd.Series(raw_regimes, index=returns.index)
                if self.config.smoothing_window > 0:
                    regimes = self._smooth_regimes(regimes)
                return regimes
            else:
                raise RuntimeError("HMM model is not initialized.")
        elif self.model_type == 'bayesian_hmm':
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            if self.alt_model is not None and hasattr(self.alt_model, 'predict'):
                raw_regimes = self.alt_model.predict(X)
                regimes = pd.Series(raw_regimes, index=returns.index)
                return regimes
            else:
                raise RuntimeError("Bayesian HMM model is not initialized.")
        elif self.model_type == 'markov_switching':
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            # Only access smoothed_marginal_probabilities if present and not RandomForestClassifier/MarkovRegression/ndarray
            if self.alt_model is not None and hasattr(self.alt_model, 'smoothed_marginal_probabilities') and not isinstance(self.alt_model, (np.ndarray,)) and self.alt_model.__class__.__name__ not in ['RandomForestClassifier', 'MarkovRegression']:
                smoothed = self.alt_model.smoothed_marginal_probabilities
                if isinstance(smoothed, pd.DataFrame) or isinstance(smoothed, pd.Series):
                    regimes = smoothed.idxmax(axis=1)
                else:
                    regimes = pd.Series(smoothed[1].idxmax(axis=1), index=returns.index)
                regimes = pd.Series(regimes, index=returns.index)
                return regimes
            else:
                raise RuntimeError("Markov Switching model is not initialized or missing smoothed_marginal_probabilities.")
        elif self.model_type == 'random_forest':
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            if self.alt_model is not None and hasattr(self.alt_model, 'predict') and not isinstance(self.alt_model, np.ndarray):
                # Ensure .predict is not called on ndarray
                if not isinstance(self.alt_model, np.ndarray):
                    raw_regimes = self.alt_model.predict(X)
                    regimes = pd.Series(raw_regimes, index=returns.index)
                    return regimes
            raise RuntimeError("RandomForest model is not initialized or missing predict method.")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _generate_dynamic_labels(self, returns: pd.Series, regimes: pd.Series) -> List[str]:
        """Generate regime labels dynamically based on selected metric"""
        metric = self.config.labeling_metric
        regime_metrics = {}
        for i in range(self.config.n_regimes):
            mask = regimes == i
            if metric == 'volatility':
                regime_metrics[i] = returns[mask].std()
            elif metric == 'sharpe':
                r = returns[mask]
                regime_metrics[i] = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else float('-inf')
            elif metric == 'drawdown':
                r = returns[mask]
                cumulative = (1 + r).cumprod()
                drawdown = (cumulative / cumulative.cummax() - 1).min() if not cumulative.empty else 0  # type: ignore
                regime_metrics[i] = drawdown
            else:
                regime_metrics[i] = returns[mask].std()
        # Sort regimes by metric (ascending for volatility, descending for sharpe, ascending for drawdown)
        if metric == 'sharpe':
            sorted_regimes = sorted(regime_metrics.items(), key=lambda x: -x[1])
        else:
            sorted_regimes = sorted(regime_metrics.items(), key=lambda x: x[1])
        
        # Create proper labels for the number of regimes
        if self.config.n_regimes == 3:
            base_labels = ["Low Vol", "Medium Vol", "High Vol"]
        elif self.config.n_regimes == 4:
            base_labels = ["Very Low Vol", "Low Vol", "Medium Vol", "High Vol"]
        elif self.config.n_regimes == 5:
            base_labels = ["Very Low Vol", "Low Vol", "Medium Vol", "High Vol", "Very High Vol"]
        else:
            base_labels = [f"Regime {i}" for i in range(self.config.n_regimes)]
        
        labels = [base_labels[i] for i in range(len(sorted_regimes))]
        regime_label_map = {regime_idx: label for (regime_idx, _), label in zip(sorted_regimes, labels)}
        return [regime_label_map.get(i, f"Regime {i}") for i in range(self.config.n_regimes)]

    def _combine_predictions(self, hmm_regimes: pd.Series, lstm_regimes: pd.Series, hmm_probs: pd.DataFrame = None, lstm_probs: pd.DataFrame = None) -> pd.Series:
        method = self.config.ensemble_method
        # Only use DataFrame if not None
        if method == 'majority_vote':
            combined = hmm_regimes.copy()
            for idx in combined.index:
                votes = [hmm_regimes[idx], lstm_regimes[idx]]
                combined[idx] = max(set(votes), key=votes.count)
            return combined
        elif method == 'weighted_average' and hmm_probs is not None and lstm_probs is not None:
            avg_probs = (hmm_probs + lstm_probs) / 2  # type: ignore
            if hasattr(avg_probs, 'idxmax'):
                return avg_probs.idxmax(axis=1)
            else:
                return pd.Series(np.argmax(np.asarray(avg_probs), axis=1), index=hmm_probs.index)
        else:  # Default: confidence_blend
            if hmm_probs is not None and lstm_probs is not None:
                avg_probs = (hmm_probs + lstm_probs) / 2  # type: ignore
                if hasattr(avg_probs, 'idxmax'):
                    return avg_probs.idxmax(axis=1)
                else:
                    return pd.Series(np.argmax(np.asarray(avg_probs), axis=1), index=hmm_probs.index)
            agreement = hmm_regimes == lstm_regimes
            combined = hmm_regimes.copy()
            combined[~agreement] = np.maximum(hmm_regimes[~agreement], lstm_regimes[~agreement])
            return combined

    def _should_switch_regime(self, current_regime, new_regime, confidence, flip_history, threshold=None, max_flips=None):
        if threshold is None:
            threshold = self.config.min_confidence
        if max_flips is None:
            max_flips = self.config.max_flips
        if new_regime != current_regime:
            recent_flips = sum(flip_history[-self.config.transition_window:]) if flip_history is not None else 0
            if confidence < threshold or recent_flips > max_flips:
                return False
        return True

    def dynamic_regime_assignment(self, returns: pd.Series) -> pd.Series:
        """
        Assign regimes dynamically using the selected method in config.
        Methods: 'rolling_quantile', 'rolling_kmeans'
        Returns a pd.Series of regime labels.
        """
        method = self.config.dynamic_assignment_method
        window = self.config.dynamic_assignment_window
        n_regimes = self.config.n_regimes
        if method == 'rolling_quantile':
            # Use rolling quantiles of volatility to assign regimes
            vol = returns.rolling(window).std()
            quantiles = [vol.quantile(q) for q in np.linspace(0, 1, n_regimes + 1)]
            regime = pd.Series(index=returns.index, dtype=int)
            for i in range(n_regimes):
                mask = (vol >= quantiles[i]) & (vol < quantiles[i+1])
                regime[mask] = i
            regime = regime.fillna(method='bfill').fillna(method='ffill').astype(int)
            return regime
        elif method == 'rolling_kmeans':
            # Use rolling KMeans clustering on volatility
            from sklearn.cluster import KMeans
            vol = returns.rolling(window).std().fillna(0)
            regime = pd.Series(index=returns.index, dtype=int)
            for i in range(window, len(returns)):
                window_vol = vol.iloc[i-window:i].values.reshape(-1, 1)
                if np.all(window_vol == 0):
                    regime.iloc[i] = 0
                    continue
                kmeans = KMeans(n_clusters=n_regimes, random_state=self.config.random_state, n_init='auto')
                labels = kmeans.fit_predict(window_vol)
                regime.iloc[i] = labels[-1]
            regime = regime.fillna(method='bfill').fillna(method='ffill').astype(int)
            return regime
        else:
            raise ValueError(f"Unknown dynamic_assignment_method: {method}")

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Enhanced fit and predict with all model types, dynamic labeling, improved ensemble, and dynamic regime assignment if enabled."""
        if self.config.dynamic_assignment_method is not None:
            regimes = self.dynamic_regime_assignment(returns)
            self.regime_labels = self._generate_dynamic_labels(returns, regimes)
            return regimes
        self.fit(returns)
        regimes = self.predict(returns)
        self.regime_labels = self._generate_dynamic_labels(returns, regimes)
        if self.config.use_deep_learning and self.lstm_model is not None:
            try:
                regimes_np = np.asarray(regimes)
                self.lstm_model.fit(returns, regimes_np.astype(int))
                lstm_regimes = self.lstm_model.predict(returns)
                hmm_probs = self.predict_proba(returns)
                lstm_probs = None
                if hasattr(self.lstm_model, 'predict_proba'):
                    try:
                        lstm_probs = self.lstm_model.predict_proba(returns)
                        if hasattr(lstm_probs, 'columns') and hasattr(hmm_probs, 'columns') and list(lstm_probs.columns) != list(hmm_probs.columns):
                            lstm_probs.columns = hmm_probs.columns  # type: ignore
                    except Exception:
                        lstm_probs = None
                combined_regimes = self._combine_predictions(regimes, lstm_regimes, hmm_probs, lstm_probs)  # type: ignore
                self.regime_labels = self._generate_dynamic_labels(returns, combined_regimes)
                return combined_regimes
            except Exception as e:
                print(f"Deep learning prediction failed: {str(e)}")
                print("Falling back to main model predictions")
                return regimes
        return regimes

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return transition probability matrix"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting transition matrix")
        if self.hmm_model is not None and hasattr(self.hmm_model, 'transmat_'):
            index = pd.Index(self.regime_labels)
            trans_mat = pd.DataFrame(
                self.hmm_model.transmat_,
                index=index,
                columns=index,
            )
            return trans_mat
        else:
            raise RuntimeError("HMM model is not initialized or missing transmat_.")

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
        log_likelihood = None
        if self.hmm_model is not None and hasattr(self.hmm_model, 'score'):
            log_likelihood = self.hmm_model.score(X)

        # 4. AIC and BIC
        n_params = (self.config.n_regimes * self.config.n_regimes - 1) + (
            self.config.n_regimes * 2
        )  # transition probs + means/vars
        n_samples = len(returns)
        aic = -2 * log_likelihood + 2 * n_params if log_likelihood is not None else None
        bic = -2 * log_likelihood + np.log(n_samples) * n_params if log_likelihood is not None else None

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

    def predict_proba(self, returns: pd.Series) -> pd.DataFrame:
        """Predict regime probabilities for each state"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        start_time = time.time()
        try:
            X = self._prepare_features(returns)
            # Get state probabilities from HMM
            if self.hmm_model is not None and hasattr(self.hmm_model, 'predict_proba'):
                hmm_probs = self.hmm_model.predict_proba(X)
            else:
                raise RuntimeError("HMM model is not initialized or missing predict_proba.")
            # Create DataFrame with proper index
            if not hasattr(self, 'regime_labels') or self.regime_labels is None:
                self.regime_labels = [f"Regime_{i}" for i in range(self.config.n_regimes)]
            
            probs_df = pd.DataFrame(
                hmm_probs,
                index=pd.Index(returns.index),
                columns=pd.Index(self.regime_labels)
            )
            # Safety check for all-NaN probabilities
            if probs_df.dropna(how="all").empty:
                logger.error("All regime probabilities are NaN. Verify model output and input features.")
                raise ValueError("All probabilities are NaN. Verify model output and input features.")
            if self.config.use_deep_learning and self.lstm_model is not None:
                try:
                    # Get LSTM probabilities
                    lstm_probs = self.lstm_model.predict_proba(returns)
                    # Combine probabilities (simple average)
                    if hasattr(lstm_probs, 'columns') and hasattr(probs_df, 'columns') and list(lstm_probs.columns) != list(probs_df.columns):
                        lstm_probs.columns = probs_df.columns
                    probs_df = (probs_df + lstm_probs) / 2
                except Exception as e:
                    logger.warning(f"LSTM prediction failed, using HMM only: {str(e)}")
            prediction_time = time.time() - start_time
            self._performance.prediction_times.append(prediction_time)
            return probs_df
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def update_real_time(self, new_returns: pd.Series, mode: str = 'auto') -> Dict:
        """
        Update regime detection in real-time with new data, with transition penalty logic.
        mode: 'auto' (default) - use HMM if fitted, else LSTM if fitted
              'hmm' - force HMM
              'lstm' - force LSTM
        """
        start_time = time.time()
        self._performance.total_updates += 1
        try:
            # Input validation
            if not isinstance(new_returns, pd.Series):
                raise ValueError("new_returns must be a pandas Series")
            # Select model
            use_hmm = (mode == 'hmm') or (mode == 'auto' and self._is_fitted)
            use_lstm = (mode == 'lstm') or (mode == 'auto' and not self._is_fitted and self.lstm_model is not None and getattr(self.lstm_model, '_is_fitted', False))
            if not use_hmm and not use_lstm:
                raise RuntimeError("No fitted model available for real-time update (HMM or LSTM)")
            if len(new_returns) < self.config.window_size:
                raise ValueError(f"Input data must have at least {self.config.window_size} observations")
            if new_returns.isnull().any():
                raise RuntimeError("Input data contains missing values")
            if new_returns.index.duplicated().any():
                raise RuntimeError("Input data contains duplicate timestamps")
            if self._last_update_time is not None:
                # Only compare if types are compatible
                if isinstance(new_returns.index[-1], pd.Timestamp) and isinstance(self._last_update_time, pd.Timestamp):
                    if pd.Timestamp(new_returns.index[-1]) <= pd.Timestamp(self._last_update_time):
                        raise ValueError("New data must be more recent than last update")
            if use_hmm:
                # Predict regime probabilities with HMM
                probs = self.predict_proba(new_returns)
                current_probs = probs.iloc[-1]
                current_regime = self.regime_labels[int(np.argmax(np.asarray(current_probs)))]
                confidence = np.max(np.asarray(current_probs))
            elif use_lstm:
                # Predict regime probabilities with LSTM
                if self.lstm_model is not None:
                    current_probs = self.lstm_model.predict_latest(new_returns)
                    current_regime = self.regime_labels[int(np.argmax(current_probs))]
                    confidence = np.max(current_probs)
                else:
                    raise RuntimeError("LSTM model is not initialized.")
            else:
                raise RuntimeError("No fitted model available for real-time update (HMM or LSTM)")
            # Check confidence threshold
            if confidence < self.config.min_confidence:
                current_regime = "Uncertain"
                logger.warning(f"Low confidence regime detection: {confidence:.2f}")
            # Transition penalty logic (only for HMM)
            flip_history = getattr(self, '_flip_history', [])
            regime_change = (self._current_regime is not None and current_regime != self._current_regime)
            suppress_switch = False
            if use_hmm and regime_change:
                if not self._should_switch_regime(self._current_regime, current_regime, confidence, flip_history, threshold=self.config.min_confidence, max_flips=self.config.max_flips):
                    suppress_switch = True
            if suppress_switch:
                current_regime = self._current_regime
            # Track flip history
            if not hasattr(self, '_flip_history'):
                self._flip_history = []
            self._flip_history.append(int(regime_change and not suppress_switch))
            if len(self._flip_history) > 100:
                self._flip_history = self._flip_history[-100:]
            # Calculate transition probabilities if we have a previous regime (HMM only)
            transition_alert = None
            if use_hmm and self._current_regime is not None and self._current_regime != "Uncertain":
                trans_mat = self.get_transition_matrix()
                current_idx = list(trans_mat.index).index(self._current_regime)
                transition_probs = trans_mat.iloc[current_idx]
                for next_regime, prob in transition_probs.items():
                    if (prob > self.config.alert_threshold and next_regime != self._current_regime):
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
            }, index=pd.Index([self._last_update_time]))
            self._regime_history = pd.concat([self._regime_history, history_entry])  # type: ignore
            # Maintain history size limit
            self._regime_history = self._regime_history.iloc[-self.config.history_size:]  # type: ignore
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
                'probabilities': current_probs if use_hmm else current_probs.tolist(),
                'timestamp': self._last_update_time,
                'data_quality': {
                    'sample_size': len(new_returns),
                    'missing_values': 0,
                    'duplicates': 0
                },
                'performance': {
                    'success_rate': (self._performance.successful_updates /
                                   self._performance.total_updates)
                },
                'model_used': 'HMM' if use_hmm else 'LSTM'
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
            if hasattr(self._regime_history, 'iloc'):
                # If iloc is on ndarray, convert to Series first
                if isinstance(self._regime_history, np.ndarray):
                    return pd.Series(self._regime_history).iloc[-lookback_periods:]
                return self._regime_history.iloc[-lookback_periods:]
            else:
                return pd.DataFrame(self._regime_history)
        if isinstance(self._regime_history, pd.DataFrame):
            return self._regime_history
        else:
            return pd.DataFrame(self._regime_history)

    def get_confidence_metrics(self) -> Dict:
        """Calculate confidence metrics for current regime detection
        
        Returns:
            Dict: Various confidence metrics
        """
        if not self._is_fitted or self._current_regime is None:
            raise ValueError("Model must be fitted and have current regime")
            
        # Get recent history
        if hasattr(self._regime_history, 'iloc'):
            if isinstance(self._regime_history, np.ndarray):
                recent_history = pd.Series(self._regime_history).iloc[-self.config.window_size:]
            else:
                recent_history = self._regime_history.iloc[-self.config.window_size:]
        else:
            recent_history = pd.DataFrame(self._regime_history)
        
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
        self._regime_history = pd.DataFrame(columns=['regime', 'confidence', 'transition_probability'])  # type: ignore
        self._performance = PerformanceMetrics()
        
        # Reset state
        self._current_regime = None
        self._last_update_time = None
        self._is_fitted = False
        
        # Clear model parameters
        if self.hmm_model is not None:
            if hasattr(self.hmm_model, 'means_'):
                try:
                    del self.hmm_model.means_
                except Exception:
                    pass
            if hasattr(self.hmm_model, 'covars_'):
                try:
                    del self.hmm_model.covars_
                except Exception:
                    pass
            if hasattr(self.hmm_model, 'transmat_'):
                try:
                    del self.hmm_model.transmat_
                except Exception:
                    pass
        
        # Clear LSTM model if present
        if self.lstm_model is not None:
            self.lstm_model = None
            
        # Clear Transformer model if present
        if self.transformer_model is not None:
            self.transformer_model = None
            
        logger.info("Cleaned up MarketRegimeDetector resources")
        
    def trim_history(self, max_age_days: Optional[int] = None):
        """Trim regime history to manage memory
        
        Parameters:
            max_age_days (int, optional): Maximum age of history entries in days
        """
        if max_age_days is not None:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
            self._regime_history = self._regime_history[self._regime_history.index > cutoff_date]  # type: ignore
        
        # Ensure we don't exceed history size limit
        self._regime_history = self._regime_history.iloc[-self.config.history_size:]  # type: ignore
            
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
        
        # Add Transformer model size if present
        if self.transformer_model is not None:
            try:
                transformer_size = sys.getsizeof(self.transformer_model) / (1024 * 1024)
                memory_usage['transformer_model'] = transformer_size
            except:
                memory_usage['transformer_model'] = 0
        
        memory_usage['total'] = sum(memory_usage.values())
        return memory_usage

    def save_model(self, path: str):
        """Save HMM, scaler, and LSTM model (if present) to disk"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.hmm_model, os.path.join(path, "hmm.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.config, os.path.join(path, "config.pkl"))
        if self.lstm_model is not None and hasattr(self.lstm_model, 'save'):
            self.lstm_model.save(os.path.join(path, "lstm/"))
        if self.transformer_model is not None and hasattr(self.transformer_model, 'save'):
            self.transformer_model.save(os.path.join(path, "transformer/"))

    def load_model(self, path: str):
        """Load HMM, scaler, and LSTM model (if present) from disk"""
        self.hmm_model = joblib.load(os.path.join(path, "hmm.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.config = joblib.load(os.path.join(path, "config.pkl"))
        if self.config.use_deep_learning and self.lstm_model is not None and hasattr(self.lstm_model, 'load'):
            self.lstm_model.load(os.path.join(path, "lstm/"))
        self._is_fitted = True
        if self.transformer_model is not None and hasattr(self.transformer_model, 'load'):
            self.transformer_model.load(os.path.join(path, "transformer/"))

    def get_transition_probabilities_over_time(self, returns: pd.Series) -> pd.DataFrame:
        """Return a DataFrame of transition probabilities for each time step"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting transition probabilities")
        X = self._prepare_features(returns)
        if self.hmm_model is not None and hasattr(self.hmm_model, 'predict_proba') and hasattr(self.hmm_model, 'transmat_'):
            posteriors = self.hmm_model.predict_proba(X)
            n = len(posteriors)
            trans_probs = []
            for t in range(1, n):
                prev = posteriors[t-1]
                curr = posteriors[t]
                # Estimate transition as dot product with transition matrix
                trans = prev @ self.hmm_model.transmat_
                trans_probs.append(trans)
            # Use list(returns.index[1:]) for index
            idx = [int(i) if isinstance(i, (np.integer, int)) else i for i in list(returns.index[1:])]
            cols = [str(l) for l in list(self.regime_labels)]
            df = pd.DataFrame(trans_probs, index=idx, columns=cols)
            return df
        else:
            return pd.DataFrame([])
    def plot_transition_probabilities(self, returns: pd.Series, ax=None):
        """Plot transition probabilities over time"""
        import matplotlib.pyplot as plt
        df = self.get_transition_probabilities_over_time(returns)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        df.plot(ax=ax)
        ax.set_title('Regime Transition Probabilities Over Time')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Time')
        plt.tight_layout()
        return ax

    def fit_predict_batch(self, returns_df: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
        """Run regime detection in parallel for multiple assets (columns)"""
        from joblib import Parallel, delayed
        def process_col(col):
            # Only pass Series to fit_predict
            series = returns_df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return self.fit_predict(series)
        results = Parallel(n_jobs=n_jobs)(delayed(process_col)(col) for col in returns_df.columns)
        regimes_df = pd.DataFrame({col: res for col, res in zip(returns_df.columns, results)}, index=returns_df.index)  # type: ignore
        # --- LSTM real-time update fix: fit main detector's LSTM on first asset ---
        if self.lstm_model is not None and not getattr(self.lstm_model, '_is_fitted', False):
            first_asset = returns_df.columns[0]
            returns = returns_df[first_asset]
            regimes = regimes_df[first_asset]
            # Filter out NaN regimes
            valid_mask = ~regimes.isna()
            returns_valid = returns[valid_mask]
            regimes_valid = regimes[valid_mask]
            # Map string regime labels to integer indices if needed
            if hasattr(regimes_valid, 'map'):
                label_to_idx = {label: idx for idx, label in enumerate(self.regime_labels)}
                regimes_int = regimes_valid.map(label_to_idx)
            else:
                regimes_int = regimes_valid
            hmm_regimes_np = np.asarray(regimes_int)
            # Only fit if enough samples remain
            if len(returns_valid) > self.lstm_model.config.sequence_length:
                try:
                    if isinstance(returns_valid, pd.Series):
                        self.lstm_model.fit(returns_valid, hmm_regimes_np.astype(int))
                except Exception as e:
                    logger.warning(f"LSTM fit for real-time update failed: {e}")
            else:
                logger.warning("Not enough valid samples to fit LSTM for real-time update.")
        return regimes_df if isinstance(regimes_df, pd.DataFrame) else pd.DataFrame(regimes_df)

def tune_regime_parameters(
    returns: pd.Series,
    param_grid: Optional[Dict[str, List]] = None,
    scoring_func: Optional[Callable[[Dict], float]] = None,
    verbose: bool = True,
    use_random: bool = False,
    n_iter: int = 10,
) -> pd.DataFrame:
    """
    Tune regime detection parameters using grid or random search.
    Args:
        returns: pd.Series of returns
        param_grid: Dict of parameter lists (n_regimes, window_size, smoothing_window)
        scoring_func: Function to score validation dict (higher is better)
        verbose: Print progress
        use_random: If True, use random search
        n_iter: Number of random samples if use_random
    Returns:
        pd.DataFrame with parameter sets and validation metrics
    """
    import random
    if param_grid is None:
        param_grid = {
            'n_regimes': [2, 3, 4],
            'window_size': [10, 21, 42],
            'smoothing_window': [3, 5, 7],
        }
    keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    if use_random:
        all_combinations = random.sample(all_combinations, min(n_iter, len(all_combinations)))
    results = []
    for combo in all_combinations:
        params = dict(zip(keys, combo))
        config = RegimeConfig(
            n_regimes=params['n_regimes'],
            window_size=params['window_size'],
            smoothing_window=params['smoothing_window'],
        )
        detector = MarketRegimeDetector(config)
        try:
            regimes = detector.fit_predict(returns)
            validation = detector.validate_model(returns, regimes)
            score = scoring_func(validation) if scoring_func else validation.get('regime_persistence', 0)
            result = {**params, **validation, 'score': score}
            if verbose:
                print(f"Params: {params}, Score: {score:.4f}")
        except Exception as e:
            result = {**params, 'error': str(e)}
            if verbose:
                print(f"Params: {params}, Error: {e}")
        results.append(result)
    return pd.DataFrame(results)

import matplotlib.pyplot as plt

def compare_regime_models(
    returns: pd.Series,
    model_configs: list,
    plot: bool = True,
    metrics: list = None,
) -> pd.DataFrame:
    """
    Compare multiple regime detection models/configs on the same returns.
    Args:
        returns: pd.Series of returns
        model_configs: list of RegimeConfig objects
        plot: If True, plot regime assignments
        metrics: List of validation metrics to aggregate (default: all)
    Returns:
        pd.DataFrame with validation metrics for each model
    """
    if metrics is None:
        metrics = []
    results = []
    regimes_dict = {}
    for config in model_configs:
        detector = MarketRegimeDetector(config)
        regimes = detector.fit_predict(returns)
        regimes_dict[config.model_type] = regimes
        validation = detector.validate_model(returns, regimes)
        row = {'model_type': config.model_type}
        row.update(validation)
        results.append(row)
    df = pd.DataFrame(results)
    if plot:
        fig, ax = plt.subplots(figsize=(12, 3))
        for model_type, regimes in regimes_dict.items():
            ax.plot(regimes.index, regimes, label=model_type, alpha=0.7)
        ax.set_title('Regime Assignments by Model')
        ax.set_ylabel('Regime')
        ax.legend()
        plt.tight_layout()
        plt.show()
    return df

class AdaptiveRegimeDetector:
    """Enhanced adaptive regime detection with quarterly retraining and confidence-based sizing"""
    
    def __init__(self, base_config: RegimeConfig, retrain_window: int = 252, retrain_freq: int = 63):
        """
        Initialize enhanced adaptive regime detector
        
        Args:
            base_config: Base configuration for regime detection
            retrain_window: Number of days for training window (default: 252 = 1 year)
            retrain_freq: Frequency of retraining in days (default: 63 = quarterly)
        """
        self.base_config = base_config
        self.retrain_window = retrain_window
        self.retrain_freq = retrain_freq
        self.detectors = {}  # Store detectors for each retraining period
        self.regime_history = {}  # Store regime predictions for each period
        self.confidence_history = {}  # Store confidence scores
        self.performance_history = {}  # Store model performance metrics
        self.current_detector = None
        self.initialized = False
        
    def adaptive_walkforward_backtest(self, train_data: pd.DataFrame, test_data: pd.DataFrame, retrain_freq: int = 63):
        """
        The CRITICAL fix: Retrain every quarter during test period
        
        Args:
            train_data: Initial training data (2020-2021)
            test_data: Test data for adaptive backtesting (2022-2023)
            retrain_freq: Days between retraining (default: 63 = quarterly)
            
        Returns:
            List of results for each testing period
        """
        results = []
        
        # Initial training on 2020-2021 data
        logger.info("Initial training on 2020-2021 data...")
        self.fit(train_data)
        
        # Walk forward with retraining every 63 days (quarterly)
        for i in range(0, len(test_data), retrain_freq):
            period_start = i
            period_end = min(i + retrain_freq, len(test_data))
            
            logger.info(f"Testing period {len(results) + 1}: {test_data.index[period_start]} to {test_data.index[period_end-1]}")
            
            # CRITICAL: Retrain with recent data
            if i >= 126:  # After 6 months, start retraining
                # Use expanding window: original training + test data so far
                retrain_data = pd.concat([
                    train_data.iloc[-126:],  # Last 6 months of training
                    test_data.iloc[max(0, i-252):i]  # Last year of test data
                ])
                
                logger.info(f"Retraining on {len(retrain_data)} days of data (period {len(results) + 1})...")
                
                # Retrain both HMM and LSTM models
                self.fit(retrain_data)
            
            # Generate predictions for next quarter
            period_data = test_data.iloc[period_start:period_end]
            period_results = self.predict_and_evaluate(period_data)
            results.append(period_results)
            
            logger.info(f"Period {len(results)} completed. Regimes detected: {len(period_results['regimes'].unique())}")
        
        return results

    def adaptive_backtest(self, train_data: pd.DataFrame, test_data: pd.DataFrame, retrain_frequency: int = 63):
        """
        Backtest with adaptive retraining every quarter (legacy method)
        
        Args:
            train_data: Initial training data
            test_data: Test data for adaptive backtesting
            retrain_frequency: Days between retraining (default: 63 = quarterly)
            
        Returns:
            List of results for each testing period
        """
        # Use the new walkforward method
        return self.adaptive_walkforward_backtest(train_data, test_data, retrain_frequency)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the regime detector on training data"""
        if self.current_detector is None:
            self.current_detector = MarketRegimeDetector(self.base_config)
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Fit the detector
        self.current_detector.fit(features)
        self.initialized = True
        
        logger.info(f"Fitted regime detector on {len(data)} days of data")
    
    def predict_and_evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict regimes and evaluate performance for a period
        
        Args:
            data: Data for prediction and evaluation
            
        Returns:
            Dictionary with predictions, confidence, and performance metrics
        """
        if not self.initialized:
            raise ValueError("Detector must be fitted before prediction")
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Get regime predictions for the first asset (portfolio-level)
        portfolio_returns = features['returns']
        regimes = self.current_detector.predict(portfolio_returns)
        
        # Calculate regime probabilities and confidence
        try:
            regime_probs = self.current_detector.predict_proba(portfolio_returns)
            confidence = self._calculate_regime_confidence(regime_probs)
        except Exception as e:
            logger.warning(f"Could not calculate regime probabilities: {e}")
            # Use default confidence
            confidence = pd.Series(0.5, index=data.index)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(data, regimes, confidence)
        
        return {
            'regimes': regimes,
            'confidence': confidence,
            'regime_probs': regime_probs if 'regime_probs' in locals() else None,
            'performance': performance,
            'period_start': data.index[0],
            'period_end': data.index[-1]
        }
    
    def confidence_based_sizing(self, signals: pd.DataFrame, regime_probs: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce position size when regime detection is uncertain
        
        Args:
            signals: Raw trading signals
            regime_probs: Regime probability matrix
            
        Returns:
            Adjusted signals with confidence-based sizing
        """
        # Calculate regime confidence (higher when one regime dominates)
        max_prob = regime_probs.max(axis=1)
        confidence = max_prob
        
        # Scale positions by confidence
        base_size = 0.1  # 10% base allocation
        adjusted_signals = signals * base_size * confidence
        
        # Apply position bounds
        adjusted_signals = adjusted_signals.clip(-0.5, 0.5)  # -50% to +50%
        
        logger.info(f"Applied confidence-based sizing. Average confidence: {confidence.mean():.3f}")
        logger.info(f"Position size range: {adjusted_signals.min().min():.3f} to {adjusted_signals.max().max():.3f}")
        
        return adjusted_signals
    
    def _calculate_regime_confidence(self, regime_probs: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on regime probability distribution"""
        if regime_probs is None or regime_probs.empty:
            # Return default confidence if no probabilities available
            return pd.Series(0.5, index=regime_probs.index if regime_probs is not None else pd.DatetimeIndex([]))
        
        # Higher confidence when one regime dominates
        max_probs = regime_probs.max(axis=1)
        
        # Normalize to 0-1 range
        confidence = (max_probs - 1/regime_probs.shape[1]) / (1 - 1/regime_probs.shape[1])
        confidence = confidence.clip(0, 1)
        
        return confidence
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, regimes: pd.Series, confidence: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for the period"""
        # Calculate returns if not already present
        if 'returns' not in data.columns:
            returns = data.pct_change().dropna()
        else:
            returns = data['returns']
        
        # Regime stability metrics
        regime_changes = (regimes != regimes.shift()).sum()
        regime_stability = 1.0 / (1.0 + regime_changes)
        
        # Confidence metrics
        avg_confidence = confidence.mean()
        low_confidence_pct = (confidence < 0.5).mean()
        high_confidence_pct = (confidence > 0.7).mean()
        
        # Return metrics - handle multi-asset data
        if isinstance(returns, pd.DataFrame):
            # For multi-asset data, calculate portfolio metrics
            portfolio_returns = returns.mean(axis=1)
            period_return = (1 + portfolio_returns).prod() - 1
            period_vol = portfolio_returns.std() * np.sqrt(252)
        else:
            # Single asset data
            period_return = (1 + returns).prod() - 1
            period_vol = returns.std() * np.sqrt(252)
        
        sharpe_ratio = period_return / period_vol if period_vol > 0 else 0
        
        return {
            'regime_stability': regime_stability,
            'avg_confidence': avg_confidence,
            'low_confidence_pct': low_confidence_pct,
            'high_confidence_pct': high_confidence_pct,
            'period_return': period_return,
            'period_vol': period_vol,
            'sharpe_ratio': sharpe_ratio,
            'regime_changes': regime_changes
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Handle multi-asset data by taking the mean across assets
        if len(data.columns) > 1:
            # For multi-asset data, use portfolio returns
            returns = data.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)
        else:
            # Single asset data
            if 'returns' in data.columns:
                portfolio_returns = data['returns']
            else:
                portfolio_returns = data.pct_change().dropna()
        
        features['returns'] = portfolio_returns
        
        # Volatility feature
        features['volatility'] = features['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Add macro features if available
        if 'vix' in data.columns:
            features['vix'] = data['vix']
        if 'yield_spread' in data.columns:
            features['yield_spread'] = data['yield_spread']
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def get_adaptive_metrics(self, data: pd.DataFrame, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive adaptive metrics"""
        metrics = {}
        
        # Transition analysis
        transition_counts = {}
        persistence_scores = {}
        regime_distributions = {}
        
        for asset in regimes.columns:
            asset_regimes = regimes[asset]
            
            # Count transitions
            transitions = (asset_regimes != asset_regimes.shift()).sum()
            transition_counts[asset] = transitions
            
            # Calculate persistence (inverse of transitions)
            persistence = 1.0 / (1.0 + transitions)
            persistence_scores[asset] = persistence
            
            # Regime distribution
            regime_dist = asset_regimes.value_counts(normalize=True)
            regime_distributions[asset] = regime_dist
        
        metrics['transition_counts'] = transition_counts
        metrics['avg_transitions_per_asset'] = np.mean(list(transition_counts.values()))
        metrics['persistence_scores'] = persistence_scores
        metrics['avg_persistence'] = np.mean(list(persistence_scores.values()))
        metrics['regime_distributions'] = regime_distributions
        metrics['num_retraining_periods'] = len(self.detectors)
        metrics['avg_window_size'] = self.retrain_window
        metrics['retrain_frequency'] = self.retrain_freq
        
        return metrics
    
    def get_regime_confidence(self, data: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
        """Get regime confidence scores"""
        if not self.initialized or self.current_detector is None:
            # Return default confidence if detector not available
            return pd.DataFrame(0.5, index=data.index, columns=regimes.columns)
        
        features = self._prepare_features(data)
        confidence = pd.DataFrame(index=data.index, columns=regimes.columns)
        
        for asset in regimes.columns:
            try:
                # Get regime probabilities for this asset
                asset_features = features.copy()
                if asset in data.columns:
                    asset_features['returns'] = data[asset].pct_change()
                    asset_features['volatility'] = asset_features['returns'].rolling(window=21).std() * np.sqrt(252)
                
                asset_features = asset_features.fillna(method='ffill').fillna(0)
                
                # Get probabilities and calculate confidence
                probs = self.current_detector.predict_proba(asset_features)
                max_probs = probs.max(axis=1)
                asset_confidence = (max_probs - 1/probs.shape[1]) / (1 - 1/probs.shape[1])
                asset_confidence = asset_confidence.clip(0, 1)
                
                confidence[asset] = asset_confidence
                
            except Exception as e:
                logger.warning(f"Could not calculate confidence for {asset}: {e}")
                confidence[asset] = 0.5
        
        return confidence
    
    def dynamic_position_sizing(self, signals: pd.DataFrame, confidence: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic position sizing based on regime confidence and volatility"""
        logger.info("Applying dynamic position sizing based on regime confidence and volatility")
        
        # Base position size
        base_size = 0.1  # 10% base allocation
        
        # Confidence adjustment (reduce size when uncertain)
        confidence_adj = confidence.clip(0.3, 1.0)
        
        # Volatility targeting (12% annual target)
        vol_target = 0.12
        vol_adj = vol_target / (volatility + 1e-6)  # Avoid division by zero
        vol_adj = vol_adj.clip(0.5, 2.0)  # Limit volatility adjustment
        
        # Apply adjustments
        adjusted_positions = signals * base_size * confidence_adj * vol_adj
        
        # Apply position bounds
        adjusted_positions = adjusted_positions.clip(-0.5, 0.5)
        
        logger.info(f"Dynamic position sizing completed. Shape: {adjusted_positions.shape}")
        
        return adjusted_positions

    def adaptive_regime_detection(self, returns_df: pd.DataFrame, retrain_frequency: int = 63) -> pd.DataFrame:
        """
        Run adaptive regime detection with rolling window retraining
        
        Args:
            returns_df: DataFrame with asset returns
            retrain_frequency: Days between retraining (default: 63 = quarterly)
            
        Returns:
            DataFrame with regime predictions for each asset
        """
        logger.info(f"Starting adaptive regime detection on {returns_df.shape[1]} assets")
        
        # Initialize results DataFrame
        regimes_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        # Split data into training and testing periods
        train_size = min(self.retrain_window, len(returns_df) // 2)
        train_data = returns_df.iloc[:train_size]
        test_data = returns_df.iloc[train_size:]
        
        logger.info(f"Training on {len(train_data)} days, testing on {len(test_data)} days")
        
        # Initial training
        self.fit(train_data)
        
        # Process each asset
        for asset in returns_df.columns:
            logger.info(f"Processing asset: {asset}")
            
            # Get asset-specific data
            asset_returns = returns_df[asset]
            
            # Initial prediction on training data
            train_regimes = self.current_detector.fit_predict(asset_returns.iloc[:train_size])
            regimes_df.loc[train_data.index, asset] = train_regimes
            
            # Adaptive prediction on test data with retraining
            for i in range(0, len(test_data), retrain_frequency):
                period_start = train_size + i
                period_end = min(period_start + retrain_frequency, len(returns_df))
                
                if period_start >= len(returns_df):
                    break
                
                # Retrain if not the first period
                if i > 0:
                    # Use expanding window for retraining
                    retrain_end = period_start
                    retrain_start = max(0, retrain_end - self.retrain_window)
                    retrain_data = returns_df.iloc[retrain_start:retrain_end]
                    
                    logger.info(f"Retraining on {len(retrain_data)} days for period {i//retrain_frequency + 1}")
                    self.fit(retrain_data)
                
                # Predict on current period
                period_data = asset_returns.iloc[period_start:period_end]
                if len(period_data) > 0:
                    period_regimes = self.current_detector.predict(period_data)
                    regimes_df.loc[period_data.index, asset] = period_regimes
        
        logger.info("Adaptive regime detection completed")
        return regimes_df

    def calculate_regime_confidence(self, regime_probabilities: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence based on regime probability distribution
        
        Args:
            regime_probabilities: DataFrame with regime probabilities
            
        Returns:
            Series with confidence scores (0-1)
        """
        # Higher confidence when one regime dominates
        max_prob = regime_probabilities.max(axis=1)
        
        # Calculate entropy (lower entropy = higher confidence)
        entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
        normalized_entropy = entropy / np.log(len(regime_probabilities.columns))
        
        # Confidence score (0-1)
        confidence = max_prob * (1 - normalized_entropy)
        
        return confidence

    def adaptive_position_sizing(self, base_signals: pd.DataFrame, regime_confidence: pd.Series, market_volatility: pd.Series) -> pd.DataFrame:
        """
        Scale position size based on regime confidence and market conditions
        
        Args:
            base_signals: Raw trading signals
            regime_confidence: Confidence scores from regime detection
            market_volatility: Market volatility series
            
        Returns:
            Adjusted signals with confidence-based sizing
        """
        # Base allocation (e.g., 10% of capital)
        base_allocation = 0.10
        
        # Confidence multiplier (0.2 to 1.0)
        confidence_multiplier = np.clip(regime_confidence, 0.2, 1.0)
        
        # Volatility adjustment (reduce size in high vol periods)
        vol_adjustment = np.clip(0.15 / market_volatility, 0.3, 1.5)
        
        # Final position size
        position_size = base_signals * base_allocation * confidence_multiplier * vol_adjustment
        
        return position_size

    def regime_aware_risk_controls(self, current_regime: str, regime_confidence: float, portfolio_equity: float) -> Dict[str, float]:
        """
        Adjust risk parameters based on regime state
        
        Args:
            current_regime: Current market regime
            regime_confidence: Confidence in regime detection
            portfolio_equity: Current portfolio equity
            
        Returns:
            Dictionary with risk parameters
        """
        if current_regime == 'High Vol' or regime_confidence < 0.7:
            # Reduce exposure during high volatility or uncertainty
            max_position_size = 0.05  # 5% max per position
            portfolio_heat = 0.15     # 15% total portfolio risk
        
        elif current_regime == 'Low Vol' and regime_confidence > 0.8:
            # Increase exposure when confident in low volatility
            max_position_size = 0.15  # 15% max per position
            portfolio_heat = 0.25     # 25% total portfolio risk
        
        else:
            # Normal sizing
            max_position_size = 0.10  # 10% max per position
            portfolio_heat = 0.20     # 20% total portfolio risk
        
        return {
            'max_position_size': max_position_size,
            'portfolio_heat': portfolio_heat,
            'regime': current_regime,
            'confidence': regime_confidence
        }

    def enhanced_confidence_based_sizing(self, signals: pd.DataFrame, regime_probs: pd.DataFrame, volatility: pd.DataFrame = None) -> pd.DataFrame:
        """
        Enhanced confidence-based position sizing with volatility adjustment
        
        Args:
            signals: Raw trading signals
            regime_probs: Regime probability matrix
            volatility: Market volatility (optional)
            
        Returns:
            Adjusted signals with enhanced confidence-based sizing
        """
        # Calculate regime confidence using entropy-based method
        confidence = self.calculate_regime_confidence(regime_probs)
        
        # Base allocation
        base_allocation = 0.10  # 10% base allocation
        
        # Confidence adjustment (reduce size when uncertain)
        confidence_adj = confidence.clip(0.2, 1.0)
        
        # Volatility adjustment if provided
        if volatility is not None:
            vol_target = 0.12  # 12% annual volatility target
            vol_adj = vol_target / (volatility + 1e-6)
            vol_adj = vol_adj.clip(0.3, 2.0)
        else:
            vol_adj = 1.0
        
        # Apply adjustments
        adjusted_signals = signals * base_allocation * confidence_adj * vol_adj
        
        # Apply position bounds
        adjusted_signals = adjusted_signals.clip(-0.5, 0.5)
        
        logger.info(f"Enhanced confidence-based sizing applied. Average confidence: {confidence.mean():.3f}")
        logger.info(f"Position size range: {adjusted_signals.min().min():.3f} to {adjusted_signals.max().max():.3f}")
        
        return adjusted_signals
