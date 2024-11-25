import numpy as np
import pandas as pd
from hmmlearn import hmm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

@dataclass
class RegimeConfig:
    n_regimes: int = 3
    n_iter: int = 100
    random_state: int = 42
    window_size: int = 21
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['returns', 'volatility']
        if self.n_regimes < 2:
            raise ValueError("Number of regimes must be at least 2")

class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.model = hmm.GaussianHMM(
            n_components=config.n_regimes,
            n_iter=1000,
            random_state=42,
            covariance_type="diag",
            init_params="",
            params="stmc"
        )
        self.regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
        
    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for HMM"""
        features = []
        
        if 'returns' in self.config.features:
            # Winsorize returns
            returns_clean = returns.clip(
                lower=returns.quantile(0.01),
                upper=returns.quantile(0.99)
            )
            features.append(returns_clean.values.reshape(-1, 1))
        
        if 'volatility' in self.config.features:
            # Use EWMA volatility with proper fillna
            vol = returns.ewm(span=self.config.window_size).std()
            vol = vol.bfill()  # Using bfill() instead of fillna(method='bfill')
            features.append(vol.values.reshape(-1, 1))
        
        # Combine and standardize features
        X = np.hstack(features)
        X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return X
    
    def _initialize_hmm_params(self, X: np.ndarray):
        """Initialize HMM parameters explicitly"""
        n_samples, n_features = X.shape
        
        # Initialize means using kmeans
        kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
        kmeans.fit(X)
        self.model.means_ = kmeans.cluster_centers_
        
        # Initialize starting probabilities
        self.model.startprob_ = np.ones(self.config.n_regimes) / self.config.n_regimes
        
        # Initialize transition matrix
        self.model.transmat_ = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        # Initialize covariances with correct shape
        variances = np.var(X, axis=0)
        self.model.covars_ = np.tile(variances, (self.config.n_regimes, 1))
    
    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Fit HMM and predict regimes"""
        try:
            # Prepare features
            X = self._prepare_features(returns)
            
            # Initialize parameters
            self._initialize_hmm_params(X)
            
            # Fit model
            self.model.fit(X)
            
            # Predict regimes
            states = self.model.predict(X)
            
            # Map states to regime labels based on volatility
            regimes = pd.Series(states, index=returns.index)
            state_vols = []
            
            for state in range(self.config.n_regimes):
                mask = (regimes == state)
                if mask.any():
                    state_vols.append((state, returns[mask].std()))
            
            state_vols.sort(key=lambda x: x[1])
            regime_map = {
                state: label 
                for (state, _), label in zip(state_vols, self.regime_labels)
            }
            
            return regimes.map(regime_map)
            
        except Exception as e:
            print(f"Error in regime detection: {str(e)}")
            return pd.Series('Medium Vol', index=returns.index)
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Return transition probability matrix"""
        trans_mat = pd.DataFrame(
            self.model.transmat_,
            index=self.regime_labels,
            columns=self.regime_labels
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
            regime_mask = (aligned_regimes == regime)
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_returns) > 0:
                stats[regime] = {
                    'mean_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'frequency': len(regime_returns) / len(aligned_returns),
                    'sharpe': (regime_returns.mean() * 252) / 
                             (regime_returns.std() * np.sqrt(252))
                }
            else:
                stats[regime] = {
                    'mean_return': np.nan,
                    'volatility': np.nan,
                    'frequency': 0.0,
                    'sharpe': np.nan
                }
        
        return stats
    
    def validate_model(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """Validate HMM model performance
        
        Parameters:
            returns (pd.Series): Portfolio returns
            regimes (pd.Series): Predicted regimes
            
        Returns:
            Dict: Validation metrics
        """
        # Ensure alignment
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # 1. Information Ratio by Regime
        ir_by_regime = {}
        for regime in self.regime_labels:
            regime_returns = returns[regimes == regime]
            if len(regime_returns) > 0:
                ir = (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252))
                ir_by_regime[regime] = ir
            else:
                ir_by_regime[regime] = np.nan
        
        # 2. Regime Persistence
        transitions = 0
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != regimes.iloc[i-1]:
                transitions += 1
        persistence = 1 - (transitions / len(regimes))
        
        # 3. Log Likelihood
        X = self._prepare_features(returns)
        log_likelihood = self.model.score(X)
        
        # 4. AIC and BIC
        n_params = (self.config.n_regimes * self.config.n_regimes - 1) + \
                   (self.config.n_regimes * 2)  # transition probs + means/vars
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
            'information_ratio': ir_by_regime,
            'regime_persistence': persistence,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'avg_regime_duration': avg_regime_duration
        }