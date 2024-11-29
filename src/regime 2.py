import numpy as np
import pandas as pd
from hmmlearn import hmm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import itertools

@dataclass
class RegimeConfig:
    n_regimes: int = 3
    n_iter: int = 100
    random_state: int = 42
    window_size: int = 21
    
    def __post_init__(self):
        if self.n_regimes < 2:
            raise ValueError("Number of regimes must be at least 2")

class MarketRegimeDetector:
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
    
    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for HMM model"""
        rolling_vol = returns.rolling(window=self.config.window_size).std() * np.sqrt(252)
        features = pd.DataFrame({
            'volatility': rolling_vol,
            'returns': returns
        })
        features = features.fillna(method='ffill').fillna(0)
        return self.scaler.fit_transform(features)
    
    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Fit HMM and predict regimes"""
        features = self._prepare_features(returns)
        
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=self.config.random_state
        )
        
        self.hmm_model.fit(features)
        regimes = pd.Series(
            self.hmm_model.predict(features),
            index=returns.index
        )
        
        # Map regimes based on volatility - Fixed groupby operation
        vol_means = pd.Series(features[:, 0], index=returns.index).groupby(regimes).mean()
        regime_map = {
            i: self.regime_labels[j] 
            for i, j in enumerate(np.argsort(vol_means))
        }
        
        return regimes.map(regime_map)