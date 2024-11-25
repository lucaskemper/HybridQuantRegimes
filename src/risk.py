# src/risk.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

@dataclass
class RiskConfig:
    """Risk management configuration parameters"""
    # Required parameters with defaults
    confidence_level: float = 0.95
    max_drawdown_limit: float = 0.20
    volatility_target: float = 0.15
    
    # Optional parameters with defaults
    weights: Optional[List[float]] = None
    rolling_windows: List[int] = field(default_factory=lambda: [21, 63])
    position_limit: float = 0.40
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.max_drawdown_limit <= 0:
            raise ValueError("Max drawdown limit must be positive")
        if self.volatility_target <= 0:
            raise ValueError("Volatility target must be positive")

class RiskManager:
    """Risk management and analysis"""
    
    def __init__(self, config: RiskConfig, risk_free_rate: float = 0.05, weights: Optional[List[float]] = None):
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.weights = weights
    
    def calculate_metrics(self, returns: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        # Use weights if provided, otherwise equal weight
        weights = (self.config.weights if self.config.weights is not None 
                  else [1/len(returns.columns)] * len(returns.columns))
        portfolio_returns = returns.dot(weights)
        
        metrics = {
            'portfolio_volatility': self._calculate_volatility(portfolio_returns),
            'var_95': self._calculate_var(portfolio_returns, 0.95),
            'var_99': self._calculate_var(portfolio_returns, 0.99),
            'expected_shortfall_95': self._calculate_expected_shortfall(portfolio_returns, 0.95),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
            'rolling_volatility': self._calculate_rolling_volatility(returns),
            'correlation': returns.corr()
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio using specified risk-free rate"""
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
    
    def _calculate_rolling_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatility for multiple windows"""
        rolling_vol = pd.DataFrame(index=returns.index)
        
        # Calculate portfolio returns first
        portfolio_returns = returns.dot(self.weights) if self.weights is not None else returns.mean(axis=1)
        
        # Calculate rolling volatility for different windows
        rolling_vol['21d'] = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol['63d'] = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        
        return rolling_vol