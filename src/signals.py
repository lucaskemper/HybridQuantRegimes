# src/signals.py
import pandas as pd
import numpy as np
from typing import Dict

class SignalGenerator:
    def __init__(self, lookback_fast: int = 20, lookback_slow: int = 50):
        self.lookback_fast = lookback_fast
        self.lookback_slow = lookback_slow
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals between -1 and 1"""
        returns = market_data['returns']
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        for ticker in returns.columns:
            # Calculate moving averages
            fast_ma = returns[ticker].rolling(window=self.lookback_fast).mean()
            slow_ma = returns[ticker].rolling(window=self.lookback_slow).mean()
            
            # Generate base signals
            signals[ticker] = np.where(fast_ma > slow_ma, 0.5, -0.5)
            
            # Add momentum component
            momentum = returns[ticker].rolling(window=20).mean()
            signals[ticker] += np.clip(momentum, -0.5, 0.5)
            
            # Ensure bounds and fill NaN values
            signals[ticker] = np.clip(signals[ticker], -1, 1)
            signals[ticker] = signals[ticker].fillna(0)  # Fill NaN with neutral signal
        
        return signals

    def _calculate_technical_indicators(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate technical indicators for signal generation"""
        indicators = pd.DataFrame(index=returns.index)
        
        # Moving averages
        indicators['fast_ma'] = returns.rolling(window=self.lookback_fast).mean()
        indicators['slow_ma'] = returns.rolling(window=self.lookback_slow).mean()
        
        # Volatility
        indicators['volatility'] = returns.rolling(window=21).std()
        
        # Momentum
        indicators['momentum'] = returns.rolling(window=20).mean()
        
        # Fill NaN values with 0
        indicators = indicators.fillna(0)
        
        return indicators