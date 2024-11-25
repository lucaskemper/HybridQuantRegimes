# src/data.py
from dataclasses import dataclass, field
from typing import List, Dict
import yfinance as yf
import pandas as pd
import numpy as np

@dataclass
class PortfolioConfig:
    """Configuration for portfolio simulation"""
    tickers: List[str] = field(default_factory=lambda: ['NVDA', 'AMD', 'INTC'])
    weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    start_date: str = '2022-01-01'
    end_date: str = '2024-05-01'
    
    def __post_init__(self):
        assert len(self.tickers) == len(self.weights), "Tickers and weights must match"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1"

class DataLoader:
    """Loads and processes stock data"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
    
    def load_stocks(self) -> Dict:
        """Load and process stock data according to requirements"""
        try:
            print(f"\nLoading data for {', '.join(self.config.tickers)}...")
            
            # Load raw data
            data = yf.download(
                tickers=self.config.tickers,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False
            )
            
            # Prepare required data structure
            market_data = {
                'close': data['Adj Close'],
                'returns': data['Adj Close'].pct_change().dropna(),
                'volume': data['Volume'],
                'metadata': {
                    'start_date': data.index[0].strftime('%Y-%m-%d'),
                    'end_date': data.index[-1].strftime('%Y-%m-%d'),
                    'number_of_trading_days': len(data),
                    'missing_data_percentage': (data.isnull().sum().sum() / 
                        (data.shape[0] * data.shape[1]) * 100)
                }
            }
            
            if self.validate_data(market_data):
                return market_data
            else:
                raise ValueError("Data validation failed")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, data: Dict) -> bool:
        """Validate data according to requirements"""
        try:
            # 1. Check for missing values
            if data['close'].isnull().any().any() or \
               data['returns'].isnull().any().any() or \
               data['volume'].isnull().any().any():
                print("Error: Missing values detected")
                return False

            # 2. Check for positive prices
            if (data['close'] <= 0).any().any():
                print("Error: Non-positive prices detected")
                return False

            # 3. Check for reasonable trading day gaps
            date_diffs = pd.Series(data['close'].index).diff().dt.days
            if (date_diffs > 4).any():  # Allow up to 4 days gap (long weekends/holidays)
                print("Error: Unusual gaps in trading days detected")
                return False

            # 4. Check returns within [-50%, +50%]
            if ((data['returns'] < -0.5) | (data['returns'] > 0.5)).any().any():
                print("Error: Daily returns outside [-50%, +50%] range")
                return False

            # 5. Check for non-negative volume
            if (data['volume'] < 0).any().any():
                print("Error: Negative volume detected")
                return False

            return True

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def get_summary_statistics(self, data: Dict) -> Dict:
        """Calculate summary statistics for the portfolio
        
        Returns:
            Dict containing:
            - returns: mean (annualized), std (annualized), skewness, kurtosis
            - correlation: correlation matrix between assets
            - daily_volume_avg: average daily trading volume
            - price_metrics: starting price, ending price, total return
        """
        stats = {
            'returns': {
                'mean': data['returns'].mean() * 252,  # Annualized returns
                'std': data['returns'].std() * np.sqrt(252),  # Annualized volatility
                'skew': data['returns'].skew(),
                'kurt': data['returns'].kurtosis()
            },
            'correlation': data['returns'].corr().round(3),
            'daily_volume_avg': data['volume'].mean(),
            'price_metrics': {
                'start_price': data['close'].iloc[0],
                'end_price': data['close'].iloc[-1],
                'return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
            }
        }
        return stats