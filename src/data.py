# src/data.py
from dataclasses import dataclass, field
from typing import List, Dict
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PortfolioConfig:
    """Configuration for portfolio simulation"""
    tickers: List[str] = field(default_factory=lambda: ['NVDA', 'AMD', 'INTC'])
    weights: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])
    start_date: str = '2022-01-01'
    end_date: str = '2024-05-01'
    alpaca_key_id: str = field(default_factory=lambda: os.getenv('ALPACA_KEY_ID', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    paper_trading: bool = field(default_factory=lambda: os.getenv('PAPER_TRADING', 'true').lower() == 'true')
    
    def __post_init__(self):
        assert len(self.tickers) == len(self.weights), "Tickers and weights must match"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1"
        assert self.alpaca_key_id, "Alpaca API key ID is required"
        assert self.alpaca_secret_key, "Alpaca secret key is required"

class DataLoader:
    """Loads and processes stock data using Alpaca API"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.api = tradeapi.REST(
            key_id=config.alpaca_key_id,
            secret_key=config.alpaca_secret_key,
            base_url='https://paper-api.alpaca.markets' if config.paper_trading 
                    else 'https://api.alpaca.markets'
        )
    
    def load_stocks(self) -> Dict:
        """Load and process stock data according to requirements"""
        try:
            print(f"\nLoading data for {', '.join(self.config.tickers)}...")
            
            # Load raw data from Alpaca
            data = {}
            for ticker in self.config.tickers:
                # Handle market indices (e.g., '^IXIC') by removing the '^' prefix
                clean_ticker = ticker.replace('^', '')
                bars = self.api.get_bars(
                    clean_ticker,
                    tradeapi.TimeFrame.Day,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    adjustment='all'
                ).df
                data[ticker] = bars
            
            # Align all data to common dates
            close_prices = pd.DataFrame({
                ticker: data[ticker]['close'] for ticker in self.config.tickers
            })
            volumes = pd.DataFrame({
                ticker: data[ticker]['volume'] for ticker in self.config.tickers
            })
            
            # Prepare required data structure (maintaining same format as before)
            market_data = {
                'close': close_prices,
                'returns': close_prices.pct_change().dropna(),
                'volume': volumes,
                'metadata': {
                    'start_date': close_prices.index[0].strftime('%Y-%m-%d'),
                    'end_date': close_prices.index[-1].strftime('%Y-%m-%d'),
                    'number_of_trading_days': len(close_prices),
                    'missing_data_percentage': (close_prices.isnull().sum().sum() / 
                        (close_prices.shape[0] * close_prices.shape[1]) * 100)
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
        """Calculate summary statistics for the portfolio"""
        stats = {
            'returns': {
                'mean': data['returns'].mean() * 252,
                'std': data['returns'].std() * np.sqrt(252),
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