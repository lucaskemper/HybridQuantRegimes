# src/data.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

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
        self._setup_logging()
        self._setup_api()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _setup_api(self):
        """Initialize Alpaca API connection"""
        try:
            self.logger.info(f"Connecting to Alpaca API...")
            self.logger.info(f"Using paper trading: {self.config.paper_trading}")
            
            # Use data API endpoint
            base_url = 'https://data.alpaca.markets'
            
            self.api = tradeapi.REST(
                key_id=self.config.alpaca_key_id,
                secret_key=self.config.alpaca_secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection with a simple market data request
            self.api.get_latest_trade('AAPL')
            self.logger.info("Successfully connected to Alpaca Data API")
            
        except Exception as e:
            self.logger.error(f"API setup failed: {str(e)}")
            raise ConnectionError(f"Failed to connect to Alpaca API: {str(e)}")
    
    def load_stocks(self, retries: int = 3) -> Dict:
        """Load and process stock data with retry mechanism"""
        for attempt in range(retries):
            try:
                self.logger.info(f"Loading data for {', '.join(self.config.tickers)}...")
                
                # Fetch data directly without market date verification
                data = self._fetch_data()
                market_data = self._process_data(data)
                
                if self.validate_data(market_data):
                    self.logger.info("Data validation successful")
                    return market_data
                raise ValueError("Data validation failed")
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
    
    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch raw data from Alpaca API"""
        data = {}
        for ticker in self.config.tickers:
            try:
                bars = self.api.get_bars(
                    ticker,
                    tradeapi.TimeFrame.Day,
                    start=pd.Timestamp(self.config.start_date).strftime('%Y-%m-%d'),
                    end=pd.Timestamp(self.config.end_date).strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    raise ValueError(f"No data returned for {ticker}")
                    
                data[ticker] = bars
                self.logger.info(f"Successfully loaded data for {ticker}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
                raise
        
        return data
    
    def _process_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict:
        """Process raw data into required format"""
        # Align all data to common dates
        close_prices = pd.DataFrame({
            ticker: data['close'] for ticker, data in raw_data.items()
        })
        volumes = pd.DataFrame({
            ticker: data['volume'] for ticker, data in raw_data.items()
        })
        
        return {
            'close': close_prices,
            'returns': close_prices.pct_change().dropna(),
            'volume': volumes,
            'metadata': self._generate_metadata(close_prices)
        }
    
    def _generate_metadata(self, prices: pd.DataFrame) -> Dict:
        """Generate metadata for the dataset"""
        return {
            'start_date': prices.index[0].strftime('%Y-%m-%d'),
            'end_date': prices.index[-1].strftime('%Y-%m-%d'),
            'number_of_trading_days': len(prices),
            'missing_data_percentage': (prices.isnull().sum().sum() / 
                (prices.shape[0] * prices.shape[1]) * 100),
            'tickers': list(prices.columns),
            'data_provider': 'alpaca'
        }
    
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
        try:
            # Calculate portfolio returns using weights
            weights_series = pd.Series(self.config.weights, index=self.config.tickers)
            portfolio_returns = (data['returns'] * weights_series).sum(axis=1)
            
            # Calculate portfolio-level statistics
            stats = {
                'portfolio': {
                    'annual_return': float(portfolio_returns.mean() * 252),
                    'annual_volatility': float(portfolio_returns.std() * np.sqrt(252)),
                    'skewness': float(portfolio_returns.skew()),
                    'kurtosis': float(portfolio_returns.kurtosis()),
                    'sharpe_ratio': float((portfolio_returns.mean() * 252) / 
                                        (portfolio_returns.std() * np.sqrt(252)))
                },
                'individual_assets': {},
                'correlation': data['returns'].corr().round(3).to_dict()
            }
            
            # Calculate individual asset statistics
            for ticker in self.config.tickers:
                returns = data['returns'][ticker]
                volume = data['volume'][ticker]
                weight = weights_series[ticker]
                
                stats['individual_assets'][ticker] = {
                    'weight': float(weight),
                    'annual_return': float(returns.mean() * 252),
                    'annual_volatility': float(returns.std() * np.sqrt(252)),
                    'sharpe_ratio': float((returns.mean() * 252) / 
                                        (returns.std() * np.sqrt(252))),
                    'avg_daily_volume': float(volume.mean()),
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            raise