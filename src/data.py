# src/data.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, TypedDict
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import functools
import time

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

class MarketData(TypedDict):
    """Type definition for market data structure"""
    close: pd.DataFrame
    returns: pd.DataFrame
    volume: pd.DataFrame
    metadata: Dict[str, Union[str, int, float, List[str]]]

class DataLoader:
    """Loads and processes stock data using Alpaca API"""
    
    # Class-level constants for validation
    MAX_TRADING_GAP_DAYS: int = 4
    MAX_RETURN_THRESHOLD: float = 0.5
    MIN_TRADING_DAYS: int = 20
    RETRY_DELAY_SECONDS: int = 2
    
    # Class-level logger
    logger = logging.getLogger(__name__)
    
    def __init__(self, config: PortfolioConfig):
        """Initialize DataLoader with configuration"""
        self.config = config
        self._setup_logging()
        self._setup_api()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration with proper formatting"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _setup_api(self) -> None:
        """Initialize Alpaca API connection with proper error handling"""
        try:
            self.logger.info("Connecting to Alpaca API (Paper Trading: %s)", 
                           self.config.paper_trading)
            
            base_url = 'https://data.alpaca.markets'
            
            self.api = tradeapi.REST(
                key_id=self.config.alpaca_key_id,
                secret_key=self.config.alpaca_secret_key,
                base_url=base_url,
                api_version='v2'
            )
            
            # Verify connection with minimal API call
            self._verify_api_connection()
            
        except Exception as e:
            self.logger.error("API setup failed: %s", str(e))
            raise ConnectionError(f"Failed to connect to Alpaca API: {str(e)}")
    
    def _verify_api_connection(self) -> None:
        """Verify API connection with a test request"""
        try:
            self.api.get_latest_trade('AAPL')
            self.logger.info("Successfully connected to Alpaca Data API")
        except Exception as e:
            raise ConnectionError(f"API connection verification failed: {str(e)}")

    @functools.lru_cache(maxsize=32)
    def load_stocks(self, retries: int = 3) -> MarketData:
        """
        Load and process stock data with retry mechanism and caching
        
        Args:
            retries: Number of retry attempts for failed requests
            
        Returns:
            MarketData containing processed stock data and metadata
            
        Raises:
            ValueError: If data validation fails
            ConnectionError: If API connection fails
        """
        for attempt in range(retries):
            try:
                self.logger.info("Loading data for %s...", 
                               ', '.join(self.config.tickers))
                
                data = self._fetch_data()
                market_data = self._process_data(data)
                
                if self.validate_data(market_data):
                    self.logger.info("Data validation successful")
                    return market_data
                raise ValueError("Data validation failed")
                
            except Exception as e:
                self.logger.error("Attempt %d/%d failed: %s", 
                                attempt + 1, retries, str(e))
                if attempt == retries - 1:
                    raise
                time.sleep(self.RETRY_DELAY_SECONDS)
    
    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch raw data from Alpaca API with parallel processing
        """
        data = {}
        failed_tickers = []
        
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
                self.logger.info("Successfully loaded data for %s", ticker)
                
            except Exception as e:
                failed_tickers.append(ticker)
                self.logger.error("Failed to fetch data for %s: %s", 
                                ticker, str(e))
        
        if failed_tickers:
            raise ValueError(f"Failed to fetch data for tickers: {failed_tickers}")
        
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
    
    def validate_data(self, data: MarketData) -> bool:
        """
        Validate data according to requirements with detailed error messages
        """
        try:
            validation_checks = [
                self._check_missing_values(data),
                self._check_positive_prices(data),
                self._check_trading_gaps(data),
                self._check_return_ranges(data),
                self._check_volumes(data),
                self._check_minimum_trading_days(data)
            ]
            
            return all(validation_checks)
            
        except Exception as e:
            self.logger.error("Validation error: %s", str(e))
            return False
    
    def _check_missing_values(self, data: MarketData) -> bool:
        """Check for missing values in all data components"""
        for component in ['close', 'returns', 'volume']:
            if data[component].isnull().any().any():
                self.logger.error("Missing values detected in %s data", component)
                return False
        return True
    
    def _check_positive_prices(self, data: MarketData) -> bool:
        """Verify all prices are positive"""
        if (data['close'] <= 0).any().any():
            self.logger.error("Non-positive prices detected")
            return False
        return True
    
    def _check_trading_gaps(self, data: MarketData) -> bool:
        """Check for unusual gaps between trading days"""
        date_diffs = pd.Series(data['close'].index).diff().dt.days
        if (date_diffs > self.MAX_TRADING_GAP_DAYS).any():
            self.logger.error(
                f"Unusual gaps in trading days detected (max allowed: {self.MAX_TRADING_GAP_DAYS} days)"
            )
            return False
        return True
    
    def _check_return_ranges(self, data: MarketData) -> bool:
        """Verify returns are within acceptable ranges"""
        if ((data['returns'].abs() > self.MAX_RETURN_THRESHOLD)).any().any():
            self.logger.error(
                f"Daily returns outside [-{self.MAX_RETURN_THRESHOLD}, {self.MAX_RETURN_THRESHOLD}] range"
            )
            return False
        return True
    
    def _check_volumes(self, data: MarketData) -> bool:
        """Verify trading volumes are non-negative"""
        if (data['volume'] < 0).any().any():
            self.logger.error("Negative trading volumes detected")
            return False
        return True
    
    def _check_minimum_trading_days(self, data: MarketData) -> bool:
        """Verify minimum number of trading days"""
        if len(data['close']) < self.MIN_TRADING_DAYS:
            self.logger.error(
                f"Insufficient trading days: {len(data['close'])} (minimum: {self.MIN_TRADING_DAYS})"
            )
            return False
        return True

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