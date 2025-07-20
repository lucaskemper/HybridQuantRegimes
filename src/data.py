# src/data.py
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf  # New import
from dotenv import load_dotenv
import numpy as np

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio configuration parameters"""

    tickers: List[str]
    start_date: str
    end_date: str
    weights: Optional[List[float]] = None
    ohlcv: bool = False  # If True, load full OHLCV data
    normalize: bool = False  # If True, normalize features
    frequency: str = 'D'  # Resampling frequency: 'D', 'W', 'M', etc.
    use_cache: bool = False  # If True, use local cache for data
    include_macro: bool = True  # If True, include VIX and yield curve data
    macro_tickers: Optional[List[str]] = None  # Macro indicators to include

    def __post_init__(self):
        """Validate configuration parameters"""
        if not self.tickers:
            raise ValueError("Tickers list cannot be empty")
        if not self.start_date or not self.end_date:
            raise ValueError("Start and end dates must be specified")
        if self.weights is not None:
            if len(self.weights) != len(self.tickers):
                raise ValueError("Number of weights must match number of tickers")
            if not all(0 <= w <= 1 for w in self.weights):
                raise ValueError("Weights must be between 0 and 1")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1")
        if self.frequency not in ['D', 'W', 'M']:
            raise ValueError("Frequency must be one of 'D', 'W', or 'M'")
        if self.macro_tickers is None:
            self.macro_tickers = ['^VIX', '^TNX', '^TYX']  # VIX, 10Y, 30Y yields


class DataLoader:
    """Data loading and preprocessing using Yahoo Finance"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        load_dotenv()  # Keep this in case we need other env variables later
        self.cache_dir = "data_cache"
        if self.config.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess market data from Yahoo Finance, with optional OHLCV, caching, and resampling."""
        try:
            self.logger.info("Loading market data...")

            start_dt = self.config.start_date
            end_dt = self.config.end_date

            prices_dict = {}
            for ticker in self.config.tickers:
                cache_path = os.path.join(self.cache_dir, f"{ticker}_{start_dt}_{end_dt}.csv")
                if self.config.use_cache and os.path.exists(cache_path):
                    self.logger.info(f"Loading {ticker} from cache.")
                    ticker_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                else:
                    ticker_data = yf.download(
                        ticker,
                        start=start_dt,
                        end=end_dt,
                        progress=False
                    )
                    if ticker_data is not None:
                        # Flatten MultiIndex columns if present
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            ticker_data.columns = [' '.join([str(c) for c in col if c]) for col in ticker_data.columns.values]
                        print(f"Ticker: {ticker}, Data shape: {ticker_data.shape}, Columns: {ticker_data.columns}")
                        print(ticker_data.head())
                    if self.config.use_cache and ticker_data is not None and not ticker_data.empty:
                        ticker_data.to_csv(cache_path)
                if ticker_data is not None and not ticker_data.empty:
                    if self.config.ohlcv:
                        prices_dict[ticker] = ticker_data  # Store full OHLCV
                    else:
                        # Use 'Adj Close {ticker}' if available, otherwise fall back to 'Close {ticker}'
                        adj_col = f'Adj Close {ticker}'
                        close_col = f'Close {ticker}'
                        volume_col = f'Volume {ticker}'
                        if adj_col in ticker_data.columns:
                            price_col = adj_col
                        elif close_col in ticker_data.columns:
                            price_col = close_col
                        else:
                            raise ValueError(f"No price column found for {ticker} in columns: {ticker_data.columns}")
                        cols = [price_col]
                        if volume_col in ticker_data.columns:
                            cols.append(volume_col)
                        prices_dict[ticker] = ticker_data[cols]
                else:
                    self.logger.warning(f"No data for ticker {ticker}, filling with NaN.")
                    date_index = pd.date_range(start=start_dt, end=end_dt)
                    if self.config.ohlcv:
                        prices_dict[ticker] = pd.DataFrame(index=date_index, columns=pd.Index(['Open','High','Low','Close','Adj Close','Volume']))
                    else:
                        prices_dict[ticker] = pd.DataFrame(index=date_index, columns=pd.Index(['Adj Close','Volume']))

            # Load macro data if requested
            macro_data = {}
            if self.config.include_macro and self.config.macro_tickers is not None:
                macro_data = self._load_macro_data(start_dt, end_dt)

            # Combine Adj Close or Close for returns calculation
            prices = pd.DataFrame({
                t: (df[f'Adj Close {t}'] if f'Adj Close {t}' in df.columns else df[f'Close {t}'])
                for t, df in prices_dict.items()
                if (df is not None and not df.empty and (f'Adj Close {t}' in df.columns or f'Close {t}' in df.columns))
            })
            prices = prices.ffill().bfill()
            returns = prices.pct_change().dropna()

            # Resample if needed
            if self.config.frequency != 'D':
                prices = DataLoader.resample_data(prices, self.config.frequency)
                returns = DataLoader.resample_data(returns, self.config.frequency)
                for t in prices_dict:
                    prices_dict[t] = DataLoader.resample_data(prices_dict[t], self.config.frequency)
                if macro_data:
                    macro_data = {k: DataLoader.resample_data(v, self.config.frequency) for k, v in macro_data.items()}

            # Process features
            features = DataLoader.process_data(returns, normalize=self.config.normalize, macro_data=macro_data)

            # Data integrity checks
            DataLoader._integrity_checks(prices, returns)

            market_data = {
                "prices": prices,
                "returns": returns,
                "features": features,
                "ohlcv": prices_dict if self.config.ohlcv else None,
                "macro": macro_data if macro_data else None,
                "metadata": {
                    "tickers": self.config.tickers,
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "weights": self.config.weights,
                    "ohlcv": self.config.ohlcv,
                    "normalize": self.config.normalize,
                    "frequency": self.config.frequency,
                    "use_cache": self.config.use_cache,
                    "include_macro": self.config.include_macro,
                    "macro_tickers": self.config.macro_tickers,
                },
            }

            self.logger.info(
                f"Successfully loaded data for {len(self.config.tickers)} tickers"
            )
            return market_data

        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise

    def _load_macro_data(self, start_dt: str, end_dt: str) -> Dict[str, pd.DataFrame]:
        """Load macro indicators (VIX, yield curve)"""
        macro_data = {}
        
        if self.config.macro_tickers is None:
            return macro_data
            
        for ticker in self.config.macro_tickers:
            cache_path = os.path.join(self.cache_dir, f"{ticker}_{start_dt}_{end_dt}.csv")
            
            if self.config.use_cache and os.path.exists(cache_path):
                self.logger.info(f"Loading macro {ticker} from cache.")
                ticker_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            else:
                ticker_data = yf.download(
                    ticker,
                    start=start_dt,
                    end=end_dt,
                    progress=False
                )
                if self.config.use_cache and ticker_data is not None and not ticker_data.empty:
                    ticker_data.to_csv(cache_path)
            
            if ticker_data is not None and not ticker_data.empty:
                # Flatten MultiIndex columns if present
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = [' '.join([str(c) for c in col if c]) for col in ticker_data.columns.values]
                
                # For macro indicators, the column structure is different
                # They have: Price, Close, High, Low, Open, Volume
                # Where 'Price' is actually the Close price and 'Close' is also available
                if 'Close' in ticker_data.columns:
                    price_col = 'Close'
                elif 'Price' in ticker_data.columns:
                    # For macro indicators, 'Price' is often the Close price
                    price_col = 'Price'
                else:
                    self.logger.warning(f"No price column found for macro {ticker}. Available columns: {ticker_data.columns.tolist()}")
                    continue
                
                macro_data[ticker] = ticker_data[[price_col]]
                self.logger.info(f"Loaded macro data for {ticker}: {ticker_data.shape}, using column: {price_col}")
            else:
                self.logger.warning(f"No data for macro ticker {ticker}")
        
        return macro_data

    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """Resample DataFrame to given frequency using last value."""
        return df.resample(freq).last()

    @staticmethod
    def process_data(returns: pd.DataFrame, normalize: bool = False, macro_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute a comprehensive set of features for each asset:
        - Volatility (rolling, EWMA)
        - Momentum
        - Moving averages (fast/slow)
        - Skewness
        - Kurtosis
        - RSI
        - Macro indicators (VIX, yield curve)
        Optionally normalize features.
        Returns a dict of DataFrames, one per feature.
        """
        features = {}
        window_fast = 20
        window_slow = 50
        window_vol = 21
        for ticker in returns.columns:
            r = returns[ticker]
            if not isinstance(r, pd.Series):
                r = pd.Series(r)
            df = pd.DataFrame(index=returns.index)
            # Returns
            df['returns'] = r
            # Volatility
            df['volatility'] = r.rolling(window=window_vol).std()
            df['ewm_volatility'] = r.ewm(span=window_vol).std()
            # Momentum
            df['momentum'] = r.rolling(window=window_fast).mean()
            # Moving averages
            df['fast_ma'] = r.rolling(window=window_fast).mean()
            df['slow_ma'] = r.rolling(window=window_slow).mean()
            # Skewness and kurtosis
            df['skewness'] = r.rolling(window=window_vol).skew()
            df['kurtosis'] = r.rolling(window=window_vol).kurt()
            # RSI
            df['rsi'] = DataLoader._calculate_rsi(r)
            # MACD
            ema12 = r.ewm(span=12).mean()
            ema26 = r.ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Add macro indicators if available
            if macro_data:
                for macro_ticker, macro_df in macro_data.items():
                    if macro_df is not None and not macro_df.empty:
                        # Get the price column - macro data has 'Close' or 'Price' columns
                        price_col = None
                        if 'Close' in macro_df.columns:
                            price_col = 'Close'
                        elif 'Price' in macro_df.columns:
                            price_col = 'Price'
                        else:
                            # Fallback: look for any column with 'Close' or 'Adj Close' in name
                            for col in macro_df.columns:
                                if 'Close' in col or 'Adj Close' in col:
                                    price_col = col
                                    break
                        
                        if price_col:
                            macro_series = macro_df[price_col].reindex(df.index).fillna(method='ffill')
                            
                            if macro_ticker == '^VIX':
                                # VIX features
                                df['vix'] = macro_series
                                df['vix_ma'] = macro_series.rolling(window=21).mean()
                                df['vix_ratio'] = macro_series / df['vix_ma']
                                df['vix_regime'] = (macro_series > macro_series.rolling(window=252).quantile(0.75)).astype(int)
                            elif macro_ticker in ['^TNX', '^TYX']:
                                # Yield curve features
                                df[f'{macro_ticker.lower()}_yield'] = macro_series
                                df[f'{macro_ticker.lower()}_yield_ma'] = macro_series.rolling(window=21).mean()
                                df[f'{macro_ticker.lower()}_yield_spread'] = macro_series - macro_series.rolling(window=252).mean()
            
            # Clean and normalize if requested
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            if normalize:
                df = (df - df.mean()) / (df.std() + 1e-8)
            
            features[ticker] = df

        return features

    @staticmethod
    def _calculate_rsi(returns: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI) using Wilder's smoothing (EMA)."""
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        delta = returns.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/periods, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/periods, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if not isinstance(rsi, pd.Series):
            rsi = pd.Series(rsi, index=returns.index)
        return rsi.fillna(50)

    @staticmethod
    def _integrity_checks(prices: pd.DataFrame, returns: pd.DataFrame):
        """Perform data integrity checks and raise errors if issues are found."""
        if returns.isnull().values.any():
            raise ValueError("Returns dataframe has NaNs")
        if (prices.nunique() <= 1).any():
            raise ValueError("Some price series are constant or ill-behaved")
