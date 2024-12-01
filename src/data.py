# src/data.py
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

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


class DataLoader:
    """Data loading and preprocessing using Alpaca"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()

        # Initialize Alpaca client with correct env variable names
        self.api_key = os.getenv("ALPACA_KEY_ID")  # Changed from ALPACA_API_KEY
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.api_secret:
            self.logger.error("API Key:", self.api_key)
            self.logger.error("Secret Key:", self.api_secret)
            raise ValueError(
                "Alpaca API credentials not found in environment variables. "
                "Please check your .env file contains ALPACA_KEY_ID and ALPACA_SECRET_KEY"
            )

        self.client = StockHistoricalDataClient(
            api_key=self.api_key, secret_key=self.api_secret
        )
        self.logger.info("Connecting to Alpaca API (Paper Trading: True)")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess market data from Alpaca"""
        try:
            self.logger.info("Loading market data...")

            # Convert dates to datetime
            start_dt = pd.Timestamp(self.config.start_date).tz_localize("UTC")
            end_dt = pd.Timestamp(self.config.end_date).tz_localize("UTC")

            # Request parameters
            request_params = StockBarsRequest(
                symbol_or_symbols=self.config.tickers,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
            )

            # Get the data
            bars = self.client.get_stock_bars(request_params)

            # Convert to DataFrame
            df = bars.df

            # Process the multi-level DataFrame
            prices = pd.DataFrame()
            for ticker in self.config.tickers:
                prices[ticker] = df.loc[ticker]["close"]

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Create market data dictionary
            market_data = {
                "prices": prices,
                "returns": returns,
                "metadata": {
                    "tickers": self.config.tickers,
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "weights": self.config.weights,
                },
            }

            self.logger.info(
                f"Successfully loaded data for {len(self.config.tickers)} tickers"
            )
            return market_data

        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise
