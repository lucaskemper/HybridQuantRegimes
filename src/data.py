# src/data.py
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf  # New import
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
    """Data loading and preprocessing using Yahoo Finance"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Remove Alpaca-specific code since we're using Yahoo Finance
        load_dotenv()  # Keep this in case we need other env variables later

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess market data from Yahoo Finance"""
        try:
            self.logger.info("Loading market data...")

            # Convert dates to datetime
            start_dt = self.config.start_date
            end_dt = self.config.end_date

            # Download data from Yahoo Finance
            prices = pd.DataFrame()
            for ticker in self.config.tickers:
                ticker_data = yf.download(
                    ticker,
                    start=start_dt,
                    end=end_dt,
                    progress=False
                )
                prices[ticker] = ticker_data['Adj Close']

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
