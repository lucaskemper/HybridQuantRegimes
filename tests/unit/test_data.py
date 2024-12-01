# tests/unit/test_data.py
import os
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.data import DataLoader, PortfolioConfig


def is_integration_test():
    """Check if we should run integration tests"""
    return os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


@pytest.fixture
def sample_portfolio_config():
    """Create a sample portfolio configuration for testing"""
    return PortfolioConfig(
        tickers=["NVDA", "AMD", "INTC"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        weights=[0.4, 0.4, 0.2],
    )


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing"""
    dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="B")
    tickers = ["NVDA", "AMD", "INTC"]

    # Create sample price data
    prices = pd.DataFrame(
        np.random.uniform(100, 200, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )

    # Calculate returns
    returns = prices.pct_change().fillna(0)

    return {
        "prices": prices,
        "returns": returns,
        "metadata": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "tickers": tickers,
            "weights": [0.4, 0.4, 0.2],
        },
    }


class TestDataLoader:
    """Test suite for DataLoader class"""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Setup environment variables for tests"""
        # Load .env file
        load_dotenv()

        # Get actual API keys from environment
        api_key = os.getenv("ALPACA_KEY_ID")
        api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not api_secret:
            pytest.skip("Alpaca API credentials not found in environment")

        # Use real credentials for integration tests
        if is_integration_test():
            monkeypatch.setenv("ALPACA_KEY_ID", api_key)
            monkeypatch.setenv("ALPACA_SECRET_KEY", api_secret)
        else:
            monkeypatch.setenv("ALPACA_KEY_ID", "test_key")
            monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")

    def test_data_loader_initialization(self, sample_portfolio_config):
        """Test DataLoader initialization"""
        loader = DataLoader(sample_portfolio_config)
        assert loader is not None
        assert loader.config == sample_portfolio_config
        assert hasattr(loader, "logger")
        assert hasattr(loader, "client")

    @patch("alpaca.data.historical.StockHistoricalDataClient")
    def test_load_data(self, mock_client, sample_portfolio_config):
        """Test load_data method with mocked API response"""
        # Create mock response data
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        mock_data = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [sample_portfolio_config.tickers, dates], names=["symbol", "timestamp"]
            )
        )
        mock_data["close"] = np.random.uniform(100, 200, size=len(mock_data))

        # Setup mock client
        mock_client_instance = Mock()
        mock_client_instance.get_stock_bars.return_value.df = mock_data
        mock_client.return_value = mock_client_instance

        # Test data loading
        loader = DataLoader(sample_portfolio_config)
        market_data = loader.load_data()

        # Verify the structure and content of returned data
        assert isinstance(market_data, dict)
        assert all(key in market_data for key in ["prices", "returns", "metadata"])
        assert all(
            ticker in market_data["prices"].columns
            for ticker in sample_portfolio_config.tickers
        )
        assert (
            market_data["metadata"]["start_date"] == sample_portfolio_config.start_date
        )
        assert market_data["metadata"]["end_date"] == sample_portfolio_config.end_date

    def test_portfolio_config_validation(self):
        """Test PortfolioConfig validation"""
        # Test valid configuration
        valid_config = PortfolioConfig(
            tickers=["AAPL", "GOOGL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            weights=[0.6, 0.4],
        )
        assert valid_config is not None

        # Test invalid configurations
        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=[],  # Empty tickers list
                start_date="2024-01-01",
                end_date="2024-03-01",
            )

        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=["AAPL", "GOOGL"],
                start_date="2024-01-01",
                end_date="2024-03-01",
                weights=[0.6, 0.3],  # Weights don't sum to 1
            )

        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-03-01",
                weights=[0.5, 0.5],  # Number of weights doesn't match tickers
            )

    @pytest.mark.skipif(
        not is_integration_test(), reason="Integration test requires API keys"
    )
    def test_live_data_loading(self, sample_portfolio_config, setup_environment):
        """Test live data loading from Alpaca API"""
        # Add debug logging
        loader = DataLoader(sample_portfolio_config)
        print(
            f"Using API Key: {loader.api_key[:5]}..."
        )  # Print first 5 chars for verification

        market_data = loader.load_data()

        # Verify the structure and content of returned data
        assert isinstance(market_data, dict)
        assert all(key in market_data for key in ["prices", "returns", "metadata"])
        assert all(
            ticker in market_data["prices"].columns
            for ticker in sample_portfolio_config.tickers
        )
        assert len(market_data["prices"]) > 0
        assert len(market_data["returns"]) > 0
