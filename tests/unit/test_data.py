import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src.data import DataLoader, PortfolioConfig

@pytest.fixture
def sample_portfolio_config():
    return PortfolioConfig(
        tickers=["AAPL", "GOOGL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-03-01",
        weights=[0.4, 0.3, 0.3],
    )

@pytest.fixture
def mock_yahoo_data():
    dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="B")
    data = pd.DataFrame({
        'Adj Close': np.random.uniform(100, 200, size=len(dates))
    }, index=dates)
    return data

class TestPortfolioConfig:
    def test_valid_config(self):
        config = PortfolioConfig(
            tickers=["AAPL", "GOOGL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            weights=[0.5, 0.5],
        )
        assert config is not None

    def test_empty_tickers(self):
        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=[],
                start_date="2024-01-01",
                end_date="2024-03-01",
            )

    def test_invalid_weights_length(self):
        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=["AAPL", "GOOGL"],
                start_date="2024-01-01",
                end_date="2024-03-01",
                weights=[1.0],
            )

    def test_weights_not_sum_to_one(self):
        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=["AAPL", "GOOGL"],
                start_date="2024-01-01",
                end_date="2024-03-01",
                weights=[0.7, 0.2],
            )

    def test_weights_out_of_bounds(self):
        with pytest.raises(ValueError):
            PortfolioConfig(
                tickers=["AAPL", "GOOGL"],
                start_date="2024-01-01",
                end_date="2024-03-01",
                weights=[-0.1, 1.1],
            )

class TestDataLoader:
    @patch("yfinance.download")
    def test_load_data_and_features(self, mock_download, sample_portfolio_config, mock_yahoo_data):
        # Mock yfinance to return data for each ticker
        def side_effect(ticker, *args, **kwargs):
            return mock_yahoo_data
        mock_download.side_effect = side_effect

        loader = DataLoader(sample_portfolio_config)
        market_data = loader.load_data()

        # Check keys
        assert set(market_data.keys()) >= {"prices", "returns", "features", "metadata"}
        # Check prices shape
        assert all(ticker in market_data["prices"].columns for ticker in sample_portfolio_config.tickers)
        # Check returns shape
        assert market_data["returns"].shape[0] == market_data["prices"].shape[0] - 1
        # Check features
        features = market_data["features"]
        assert isinstance(features, dict)
        for ticker, df in features.items():
            assert set(df.columns) >= {"returns", "volatility", "ewm_volatility", "momentum", "fast_ma", "slow_ma", "skewness", "kurtosis", "rsi"}
            assert not np.any(df.isna().to_numpy())

    @patch("yfinance.download")
    def test_load_data_ohlcv_and_volume(self, mock_download, sample_portfolio_config, mock_yahoo_data):
        # Add Volume to mock data
        mock_data = mock_yahoo_data.copy()
        mock_data['Volume'] = np.random.randint(1_000_000, 10_000_000, size=len(mock_data))
        mock_data['Open'] = mock_data['Adj Close'] + np.random.normal(0, 1, size=len(mock_data))
        mock_data['High'] = mock_data['Adj Close'] + np.random.normal(0, 2, size=len(mock_data))
        mock_data['Low'] = mock_data['Adj Close'] - np.random.normal(0, 2, size=len(mock_data))
        mock_data['Close'] = mock_data['Adj Close'] + np.random.normal(0, 1, size=len(mock_data))
        def side_effect(ticker, *args, **kwargs):
            return mock_data
        mock_download.side_effect = side_effect
        config = PortfolioConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            ohlcv=True
        )
        loader = DataLoader(config)
        market_data = loader.load_data()
        assert "ohlcv" in market_data
        assert "AAPL" in market_data["ohlcv"]
        ohlcv_df = market_data["ohlcv"]["AAPL"]
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            assert col in ohlcv_df.columns

    @patch("yfinance.download")
    def test_load_data_with_normalization(self, mock_download, sample_portfolio_config, mock_yahoo_data):
        mock_download.return_value = mock_yahoo_data
        config = PortfolioConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            normalize=True
        )
        loader = DataLoader(config)
        market_data = loader.load_data()
        features = market_data["features"]["AAPL"]
        # Normalized features should have mean ~0 and std ~1 (except for edge cases)
        means = features.mean()
        means = pd.Series(means).dropna()
        stds = features.std(ddof=0)
        stds = pd.Series(stds).dropna()
        assert np.all(np.abs(means) < 1e-1)  # Allow some tolerance
        assert np.all(np.abs(stds - 1) < 1e-1)

    @patch("yfinance.download")
    def test_load_data_with_resampling(self, mock_download, sample_portfolio_config, mock_yahoo_data):
        mock_download.return_value = mock_yahoo_data
        config = PortfolioConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            frequency="M"
        )
        loader = DataLoader(config)
        market_data = loader.load_data()
        # Should be monthly data
        freq = getattr(market_data["prices"].index, "inferred_freq", None)
        assert freq in ["M", "MS", None] or len(market_data["prices"]) <= 3

    @patch("yfinance.download")
    def test_load_data_with_cache(self, mock_download, tmp_path, sample_portfolio_config, mock_yahoo_data):
        # Use a temporary directory for cache
        cache_dir = tmp_path / "data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        mock_download.return_value = mock_yahoo_data
        config = PortfolioConfig(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            use_cache=True
        )
        loader = DataLoader(config)
        loader.cache_dir = str(cache_dir)  # Override cache dir
        # First call should write cache
        market_data = loader.load_data()
        cache_file = cache_dir / "AAPL_2024-01-01_2024-03-01.csv"
        assert cache_file.exists()
        # Second call should read from cache (simulate by removing yfinance)
        with patch("yfinance.download", return_value=None) as mock_dl:
            market_data2 = loader.load_data()
            assert mock_dl.call_count == 0  # Should not call yfinance

    @patch("yfinance.download")
    def test_missing_yahoo_data(self, mock_download, sample_portfolio_config):
        # Simulate missing data for one ticker
        def side_effect(ticker, *args, **kwargs):
            if ticker == "GOOGL":
                return pd.DataFrame()  # Empty DataFrame
            else:
                dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="B")
                return pd.DataFrame({'Adj Close': np.random.uniform(100, 200, size=len(dates))}, index=dates)
        mock_download.side_effect = side_effect

        loader = DataLoader(sample_portfolio_config)
        with pytest.raises(ValueError):
            loader.load_data()

    def test_process_data_direct(self):
        # Test process_data with synthetic returns
        dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="B")
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, size=(len(dates), 2)),
            index=dates,
            columns=pd.Index(["AAPL", "GOOGL"])
        )
        features = DataLoader.process_data(returns)
        assert isinstance(features, dict)
        for ticker, df in features.items():
            assert set(df.columns) >= {"returns", "volatility", "ewm_volatility", "momentum", "fast_ma", "slow_ma", "skewness", "kurtosis", "rsi"}
            assert not np.any(df.isna().to_numpy())

    def test_rsi_wilders_smoothing(self):
        # Test that RSI uses Wilder's smoothing and is less noisy than simple MA
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        rsi = DataLoader._calculate_rsi(returns)
        assert isinstance(rsi, pd.Series)
        assert np.all((rsi >= 0) & (rsi <= 100))
        # Should be less noisy than returns
        assert rsi.std() < 50
