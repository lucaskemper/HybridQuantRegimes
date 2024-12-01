# tests/unit/test_data.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.data import DataLoader, PortfolioConfig, MarketData

@pytest.fixture
def sample_portfolio_config():
    """Create a sample portfolio configuration for testing"""
    return PortfolioConfig(
        tickers=['NVDA', 'AMD', 'INTC'],
        weights=[0.4, 0.4, 0.2],
        start_date='2024-01-01',
        end_date='2024-03-01',
        alpaca_key_id='test_key',
        alpaca_secret_key='test_secret',
        paper_trading=True
    )

@pytest.fixture
def mock_market_data() -> MarketData:
    """Create mock market data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='B')
    tickers = ['NVDA', 'AMD', 'INTC']
    
    # Create sample price data
    close_data = pd.DataFrame(
        np.random.uniform(100, 200, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    
    # Calculate returns
    returns_data = close_data.pct_change().fillna(0)
    
    # Create sample volume data
    volume_data = pd.DataFrame(
        np.random.randint(1000000, 5000000, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    
    return {
        'close': close_data,
        'returns': returns_data,
        'volume': volume_data,
        'metadata': {
            'start_date': dates[0].strftime('%Y-%m-%d'),
            'end_date': dates[-1].strftime('%Y-%m-%d'),
            'number_of_trading_days': len(dates),
            'missing_data_percentage': 0.0,
            'tickers': tickers,
            'data_provider': 'alpaca'
        }
    }

def generate_valid_prices(base_price=100, size=40):
    """Generate a sequence of prices with controlled changes to ensure valid returns"""
    prices = []
    current_price = base_price
    for _ in range(size):
        # Generate small price changes (max ±5%)
        change = np.random.uniform(-0.05, 0.05)
        current_price = current_price * (1 + change)
        prices.append(current_price)
    return prices

class TestDataLoader:
    """Test suite for DataLoader class"""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Setup mocks for all tests"""
        mock_api = Mock()
        mock_api.get_latest_trade.return_value = Mock()
        monkeypatch.setattr('alpaca_trade_api.REST', lambda *args, **kwargs: mock_api)

    def test_data_loader_initialization(self, sample_portfolio_config):
        """Test DataLoader initialization"""
        loader = DataLoader(sample_portfolio_config)
        assert loader is not None
        assert loader.config == sample_portfolio_config
        assert hasattr(loader, 'logger')
        assert hasattr(loader, 'api')

    @patch('alpaca_trade_api.REST')
    def test_load_stocks(self, mock_rest, sample_portfolio_config):
        """Test load_stocks method with mocked API response"""
        dates = pd.date_range('2024-01-01', periods=40, freq='B')
        
        # Create mock bar data for each ticker
        data_by_ticker = {}
        for ticker in sample_portfolio_config.tickers:
            ticker_data = pd.DataFrame({
                'close': generate_valid_prices(base_price=100),
                'volume': np.random.randint(1000000, 5000000, size=40)
            }, index=dates)
            data_by_ticker[ticker] = ticker_data
        
        # Setup mock API response
        mock_api = Mock()
        mock_api.get_bars.side_effect = lambda symbol, *args, **kwargs: Mock(df=data_by_ticker[symbol])
        mock_rest.return_value = mock_api
        
        loader = DataLoader(sample_portfolio_config)
        data = loader.load_stocks()
        
        # Verify the structure and content of returned data
        assert isinstance(data, dict)
        assert all(key in data for key in ['close', 'returns', 'volume'])
        assert all(ticker in data['close'].columns 
                  for ticker in sample_portfolio_config.tickers)

    def test_data_validation(self, sample_portfolio_config):
        """Test data validation methods"""
        loader = DataLoader(sample_portfolio_config)
        
        # Create valid test data with controlled price changes
        dates = pd.date_range('2024-01-01', periods=40, freq='B')
        valid_data = {
            'close': pd.DataFrame({
                ticker: generate_valid_prices(base_price=100 + i * 10, size=len(dates))
                for i, ticker in enumerate(sample_portfolio_config.tickers)
            }, index=dates),
            'volume': pd.DataFrame({
                ticker: np.random.randint(1000000, 5000000, size=len(dates))
                for ticker in sample_portfolio_config.tickers
            }, index=dates)
        }
        
        # Calculate returns with controlled values
        valid_data['returns'] = valid_data['close'].pct_change().fillna(0)
        valid_data['metadata'] = {
            'start_date': dates[0].strftime('%Y-%m-%d'),
            'end_date': dates[-1].strftime('%Y-%m-%d'),
            'number_of_trading_days': len(dates),
            'missing_data_percentage': 0.0,
            'tickers': sample_portfolio_config.tickers,
            'data_provider': 'alpaca'
        }
        
        # Test valid data
        assert loader.validate_data(valid_data) is True
        
        # Test invalid data cases
        invalid_cases = [
            (self._create_data_with_missing_values, "Missing values"),
            (self._create_data_with_negative_prices, "Negative prices"),
            (self._create_data_with_large_gaps, "Large gaps"),
            (self._create_data_with_extreme_returns, "Extreme returns"),
            (self._create_data_with_negative_volumes, "Negative volumes"),
            (self._create_data_with_insufficient_days, "Insufficient days")
        ]
        
        for create_invalid_data, case_name in invalid_cases:
            invalid_data = create_invalid_data(valid_data.copy())
            assert loader.validate_data(invalid_data) is False, f"Failed to detect {case_name}"

    def test_get_summary_statistics(self, sample_portfolio_config, mock_market_data):
        """Test summary statistics calculation"""
        loader = DataLoader(sample_portfolio_config)
        stats = loader.get_summary_statistics(mock_market_data)
        
        # Check portfolio-level statistics
        assert 'portfolio' in stats
        assert all(key in stats['portfolio'] for key in [
            'annual_return', 'annual_volatility', 'sharpe_ratio', 
            'skewness', 'kurtosis'
        ])
        
        # Check individual asset statistics
        assert 'individual_assets' in stats
        for ticker in sample_portfolio_config.tickers:
            assert ticker in stats['individual_assets']
            asset_stats = stats['individual_assets'][ticker]
            assert all(key in asset_stats for key in [
                'weight', 'annual_return', 'annual_volatility', 'sharpe_ratio',
                'avg_daily_volume', 'skewness', 'kurtosis'
            ])
        
        # Check correlation matrix
        assert 'correlation' in stats

    # Helper methods for creating invalid test data
    @staticmethod
    def _create_data_with_missing_values(data: MarketData) -> MarketData:
        """Create test data with missing values"""
        data['close'].iloc[0, 0] = np.nan
        return data

    @staticmethod
    def _create_data_with_negative_prices(data: MarketData) -> MarketData:
        """Create test data with negative prices"""
        data['close'].iloc[0, 0] = -100
        return data

    @staticmethod
    def _create_data_with_large_gaps(data: MarketData) -> MarketData:
        """Create test data with large trading gaps"""
        data['close'] = data['close'].drop(data['close'].index[1:5])
        return data

    @staticmethod
    def _create_data_with_extreme_returns(data: MarketData) -> MarketData:
        """Create test data with extreme returns"""
        data['returns'].iloc[1, 0] = 0.6  # Above MAX_RETURN_THRESHOLD
        return data

    @staticmethod
    def _create_data_with_negative_volumes(data: MarketData) -> MarketData:
        """Create test data with negative volumes"""
        data['volume'].iloc[0, 0] = -1000
        return data

    @staticmethod
    def _create_data_with_insufficient_days(data: MarketData) -> MarketData:
        """Create test data with insufficient trading days"""
        data['close'] = data['close'].iloc[:10]  # Less than MIN_TRADING_DAYS
        data['returns'] = data['returns'].iloc[:10]
        data['volume'] = data['volume'].iloc[:10]
        return data