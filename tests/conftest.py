# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from src.data import PortfolioConfig

@pytest.fixture
def sample_portfolio_config():
    return PortfolioConfig(
        tickers=['NVDA', 'AMD', 'INTC'],
        weights=[0.4, 0.4, 0.2]
    )

@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'close': pd.DataFrame(
            np.random.random((len(dates), 3)) * 100,
            index=dates,
            columns=['NVDA', 'AMD', 'INTC']
        ),
        'returns': pd.DataFrame(
            np.random.normal(0.001, 0.02, (len(dates), 3)),
            index=dates,
            columns=['NVDA', 'AMD', 'INTC']
        )
    }
    return data