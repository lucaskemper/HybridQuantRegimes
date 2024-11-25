# tests/unit/test_data.py
import pytest
import pandas as pd
from src.data import DataLoader

def test_data_loader_initialization(sample_portfolio_config):
    loader = DataLoader(sample_portfolio_config)
    assert loader is not None
    assert loader.config == sample_portfolio_config

def test_load_stocks(sample_portfolio_config):
    loader = DataLoader(sample_portfolio_config)
    market_data = loader.load_stocks()
    
    assert isinstance(market_data, dict)
    assert 'close' in market_data
    assert 'returns' in market_data
    assert all(ticker in market_data['close'].columns 
              for ticker in sample_portfolio_config.tickers)