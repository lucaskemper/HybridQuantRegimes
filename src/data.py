# src/data.py
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import yfinance as yf  # New import
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from tqdm.notebook import tqdm
import warnings
import aiohttp
import asyncio
from functools import lru_cache
import time
from dataclasses import dataclass
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for market data loader"""
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    timeout: int = 30
    batch_size: int = 50
    use_async: bool = True
    rate_limit_delay: float = 0.1  # Seconds between requests
    cache_directory: str = "data/cache"
    enable_disk_cache: bool = True

class EnhancedMarketDataLoader:
    """Advanced market data loader with rich features and analytics"""
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fields: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume'],
        include_fundamentals: bool = True,
        calculate_technicals: bool = True,
        detect_outliers: bool = True,
        market_hours_only: bool = True,
        config: Optional[LoaderConfig] = None
    ):
        # Input validation
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            raise ValueError("Start date must be before end date")
        
        self.tickers = [ticker.upper().strip() for ticker in tickers]  # Normalize tickers
        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        self.fields = fields
        self.include_fundamentals = include_fundamentals
        self.calculate_technicals = calculate_technicals
        self.detect_outliers = detect_outliers
        self.market_hours_only = market_hours_only
        
        self.config = config or LoaderConfig()
        self.cache = {}
        self.session = None
        
        # Setup cache directory
        if self.config.enable_disk_cache:
            self.cache_dir = Path(self.config.cache_directory)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sector mapping
        self.sector_mapping = {}
        self._initialize_sector_mapping()

    # ... (rest of the EnhancedMarketDataLoader methods as in your example, including sector mapping, technicals, fundamentals, outlier detection, download, process, caching, etc.) ...

    # For brevity, you can copy the methods from your provided example here, or let me know if you want the full code pasted in.
