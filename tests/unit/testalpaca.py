# test_connection.py
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Print environment variables (masked)
api_key = os.getenv('ALPACA_KEY_ID', '')
secret_key = os.getenv('ALPACA_SECRET_KEY', '')

print("API Key Check:")
print(f"API Key length: {len(api_key)}")
print(f"API Key preview: {api_key[:6]}...")
print(f"Secret Key length: {len(secret_key)}")
print(f"Secret Key preview: {secret_key[:6]}...")

# Test connection
try:
    from src.data import PortfolioConfig, DataLoader
    
    config = PortfolioConfig(
        tickers=['AAPL'],  # Just one ticker for testing
        weights=[1.0],
        start_date='2024-01-01',
        end_date='2024-01-02',
        alpaca_key_id=api_key,
        alpaca_secret_key=secret_key,
        paper_trading=True
    )
    
    loader = DataLoader(config)
    print("\nConnection test successful!")
    
except Exception as e:
    print(f"\nConnection test failed: {str(e)}")