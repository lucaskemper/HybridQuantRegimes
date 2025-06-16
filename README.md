<<<<<<< HEAD
# Advanced Algorithmic Trading System
=======
# Quantitative Portfolio Risk Analysis System: A Framework for Market Regime Detection and Risk Assessment in Semiconductor Equities
## Project Status
>>>>>>> 401a815b6bffb23966c02a4bcb2d3c4e4bf93177

A sophisticated quantitative trading system with deep learning, real-time risk management, and broker integration.

<<<<<<< HEAD
## Features
=======
**Note**: Alpaca API key required for market data access. Configure in `.env` file.
...
## Authors and Institutional Affiliation
- **Lucas Kemper** - Bsc Student, HEC Lausanne
## Setup Guide
>>>>>>> 401a815b6bffb23966c02a4bcb2d3c4e4bf93177

- **Interactive Brokers Integration**: Connect to real markets through IB TWS
- **Advanced Risk Management**: Real-time risk monitoring with customizable limits
- **Deep Learning Market Regime Detection**: LSTM-based analysis of market regimes
- **Monte Carlo Simulation**: Advanced risk analysis with proper statistical validation
- **Multi-factor Strategy Framework**: Combine momentum, mean-reversion, and volatility factors
- **Production-grade Architecture**: Async operations, error handling, and fault tolerance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Install Interactive Brokers TWS or Gateway:
   - Download from [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=16457)
   - Enable API connections in TWS settings
   - Configure the port to match your config (default: 7497 for paper trading)

## Configuration

Edit `config/trading_config.yaml` to configure:

- Broker connection settings
- Risk management parameters
- Trading strategy settings
- Watchlist symbols
- Market data sources

## Running the System

1. Start Interactive Brokers TWS or Gateway

2. Run the trading system:
```bash
python src/main.py config/trading_config.yaml
```

3. For paper trading (recommended for testing):
   - Use port 7497 in the config
   - Set `environment: "paper"` in the config
   - Login to TWS with a paper trading account

## Risk Warning

This software is for educational and research purposes only. Trading financial instruments carries substantial risk of loss. The authors and contributors are not responsible for any financial losses incurred while using this software.

Always start with paper trading and thoroughly test any strategy before using real money.

## System Requirements

- Python 3.7+
- Interactive Brokers account with TWS or IB Gateway
- 8GB+ RAM recommended for running deep learning components
