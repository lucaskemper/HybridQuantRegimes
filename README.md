# ResearchLucas: Dynamic Portfolio Analysis & Regime Detection

## Overview

This project provides a modular framework for advanced portfolio analysis, dynamic risk management, and market regime detection using both classical and deep learning methods. It is designed for quantitative finance research, supporting simulation, risk, and signal generation workflows.

## Features

- **Data Loading**: Fetches and preprocesses historical market data from Yahoo Finance.
- **Monte Carlo Simulation**: Simulates portfolio returns using configurable distributions and GARCH volatility forecasting.
- **Market Regime Detection**: Identifies market regimes using Hidden Markov Models (HMM) and optional LSTM-based deep learning.
- **Risk Management**: Computes advanced risk metrics (VaR, ES, drawdown, Sharpe, rolling volatility, etc.) with regime-aware options.
- **Signal Generation**: Produces trading signals based on moving averages and momentum.
- **Visualization**: Generates comprehensive plots for prices, signals, Monte Carlo paths, and return distributions.
- **Extensive Testing**: Includes unit and integration tests for all major modules.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd researchlucas
   ```

2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a `.env` file for environment variables if needed.

## Usage

### Example Workflow

```python
from src.data import DataLoader, PortfolioConfig
from src.portfolio import PortfolioAnalyzer, PortfolioAnalyzerConfig

# Configure your portfolio
config = PortfolioConfig(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    weights=[0.4, 0.3, 0.3]
)
data_loader = DataLoader(config)
market_data = data_loader.load_data()

# Analyze portfolio
analyzer = PortfolioAnalyzer(PortfolioAnalyzerConfig())
results = analyzer.analyze_portfolio(market_data)
```

### Visualization

The `PortfolioVisualizer` class in `src/visualization.py` can be used to plot results:
```python
from src.visualization import PortfolioVisualizer
visualizer = PortfolioVisualizer()
visualizer.plot_portfolio_analysis(
    market_data, results['signals'], results['monte_carlo'], results['regimes']['current_regime']
)
```

### Running Tests

```bash
pytest tests/
```

## Project Structure

- `src/data.py` – Data loading and preprocessing
- `src/monte_carlo.py` – Monte Carlo simulation engine
- `src/regime.py` – Market regime detection (HMM, LSTM)
- `src/risk.py` – Risk metrics and management
- `src/signals.py` – Trading signal generation
- `src/visualization.py` – Plotting and visualization
- `src/deep_learning.py` – LSTM regime detection
- `src/portfolio.py` – High-level portfolio analysis workflow
- `tests/` – Unit and integration tests

## Requirements

See `requirements.txt` for all dependencies, including:
- numpy, pandas, scipy, yfinance, scikit-learn, hmmlearn, tensorflow, arch, matplotlib, seaborn, etc.

## Notes

- The framework is modular: each component can be used independently or as part of the full workflow.
- For deep learning regime detection, TensorFlow is required.
- Example notebooks and additional documentation may be added in the future.
