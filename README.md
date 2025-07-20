# Enhanced Market Regime Detection and Portfolio Analysis System

A comprehensive quantitative finance pipeline for market regime detection, signal generation, risk management, and portfolio analysis using machine learning and statistical methods.

## Overview

This system provides a complete framework for:
- **Market Regime Detection**: Hidden Markov Models (HMM) and LSTM-based regime classification
- **Signal Generation**: Multi-factor signal generation with regime awareness
- **Risk Management**: Advanced risk metrics and Monte Carlo simulation
- **Portfolio Analysis**: Backtesting with dynamic position sizing and regime-aware strategies
- **Visualization**: Comprehensive plotting and analysis tools

## Features

### Core Components

- **Regime Detection**: 5-regime HMM with ensemble methods and adaptive retraining
- **Deep Learning**: LSTM-based regime detection with TensorFlow
- **Signal Generation**: Multi-factor signals with trend, momentum, mean reversion, and RSI
- **Risk Management**: VaR, Expected Shortfall, drawdown analysis, and stress testing
- **Monte Carlo Simulation**: 50,000+ simulations with GARCH volatility forecasting
- **Backtesting**: Walk-forward analysis with regime-aware position sizing
- **Data Management**: Yahoo Finance integration with caching and macro indicators

### Advanced Features

- **Adaptive Retraining**: Quarterly regime model updates with walk-forward validation
- **Dynamic Position Sizing**: Confidence-based position sizing with volatility targeting
- **Ensemble Methods**: Multiple regime detection models with confidence blending
- **Real-time Analysis**: Live regime tracking with transition alerts
- **Risk Controls**: Stop-loss, take-profit, and maximum drawdown limits
- **Performance Monitoring**: Comprehensive metrics and visualization

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **Data Processing**: numpy, pandas, scipy
- **Finance**: yfinance, pandas-datareader, alpaca-py
- **Machine Learning**: scikit-learn, hmmlearn, tensorflow
- **Statistics**: statsmodels, arch
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: pytest, black, flake8

## Quick Start

### Basic Usage

```python
from main import AnalysisPipeline, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    n_regimes=5,
    use_adaptive_retraining=True
)

# Run complete analysis
pipeline = AnalysisPipeline(config)
pipeline.run()
```

### Demo Notebook

Run the interactive demo:

```bash
jupyter notebook ProjectDemo.ipynb
```

## Project Structure

```
researchlucas/
├── main.py                          # Main analysis pipeline
├── src/
│   ├── data.py                     # Data loading and preprocessing
│   ├── regime.py                   # Market regime detection
│   ├── deep_learning.py            # LSTM-based regime detection
│   ├── signals.py                  # Signal generation
│   ├── risk.py                     # Risk management
│   ├── monte_carlo.py              # Monte Carlo simulation
│   ├── backtest.py                 # Backtesting engine
│   ├── visualization.py            # Plotting and visualization
│   └── portfolio.py                # Portfolio management
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── data_cache/                     # Cached market data
├── plots/                          # Generated plots and reports
├── requirements.txt                 # Python dependencies
└── ProjectDemo.ipynb               # Interactive demo
```

## Core Modules

### Data Management (`src/data.py`)

- **PortfolioConfig**: Configuration for data loading
- **DataLoader**: Yahoo Finance integration with caching
- Features: OHLCV data, macro indicators (VIX, yield curves), resampling

### Regime Detection (`src/regime.py`)

- **MarketRegimeDetector**: HMM-based regime detection
- **AdaptiveRegimeDetector**: Adaptive retraining with walk-forward analysis
- **RegimeConfig**: Configuration for regime detection parameters
- Features: 5-regime classification, ensemble methods, confidence scoring

### Deep Learning (`src/deep_learning.py`)

- **LSTMRegimeDetector**: LSTM-based regime classification
- **DeepLearningConfig**: Configuration for neural network parameters
- Features: Sequence modeling, early stopping, probability predictions

### Signal Generation (`src/signals.py`)

- **SignalGenerator**: Multi-factor signal generation
- Features: Trend, momentum, mean reversion, RSI, MACD, regime filtering

### Risk Management (`src/risk.py`)

- **RiskManager**: Comprehensive risk analysis
- **RiskConfig**: Risk management configuration
- Features: VaR, Expected Shortfall, drawdown analysis, stress testing

### Monte Carlo (`src/monte_carlo.py`)

- **MonteCarlo**: Advanced simulation engine
- **SimConfig**: Simulation configuration
- Features: 50,000+ simulations, GARCH volatility, Student's t-distribution

### Backtesting (`src/backtest.py`)

- **BacktestEngine**: Comprehensive backtesting framework
- Features: Walk-forward analysis, regime-aware sizing, risk controls

### Visualization (`src/visualization.py`)

- **PortfolioVisualizer**: Advanced plotting capabilities
- Features: Regime overlays, Monte Carlo paths, risk metrics dashboard

## Configuration

### Analysis Configuration

```python
config = AnalysisConfig(
    # Data settings
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    
    # Regime detection
    n_regimes=5,
    window_size=10,
    use_adaptive_retraining=True,
    
    # Risk management
    stop_loss=0.02,
    take_profit=0.30,
    max_drawdown=0.10,
    
    # Position sizing
    use_dynamic_position_sizing=True,
    base_position_size=0.10,
    confidence_threshold=0.6
)
```

### Regime Detection Configuration

```python
regime_config = RegimeConfig(
    n_regimes=5,
    window_size=10,
    features=["returns", "volatility", "momentum"],
    ensemble_method="confidence_blend",
    use_deep_learning=True
)
```

## Usage Examples

### Market Regime Detection

```python
from src.regime import MarketRegimeDetector, RegimeConfig

config = RegimeConfig(n_regimes=5, window_size=10)
detector = MarketRegimeDetector(config)
regimes = detector.fit_predict(returns)
```

### Signal Generation

```python
from src.signals import SignalGenerator

generator = SignalGenerator(use_regime=True, regime_detector=detector)
signals = generator.generate_signals(market_data)
```

### Risk Analysis

```python
from src.risk import RiskManager, RiskConfig

risk_config = RiskConfig(confidence_level=0.95)
risk_manager = RiskManager(risk_config)
metrics = risk_manager.calculate_metrics(returns)
```

### Monte Carlo Simulation

```python
from src.monte_carlo import MonteCarlo, SimConfig

sim_config = SimConfig(n_sims=50000, n_days=252)
mc = MonteCarlo(sim_config)
results = mc.simulate(market_data)
```

### Backtesting

```python
from src.backtest import BacktestEngine

engine = BacktestEngine(
    returns=returns,
    signals=signals,
    regime_confidence=regime_confidence
)
results = engine.run()
```

## Advanced Features

### Adaptive Retraining

The system supports quarterly retraining of regime models:

```python
config = AnalysisConfig(
    use_adaptive_retraining=True,
    retrain_window=252,  # 1 year initial training
    retrain_freq=63      # Quarterly retraining
)
```

### Dynamic Position Sizing

Confidence-based position sizing with volatility targeting:

```python
config = AnalysisConfig(
    use_dynamic_position_sizing=True,
    base_position_size=0.10,
    confidence_threshold=0.6,
    vol_target=0.12
)
```

### Ensemble Methods

Multiple regime detection models with confidence blending:

```python
config = RegimeConfig(
    ensemble_method="confidence_blend",
    use_deep_learning=True
)
```

## Testing

### Unit Tests

```bash
pytest tests/unit/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Test Coverage

```bash
pytest --cov=src tests/
```

## Output and Reports

The system generates comprehensive outputs:

- **Plots**: Regime analysis, Monte Carlo paths, risk metrics
- **Reports**: Performance metrics, regime statistics, backtest results
- **Data**: Cached market data, regime classifications, signal data

### Generated Files

- `plots/`: Visualization outputs
- `data_cache/`: Cached market data
- `*.txt`: Analysis reports
- `*.csv`: Grid search results

## Performance Optimization

- **Caching**: Market data caching for faster repeated analysis
- **Parallel Processing**: Multi-core regime detection
- **Memory Management**: Efficient data structures and cleanup
- **Early Stopping**: Neural network training optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is for research and educational purposes.

## Acknowledgments

- Yahoo Finance for market data
- Academic research on regime detection and portfolio optimization
- Open-source quantitative finance community

## Contact

For questions or contributions, please refer to the project documentation and test suite.

