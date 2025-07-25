# HybridQuantRegimes

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technical Details](#technical-details)
  - [1. Regime Detection (HMM + LSTM/Transformer)](#1-regime-detection-hmm--lstmtransformer)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Signal Generation](#3-signal-generation)
  - [4. Risk Management](#4-risk-management)
  - [5. Backtesting](#5-backtesting)
  - [6. Monte Carlo Simulation](#6-monte-carlo-simulation)
  - [7. Statistical Validation](#7-statistical-validation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Module Overview](#module-overview)
- [Testing](#testing)
- [Data](#data)
- [Results & Outputs](#results--outputs)
- [References](#references)
- [License](#license)

## Overview

This project implements a **hybrid market regime detection and risk management framework** for quantitative finance, with a focus on semiconductor equity markets. It combines **Hidden Markov Models (HMM)** and **deep learning (LSTM/Transformer)** to identify market regimes, generate regime-aware trading signals, manage risk dynamically, and perform robust backtesting and scenario analysis. The system is modular, extensible, and designed for both research and practical portfolio management.

## Key Features

- **Hybrid Regime Detection**: Combines HMM and LSTM/Transformer models for robust, multi-horizon regime identification with fallback and smoothing mechanisms.
- **Regime-Aware Signal Generation**: Produces trading signals that adapt to detected market regimes.
- **Dynamic Risk Management**: Implements regime-dependent risk metrics (VaR, ES, drawdown, volatility targeting, etc.) and adaptive position sizing.
- **Comprehensive Backtesting**: Realistic backtest engine with walk-forward analysis, transaction costs, slippage, leverage, and risk controls.
- **Monte Carlo Simulation**: Scenario analysis using regime-conditional and heavy-tailed distributions.
- **Statistical Validation**: Extensive validation of model performance and risk metrics.
- **Visualization**: Rich plotting and reporting of results, including regime transitions, risk, and performance.
- **Modular Design**: Easily extensible for new assets, features, or models.

## Architecture

```
DataLoader/PortfolioConfig
        ↓
MarketRegimeDetector (HMM + LSTM/Transformer)
        ↓
SignalGenerator (regime-aware signals)
        ↓
RiskManager (regime-dependent risk, position sizing)
        ↓
BacktestEngine (realistic backtesting)
        ↓
MonteCarlo (scenario analysis)
        ↓
Visualization/Reporting
```

- **src/data.py**: Data loading, preprocessing, and feature engineering (Yahoo Finance, macro data, caching).
- **src/regime.py**: Regime detection (HMM, LSTM, smoothing, Bayesian model averaging).
- **src/deep_learning.py**: LSTM/Transformer regime models (TensorFlow, attention, regularization).
- **src/signals.py**: Regime-aware signal generation (momentum, normalization, diagnostics).
- **src/risk.py**: Risk management (VaR, ES, drawdown, volatility targeting, regime-aware limits).
- **src/backtest.py**: Backtesting engine (walk-forward, constraints, risk triggers).
- **src/monte_carlo.py**: Monte Carlo simulation (normal/t/regime-conditional, multi-asset, confidence intervals).
- **src/visualization.py**: Portfolio and regime visualization.
- **src/statistical_validation.py**: Statistical validation and diagnostics.

## Technical Details

### 1. Regime Detection (HMM + LSTM/Transformer)
- **Hidden Markov Model (HMM):**
  - Detects discrete market regimes (e.g., Low, Medium, High Volatility) using rolling windows of engineered features.
  - Parameters: `n_regimes`, `window_size`, `min_regime_size`, `smoothing_window`, `features` (returns, volatility, momentum, skewness, kurtosis, etc.).
  - Smoothing: Mode filter over regime sequence to reduce spurious transitions.
  - Transition matrix: Estimated via maximum likelihood, used for regime persistence and transition alerts.
- **LSTM/Transformer Deep Learning:**
  - LSTM model with configurable layers, attention, bidirectionality, batch normalization, dropout, and residual connections.
  - Parameters: `sequence_length`, `hidden_dims`, `epochs`, `learning_rate`, `dropout_rate`, `early_stopping_patience`, `learning_rate_schedule` (cosine annealing), etc.
  - Trained to predict regime labels from engineered features, optionally using HMM output as targets.
- **Bayesian Model Averaging:**
  - Combines HMM and LSTM regime probabilities using entropy- or evidence-based weights:
    - \( P_{BMA}(y|x) = w_{HMM} P_{HMM}(y|x) + w_{LSTM} P_{LSTM}(y|x) \)
  - Weights adapt based on model confidence (entropy/log-evidence).
- **Real-time/Online Options:**
  - Alert thresholds, minimum confidence, update frequency, and regime history tracking for live applications.

### 2. Feature Engineering
- **Core Features:**
  - Returns (raw, log), rolling and EWMA volatility, realized volatility, momentum (various windows), moving averages (fast/slow), MACD, RSI (14/30), Bollinger position, Williams %R, on-balance volume, skewness, kurtosis, price position, and more.
- **Macro Features:**
  - VIX level/change, yield curve (TNX, TYX), term structure slope, dollar strength (DXY), macro regime indicators.
- **Semiconductor-Specific Features:**
  - Semiconductor PMI, memory vs. logic spread, equipment vs. design ratio.
- **Normalization:**
  - Optional z-score normalization per feature.
- **Missing Data Handling:**
  - Forward/backward fill, then zero-imputation for robustness.

### 3. Signal Generation
- **Momentum-based Signals:**
  - Short-term (3-day) and medium-term (10-day) momentum.
  - Moving average crossover (fast/slow), price position relative to recent high.
  - Combined using weighted sum and nonlinearity (tanh), clipped to [-1, 1].
- **Regime Awareness:**
  - Signals can be modulated or filtered based on detected regime (e.g., more aggressive in low-vol, defensive in high-vol).
- **Diagnostics:**
  - Signal statistics (mean, std, range, nonzero count), forward correlation with returns, and red-flag detection for weak or sparse signals.

#### Recent Changes to Signal Logic
- **Regime-Specific Feature Weights:** Signals now use regime-specific feature weights learned via Ridge regression, adapting the signal formula to each detected regime.
- **Dynamic Signal Formulation:**
  - In "High Vol" regimes, signals emphasize mean-reversion (RSI, Williams %R).
  - In "Low Vol" regimes, signals emphasize momentum.tensorflow_probability
  - In "Medium Vol" regimes, signals blend both approaches.
- **Soft Thresholding:** Weak signals are softly thresholded, retaining some edge for low-signal environments.
- **Flexible Scaling:** Signal scaling can be set to `tanh` or `clip` (see config), with a tunable scaling factor (`scaler_k`).
- **Volatility Normalization:** Optionally normalize signals by realized volatility (configurable).
- **Diagnostics:** Extensive diagnostics and debug output for signal quality, feature coverage, and regime adjustment.
- **Configurable via `main.py` and YAML:** All signal logic options (regime use, scaling, normalization, etc.) are exposed in the pipeline and config.

### 4. Risk Management
- **Risk Metrics:**
  - Annualized volatility, Value at Risk (VaR, historical/parametric/Monte Carlo), Expected Shortfall (ES), Sharpe/Sortino/Calmar/Omega ratios, max drawdown, rolling volatility, tail ratio, regime-dependent risk.
- **Dynamic Position Sizing:**
  - Proportional, fixed, regime-confidence, dynamic (volatility-adjusted), or custom (e.g., risk parity) sizing.
  - Constraints: leverage, max position size, min trade size, short selling toggle.
- **Risk Controls:**
  - Stop-loss, take-profit, max drawdown, custom risk management hooks (e.g., consecutive losses, extreme daily loss).
- **Stress Testing:**
  - Scenario analysis (e.g., 2008 crisis, COVID crash), worst month/quarter, recovery time, max consecutive loss.
- **Regime-Dependent Correlation:**
  - Correlation matrices and risk decomposition by regime.

### 5. Backtesting
- **Walk-Forward Analysis:**
  - Rolling window and step for out-of-sample validation.
- **Transaction Costs:**
  - Proportional, fixed, and slippage costs per trade.
- **Performance Metrics:**
  - Total/annualized return, volatility, Sharpe, max drawdown, win rate, turnover, Calmar ratio, and more.
- **Diagnostics:**
  - Position, trade, and equity curve tracking; confidence metrics for regime-based sizing.

### 6. Monte Carlo Simulation
- **Simulation Engine:**
  - Supports normal, t-distribution, and regime-conditional return generation.
  - GARCH(1,1) volatility forecasting per asset.
  - Cholesky decomposition for correlated asset paths.
- **Metrics:**
  - Expected/annualized return, volatility, Sharpe, VaR/CVaR, max drawdown, information/sortino ratios, success rate.
  - Confidence intervals for portfolio value at multiple quantiles.
- **Validation:**
  - Checks for positive values, correlation preservation, volatility alignment, and return reasonability.
- **Reporting:**
  - Plots of simulation paths, distribution, and risk metrics; summary statistics and JSON/CSV export.

### 7. Statistical Validation
- **Accuracy Testing:**
  - Binomial test for regime detection accuracy vs. random chance.
- **Performance Testing:**
  - Paired t-test and bootstrap confidence intervals for model outperformance (e.g., vs. baseline or HMM-only).
- **Cross-Validation:**
  - TimeSeriesSplit for regime stability and persistence.
- **Bayesian Model Averaging:**
  - Weighted combination of model probabilities based on entropy or log-evidence.
- **Extreme Value Theory:**
  - Tail index estimation for risk of rare events.

## Installation

1. **Clone the repository**

```bash
git clone <repo_url>
cd researchlucas
```

2. **Install dependencies** (Python 3.11 recommended)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **(Optional) Set up environment variables**

- If using API keys or custom data sources, create a `.env` file in the root directory.

## Quick Start

Get up and running with the default pipeline and demo data in just a few steps:

```bash
git clone <repo_url>
cd researchlucas
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

- This will run the full pipeline using the default configuration and cached demo datasets.
- Results and plots will be saved in the `results/` and `plots/` directories.
- For a step-by-step interactive demo, launch:

```bash
jupyter notebook ProjectDemo.ipynb
```

## Usage

### 1. Run the Main Pipeline

```bash
python main.py
```

This will execute the full pipeline:
- Data loading and preprocessing
- Regime detection (HMM + LSTM/Transformer)
- Signal generation
- Risk analysis
- Backtesting
- Monte Carlo simulation
- Visualization and reporting (results saved in `results/` and `plots/`)

### 2. Jupyter Notebook Demo

A demo notebook is provided:

```bash
jupyter notebook ProjectDemo.ipynb
```

This notebook demonstrates step-by-step usage, including custom portfolio configuration, regime detection, and visualization.

### 3. Customization

- **Portfolio**: Edit tickers, weights, and dates in `main.py` or the notebook.
- **Model Parameters**: Adjust regime, LSTM, and risk configs in `main.py` or via config classes.
- **Data**: Place custom CSVs in `data_cache/` or use live download.

## Configuration

- **config.yml**: (Optional) For advanced configuration.
- **.env**: (Optional) For API keys or environment variables.
- **PortfolioConfig**: Tickers, weights, dates, macro indicators.
- **RegimeConfig**: Number of regimes, window size, features, smoothing, deep learning options.
- **RiskConfig**: Risk limits, VaR/ES methods, stop-loss, take-profit, volatility target.
- **SimConfig**: Monte Carlo simulation parameters.

## Module Overview

- **src/data.py**: `DataLoader`, `PortfolioConfig` — Loads and preprocesses market and macro data, supports caching and normalization.
- **src/regime.py**: `MarketRegimeDetector`, `RegimeConfig` — Detects regimes using HMM and LSTM, supports smoothing, Bayesian averaging, and real-time tracking.
- **src/deep_learning.py**: `LSTMRegimeDetector`, `DeepLearningConfig` — Deep learning regime models with attention, regularization, and early stopping.
- **src/signals.py**: `SignalGenerator` — Generates regime-aware trading signals, supports diagnostics.
- **src/risk.py**: `RiskManager`, `RiskConfig` — Calculates risk metrics, manages position sizing, and enforces risk controls.
- **src/backtest.py**: `BacktestEngine` — Realistic backtesting with walk-forward, transaction costs, leverage, and risk triggers.
- **src/monte_carlo.py**: `MonteCarlo`, `SimConfig` — Monte Carlo scenario analysis with regime-conditional and heavy-tailed distributions.
- **src/visualization.py**: `PortfolioVisualizer` — Plots performance, regimes, risk, and simulation results.
- **src/statistical_validation.py**: `StatisticalValidator` — Validates model and risk metrics.

## Testing

- **Unit tests**: Located in `tests/unit/` (e.g., `test_data.py`).
- **Integration tests**: Located in `tests/integration/` (e.g., `test_workflow.py`).
- **Run all tests**:

```bash
pytest
```

- **Test coverage**:

```bash
pytest --cov=src
```

## Data

- **Live download**: Uses Yahoo Finance via `yfinance`.
- **Cached data**: Place CSVs in `data_cache/` for reproducibility.
- **Supported assets**: Equities, ETFs, macro indicators (VIX, ^TNX, ^TYX, etc.).

## Results & Outputs

- **Results**: Saved in `results/` and `output_conservative/`.
- **Plots**: Saved in `plots/` (e.g., equity curves, regime transitions, risk metrics).
- **Logs**: Written to `regime_detection.log`.


## License

This project is for research and educational purposes. For commercial or production use, please contact the author.

---
