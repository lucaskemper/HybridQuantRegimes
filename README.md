# Quantitative Portfolio Risk Analysis System

A sophisticated framework for market regime detection, Monte Carlo simulation, and risk assessment in semiconductor equity portfolios.

## Authors
- Lucas Kemper (HEC Lausanne)
- Antonio Schoeffel (HEC Lausanne)

## Abstract

This research implements a comprehensive quantitative system for analyzing portfolio risk in the semiconductor sector. The framework combines volatility-based regime detection, Monte Carlo simulation with heavy-tailed distributions, and advanced risk metrics to provide robust portfolio analysis and risk assessment capabilities.

## Methodology

### 1. Market Regime Detection

The system employs a volatility-based approach to identify distinct market regimes:
- Low Volatility Regime (≤33rd percentile)
- Medium Volatility Regime (33rd-67th percentile)
- High Volatility Regime (≥67th percentile)

Regime classification utilizes rolling volatility calculations with both 21-day and 63-day windows to capture different temporal dynamics.

### 2. Monte Carlo Simulation

The simulation framework incorporates:
- Multiple distribution options (Normal, Student's t)
- GARCH volatility forecasting
- Cholesky decomposition for correlation preservation
- Comprehensive validation metrics

```python
def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
    """Run Monte Carlo simulation with configurable distributions"""
    n_assets = len(returns.columns)
    paths = np.zeros((self.config.n_sims, n_assets, self.config.n_days))

    # Generate correlated returns
    for i in range(self.config.n_days):
        if self.distribution == 'normal':
            z = np.random.standard_normal((self.config.n_sims, n_assets))
        elif self.distribution == 't':
            z = scipy_stats.t.rvs(df=3, size=(self.config.n_sims, n_assets))
```

### 3. Risk Assessment

Comprehensive risk metrics calculation including:
- Value at Risk (VaR) at 95% and 99% confidence levels
- Expected Shortfall (ES)
- Maximum Drawdown
- Rolling Volatility
- Correlation Analysis
- Sharpe Ratio

## Implementation

### Core Components
```
src/
├── risk.py         # Risk metrics calculation
├── monte_carlo.py  # Monte Carlo simulation engine
├── visualization.py # Data visualization tools
└── data.py         # Data loading and preprocessing
```

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
arch>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Empirical Results

Analysis of semiconductor portfolio (NVDA, AMD, INTC) from 2020-2024 demonstrates:

| Metric | Value |
|--------|-------|
| Portfolio Volatility | 42.72% |
| VaR (95%) | -3.99% |
| Expected Shortfall (95%) | -5.74% |
| Sharpe Ratio | 0.73 |
| Annualized Return | 44.57% |

### Risk Regime Analysis

| Regime | Avg Monthly Return | Volatility | Time Distribution |
|--------|-------------------|------------|-------------------|
| Low Vol | 4.36% | 26.10% | 32.5% |
| Medium Vol | 2.81% | 37.00% | 33.4% |
| High Vol | -0.25% | 57.06% | 32.5% |

## Usage

```python
from src.data import DataLoader, PortfolioConfig
from src.risk import RiskManager, RiskConfig
from src.visualization import RiskVisualizer

# Configure portfolio
portfolio_config = PortfolioConfig(
    tickers=['NVDA', 'AMD', 'INTC'],
    weights=[0.4, 0.4, 0.2],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Initialize risk analysis
risk_config = RiskConfig(
    confidence_level=0.95,
    max_drawdown_limit=0.20,
    volatility_target=0.15
)

# Calculate risk metrics
risk_manager = RiskManager(risk_config)
risk_metrics = risk_manager.calculate_metrics(market_data['returns'])
```

## References

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
2. McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative risk management: Concepts, techniques and tools. Princeton University Press.
3. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. The Review of Financial Studies, 15(4), 1137-1187.