# Quantitative Portfolio Risk Analysis System: A Framework for Market Regime Detection and Risk Assessment in Semiconductor Equities
(Classwork -due december, scope / content might change at any time)  v0.1
Current state:
Data : Working
Monte_carlo : Working
Visualization : Check
Signals : Not Working
...

## Authors and Institutional Affiliation
- **Lucas Kemper** - MscF Student, HEC Lausanne
- **Antonio Schoeffel** -  MscF Student, HEC Lausanne
## Setup Guide

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- Git (optional)

### Installation

#### 1. Basic Installation
```bash
# Clone repository
git clone <repository-url>
cd <repository-directory>

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install package
pip install -e .
```

#### 2. Development Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Configuration
The system requires configuration of risk parameters and portfolio settings. See `config.yml` for customization options.

### Validation
```bash
# Run test suite
pytest tests/

# Run type checking
mypy src/
```

## Abstract
This research presents a comprehensive quantitative framework for analyzing portfolio risk in the semiconductor sector, with particular emphasis on regime detection methodologies and Monte Carlo simulations incorporating heavy-tailed distributions. The system implements sophisticated statistical approaches for risk assessment, including Value at Risk (VaR) and Expected Shortfall (ES) calculations, while accounting for regime-dependent volatility dynamics.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Methodological Framework

### 1. Market Regime Detection Methodology
The framework employs a sophisticated volatility-based approach for regime identification, utilizing both short-term (21-day) and medium-term (63-day) rolling windows to capture temporal dynamics in market conditions. The regime classification methodology follows:

```python
def detect_regime(self, volatility: np.ndarray) -> str:
    """
    Implements regime classification based on rolling volatility measures
    utilizing empirical quantile thresholds.
    
    Parameters:
        volatility (np.ndarray): Time series of realized volatilities
        
    Returns:
        str: Classified regime state
    """
    percentile_33 = np.percentile(volatility, 33)
    percentile_67 = np.percentile(volatility, 67)
    
    if volatility[-1] <= percentile_33:
        return "Low Volatility"
    elif volatility[-1] >= percentile_67:
        return "High Volatility"
    return "Medium Volatility"
```

### 2. Monte Carlo Simulation Framework
The simulation methodology incorporates:

```python
def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Implements Monte Carlo simulation with configurable distributional assumptions
    and GARCH volatility processes.
    
    Parameters:
        market_data (Dict[str, pd.DataFrame]): Historical market data
        
    Returns:
        Dict: Simulation results and validation metrics
    """
    n_assets = len(returns.columns)
    paths = np.zeros((self.config.n_sims, n_assets, self.config.n_days))

    for i in range(self.config.n_days):
        if self.distribution == 'normal':
            z = np.random.standard_normal((self.config.n_sims, n_assets))
        elif self.distribution == 't':
            z = scipy_stats.t.rvs(df=3, size=(self.config.n_sims, n_assets))
```

## Empirical Results and Statistical Analysis

### 1. Portfolio Performance Metrics
| Metric | Value | 95% Confidence Interval | Statistical Significance |
|--------|-------|------------------------|-------------------------|
| Portfolio Volatility | 42.72% | [38.5%, 46.9%] | p < 0.001 |
| Value at Risk (95%) | -3.99% | [-4.2%, -3.7%] | p < 0.001 |
| Expected Shortfall | -5.74% | [-6.1%, -5.4%] | p < 0.001 |
| Sharpe Ratio | 0.73 | [0.65, 0.81] | p < 0.05 |
| Information Ratio | 0.68 | [0.61, 0.75] | p < 0.05 |

### 2. Regime Characteristics and Transition Dynamics
| Regime | Mean Return (μ) | Volatility (σ) | Stationary Distribution | Transition Matrix Row |
|--------|----------------|----------------|------------------------|---------------------|
| Low | 4.36% (t=3.42)* | 26.10% | 32.5% | [0.81, 0.12, 0.07] |
| Medium | 2.81% (t=2.15)* | 37.00% | 33.4% | [0.09, 0.82, 0.09] |
| High | -0.25% (t=-0.18) | 57.06% | 32.5% | [0.08, 0.11, 0.81] |
*Statistically significant at p < 0.05

## Implementation Architecture
```
src/
├── risk.py          # Risk metric implementations
├── monte_carlo.py   # Simulation engine
├── visualization.py # Statistical visualization
└── data.py         # Data preprocessing
```

## Methodological Implementation
```python
from src.risk import RiskManager, RiskConfig

risk_config = RiskConfig(
    confidence_level=0.95,
    distribution='student_t',
    df=3,  # Degrees of freedom for t-distribution
    garch_order=(1,1),
    vol_window=63  # Rolling window for volatility estimation
)

risk_manager = RiskManager(risk_config)
results = risk_manager.analyze_portfolio(market_data)
```

## References
1. Ang, A., & Bekaert, G. (2002). "Regime Shifts in Asset Allocation." *The Review of Financial Studies*, 15(4), 1137-1187. https://doi.org/10.1093/rfs/15.4.1137

2. McNeil, A. J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk Management: Concepts, Techniques and Tools." *Princeton University Press*. ISBN: 978-0691166278

3. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327. https://doi.org/10.1016/0304-4076(86)90063-1
5.  McNeil, A.J., Frey, R. (2000). "Estimation of Tail-Related Risk Measures"
3. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Time Series"
4. Ang, A., Bekaert, G. (2002). "Regime Switches in Interest Rates"
5. Maïnassara, Y.B., Kadmiri O., Saussereau B. (2022). "Estimation of multivariate asymmetric power GARCH models"
6. Cunchala, A. (2024). "A Basic Overview of Various Stochastic Approaches to Financial Modeling With Examples"
7. Goyal, A., Welch, I., "A Comprehensive 2022 Look at the Empirical Performance of Equity Premium Prediction"
8. Jondeau, E., Rockinger, M. (2006). "The copula-garch model of conditional dependencies: An international stock market application"


## License and Distribution
This research implementation is distributed under the MIT License - see [LICENSE.md](LICENSE.md)
