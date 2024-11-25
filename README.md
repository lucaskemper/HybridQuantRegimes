# 📊 Quantitative Portfolio Risk Analysis System
> Advanced framework for semiconductor portfolio risk assessment combining regime detection, Monte Carlo simulation, and sophisticated risk metrics.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/username/repo/issues)

## 🎯 Key Features
- **Market Regime Detection**: Volatility-based classification (Low/Medium/High)
- **Advanced Monte Carlo**: Heavy-tailed distributions, GARCH modeling
- **Comprehensive Risk Metrics**: VaR, ES, Maximum Drawdown
- **Real-time Analysis**: Live semiconductor portfolio monitoring

## 🔬 Research & Implementation
### Market Regime Detection
```python
def detect_regime(self, volatility: np.ndarray) -> str:
    """
    Classify market regime based on rolling volatility
    Low: ≤33rd percentile
    Medium: 33rd-67th percentile
    High: ≥67th percentile
    """
    percentile_33 = np.percentile(volatility, 33)
    percentile_67 = np.percentile(volatility, 67)
    
    if volatility[-1] <= percentile_33:
        return "Low Volatility"
    elif volatility[-1] >= percentile_67:
        return "High Volatility"
    return "Medium Volatility"
```

### Monte Carlo Engine
```python
def simulate(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
    """Monte Carlo simulation with configurable distributions"""
    n_assets = len(returns.columns)
    paths = np.zeros((self.config.n_sims, n_assets, self.config.n_days))

    # Generate correlated returns
    for i in range(self.config.n_days):
        if self.distribution == 'normal':
            z = np.random.standard_normal((self.config.n_sims, n_assets))
        elif self.distribution == 't':
            z = scipy_stats.t.rvs(df=3, size=(self.config.n_sims, n_assets))
```

## 📈 Empirical Results
### Performance Metrics
| Metric | Value | Confidence Interval |
|--------|-------|-------------------|
| Portfolio Volatility | 42.72% | [38.5%, 46.9%] |
| VaR (95%) | -3.99% | [-4.2%, -3.7%] |
| Expected Shortfall | -5.74% | [-6.1%, -5.4%] |
| Sharpe Ratio | 0.73 | [0.65, 0.81] |
| Information Ratio | 0.68 | [0.61, 0.75] |

### Regime Analysis
| Regime | Return | Vol | Distribution | Transition Prob |
|--------|---------|-----|--------------|-----------------|
| Low | 4.36% | 26.10% | 32.5% | [0.81, 0.12, 0.07] |
| Medium | 2.81% | 37.00% | 33.4% | [0.09, 0.82, 0.09] |
| High | -0.25% | 57.06% | 32.5% | [0.08, 0.11, 0.81] |

## 🚀 Quick Start
```python
from src.risk import RiskManager, RiskConfig

# Initialize with advanced configuration
risk_config = RiskConfig(
    confidence_level=0.95,
    distribution='student_t',
    df=3,
    garch_order=(1,1),
    vol_window=63
)

# Run analysis
risk_manager = RiskManager(risk_config)
results = risk_manager.analyze_portfolio(market_data)
```

## 📦 Project Structure
```
📦 src
 ┣ 📜 risk.py          # Core risk metrics
 ┣ 📜 monte_carlo.py   # Simulation engine
 ┣ 📜 visualization.py # Interactive plots
 ┗ 📜 data.py         # Data processing
```

## 📚 References & Research
- [Regime Shifts in Asset Allocation](https://doi.org/10.1093/rfs/15.4.1137)
- [Quantitative Risk Management](https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management)
- [GARCH Models](https://doi.org/10.1016/0304-4076(86)90063-1)

## 👥 Authors
- **Lucas Kemper** - *HEC Lausanne* - [🔗](https://github.com/lucaskemper)
- **Antonio Schoeffel** - *HEC Lausanne* - [🔗](https://github.com/antonioschoeffel)

## 📄 License
MIT License - see [LICENSE.md](LICENSE.md)
