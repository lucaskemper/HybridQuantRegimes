# src/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def plot_portfolio_analysis(market_data: Dict[str, pd.DataFrame], 
                          signals: pd.DataFrame,
                          mc_results: Dict) -> None:
    """Create comprehensive portfolio analysis plots"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Price Movement
    plt.subplot(2, 2, 1)
    for col in market_data['close'].columns:
        plt.plot(market_data['close'][col] / market_data['close'][col].iloc[0],
                label=col)
    plt.title('Normalized Price Movement')
    plt.legend()
    
    # 2. Signal Heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(signals.iloc[-20:], center=0, cmap='RdYlGn')
    plt.title('Recent Trading Signals')
    
    # 3. Monte Carlo Paths
    plt.subplot(2, 2, 3)
    paths = mc_results['paths']
    plt.plot(paths[:, :100].T, alpha=0.1, color='blue')
    plt.axhline(y=1, color='white', linestyle='--')
    plt.title('Monte Carlo Simulation Paths')
    
    # 4. Return Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(mc_results['final_values'], bins=50)
    plt.axvline(mc_results['var'], color='red', linestyle='--', 
                label=f"VaR: ${mc_results['var']:,.0f}")
    plt.title('Final Value Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(returns: pd.DataFrame) -> Dict:
    """Calculate portfolio performance metrics"""
    metrics = {}
    
    # Return metrics
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_returns / annual_vol
    
    # Risk metrics
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    var_95 = returns.quantile(0.05)
    
    metrics = {
        'annual_returns': annual_returns,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'correlation': returns.corr()
    }
    
    return metrics

def calculate_tracking_error(returns: pd.DataFrame, benchmark_returns: pd.Series) -> float:
    """Calculate tracking error against benchmark"""
    return np.std((returns - benchmark_returns)) * np.sqrt(252)

def calculate_information_ratio(returns: pd.DataFrame, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio"""
    excess_returns = returns - benchmark_returns
    tracking_error = calculate_tracking_error(returns, benchmark_returns)
    return (excess_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0

def calculate_risk_contribution(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Calculate risk contribution of each asset"""
    cov = returns.cov() * 252
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)
    marginal_contrib = cov @ weights
    return pd.Series(weights * marginal_contrib / portfolio_vol, index=returns.columns)