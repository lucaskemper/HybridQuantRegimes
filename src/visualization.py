# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import pandas as pd
import numpy as np

class RiskVisualizer:
    """Visualization tools for risk metrics"""
    
    def __init__(self):
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        self.colors = sns.color_palette("husl")
    
    def create_risk_dashboard(self, market_data: Dict, risk_metrics: Dict):
        """Create comprehensive risk dashboard"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Asset Prices
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_prices(market_data['close'], ax1)
        
        # 2. Rolling Volatility
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_rolling_volatility(risk_metrics['rolling_volatility'], ax2)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_returns_dist(market_data['returns'], risk_metrics, ax3)
        
        # 4. Correlation Matrix
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_correlation(risk_metrics['correlation'], ax4)
        
        # 5. Risk Metrics Table
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_metrics_table(risk_metrics, ax5)
        
        # 6. Drawdown
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_drawdown(market_data['returns'], ax6)
        
        plt.tight_layout()
        return fig
    
    def _plot_prices(self, prices: pd.DataFrame, ax):
        """Plot normalized prices"""
        normalized = prices / prices.iloc[0]
        normalized.plot(ax=ax)
        ax.set_title('Normalized Asset Prices')
        ax.set_ylabel('Price (Normalized)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_rolling_volatility(self, rolling_vol: pd.DataFrame, ax):
        """Plot rolling volatility"""
        if not isinstance(rolling_vol, pd.DataFrame):
            raise ValueError("Rolling volatility must be a DataFrame")
        
        # Plot rolling volatilities
        for col in rolling_vol.columns:
            ax.plot(rolling_vol.index, rolling_vol[col], label=col)
        
        ax.set_title('Portfolio Rolling Volatility')
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.legend()
        ax.grid(True)
    
    def _plot_returns_dist(self, returns: pd.DataFrame, metrics: Dict, ax):
        """Plot returns distribution with VaR"""
        portfolio_returns = returns.sum(axis=1)
        sns.histplot(data=portfolio_returns, ax=ax, bins=50)
        
        # Add VaR lines
        ax.axvline(metrics['var_95'], color='r', linestyle='--', 
                  label=f"95% VaR: {metrics['var_95']:.2%}")
        ax.axvline(metrics['var_99'], color='darkred', linestyle='--',
                  label=f"99% VaR: {metrics['var_99']:.2%}")
        
        ax.set_title('Portfolio Returns Distribution')
        ax.legend()
    
    def _plot_correlation(self, corr: pd.DataFrame, ax):
        """Plot correlation heatmap"""
        sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
    
    def _plot_metrics_table(self, metrics: Dict, ax):
        """Plot risk metrics table"""
        metrics_display = {
            'Portfolio Volatility': f"{metrics['portfolio_volatility']:.2%}",
            'VaR (95%)': f"{metrics['var_95']:.2%}",
            'VaR (99%)': f"{metrics['var_99']:.2%}",
            'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}"
        }
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=[[k, v] for k, v in metrics_display.items()],
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Risk Metrics Summary')
    
    def _plot_drawdown(self, returns: pd.DataFrame, ax):
        """Plot drawdown"""
        portfolio_returns = returns.sum(axis=1)
        cum_returns = (1 + portfolio_returns).cumprod()
        drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
        drawdown.plot(ax=ax)
        ax.set_title('Portfolio Drawdown')
        ax.set_ylabel('Drawdown %')