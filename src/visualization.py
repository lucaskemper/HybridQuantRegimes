from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PortfolioVisualizer:
    """Comprehensive portfolio visualization and metrics calculation"""

    def __init__(self):
        self.colors = {
            "primary": "#2962FF",
            "secondary": "#FF6D00",
            "accent": "#00C853",
            "background": "#1E1E1E",
            "text": "#FFFFFF",
            "grid": "#333333",
        }
        # Set default style
        plt.style.use("dark_background")

    def plot_portfolio_analysis(
        self,
        market_data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        mc_results: Dict,
        regimes: Optional[pd.Series] = None,
    ) -> None:
        """Create comprehensive portfolio analysis plots

        Parameters:
            market_data (Dict): Dictionary containing market data ('prices', 'returns')
            signals (pd.DataFrame): Trading signals for each asset
            mc_results (Dict): Monte Carlo simulation results
            regimes (pd.Series, optional): Market regime labels
        """
        fig = plt.figure(figsize=(15, 10))

        # 1. Price Movement with Returns and Regimes
        ax1 = plt.subplot(2, 2, 1)
        self._plot_price_evolution(market_data["prices"], ax1, regimes)

        # 2. Signal Heatmap
        ax2 = plt.subplot(2, 2, 2)
        self._plot_signals_heatmap(signals, ax2)

        # 3. Monte Carlo Paths
        ax3 = plt.subplot(2, 2, 3)
        self._plot_monte_carlo_paths(mc_results, ax3)

        # 4. Return Distribution
        ax4 = plt.subplot(2, 2, 4)
        self._plot_return_distribution(mc_results, ax4)

        plt.tight_layout()
        plt.show()

    def _plot_price_evolution(
        self, prices: pd.DataFrame, ax: plt.Axes, regimes: Optional[pd.Series] = None
    ) -> None:
        """Plot price evolution with optional regime overlay"""
        normalized = prices / prices.iloc[0]

        for i, col in enumerate(normalized.columns):
            ax.plot(
                normalized.index,
                normalized[col],
                label=col,
                color=list(self.colors.values())[i % 3],
                linewidth=1.5,
                alpha=0.9,
            )

        if regimes is not None:
            self._add_regime_overlay(ax, normalized.index, regimes)

        ax.set_title("Asset Price Evolution", pad=20)
        ax.legend(frameon=True, facecolor=self.colors["background"])
        ax.grid(True, color=self.colors["grid"], alpha=0.2)

    def _plot_signals_heatmap(self, signals: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot signal heatmap"""
        sns.heatmap(
            signals.iloc[-20:],
            center=0,
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Signal Strength"},
        )
        ax.set_title("Recent Trading Signals")

    def _plot_monte_carlo_paths(self, mc_results: Dict, ax: plt.Axes) -> None:
        """Plot Monte Carlo simulation paths"""
        paths = mc_results["paths"]
        ax.plot(paths[:, :100].T, alpha=0.1, color=self.colors["primary"])
        ax.axhline(y=1, color=self.colors["text"], linestyle="--")
        ax.set_title("Monte Carlo Simulation Paths")
        ax.grid(True, color=self.colors["grid"], alpha=0.2)

    def _plot_return_distribution(self, mc_results: Dict, ax: plt.Axes) -> None:
        """Plot return distribution with VaR"""
        sns.histplot(mc_results["final_values"], bins=50, ax=ax)
        ax.axvline(
            mc_results["var_95"],
            color=self.colors["secondary"],
            linestyle="--",
            label=f"VaR (95%): {format_currency(mc_results['var_95'])}",
        )
        ax.set_title("Final Value Distribution")
        ax.legend()
        ax.grid(True, color=self.colors["grid"], alpha=0.2)

    def _add_regime_overlay(
        self, ax: plt.Axes, dates: pd.DatetimeIndex, regimes: pd.Series
    ) -> None:
        """Add regime overlay to price evolution plot"""
        regime_colors = {"Low Vol": "green", "Medium Vol": "yellow", "High Vol": "red"}

        ylim = ax.get_ylim()
        regime_changes = regimes.ne(regimes.shift()).cumsum()

        for regime in regime_colors:
            mask = regimes == regime
            if mask.any():
                ax.fill_between(
                    dates,
                    ylim[0],
                    ylim[1],
                    where=mask,
                    color=regime_colors[regime],
                    alpha=0.1,
                    label=f"{regime} Regime",
                )

    def plot_risk_metrics(self, metrics: Dict) -> None:
        """Plot risk metrics dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        self._plot_risk_summary(metrics, axes[0, 0])
        self._plot_correlation_matrix(metrics["correlation"], axes[0, 1])
        self._plot_rolling_metrics(metrics, axes[1, 0])
        self._plot_drawdown(metrics, axes[1, 1])

        plt.tight_layout()
        plt.show()

    def _plot_risk_summary(self, metrics: Dict, ax: plt.Axes) -> None:
        """Plot risk metrics summary"""
        key_metrics = {
            "Volatility": f"{metrics['annual_vol']:.1%}",
            "Sharpe": f"{metrics['sharpe']:.2f}",
            "VaR (95%)": f"{metrics['var_95']:.1%}",
            "Max DD": f"{metrics['max_drawdown']:.1%}",
        }

        y_pos = np.arange(len(key_metrics))
        values = [float(str(v).rstrip("%")) for v in key_metrics.values()]

        ax.barh(y_pos, np.abs(values), color=self.colors["primary"], alpha=0.3)

        for i, (key, value) in enumerate(key_metrics.items()):
            ax.text(0, i, f"{key}: {value}", va="center")

        ax.set_title("Key Risk Metrics")
        ax.set_yticks([])
        ax.set_xticks([])


# Utility functions moved from utils.py
def calculate_metrics(returns: pd.DataFrame) -> Dict:
    """Calculate portfolio performance metrics"""
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_returns / annual_vol
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    var_95 = returns.quantile(0.05)

    return {
        "annual_returns": annual_returns,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "correlation": returns.corr(),
    }


def calculate_tracking_error(
    returns: pd.DataFrame, benchmark_returns: pd.Series
) -> float:
    """Calculate tracking error against benchmark"""
    return np.std((returns - benchmark_returns)) * np.sqrt(252)


def calculate_information_ratio(
    returns: pd.DataFrame, benchmark_returns: pd.Series
) -> float:
    """Calculate information ratio"""
    excess_returns = returns - benchmark_returns
    tracking_error = calculate_tracking_error(returns, benchmark_returns)
    return (excess_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0


def calculate_risk_contribution(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Calculate risk contribution of each asset"""
    cov = returns.cov() * 252
    portfolio_vol = np.sqrt(weights.T @ cov @ weights)
    marginal_contrib = cov @ weights
    return pd.Series(weights * marginal_contrib / portfolio_vol, index=returns.columns)


def format_percentage(value: float) -> str:
    """Format float as percentage string"""
    return f"{value:.2%}"


def format_currency(value: float, currency: str = "$") -> str:
    """Format float as currency string"""
    return f"{currency}{value:,.2f}"


def save_analysis_results(
    analysis: Dict, file_path: str, include_plots: bool = True
) -> None:
    """Save analysis results to file"""
    results_df = pd.DataFrame(analysis)
    results_df.to_csv(file_path)

    if include_plots:
        plt.savefig(file_path.replace(".csv", ".png"))
