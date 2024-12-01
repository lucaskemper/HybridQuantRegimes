from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats


class RiskVisualizer:
    """Modern portfolio risk visualization dashboard"""

    def __init__(self):
        # Set style to minimal and modern
        plt.style.use("seaborn-v0_8-white")

        # Custom color palette inspired by modern fintech apps
        self.colors = {
            "primary": "#1a56db",  # Darker blue for better visibility
            "secondary": "#3b82f6",  # Keep current blue
            "accent": "#0ea5e9",  # Brighter blue for emphasis
            "warning": "#ef4444",  # Brighter red for risks
            "success": "#10b981",  # Brighter green
            "neutral": "#64748b",  # Darker gray for better contrast
            "background": "#ffffff",  # Keep white
            "text": "#0f172a",  # Darker text for better readability
            "grid": "#e2e8f0",  # Slightly darker grid
        }

        # Modern typography and layout settings
        plt.rcParams.update(
            {
                "figure.figsize": [18, 11],  # Slightly wider for better readability
                "figure.facecolor": self.colors["background"],
                "axes.facecolor": self.colors["background"],
                "axes.grid": True,
                "grid.color": self.colors["grid"],
                "grid.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.titlesize": 13,  # Larger titles
                "axes.labelsize": 11,  # Larger labels
                "xtick.labelsize": 10,  # Larger ticks
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 150,
                "font.family": ["Helvetica", "Arial", "sans-serif"],
            }
        )

    def create_dashboard(
        self, market_data: Dict, risk_metrics: Dict, regimes: Optional[pd.Series] = None
    ) -> plt.Figure:
        """Create a modern risk analytics dashboard"""

        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = GridSpec(3, 3, figure=fig)

        # 1. Price Evolution (larger plot)
        ax_price = fig.add_subplot(gs[0, :2])
        self._plot_price_evolution(market_data["close"], ax_price, regimes)

        # 2. Risk Metrics Summary (top right)
        ax_metrics = fig.add_subplot(gs[0, 2])
        self._plot_risk_summary(risk_metrics, ax_metrics)

        # 3. Returns Distribution
        ax_dist = fig.add_subplot(gs[1, 0])
        self._plot_returns_dist(market_data["returns"], risk_metrics, ax_dist)

        # 4. Rolling Volatility
        ax_vol = fig.add_subplot(gs[1, 1])
        self._plot_rolling_vol(market_data["returns"], ax_vol)

        # 5. Correlation Matrix
        ax_corr = fig.add_subplot(gs[1, 2])
        self._plot_correlation(risk_metrics["correlation"], ax_corr)

        # 6. Drawdown Analysis (wider plot)
        ax_dd = fig.add_subplot(gs[2, :])
        self._plot_drawdown(market_data["returns"], risk_metrics, ax_dd)

        # Add dashboard title
        fig.suptitle(
            "Portfolio Risk Analytics", fontsize=14, color=self.colors["text"], y=0.95
        )

        return fig

    def _plot_price_evolution(
        self, prices: pd.DataFrame, ax: plt.Axes, regimes: Optional[pd.Series] = None
    ):
        """Improved price evolution plot"""
        # Normalize prices
        normalized = prices / prices.iloc[0]

        # Use different colors for each asset
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["accent"],
        ]

        # Plot each asset
        for i, col in enumerate(normalized.columns):
            ax.plot(
                normalized.index,
                normalized[col],
                label=col,
                color=colors[i % len(colors)],
                linewidth=1.5,
                alpha=0.9,
            )

        # Add regime overlay if available
        if regimes is not None:
            self._add_regime_overlay(ax, normalized.index, regimes)

        ax.set_title("Asset Price Evolution", pad=20)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

    def _plot_risk_summary(self, metrics: Dict, ax: plt.Axes):
        """Create modern risk metrics summary"""
        key_metrics = {
            "Volatility": f"{metrics['portfolio_volatility']:.1%}",
            "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
            "VaR (95%)": f"{metrics['var_95']:.1%}",
            "Max DD": f"{metrics['max_drawdown']:.1%}",
        }

        # Create horizontal bars
        y_pos = np.arange(len(key_metrics))
        values = [float(v.strip("%").strip()) for v in key_metrics.values()]

        # Plot bars
        bars = ax.barh(y_pos, np.abs(values), color=self.colors["primary"], alpha=0.3)

        # Add value labels
        for i, v in enumerate(key_metrics.values()):
            ax.text(
                0,
                i,
                f"{list(key_metrics.keys())[i]}: {v}",
                va="center",
                color=self.colors["text"],
            )

        ax.set_title("Key Risk Metrics", pad=20)
        ax.set_yticks([])
        ax.set_xticks([])

    def _plot_returns_dist(self, returns: pd.DataFrame, metrics: Dict, ax: plt.Axes):
        """Enhanced returns distribution"""
        portfolio_returns = returns.mean(axis=1)

        # Use more bins for better distribution visualization
        sns.histplot(
            data=portfolio_returns,
            stat="density",
            bins=50,  # More bins
            color=self.colors["primary"],
            alpha=0.4,
            ax=ax,
        )

        # Add mean line
        mean = portfolio_returns.mean()
        ax.axvline(
            mean,
            color=self.colors["success"],
            linestyle="-",
            label=f"Mean: {mean:.1%}",
        )

        # Add VaR and ES lines
        var_95 = metrics.get("var_95", portfolio_returns.quantile(0.05))
        es_95 = metrics.get(
            "expected_shortfall_95",
            portfolio_returns[portfolio_returns <= var_95].mean(),
        )

        ax.axvline(
            var_95,
            color=self.colors["warning"],
            linestyle="--",
            label=f"VaR (95%): {var_95:.1%}",
        )

        ax.axvline(
            es_95,
            color=self.colors["warning"],
            linestyle=":",
            label=f"ES (95%): {es_95:.1%}",
        )

        ax.set_title("Returns Distribution", pad=20)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

    def _plot_rolling_vol(self, returns: pd.DataFrame, ax: plt.Axes):
        """Plot rolling volatility with modern styling"""
        # Calculate rolling volatility (21-day window)
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)

        # Plot for each asset
        for col in rolling_vol.columns:
            ax.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.7)

        ax.set_title("Rolling Volatility (21D)", pad=20)
        ax.set_ylabel("Annualized Volatility")
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

    def _plot_correlation(self, correlation: pd.DataFrame, ax: plt.Axes):
        """Plot correlation matrix with modern styling"""
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation), k=1)

        # Custom diverging colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        # Plot heatmap
        sns.heatmap(
            correlation,
            mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            ax=ax,
        )

        ax.set_title("Correlation Matrix", pad=20)

    def _plot_drawdown(self, returns: pd.DataFrame, metrics: Dict, ax: plt.Axes):
        """Plot enhanced drawdown analysis with better styling and annotations"""
        # Calculate drawdown
        portfolio_returns = returns.mean(axis=1)
        cum_returns = (1 + portfolio_returns).cumprod()
        drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()

        # Create gradient color effect for fill
        gradient_alpha = np.linspace(0.1, 0.3, len(drawdown))

        # Plot drawdown with gradient fill
        for i in range(len(drawdown) - 1):
            ax.fill_between(
                drawdown.index[i : i + 2],
                0,
                drawdown.iloc[i : i + 2],
                color=self.colors["warning"],
                alpha=gradient_alpha[i],
            )

        # Plot drawdown line with better styling
        ax.plot(
            drawdown.index,
            drawdown,
            color=self.colors["warning"],
            label="Drawdown",
            linewidth=1.2,
            zorder=2,
        )

        # Add max drawdown line with improved styling
        max_dd = metrics.get("max_drawdown", drawdown.min())
        ax.axhline(
            max_dd,
            color=self.colors["warning"],
            linestyle="--",
            alpha=0.9,
            linewidth=1.5,
            label=f"Max Drawdown: {max_dd:.1%}",
            zorder=1,
        )

        # Add recovery periods with improved annotations
        self._add_recovery_periods(ax, drawdown)

        # Add underwater periods shading
        self._add_underwater_periods(ax, drawdown)

        # Improve axis styling
        ax.set_title("Drawdown Analysis", pad=20, fontsize=12)
        ax.set_ylabel("Drawdown %", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":")

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

        # Enhance legend
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
        )

        # Add annotations for significant drawdowns
        self._annotate_significant_drawdowns(ax, drawdown)

    def _add_recovery_periods(self, ax: plt.Axes, drawdown: pd.Series):
        """Add recovery period annotations only for major drawdowns"""
        threshold = (
            -0.15
        )  # Changed from -0.1 to -0.15 to show only significant drawdowns
        in_drawdown = False
        start_date = None
        min_duration_days = 30  # Only show recovery periods longer than 30 days

        for date, value in drawdown.items():
            if value < threshold and not in_drawdown:
                start_date = date
                in_drawdown = True
            elif value > threshold and in_drawdown:
                end_date = date
                duration = (end_date - start_date).days
                if duration >= min_duration_days:
                    # Position the annotation at the deepest drawdown point in this period
                    drawdown_period = drawdown[start_date:end_date]
                    min_date = drawdown_period.idxmin()
                    min_value = drawdown_period.min()

                    # Add recovery annotation with improved styling
                    ax.annotate(
                        f"{duration}d",
                        xy=(min_date, min_value),
                        xytext=(0, -25),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        color=self.colors["text"],
                        fontsize=8,
                        bbox=dict(
                            facecolor="white",
                            edgecolor=self.colors["warning"],
                            alpha=0.7,
                            boxstyle="round,pad=0.3",
                            linewidth=0.5,
                        ),
                        zorder=4,
                    )
                in_drawdown = False

    def _add_regime_overlay(
        self, ax: plt.Axes, dates: pd.DatetimeIndex, regimes: pd.Series
    ):
        """Add regime overlay to price plot"""
        # Define regime colors
        regime_colors = {
            "Low Vol": self.colors["success"],
            "Medium Vol": self.colors["neutral"],
            "High Vol": self.colors["warning"],
        }

        # Plot regime backgrounds
        prev_date = dates[0]
        prev_regime = regimes.iloc[0]

        for date, regime in regimes.items():
            if regime != prev_regime:
                ax.axvspan(
                    prev_date,
                    date,
                    alpha=0.1,
                    color=regime_colors[prev_regime],
                    label=(
                        prev_regime
                        if prev_regime not in ax.get_legend_handles_labels()[1]
                        else ""
                    ),
                )
                prev_date = date
                prev_regime = regime

        # Add final regime
        ax.axvspan(
            prev_date,
            dates[-1],
            alpha=0.1,
            color=regime_colors[prev_regime],
            label=(
                prev_regime
                if prev_regime not in ax.get_legend_handles_labels()[1]
                else ""
            ),
        )

    def _add_underwater_periods(self, ax: plt.Axes, drawdown: pd.Series):
        """Add shaded regions for underwater periods"""
        underwater = drawdown < 0
        start_idx = None

        for i, is_underwater in enumerate(underwater):
            if is_underwater and start_idx is None:
                start_idx = i
            elif not is_underwater and start_idx is not None:
                ax.axvspan(
                    drawdown.index[start_idx],
                    drawdown.index[i],
                    color=self.colors["warning"],
                    alpha=0.05,
                    zorder=0,
                )
                start_idx = None

        # Handle case where we end in drawdown
        if start_idx is not None:
            ax.axvspan(
                drawdown.index[start_idx],
                drawdown.index[-1],
                color=self.colors["warning"],
                alpha=0.05,
                zorder=0,
            )

    def _annotate_significant_drawdowns(self, ax: plt.Axes, drawdown: pd.Series):
        """Add annotations for significant drawdown points"""
        # Find significant drawdowns (less than -15%)
        significant_dd = drawdown[drawdown < -0.15]

        for date, value in significant_dd.items():
            # Only annotate local minima
            if (
                value
                == drawdown[
                    max(0, drawdown.index.get_loc(date) - 10) : min(
                        len(drawdown), drawdown.index.get_loc(date) + 10
                    )
                ].min()
            ):
                ax.annotate(
                    f"{value:.1%}",
                    xy=(date, value),
                    xytext=(10, -10),
                    textcoords="offset points",
                    color=self.colors["text"],
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        edgecolor=self.colors["warning"],
                        alpha=0.7,
                        boxstyle="round,pad=0.5",
                    ),
                    zorder=3,
                )
