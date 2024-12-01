# src/risk.py
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


@dataclass
class RiskConfig:
    """Risk management configuration parameters"""

    # Required parameters with defaults
    confidence_level: float = 0.95
    max_drawdown_limit: float = 0.20
    volatility_target: float = 0.15

    # Optional parameters with defaults
    weights: Optional[List[float]] = None
    rolling_windows: List[int] = field(default_factory=lambda: [21, 63])
    position_limit: float = 0.40

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.max_drawdown_limit <= 0:
            raise ValueError("Max drawdown limit must be positive")
        if self.volatility_target <= 0:
            raise ValueError("Volatility target must be positive")


class RiskManager:
    """Risk management and analysis"""

    def __init__(
        self,
        config: RiskConfig,
        risk_free_rate: float = 0.05,
        weights: Optional[List[float]] = None,
    ):
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.weights = weights

    def calculate_metrics(self, returns: pd.DataFrame) -> Dict:
        """Enhanced risk metrics calculation"""
        # Validate input
        if len(returns.columns) == 0:
            raise ValueError("Returns DataFrame must have at least one column")
        if len(returns.index) == 0:
            raise ValueError("Returns DataFrame must have at least one row")

        weights = (
            self.config.weights
            if self.config.weights is not None
            else [1 / len(returns.columns)] * len(returns.columns)
        )
        portfolio_returns = returns.dot(weights)

        # Basic metrics
        metrics = {
            "portfolio_volatility": self._calculate_volatility(portfolio_returns),
            "var_95": self._calculate_var(portfolio_returns, 0.95),
            "expected_shortfall_95": self._calculate_expected_shortfall(
                portfolio_returns, 0.95
            ),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_returns),
            "rolling_volatility": self._calculate_rolling_volatility(returns),
            "correlation": returns.corr(),
        }

        # Add risk management metrics if enough data
        if len(portfolio_returns) > 21:  # Need at least 21 days for ratios
            metrics.update(
                {
                    "volatility_ratio": self._calculate_volatility_ratio(
                        portfolio_returns
                    ),
                    "drawdown_ratio": self._calculate_drawdown_ratio(portfolio_returns),
                    "risk_adjusted_position": self._calculate_risk_adjusted_position(
                        portfolio_returns
                    ),
                    "stress_test": self._perform_stress_test(portfolio_returns),
                }
            )
        else:
            metrics.update(
                {
                    "volatility_ratio": 1.0,
                    "drawdown_ratio": 0.0,
                    "risk_adjusted_position": 1.0,
                    "stress_test": {
                        "worst_month": 0.0,
                        "worst_quarter": 0.0,
                        "recovery_time": 0.0,
                        "max_consecutive_loss": 0,
                    },
                }
            )

        # Enhanced metrics
        metrics.update(
            {
                "sortino_ratio": self._calculate_sortino_ratio(portfolio_returns),
                "calmar_ratio": self._calculate_calmar_ratio(portfolio_returns),
                "omega_ratio": self._calculate_omega_ratio(portfolio_returns),
                "tail_ratio": self._calculate_tail_ratio(portfolio_returns),
                "market_regime": self._detect_market_regime(portfolio_returns),
                "risk_decomposition": (
                    self._decompose_risk(returns) if len(returns.columns) > 1 else None
                ),
            }
        )

        return metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio using specified risk-free rate"""
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_expected_shortfall(
        self, returns: pd.Series, confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()

    def _calculate_rolling_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatility for multiple windows"""
        rolling_vol = pd.DataFrame(index=returns.index)

        # Calculate portfolio returns first
        portfolio_returns = (
            returns.dot(self.weights)
            if self.weights is not None
            else returns.mean(axis=1)
        )

        # Calculate rolling volatility for different windows
        rolling_vol.loc[:, "21d"] = portfolio_returns.rolling(
            window=21
        ).std() * np.sqrt(252)
        rolling_vol.loc[:, "63d"] = portfolio_returns.rolling(
            window=63
        ).std() * np.sqrt(252)

        return rolling_vol

    def _calculate_volatility_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio of recent to historical volatility"""
        recent_returns = returns.iloc[-21:]
        historical_returns = returns.iloc[:-21]

        recent_vol = recent_returns.std() * np.sqrt(252)
        historical_vol = historical_returns.std() * np.sqrt(252)

        return recent_vol / historical_vol if historical_vol != 0 else 1.0

    def _calculate_drawdown_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio of current drawdown to max allowed"""
        current_dd = self._calculate_current_drawdown(returns)
        return abs(current_dd) / self.config.max_drawdown_limit

    def _calculate_current_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown from peak"""
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()

        return (cum_returns.iloc[-1] / peak.iloc[-1]) - 1

    def _calculate_risk_adjusted_position(self, returns: pd.Series) -> float:
        """Calculate suggested position size based on risk metrics"""
        vol_ratio = self._calculate_volatility_ratio(returns)
        dd_ratio = self._calculate_drawdown_ratio(returns)

        # More aggressive position scaling
        position_scale = min(
            1.0,
            1.0 / (vol_ratio**1.5) if vol_ratio > 1.1 else 1.0,
            1.0 / (dd_ratio**1.5) if dd_ratio > 0.6 else 1.0,
        )

        return max(0.2, min(position_scale, 1.0))  # Ensure minimum 20% position

    def _calculate_max_consecutive_loss(self, returns: pd.Series) -> int:
        """Calculate maximum number of consecutive losing days"""
        losses = returns < 0
        max_consecutive = 0
        current_consecutive = 0

        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _perform_stress_test(self, returns: pd.Series) -> Dict:
        """Perform basic stress testing"""
        monthly_returns = returns.resample("ME").sum()
        quarterly_returns = returns.resample("QE").sum()

        return {
            "worst_month": monthly_returns.min(),
            "worst_quarter": quarterly_returns.min(),
            "recovery_time": self._calculate_recovery_time(returns),
            "max_consecutive_loss": self._calculate_max_consecutive_loss(returns),
        }

    def _calculate_recovery_time(self, returns: pd.Series) -> int:
        """Calculate average days to recover from drawdowns"""
        cum_returns = (1 + returns).cumprod()
        peaks = cum_returns.expanding().max()
        drawdowns = cum_returns < peaks

        recovery_periods = []
        in_drawdown = False
        drawdown_start = 0

        for i in range(len(drawdowns)):
            is_drawdown = drawdowns.iloc[i]
            if is_drawdown and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif not is_drawdown and in_drawdown:
                in_drawdown = False
                recovery_periods.append(i - drawdown_start)

        return np.mean(recovery_periods) if recovery_periods else 0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation"""
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(252) * np.sqrt(np.mean(downside_returns**2))
        return excess_returns / downside_std if downside_std != 0 else 0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        max_dd = abs(self._calculate_max_drawdown(returns))
        return excess_returns / max_dd if max_dd != 0 else 0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else np.inf

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio between right and left tail"""
        left_tail = np.abs(np.percentile(returns, 5))
        right_tail = np.abs(np.percentile(returns, 95))
        return right_tail / left_tail if left_tail != 0 else np.inf

    def _detect_market_regime(self, returns: pd.Series) -> Dict[str, float]:
        """Enhanced market regime detection"""
        vol_ratio = self._calculate_volatility_ratio(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        regimes = {
            "volatility_regime": (
                "high" if vol_ratio > 1.5 else "low" if vol_ratio < 0.7 else "normal"
            ),
            "skewness_regime": (
                "negative" if skew < -0.5 else "positive" if skew > 0.5 else "neutral"
            ),
            "tail_regime": "fat" if kurt > 3 else "thin" if kurt < -0.5 else "normal",
            "confidence": min(1.0, max(0.6, 1 - abs(vol_ratio - 1))),
        }
        return regimes

    def _decompose_risk(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """Decompose portfolio risk into components"""
        # Calculate correlation-based risk decomposition
        corr = returns.corr()
        vol = returns.std()

        # Calculate contribution to portfolio risk
        weights = (
            self.weights
            if self.weights is not None
            else [1 / len(returns.columns)] * len(returns.columns)
        )
        weight_vector = np.array(weights)

        # Component risk contributions
        marginal_risk = np.dot(corr, weight_vector) * vol
        component_risk = weight_vector * marginal_risk
        total_risk = np.sqrt(np.dot(weight_vector, component_risk))

        # Normalize to get percentage contributions
        risk_contrib = pd.Series(
            component_risk / total_risk if total_risk != 0 else component_risk,
            index=returns.columns,
        )

        return {
            "risk_contribution": risk_contrib,
            "diversification_score": 1
            - (risk_contrib.max() if len(risk_contrib) > 0 else 0),
            "concentration_score": (risk_contrib**2).sum(),
        }

    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> None:
        """Enhanced return data validation"""
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be a pandas DataFrame")

        # Check for missing values
        if returns.isna().any().any():
            warnings.warn(
                "Returns contain missing values. Consider handling them first."
            )

        # Check for extreme values
        z_scores = np.abs(stats.zscore(returns, nan_policy="omit"))
        if (z_scores > 5).any().any():
            warnings.warn(
                "Returns contain extreme values (z-score > 5). Consider investigating outliers."
            )
