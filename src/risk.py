# src/risk.py
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

import statsmodels.api as sm
from typing import Sequence
# For Bayesian and copula modeling
try:
    import pymc3 as pm
except ImportError:
    pm = None  # Optional, for Bayesian AR(1)
try:
    from copulas.multivariate import GaussianMultivariate, StudentMultivariate
except ImportError:
    GaussianMultivariate = StudentMultivariate = None

from src.regime import MarketRegimeDetector, RegimeConfig
from src.deep_learning import BayesianLSTMRegimeForecaster
from sklearn.decomposition import PCA


@dataclass
class RiskConfig:
    """Enhanced risk management configuration"""

    # Required parameters with defaults
    confidence_level: float = 0.95
    max_drawdown_limit: float = 0.20  # Raised from 0.10 to 0.20 for less aggressive de-risking
    volatility_target: float = 0.15
    
    # Risk limits from grid search
    stop_loss: float = 0.02  # 2% stop-loss from best config
    take_profit: float = 0.50  # 30% take-profit from best config

    # Optional parameters with defaultsw
    var_calculation_method: str = (
        "historical"  # ['historical', 'parametric', 'monte_carlo']
    )
    es_calculation_method: str = "historical"
    regime_detection_method: str = (
        "volatility"  # ['volatility', 'markov', 'clustering']
    )
    stress_scenarios: List[str] = field(
        default_factory=lambda: ["2008_crisis", "covid_crash", "tech_bubble"]
    )
    correlation_regime: bool = True
    tail_risk_measure: str = "evt"  # ['evt', 'copula', 'empirical']
    _weights: Optional[List[float]] = field(default=None, repr=False)

    # Add regime detection configuration
    regime_config: RegimeConfig = field(
        default_factory=lambda: RegimeConfig(
            n_regimes=5, window_size=10, features=["returns", "volatility"]  # Updated to best config
        )
    )

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.max_drawdown_limit <= 0:
            raise ValueError("Max drawdown limit must be positive")
        if self.volatility_target <= 0:
            raise ValueError("Volatility target must be positive")

    @property
    def weights(self) -> Optional[List[float]]:
        return self._weights

    @weights.setter
    def weights(self, value: Optional[List[float]]):
        """Validate and set portfolio weights"""
        if value is not None:
            if not isinstance(value, (list, np.ndarray)):
                raise TypeError("Weights must be a list or numpy array")
            if any(w < 0 for w in value):
                raise ValueError("Weights cannot be negative")
            if not np.isclose(sum(value), 1.0):
                raise ValueError("Weights must sum to 1.0")
        self._weights = value


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
        # FIX: Use config weights if provided, otherwise use parameter weights
        self.weights = config.weights if config.weights is not None else weights
        self.regime_detector = MarketRegimeDetector(config.regime_config)

    def calculate_metrics(self, returns: Union[pd.Series, pd.DataFrame], n_bootstraps: int = 0) -> Dict[str, Any]:
        """Enhanced risk metrics calculation with optional bootstrapped CIs"""
        # Robustly handle DataFrame/Series
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                returns = returns.iloc[:, 0]
            else:
                returns = returns.mean(axis=1)
        returns = returns.dropna()
        # If all zeros, warn
        if (returns != 0).sum() == 0:
            print("⚠️  WARNING: All returns are zero! Metrics will be zero.")
        # Calculate metrics
        metrics = {}
        metrics['mean_return'] = returns.mean()
        metrics['std_return'] = returns.std()
        metrics['portfolio_volatility'] = returns.std() * np.sqrt(252)
        metrics['volatility'] = metrics['portfolio_volatility']  # Ensure alias for summary compatibility
        metrics['var_95'] = np.percentile(returns.dropna(), 5) if not returns.empty else 0
        metrics['expected_shortfall_95'] = returns[returns <= metrics['var_95']].mean() if not returns.empty else 0
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        metrics['sharpe_ratio'] = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if returns.std() > 0 else 0
        # Bootstrapped CIs
        if n_bootstraps > 0:
            metrics['sharpe_ci'] = self.bootstrap_metric(lambda r: (r.mean() / (r.std() + 1e-8)) * np.sqrt(252), returns, n=n_bootstraps)
            metrics['var_95_ci'] = self.bootstrap_metric(lambda r: np.percentile(r.dropna(), 5), returns, n=n_bootstraps)
            metrics['max_drawdown_ci'] = self.bootstrap_metric(self._calculate_max_drawdown, returns, n=n_bootstraps)
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
        """Enhanced VaR calculation with multiple methodologies"""
        if self.config.var_calculation_method == "parametric":
            return self._calculate_parametric_var(returns, confidence_level)
        elif self.config.var_calculation_method == "monte_carlo":
            return self._calculate_monte_carlo_var(returns, confidence_level)
        else:
            return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_parametric_var(
        self, returns: pd.Series, confidence_level: float
    ) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mu = returns.mean()
        sigma = returns.std()
        return norm.ppf(1 - confidence_level, mu, sigma)

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

    def _calculate_rolling_volatility(self, portfolio_returns: pd.Series) -> pd.DataFrame:
        """Calculate rolling volatility for portfolio returns (Series input only)"""
        rolling_vol = pd.DataFrame(index=portfolio_returns.index)
        rolling_vol.loc[:, "21d"] = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol.loc[:, "63d"] = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
        return rolling_vol

    def _calculate_volatility_ratio(self, returns: pd.DataFrame) -> float:
        """Calculate ratio of recent to historical volatility"""
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)  # Convert to series if DataFrame

        recent_vol = returns[-21:].std() * np.sqrt(252)  # Last month
        historical_vol = returns.std() * np.sqrt(252)  # Full period

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
        """Calculate suggested position size based on risk metrics (flattened penalty)"""
        vol_ratio = self._calculate_volatility_ratio(returns)
        dd_ratio = self._calculate_drawdown_ratio(returns)

        # Flattened penalty: use exponent 1.0 instead of 1.5, and log-based penalty for vol_ratio
        penalty = np.log1p(vol_ratio - 1) if vol_ratio > 1 else 0
        position_scale = 1.0 - penalty
        # Drawdown penalty (linear, not exponential)
        if dd_ratio > 0.6:
            position_scale *= max(0.5, 1.0 - (dd_ratio - 0.6))
        # Ensure minimum 20% position
        return max(0.2, min(position_scale, 1.0))

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
        monthly_returns = returns.resample("M").sum()
        quarterly_returns = returns.resample("Q").sum()

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

    def _calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Monte Carlo VaR calculation using bootstrap simulation
        Args:
            returns: Historical return series
            confidence_level: Confidence level for VaR calculation
        Returns:
            Monte Carlo VaR estimate
        """
        n_simulations = 10000
        simulated_returns = np.random.choice(
            returns.dropna().values,
            size=(n_simulations, len(returns.dropna())),
            replace=True
        )
        portfolio_returns = np.mean(simulated_returns, axis=1)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def _calculate_risk_score(self, returns: pd.Series) -> float:
        """
        Calculate overall risk score (0-100 scale)
        Less punitive: flatten scoring, regime-aware (optional)
        """
        vol = self._calculate_volatility(returns)
        max_dd = abs(self._calculate_max_drawdown(returns))
        var_95 = abs(self._calculate_var(returns, 0.95))
        # Flattened scores, less aggressive
        vol_score = min(vol / 0.30, 1.0) * 30
        dd_score = min(max_dd / 0.30, 1.0) * 30
        var_score = min(var_95 / 0.10, 1.0) * 20
        total_score = vol_score + dd_score + var_score
        return min(total_score, 80.0)  # Cap at 80, not 100

    def generate_risk_report(self, returns: pd.Series, n_bootstraps: int = 0) -> Dict[str, Any]:
        """
        Generate comprehensive risk report with assessment and recommendations, with optional bootstrapped CIs.
        Args:
            returns: Portfolio return series
            n_bootstraps: number of bootstrap samples for CIs
        Returns:
            Complete risk analysis report
        """
        # Robustly handle DataFrame/Series
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                returns = returns.iloc[:, 0]
            else:
                returns = returns.mean(axis=1)
        # Print debug info
        print("\n=== RISK INPUT RETURNS SUMMARY ===")
        print(returns.describe())
        print("Nonzero count:", (returns != 0).sum())
        print("NaN count:", returns.isna().sum())
        if (returns != 0).sum() == 0 or returns.isna().all():
            print("⚠️  WARNING: All returns are zero or NaN! Risk metrics will be zero.")
        metrics = self.calculate_metrics(returns, n_bootstraps=n_bootstraps)
        risk_score = self._calculate_risk_score(returns)
        metrics['risk_score'] = risk_score
        # Regime-aware position recommendation
        min_position = 0.8
        if risk_score < 30:
            risk_level = "Low"
            position_recommendation = 1.0
            key_risks = ["Market risk within acceptable bounds"]
        elif risk_score < 60:
            risk_level = "Medium"
            position_recommendation = 0.9
            key_risks = ["Moderate volatility", "Some drawdown risk"]
        else:
            risk_level = "High"
            position_recommendation = 0.8
            key_risks = ["High volatility", "Significant drawdown risk", "Tail risk concerns"]
        # Enforce minimum position recommendation
        position_recommendation = max(position_recommendation, min_position)
        # Regime feedback loop
        regime_info = self._detect_market_regime(returns)
        current_regime = regime_info.get('current_regime', 'Unknown')
        if current_regime == "High Vol":
            position_recommendation *= 0.8
        elif current_regime == "Low Vol":
            position_recommendation *= 1.2
        position_recommendation = min(max(position_recommendation, min_position), 1.2)
        recommendations = []
        if metrics['max_drawdown'] < -0.15:
            recommendations.append("Consider reducing position sizes due to high drawdown")
        if metrics['sharpe_ratio'] < 0.5:
            recommendations.append("Review signal quality - low risk-adjusted returns")
        if metrics.get('portfolio_volatility', 0) > 0.30:
            recommendations.append("High volatility detected - consider volatility targeting")
        if risk_score > 70:
            recommendations.append("Overall risk is elevated - implement stricter risk controls")
        if current_regime == "High Vol":
            recommendations.append("Currently in high volatility regime - reduce exposure")
        elif current_regime == "Low Vol":
            recommendations.append("Low volatility environment - potential for higher allocation")
        return {
            'risk_metrics': metrics,
            'risk_assessment': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'position_recommendation': position_recommendation,
                'key_risks': key_risks,
                'current_regime': current_regime
            },
            'recommendations': recommendations,
            'summary': f"Risk Level: {risk_level} ({risk_score:.1f}/80)"
        }

    def _detect_market_regime(self, returns: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced market regime detection using HMM"""
        # Ensure returns is a Series for LSTM regime detection
        if isinstance(returns, pd.DataFrame):
            if len(returns.columns) == 1:
                returns_series = returns.iloc[:, 0]
            elif self.config.weights is not None:
                returns_series = returns.dot(self.config.weights)
            else:
                returns_series = returns.mean(axis=1)
        else:
            returns_series = returns

        from src.features import calculate_enhanced_features
        features = calculate_enhanced_features(returns_series)
        regimes = self.regime_detector.fit_predict(features, returns_series)

        # Get regime statistics
        regime_stats = self.regime_detector.get_regime_stats(returns, regimes)

        # Get validation metrics
        validation = self.regime_detector.validate_model(returns, regimes)

        # Get transition probabilities
        transitions = self.regime_detector.get_transition_matrix()

        current_regime = regimes.iloc[-1]
        regime_mask = regimes == current_regime

        return {
            "current_regime": current_regime,
            "regime_mask": regime_mask,
            "regime_stats": regime_stats,
            "model_validation": validation,
            "transition_probs": transitions.to_dict() if hasattr(transitions, 'to_dict') else transitions,
            "confidence": validation.get("regime_persistence", None),
        }

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
        # Check for missing values
        if returns.isna().any().any():
            warnings.warn("Returns contain missing values", UserWarning)

        # Check for extreme values (z-score > 5)
        z_scores = np.abs(stats.zscore(returns, nan_policy="omit"))
        if (z_scores > 5).any().any():
            warnings.warn("Returns contain extreme values (z-score > 5)", UserWarning)

    def _calculate_conditional_metrics(self, returns: pd.Series) -> Dict:
        """Calculate regime-dependent risk metrics"""
        regime = self._detect_market_regime(returns)
        return {
            f"{regime}_volatility": self._calculate_volatility(returns),
            f"{regime}_var": self._calculate_var(returns, self.config.confidence_level),
            f"{regime}_es": self._calculate_expected_shortfall(
                returns, self.config.confidence_level
            ),
        }

    def _extreme_value_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform Extreme Value Theory analysis"""
        threshold = np.percentile(returns, 5)  # 5th percentile as threshold
        exceedances = returns[returns < threshold]
        from scipy.stats import genpareto
        try:
            shape, loc, scale = genpareto.fit(abs(exceedances))
            return {
                "tail_index": shape,
                "scale": scale,
                "threshold": threshold,
                "exceedance_rate": len(exceedances) / len(returns),
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_dynamic_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime-dependent correlation matrix
        Args:
            returns: Multi-asset return DataFrame
        Returns:
            Correlation matrix adjusted for current market regime
        """
        if len(returns.columns) == 1:
            return pd.DataFrame([[1.0]], index=returns.columns, columns=returns.columns)
        try:
            portfolio_returns = returns.mean(axis=1)
            regime_info = self._detect_market_regime(portfolio_returns)
            current_regime = regime_info.get("current_regime", None)
            regime_mask = regime_info.get("regime_mask", pd.Series(True, index=returns.index))
            regime_returns = returns[regime_mask]
            if len(regime_returns) > 21:
                regime_corr = regime_returns.corr()
            else:
                regime_corr = returns.corr()
            regime_corr = regime_corr.fillna(0)
            np.fill_diagonal(regime_corr.values, 1.0)
            return regime_corr
        except Exception as e:
            print(f"Warning: Dynamic correlation calculation failed: {e}")
            return returns.corr()

    def forecast_risk_bayesian(self, returns: pd.Series, macro_features: Optional[pd.DataFrame] = None, method: str = 'ar1', forecast_horizon: int = 1, target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Bayesian risk forecasting: AR(1) or Bayesian LSTM for forward volatility/VaR prediction.
        Args:
            returns: pd.Series of returns
            macro_features: Optional DataFrame of macro variables
            method: 'ar1' or 'lstm'
            forecast_horizon: periods ahead to forecast (for AR1)
            target: np.ndarray of target risk metric (for LSTM training)
        Returns:
            Dict with predictive mean, std, and intervals for volatility/VaR
        Usage:
            - For AR(1): method='ar1' (default)
            - For Bayesian LSTM: method='lstm', provide target (e.g., next-period volatility)
        """
        if method == 'ar1':
            # Bayesian AR(1) using statsmodels (non-Bayesian fallback)
            ar1 = sm.tsa.ARIMA(returns, order=(1,0,0)).fit()
            forecast = ar1.get_forecast(steps=forecast_horizon)
            pred_mean = forecast.predicted_mean.values[-1]
            pred_std = np.sqrt(forecast.var_pred_mean.values[-1])
            return {'predicted_mean': pred_mean, 'predicted_std': pred_std}
        elif method == 'lstm':
            # Bayesian LSTM for risk forecasting
            if target is None:
                raise ValueError('For Bayesian LSTM, you must provide a target risk metric (e.g., next-period volatility)')
            bayes_lstm = BayesianLSTMRegimeForecaster(self.config)
            bayes_lstm.fit(returns, target)
            pred = bayes_lstm.predict_latest(returns, n_samples=100)
            return pred
        else:
            raise ValueError('Unknown method for Bayesian risk forecasting')

    def get_regime_path_features(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Compute cross-regime path dependency features: transition frequency, time since last regime change, cumulative drawdown since last regime, regime momentum.
        Returns:
            Dict of path-dependent regime features
        """
        regime_info = self._detect_market_regime(returns)
        regimes = self.regime_detector.fit_predict(returns)
        transitions = regimes != regimes.shift(1)
        transition_count = transitions.sum()
        time_since_last = (len(regimes) - np.where(transitions)[0][-1]) if transition_count > 0 else len(regimes)
        current_regime = regimes.iloc[-1]
        regime_duration = (regimes == current_regime)[::-1].cumsum().iloc[0]
        # Cumulative drawdown since last regime change
        last_change_idx = np.where(transitions)[0][-1] if transition_count > 0 else 0
        drawdown_since_last = self._calculate_max_drawdown(returns.iloc[last_change_idx:])
        return {
            'transition_count': int(transition_count),
            'time_since_last_transition': int(time_since_last),
            'current_regime_duration': int(regime_duration),
            'drawdown_since_last_regime': float(drawdown_since_last),
        }

    def calculate_tail_dependence_copula(self, returns: pd.DataFrame, regime_labels: Optional[Sequence] = None, copula_type: str = 'gaussian', alpha: float = 0.05, n_sim: int = 10000) -> Dict[str, Any]:
        """
        Fit copula (Gaussian or Student-t) to returns, optionally per regime. Compute tail dependence metrics (lambda).
        Args:
            returns: DataFrame of asset returns
            regime_labels: Optional regime labels (same length as returns)
            copula_type: 'gaussian' or 'student'
            alpha: tail threshold (default 0.05)
            n_sim: number of copula samples for lambda estimation
        Returns:
            Dict of copula parameters and tail dependence metrics
        """
        def compute_tail_dependence(u, v, alpha=0.05):
            return np.mean((u < alpha) & (v < alpha)) / alpha
        if copula_type == 'gaussian' and GaussianMultivariate is not None:
            copula = GaussianMultivariate()
        elif copula_type == 'student' and StudentMultivariate is not None:
            copula = StudentMultivariate()
        else:
            raise ImportError('Copula library not installed or unknown copula type')
        results = {}
        if regime_labels is not None:
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                copula.fit(returns[mask])
                params = copula.to_dict()
                # Simulate from copula
                sim = copula.sample(n_sim)
                lambdas = {}
                for i, col1 in enumerate(sim.columns):
                    for j, col2 in enumerate(sim.columns):
                        if i < j:
                            u, v = sim[col1].values, sim[col2].values
                            lambda_L = compute_tail_dependence(u, v, alpha)
                            lambdas[f"{col1}-{col2}"] = lambda_L
                results[str(regime)] = {'params': params, 'lambda_L': lambdas}
        else:
            copula.fit(returns)
            params = copula.to_dict()
            sim = copula.sample(n_sim)
            lambdas = {}
            for i, col1 in enumerate(sim.columns):
                for j, col2 in enumerate(sim.columns):
                    if i < j:
                        u, v = sim[col1].values, sim[col2].values
                        lambda_L = compute_tail_dependence(u, v, alpha)
                        lambdas[f"{col1}-{col2}"] = lambda_L
            results['all'] = {'params': params, 'lambda_L': lambdas}
        return results

    def adjust_signal_by_risk_forecast(self, signal: pd.Series, forecasted_vol: float, base_threshold: float = 1.0, penalize: bool = True) -> pd.Series:
        """
        Adjust signal weights or thresholds based on forecasted volatility (or VaR).
        Args:
            signal: pd.Series of raw signal values
            forecasted_vol: float, forecasted volatility or VaR
            base_threshold: base threshold for signal
            penalize: if True, penalize by 1/vol, else tighten threshold
        Returns:
            Adjusted signal Series
        """
        if penalize:
            return signal / (forecasted_vol + 1e-8)
        else:
            return signal.where(signal.abs() > base_threshold * forecasted_vol, 0)

    def risk_factor_attribution(self, returns: pd.DataFrame, factors: Optional[pd.DataFrame] = None, method: str = 'pca', n_factors: int = 3) -> Dict[str, Any]:
        """
        Attribute portfolio risk to factors (PCA or Fama-French).
        Args:
            returns: DataFrame of asset returns
            factors: DataFrame of factor returns (for Fama-French)
            method: 'pca' or 'fama_french'
            n_factors: number of factors to use
        Returns:
            Dict with factor exposures and % risk explained
        """
        if method == 'pca':
            pca = PCA(n_components=n_factors)
            pca.fit(returns)
            exposures = pca.components_
            explained = pca.explained_variance_ratio_
            return {'factor_exposures': exposures, 'explained_variance': explained}
        elif method == 'fama_french':
            if factors is None:
                raise ValueError('Must provide factor returns for Fama-French attribution')
            exposures = {}
            for col in returns.columns:
                model = sm.OLS(returns[col], sm.add_constant(factors)).fit()
                exposures[col] = model.params.to_dict()
            return {'factor_exposures': exposures}
        else:
            raise ValueError('Unknown factor attribution method')

    def bootstrap_metric(self, metric_fn, returns: pd.Series, n: int = 1000) -> Tuple[float, float]:
        """
        Compute bootstrapped confidence interval for a risk metric.
        Args:
            metric_fn: function to compute metric
            returns: pd.Series of returns
            n: number of bootstrap samples
        Returns:
            (lower, upper) 95% CI
        """
        boot = [metric_fn(returns.sample(frac=1, replace=True)) for _ in range(n)]
        return np.percentile(boot, [2.5, 97.5])

    def decompose_expected_shortfall(self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Decompose Expected Shortfall (ES) by asset and by time period. Compute marginal ES contributions.
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights (defaults to equal)
            alpha: ES confidence level
        Returns:
            Dict with ES by asset, by period, and marginal contributions
        """
        if weights is None:
            weights = np.ones(returns.shape[1]) / returns.shape[1]
        port_returns = returns.dot(weights)
        var = np.percentile(port_returns, 100 * alpha)
        es_mask = port_returns <= var
        es_periods = returns[es_mask]
        es_by_asset = es_periods.mean()
        es_by_time = es_periods.index.tolist()
        # Marginal ES: asset's average loss during ES events, weighted
        marginal_es = (es_periods * weights).mean()
        # Correlation factor (approx): asset's correlation with portfolio during ES periods
        corr_factors = es_periods.corrwith(port_returns[es_mask])
        es_contribution = es_by_asset * weights * corr_factors
        return {
            'es_by_asset': es_by_asset.to_dict(),
            'es_periods': es_by_time,
            'marginal_es': marginal_es.to_dict() if hasattr(marginal_es, 'to_dict') else float(marginal_es),
            'es_contribution': es_contribution.to_dict(),
        }
