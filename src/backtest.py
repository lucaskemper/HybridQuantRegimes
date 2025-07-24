import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any, Union
import warnings
from datetime import datetime
import logging
from scipy.special import softmax as scipy_softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtest engine for portfolio strategies. By default, short selling is enabled (allow_short=True).
    """
    def __init__(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float = 1.0,
        rebalance_freq: str = 'W',  # Default to weekly
        transaction_cost: float = 0.001,  # 0.1%
        fixed_cost: float = 0.0,
        leverage: float = 1.0,
        position_sizing: Union[str, Callable] = 'proportional',
        slippage: float = 0.0005,  # 0.05%
        allow_short: bool = True,  # Changed default to True
        custom_position_func: Optional[Callable] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        walk_forward_window: Optional[int] = None,
        walk_forward_step: Optional[int] = None,
        risk_management_func: Optional[Callable] = None,
        regime_confidence: Optional[pd.DataFrame] = None,
        base_position_size: float = 1.0,
        confidence_threshold: float = 0.5,
        min_trade_size: float = 1e-6,  # Restore to previous value
        max_position_size: float = 1.0,  # Increased for debugging
    ):
        # Input validation
        self._validate_inputs(returns, signals, regime_confidence)
        
        # Core data - ensure alignment
        common_cols = returns.columns.intersection(signals.columns)
        if len(common_cols) == 0:
            raise ValueError("No common columns between returns and signals")
            
        self.returns = returns[common_cols].copy()
        self.signals = signals[common_cols].copy()
        
        # Align regime confidence if provided
        if regime_confidence is not None:
            common_regime_cols = common_cols.intersection(regime_confidence.columns)
            if len(common_regime_cols) > 0:
                self.regime_confidence = regime_confidence[common_regime_cols].copy()
            else:
                logger.warning("No common columns between returns and regime_confidence. Using single confidence value.")
                self.regime_confidence = regime_confidence.copy()
        else:
            self.regime_confidence = None
        
        # Parameters
        self.initial_cash = initial_cash
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = max(0, transaction_cost)
        self.fixed_cost = max(0, fixed_cost)
        self.leverage = max(0, leverage)
        self.position_sizing = position_sizing
        self.slippage = max(0, slippage)
        self.allow_short = allow_short
        self.custom_position_func = custom_position_func
        self.stop_loss = abs(stop_loss) if stop_loss is not None else None
        self.take_profit = abs(take_profit) if take_profit is not None else None
        self.max_drawdown = abs(max_drawdown) if max_drawdown is not None else None
        self.walk_forward_window = walk_forward_window
        self.walk_forward_step = walk_forward_step
        self.risk_management_func = risk_management_func
        self.base_position_size = max(0, base_position_size)
        self.confidence_threshold = np.clip(confidence_threshold, 0, 1)
        self.min_trade_size = max(0, min_trade_size)
        self.max_position_size = np.clip(max_position_size, 0, 1)
        
        # Internal state
        self._assets = list(common_cols)
        
    def _validate_inputs(self, returns: pd.DataFrame, signals: pd.DataFrame, regime_confidence: Optional[pd.DataFrame]):
        """Validate input data"""
        if not isinstance(returns, pd.DataFrame) or not isinstance(signals, pd.DataFrame):
            raise ValueError("returns and signals must be pandas DataFrames")
        
        if returns.empty or signals.empty:
            raise ValueError("returns and signals cannot be empty")
        
        if returns.isnull().all().any() or signals.isnull().all().any():
            warnings.warn("Found columns with all NaN values")
        
        if regime_confidence is not None and not isinstance(regime_confidence, pd.DataFrame):
            raise ValueError("regime_confidence must be a pandas DataFrame or None")

    def _get_positions(self, date: pd.Timestamp, context: Optional[Dict] = None) -> pd.Series:
        """Calculate position weights for given date"""
        if context is None:
            context = {}
            
        # Check if we have signals for this date
        if date not in self.signals.index:
            return pd.Series(0, index=self._assets)
        
        # Get signals for this date
        signals = self.signals.loc[date].reindex(self._assets, fill_value=0)
        
        # Handle NaN signals
        signals = signals.fillna(0)
        
        # Apply custom position function if provided
        if self.custom_position_func is not None:
            try:
                weights = self.custom_position_func(self.signals, self.returns, date, context)
                if isinstance(weights, pd.Series):
                    weights = weights.reindex(self._assets, fill_value=0)
                else:
                    weights = pd.Series(weights, index=self._assets)
                return self._apply_constraints(weights)
            except Exception as e:
                logger.warning(f"Custom position function failed at {date}: {e}. Using default sizing.")
        
        # Calculate position sizes based on method
        weights = self._calculate_position_sizes(signals, date, context)
        
        return self._apply_constraints(weights)
    
    def _calculate_position_sizes(self, signals: pd.Series, date: pd.Timestamp, context: Dict) -> pd.Series:
        """Calculate position sizes based on the selected method"""
        
        if self.position_sizing == 'proportional':
            return self._proportional_sizing(signals, date, context)
        elif self.position_sizing == 'fixed':
            return self._fixed_sizing(signals, date, context)
        elif self.position_sizing == 'regime_confidence':
            return self._regime_confidence_sizing(signals, date, context)
        elif self.position_sizing == 'dynamic':
            return self._dynamic_sizing(signals, date, context)
        elif self.position_sizing == 'softmax':
            # Softmax weighting
            abs_signals = np.abs(signals.values)
            softmax_weights = scipy_softmax(abs_signals)
            weights = softmax_weights * np.sign(signals.values)
            weights = pd.Series(weights, index=signals.index)
            return weights / np.sum(np.abs(weights)) * self.leverage if np.sum(np.abs(weights)) > 0 else weights
        elif self.position_sizing == 'quadratic':
            # Quadratic weighting
            quad_weights = (signals.values ** 2) * np.sign(signals.values)
            weights = pd.Series(quad_weights, index=signals.index)
            return weights / np.sum(np.abs(weights)) * self.leverage if np.sum(np.abs(weights)) > 0 else weights
        elif self.position_sizing == 'bayesian_risk_budget':
            # Bayesian risk budget: combine regime confidence and volatility targeting
            target_vol = 0.10
            # Use 21-day rolling volatility
            if date in self.returns.index:
                asset_vols = self.returns.rolling(21).std().loc[date]
            else:
                asset_vols = self.returns.rolling(21).std().iloc[-1]
            asset_vols = asset_vols.reindex(signals.index, fill_value=1.0).replace(0, 1.0)
            if self.regime_confidence is not None and date in self.regime_confidence.index:
                confidence = self.regime_confidence.loc[date].reindex(signals.index, fill_value=1.0)
            else:
                confidence = pd.Series(1.0, index=signals.index)
            risk_weight = confidence / (asset_vols + 1e-4)
            weights = signals * risk_weight
            # Normalize to total risk budget (leverage)
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights)) * self.leverage
            return weights
        elif callable(self.position_sizing):
            try:
                weights = self.position_sizing(signals, self.returns, date, context)
                return pd.Series(weights, index=self._assets) if not isinstance(weights, pd.Series) else weights
            except Exception as e:
                logger.warning(f"Custom position sizing failed at {date}: {e}. Using proportional sizing.")
                return self._proportional_sizing(signals, date, context)
        else:
            return signals * self.leverage
    
    def _proportional_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict) -> pd.Series:
        """Proportional position sizing"""
        if signals.abs().sum() == 0:
            return pd.Series(0, index=self._assets)
        
        weights = signals / signals.abs().sum() * self.leverage
        return weights
    
    def _fixed_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict) -> pd.Series:
        """Fixed position sizing"""
        non_zero_signals = (signals != 0).sum()
        if non_zero_signals == 0:
            return pd.Series(0, index=self._assets)
        
        fixed_size = self.leverage / non_zero_signals
        weights = signals.apply(lambda x: np.sign(x) * fixed_size if x != 0 else 0)
        return weights
    
    def _regime_confidence_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict) -> pd.Series:
        """Position sizing based on regime confidence (less conservative)"""
        if self.regime_confidence is None:
            return self._proportional_sizing(signals, date, context)
        # Get confidence for current date
        if date in self.regime_confidence.index:
            confidence = self.regime_confidence.loc[date]
            if isinstance(confidence, (int, float)):
                confidence = pd.Series(confidence, index=self._assets)
            else:
                confidence = confidence.reindex(self._assets, fill_value=0.5)
        else:
            if len(self.regime_confidence) > 0:
                confidence = self.regime_confidence.mean()
                if isinstance(confidence, (int, float)):
                    confidence = pd.Series(confidence, index=self._assets)
                else:
                    confidence = confidence.reindex(self._assets, fill_value=0.5)
            else:
                confidence = pd.Series(0.5, index=self._assets)
        # Use higher confidence floor
        confidence = confidence.clip(0.7, 1.0)  # Minimum 70% confidence
        # Calculate base weights
        if signals.abs().sum() == 0:
            return pd.Series(0, index=self._assets)
        base_weights = signals / signals.abs().sum()
        # Less conservative scaling
        adjusted_weights = base_weights * confidence * self.base_position_size * self.leverage
        self.debug_position_sizing(signals, date, context)
        return adjusted_weights
    
    def _dynamic_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict) -> pd.Series:
        """Dynamic position sizing with volatility adjustment"""
        # Start with regime confidence sizing
        weights = self._regime_confidence_sizing(signals, date, context)
        
        # Apply volatility adjustment if we have enough data
        if len(context.get('equity_curve', pd.Series())) > 21:  # Need at least 21 days
            recent_returns = context['equity_curve'].pct_change().tail(21)
            current_vol = recent_returns.std() * np.sqrt(252)
            
            if current_vol > 0:
                vol_target = 0.15  # 15% annual volatility target
                vol_adjustment = np.clip(vol_target / current_vol, 0.5, 2.0)
                weights *= vol_adjustment
        
        return weights
    
    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply position constraints"""
        weights = weights.fillna(0)
        
        # Apply short selling constraint
        if not self.allow_short:
            weights = weights.clip(lower=0)
        
        # Apply maximum position size constraint
        weights = weights.clip(-self.max_position_size, self.max_position_size)
        
        # Apply leverage constraint
        total_exposure = weights.abs().sum()
        if total_exposure > self.leverage and total_exposure > 0:
            scale_factor = self.leverage / total_exposure
            weights *= scale_factor
        
        return weights
    
    def _calculate_transaction_costs(self, new_weights: pd.Series, prev_weights: pd.Series, portfolio_value: float) -> float:
        """Calculate realistic transaction costs"""
        # Ensure same index
        new_weights = new_weights.reindex(self._assets, fill_value=0)
        prev_weights = prev_weights.reindex(self._assets, fill_value=0)
        
        # Calculate turnover
        turnover = (new_weights - prev_weights).abs()
        
        # Proportional transaction costs
        proportional_cost = (turnover * self.transaction_cost * portfolio_value).sum()
        
        # Fixed costs per trade (only for trades above minimum size)
        trade_count = (turnover > self.min_trade_size).sum()
        fixed_cost = trade_count * self.fixed_cost
        
        # Slippage costs
        slippage_cost = (turnover * self.slippage * portfolio_value).sum()
        
        return proportional_cost + fixed_cost + slippage_cost
    
    def _is_rebalance_date(self, date: pd.Timestamp, prev_date: Optional[pd.Timestamp] = None) -> bool:
        """Check if current date is a rebalancing date"""
        if isinstance(self.rebalance_freq, int):
            # Custom N-day rebalancing
            if prev_date is None:
                return True
            day_diff = (date - prev_date).days
            return day_diff >= self.rebalance_freq
        elif self.rebalance_freq == 'D':
            return True
        elif self.rebalance_freq == 'W':
            return date.weekday() == 0  # Monday
        elif self.rebalance_freq == 'M':
            # Rebalance on month end
            if prev_date is None:
                return date.is_month_end
            return date.month != prev_date.month
        elif self.rebalance_freq == 'Q':
            # Rebalance on quarter end
            if prev_date is None:
                return date.month % 3 == 0 and date.is_month_end
            return (date.month - 1) // 3 != (prev_date.month - 1) // 3
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run the backtest"""
        if self.walk_forward_window is not None and self.walk_forward_step is not None:
            return self._run_walk_forward()
        return self._run_backtest()
    
    def _run_backtest(self, custom_returns: Optional[pd.DataFrame] = None, 
                     custom_signals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run single backtest"""
        
        # Use custom data if provided (for walk-forward)
        returns_data = custom_returns if custom_returns is not None else self.returns
        signals_data = custom_signals if custom_signals is not None else self.signals
        
        # Get common dates
        common_dates = returns_data.index.intersection(signals_data.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and signals")
        
        common_dates = common_dates.sort_values()
        
        # Initialize tracking variables
        portfolio_value = self.initial_cash
        positions = pd.DataFrame(0.0, index=common_dates, columns=self._assets)
        equity_curve = pd.Series(dtype=float, index=common_dates)
        daily_returns = pd.Series(dtype=float, index=common_dates)
        
        prev_weights = pd.Series(0, index=self._assets)
        peak_value = self.initial_cash
        total_transaction_costs = 0.0
        total_trades = 0
        
        # Risk management state
        stop_triggered = False
        stop_reason = None
        
        # Track regime confidence metrics
        confidence_metrics = self._initialize_confidence_metrics()
        
        for i, date in enumerate(common_dates):
            prev_date = common_dates[i-1] if i > 0 else None
            
            # Create context for position sizing
            context = {
                'equity_curve': equity_curve[:i] if i > 0 else pd.Series(dtype=float),
                'positions': positions.iloc[:i] if i > 0 else pd.DataFrame(columns=self._assets),
                'portfolio_value': portfolio_value,
                'date': date,
                'peak_value': peak_value
            }
            
            # Check if we should rebalance
            should_rebalance = (i == 0) or self._is_rebalance_date(date, prev_date)
            
            if should_rebalance and not stop_triggered:
                # Calculate new positions
                new_weights = self._get_positions(date, context)
                
                # No Trade Zone (NTZ): skip trades with small weight changes
                ntz_threshold = 0.002  # 0.2% weight change
                if i > 0:
                    old_weights = prev_weights.copy()
                    small_change = (np.abs(new_weights - old_weights) < ntz_threshold)
                    new_weights[small_change] = old_weights[small_change]
                # Calculate transaction costs (skip first day)
                if i > 0:
                    transaction_cost = self._calculate_transaction_costs(
                        new_weights, prev_weights, portfolio_value
                    )
                    total_transaction_costs += transaction_cost
                    portfolio_value -= transaction_cost
                    
                    # Count trades
                    trades = ((new_weights - prev_weights).abs() > self.min_trade_size).sum()
                    total_trades += trades
                
                prev_weights = new_weights.copy()
                
                # Update confidence metrics
                self._update_confidence_metrics(confidence_metrics, date)
                
            else:
                new_weights = prev_weights.copy()
            
            # Store positions
            positions.loc[date] = new_weights
            
            # Calculate daily return
            if date in returns_data.index:
                asset_returns = returns_data.loc[date].reindex(self._assets, fill_value=0).fillna(0)
                daily_return = (new_weights * asset_returns).sum()
            else:
                daily_return = 0.0
            
            # Update portfolio value
            portfolio_value *= (1 + daily_return)
            equity_curve[date] = portfolio_value
            daily_returns[date] = daily_return
            
            # Update peak for drawdown calculation
            peak_value = max(peak_value, portfolio_value)
            
            # Risk management checks
            if not stop_triggered:
                stop_triggered, stop_reason = self._check_risk_management(
                    portfolio_value, peak_value, equity_curve[:i+1], positions.iloc[:i+1], date, context
                )
                
                if stop_triggered:
                    # Zero out remaining positions
                    remaining_dates = common_dates[i+1:]
                    for future_date in remaining_dates:
                        positions.loc[future_date] = 0
                        equity_curve[future_date] = portfolio_value
                        daily_returns[future_date] = 0
                    break
        
        # Calculate trades
        trades = positions.diff().fillna(positions.iloc[0])

        # Diagnostics: print positions, trades, equity curve, daily returns
        print("\n[DIAGNOSTIC] Positions head:\n", positions.head(20))
        print("\n[DIAGNOSTIC] Trades head:\n", trades.head(20))
        print("\n[DIAGNOSTIC] Equity curve head:\n", equity_curve.head(20))
        print("\n[DIAGNOSTIC] Daily returns head:\n", daily_returns.head(20))
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, daily_returns)
        
        # Add transaction cost metrics
        metrics.update({
            'total_transaction_costs': total_transaction_costs,
            'total_trades': total_trades,
            'avg_transaction_cost': total_transaction_costs / len(common_dates) if len(common_dates) > 0 else 0,
            'turnover_ratio': trades.abs().sum().sum() / len(common_dates) if len(common_dates) > 0 else 0,
        })
        
        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'trades': trades,
            'daily_returns': daily_returns,
            'final_value': portfolio_value,
            'metrics': metrics,
            'stop_reason': stop_reason,
            'confidence_metrics': confidence_metrics if confidence_metrics else None,
        }
    
    def _initialize_confidence_metrics(self) -> Optional[Dict]:
        """Initialize confidence tracking metrics"""
        if self.regime_confidence is None:
            return None
            
        return {
            'avg_confidence': [],
            'low_confidence_periods': 0,
            'high_confidence_periods': 0,
            'position_adjustments': []
        }
    
    def _update_confidence_metrics(self, confidence_metrics: Optional[Dict], date: pd.Timestamp):
        """Update confidence metrics for current date"""
        if confidence_metrics is None or self.regime_confidence is None:
            return
            
        if date in self.regime_confidence.index:
            confidence = self.regime_confidence.loc[date]
            
            if isinstance(confidence, (int, float)):
                avg_conf = confidence
            else:
                avg_conf = confidence.mean()
            
            confidence_metrics['avg_confidence'].append(avg_conf)
            
            if avg_conf < self.confidence_threshold:
                confidence_metrics['low_confidence_periods'] += 1
            elif avg_conf > 0.8:
                confidence_metrics['high_confidence_periods'] += 1
    
    def _check_risk_management(self, portfolio_value: float, peak_value: float, 
                             equity_curve: pd.Series, positions: pd.DataFrame, 
                             date: pd.Timestamp, context: Dict) -> tuple:
        """Check risk management rules"""
        
        # Stop loss check
        if self.stop_loss is not None:
            current_drawdown = (portfolio_value - peak_value) / peak_value
            if current_drawdown <= -self.stop_loss:
                return True, f'stop_loss_triggered_{self.stop_loss:.1%}'
        
        # Take profit check
        if self.take_profit is not None:
            total_return = (portfolio_value - self.initial_cash) / self.initial_cash
            if total_return >= self.take_profit:
                return True, f'take_profit_triggered_{self.take_profit:.1%}'
        
        # Max drawdown check
        if self.max_drawdown is not None:
            current_drawdown = (portfolio_value - peak_value) / peak_value
            if current_drawdown <= -self.max_drawdown:
                return True, f'max_drawdown_triggered_{self.max_drawdown:.1%}'
        
        # Custom risk management
        if self.risk_management_func is not None:
            try:
                should_stop, reason = self.risk_management_func(equity_curve, positions, date, context)
                if should_stop:
                    return True, f'custom_risk_management_{reason}'
            except Exception as e:
                logger.warning(f"Custom risk management function failed at {date}: {e}")
        
        return False, None
    
    def _run_walk_forward(self) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        if self.walk_forward_window is None or self.walk_forward_step is None:
            raise ValueError("walk_forward_window and walk_forward_step must be set")
        
        common_dates = self.returns.index.intersection(self.signals.index).sort_values()
        n_dates = len(common_dates)
        window = self.walk_forward_window
        step = self.walk_forward_step
        
        if window >= n_dates:
            raise ValueError("walk_forward_window must be less than total number of dates")
        
        results = []
        all_equity_curves = []
        all_positions = []
        all_trades = []
        all_daily_returns = []
        
        for start_idx in range(0, n_dates - window, step):
            end_idx = start_idx + window
            test_end_idx = min(end_idx + step, n_dates)
            
            if test_end_idx <= end_idx:
                break
                
            # Get test period dates
            test_dates = common_dates[end_idx:test_end_idx]
            
            if len(test_dates) == 0:
                break
            
            logger.info(f"Walk-forward period: {test_dates[0].date()} to {test_dates[-1].date()}")
            
            # Run backtest on test period
            test_returns = self.returns.loc[test_dates]
            test_signals = self.signals.loc[test_dates]
            
            # Use initial cash for each period (not compound across periods)
            result = self._run_backtest(test_returns, test_signals)
            results.append(result)
            
            # Collect results
            all_equity_curves.append(result['equity_curve'])
            all_positions.append(result['positions'])
            all_trades.append(result['trades'])
            all_daily_returns.append(result['daily_returns'])
        
        if not results:
            raise ValueError("No valid walk-forward periods found")
        
        # Combine all periods
        combined_equity = pd.concat(all_equity_curves)
        combined_positions = pd.concat(all_positions)
        combined_trades = pd.concat(all_trades)
        combined_returns = pd.concat(all_daily_returns)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(combined_equity, combined_returns)
        
        return {
            'equity_curve': combined_equity,
            'positions': combined_positions,
            'trades': combined_trades,
            'daily_returns': combined_returns,
            'final_value': combined_equity.iloc[-1],
            'metrics': overall_metrics,
            'walk_forward_results': results,
            'n_periods': len(results)
        }
    
    def _calculate_metrics(self, equity_curve: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(equity_curve) == 0 or len(daily_returns) == 0:
            return {}
        
        # Remove any NaN or infinite values
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Basic returns
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized metrics
        n_periods = len(daily_returns)
        years = n_periods / 252  # Assume 252 trading days per year
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # Volatility
        if len(daily_returns) > 1:
            daily_vol = daily_returns.std()
            annualized_vol = daily_vol * np.sqrt(252)
        else:
            annualized_vol = 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if annualized_vol > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe_ratio = 0
        
        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        positive_days = (daily_returns > 0).sum()
        win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_periods': n_periods,
            'positive_periods': positive_days,
        }
    
    @staticmethod
    def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Static method to compute metrics from equity curve"""
        if len(equity_curve) < 2:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        n_periods = len(returns)
        years = n_periods / 252
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        annualized_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        if annualized_vol > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe_ratio = 0
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
        }

    def debug_position_sizing(self, signals, date, context):
        """Debug position sizing calculations"""
        if self.regime_confidence is not None and date in self.regime_confidence.index:
            confidence = self.regime_confidence.loc[date]
            print(f"Date {date}:")
            print(f"  Raw signals: {signals.to_dict()}")
            print(f"  Confidence: {confidence}")
            print(f"  Base position size: {self.base_position_size}")
            base_weights = signals / signals.abs().sum() if signals.abs().sum() > 0 else signals * 0
            adjusted_weights = base_weights * confidence * self.base_position_size * self.leverage
            print(f"  Final weights: {adjusted_weights.to_dict()}")


# Example usage and custom functions
def risk_parity_position_sizing(signals: pd.Series, returns: pd.DataFrame, date: pd.Timestamp, context: Dict) -> pd.Series:
    """Example risk parity position sizing function"""
    lookback = 21
    
    if date not in returns.index:
        return pd.Series(0, index=signals.index)
    
    try:
        date_idx = returns.index.get_loc(date)
        if date_idx < lookback:
            return signals * 0.1  # Small initial positions
        
        # Calculate covariance matrix
        recent_returns = returns.iloc[date_idx-lookback:date_idx]
        
        if len(recent_returns) < 5:  # Need minimum data
            return signals * 0.1
        
        # Risk parity weights
        volatilities = recent_returns.std()
        volatilities = volatilities.replace(0, volatilities.median())  # Handle zero vol
        
        if volatilities.sum() > 0:
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()
            
            # Apply signal direction
            final_weights = weights * np.sign(signals)
            return final_weights.reindex(signals.index, fill_value=0)
        else:
            return signals * 0.1
            
    except Exception as e:
        logger.warning(f"Risk parity sizing failed at {date}: {e}")
        return signals * 0.1


def custom_risk_management(equity_curve: pd.Series, positions: pd.DataFrame, 
                          date: pd.Timestamp, context: Dict) -> tuple:
    """Example custom risk management function"""
    if len(equity_curve) < 10:
        return False, None
    
    # Check for consecutive losing days
    recent_returns = equity_curve.pct_change().tail(5)
    if (recent_returns < -0.01).sum() >= 4:  # 4 out of 5 days with >1% loss
        return True, "consecutive_losses"
    
    # Check for extreme single day loss
    if len(recent_returns) > 0 and recent_returns.iloc[-1] < -0.05:  # >5% single day loss
        return True, "extreme_daily_loss"
    
    return False, None


# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    # Sample returns (random for demo)
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    
    # Sample signals
    signals = pd.DataFrame(
        np.random.choice([-1, 0, 1], size=(len(dates), len(assets)), p=[0.2, 0.6, 0.2]),
        index=dates,
        columns=assets
    )
    
    # Sample regime confidence
    regime_confidence = pd.DataFrame(
        np.random.beta(2, 2, size=(len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        returns=returns,
        signals=signals,
        initial_cash=100000,
        transaction_cost=0.0015,
        position_sizing='regime_confidence',
        regime_confidence=regime_confidence,
        confidence_threshold=0.6,
        max_position_size=0.25,
        rebalance_freq='W',
        custom_position_func=risk_parity_position_sizing,
        risk_management_func=custom_risk_management,
    )
    
    # Run backtest
    results = engine.run()
    
    # Display results
    print("Backtest Results:")
    print("-" * 50)
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nFinal Portfolio Value: ${results['final_value']:,.2f}")
    if results['stop_reason']:
        print(f"Stopped due to: {results['stop_reason']}")
