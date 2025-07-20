import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any

class BacktestEngine:
    def __init__(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float = 1.0,
        rebalance_freq: str = 'D',
        transaction_cost: float = 0.001,  # Realistic 0.1% transaction cost
        fixed_cost: float = 0.0,        # fixed cost per trade
        leverage: float = 1.0,
        position_sizing: str = 'proportional',  # or 'fixed', or pass a function
        slippage: float = 0.0005,       # Realistic 0.05% slippage per trade
        allow_short: bool = False,
        custom_position_func: Optional[Callable] = None,      # function(signals, returns, date, context) -> positions
        stop_loss: Optional[float] = None,        # e.g., 0.05 for 5% stop-loss
        take_profit: Optional[float] = None,      # e.g., 0.10 for 10% take-profit
        max_drawdown: Optional[float] = None,     # e.g., 0.20 for 20% max drawdown
        walk_forward_window: Optional[int] = None, # number of periods for walk-forward (rolling OOS)
        walk_forward_step: Optional[int] = None,   # step size for walk-forward
        risk_management_func: Optional[Callable] = None,      # function(equity_curve, positions, date, context) -> (stop, reason)
        regime_confidence: Optional[pd.DataFrame] = None,     # Regime confidence scores for dynamic sizing
        base_position_size: float = 1.0,          # Base position size multiplier
        confidence_threshold: float = 0.5,        # Minimum confidence for full position
        min_trade_size: float = 0.001,           # Minimum trade size to avoid micro-trades
        max_position_size: float = 0.3,          # Maximum position size per asset
    ):
        self.returns = returns
        self.signals = signals
        self.initial_cash = initial_cash
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.fixed_cost = fixed_cost
        self.leverage = leverage
        self.position_sizing = position_sizing
        self.slippage = slippage
        self.allow_short = allow_short
        self.custom_position_func = custom_position_func
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.walk_forward_window = walk_forward_window
        self.walk_forward_step = walk_forward_step
        self.risk_management_func = risk_management_func
        self.regime_confidence = regime_confidence
        self.base_position_size = base_position_size
        self.confidence_threshold = confidence_threshold
        self.min_trade_size = min_trade_size
        self.max_position_size = max_position_size

    def _get_positions(self, date, context=None):
        if self.custom_position_func is not None:
            return self.custom_position_func(self.signals, self.returns, date, context)
        
        sig = self.signals.loc[date]
        
        if self.position_sizing == 'proportional':
            weights = sig / sig.abs().sum() if sig.abs().sum() > 0 else sig
        elif self.position_sizing == 'fixed':
            weights = sig.apply(lambda x: np.sign(x) * 1.0)
        elif self.position_sizing == 'regime_confidence':
            weights = self._get_regime_confidence_positions(sig, date)
        elif callable(self.position_sizing):
            weights = self.position_sizing(sig, self.returns, date, context)
        else:
            weights = sig  # assume user provides weights
        
        # Apply position size constraints
        if not self.allow_short:
            weights = weights.clip(lower=0)
            weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        # Apply maximum position size constraint
        weights = weights.clip(upper=self.max_position_size)
        
        # Normalize weights to ensure they sum to leverage
        if weights.abs().sum() > 0:
            weights = weights * self.leverage / weights.abs().sum()
        
        return weights

    def _get_regime_confidence_positions(self, signals: pd.Series, date: pd.Timestamp) -> pd.Series:
        """
        Calculate dynamic position sizes based on regime confidence
        
        Args:
            signals: Signal values for current date
            date: Current date
            
        Returns:
            Position weights adjusted by regime confidence
        """
        if self.regime_confidence is None or date not in self.regime_confidence.index:
            # Fallback to proportional sizing
            return signals / signals.abs().sum() if signals.abs().sum() > 0 else signals
        
        # Get confidence scores for current date
        confidence_scores = self.regime_confidence.loc[date]
        
        # Calculate position adjustments based on confidence
        position_adjustments = pd.Series(index=signals.index, dtype=float)
        
        for asset in signals.index:
            if asset in confidence_scores.index:
                confidence = confidence_scores[asset]
                
                # Adjust position size based on confidence
                if confidence >= self.confidence_threshold:
                    # High confidence: full position
                    adjustment = 1.0
                elif confidence > 0:
                    # Low confidence: scaled position
                    adjustment = confidence / self.confidence_threshold
                else:
                    # No confidence: minimal position
                    adjustment = 0.1
                
                position_adjustments[asset] = adjustment
            else:
                # No confidence data available
                position_adjustments[asset] = 0.5
        
        # Apply adjustments to signals
        adjusted_signals = signals * position_adjustments * self.base_position_size
        
        # Normalize to ensure weights sum to reasonable value
        if adjusted_signals.abs().sum() > 0:
            adjusted_signals = adjusted_signals / adjusted_signals.abs().sum()
        
        return adjusted_signals

    def _calculate_position_sizes(self, signals: pd.Series, date: pd.Timestamp, context: Dict[str, Any]) -> pd.Series:
        """Calculate position sizes based on the selected method"""
        if self.position_sizing == 'proportional':
            return signals * self.leverage
        elif self.position_sizing == 'fixed':
            return pd.Series(self.leverage / len(signals), index=signals.index) * np.sign(signals)
        elif self.position_sizing == 'regime_confidence':
            return self._regime_confidence_position_sizing(signals, date, context)
        elif self.position_sizing == 'dynamic':
            return self._dynamic_position_sizing(signals, date, context)
        elif callable(self.position_sizing):
            return self.position_sizing(signals, self.returns, date, context)
        else:
            return signals * self.leverage
    
    def _regime_confidence_position_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict[str, Any]) -> pd.Series:
        """Position sizing based on regime confidence"""
        if 'regime_confidence' not in context or context['regime_confidence'] is None:
            return signals * self.leverage
        
        confidence = context['regime_confidence']
        base_position_size = context.get('base_position_size', 1.0)
        confidence_threshold = context.get('confidence_threshold', 0.5)
        
        # Get confidence for current date
        if date in confidence.index:
            date_confidence = confidence.loc[date]
        else:
            # Use average confidence if date not found
            date_confidence = confidence.mean()
        
        # Adjust position size based on confidence
        if isinstance(date_confidence, pd.Series):
            # Multi-asset confidence
            confidence_adj = np.clip(date_confidence, 0.3, 1.0)
        else:
            # Single confidence value
            confidence_adj = np.clip(date_confidence, 0.3, 1.0)
        
        # Apply confidence adjustment
        adjusted_signals = signals * base_position_size * confidence_adj
        
        # Ensure positions are within bounds
        return np.clip(adjusted_signals, -self.leverage, self.leverage)
    
    def _dynamic_position_sizing(self, signals: pd.Series, date: pd.Timestamp, context: Dict[str, Any]) -> pd.Series:
        """Dynamic position sizing based on regime confidence and volatility"""
        if 'regime_confidence' not in context or context['regime_confidence'] is None:
            return signals * self.leverage
        
        confidence = context['regime_confidence']
        volatility = context.get('volatility', None)
        base_size = context.get('base_position_size', 0.1)  # 10% base allocation
        
        # Get confidence for current date
        if date in confidence.index:
            date_confidence = confidence.loc[date]
        else:
            date_confidence = confidence.mean()
        
        # Confidence adjustment: reduce size when regime is uncertain
        if isinstance(date_confidence, pd.Series):
            confidence_adj = np.clip(date_confidence, 0.3, 1.0)
        else:
            confidence_adj = np.clip(date_confidence, 0.3, 1.0)
        
        # Volatility adjustment if available
        vol_adj = 1.0
        if volatility is not None and date in volatility.index:
            date_vol = volatility.loc[date]
            vol_target = 0.12  # 12% annual volatility target
            if isinstance(date_vol, pd.Series):
                vol_adj = np.clip(vol_target / date_vol, 0.5, 2.0)
            else:
                vol_adj = np.clip(vol_target / date_vol, 0.5, 2.0)
        
        # Combine adjustments
        final_positions = signals * base_size * confidence_adj * vol_adj
        
        # Ensure positions are within reasonable bounds
        return np.clip(final_positions, -0.5, 0.5)

    def _calculate_transaction_costs(self, new_weights: pd.Series, prev_weights: pd.Series, portfolio_value: float) -> float:
        """
        Calculate realistic transaction costs including:
        - Proportional transaction costs
        - Fixed costs per trade
        - Slippage costs
        """
        # Calculate turnover
        turnover = (new_weights - prev_weights).abs()
        
        # Proportional transaction costs
        proportional_cost = (turnover * self.transaction_cost * portfolio_value).sum()
        
        # Fixed costs per trade (only for non-zero trades)
        trade_count = (turnover > self.min_trade_size).sum()
        fixed_cost = trade_count * self.fixed_cost
        
        # Slippage costs (proportional to trade size)
        slippage_cost = (turnover * self.slippage * portfolio_value).sum()
        
        total_cost = proportional_cost + fixed_cost + slippage_cost
        
        return total_cost

    def run(self):
        if self.walk_forward_window is not None and self.walk_forward_step is not None:
            return self._run_walk_forward()
        return self._run_backtest(self.returns, self.signals, self.initial_cash)

    def _run_backtest(self, returns, signals, initial_cash):
        dates = returns.index.intersection(signals.index)
        positions = pd.DataFrame(index=dates, columns=returns.columns)
        cash = initial_cash
        equity_curve = []
        prev_weights = pd.Series(0, index=returns.columns)
        peak = cash
        stop = False
        stop_reason = None
        
        # Track transaction costs and other metrics
        total_transaction_costs = 0.0
        total_trades = 0
        daily_returns = []
        
        # Track regime confidence metrics if available
        confidence_metrics = {}
        if self.regime_confidence is not None:
            confidence_metrics = {
                'avg_confidence': [],
                'low_confidence_periods': 0,
                'high_confidence_periods': 0
            }
        
        for i, date in enumerate(dates):
            context = {
                'equity_curve': pd.Series(equity_curve, index=dates[:i]) if i > 0 else pd.Series(dtype=float),
                'positions': positions.iloc[:i] if i > 0 else pd.DataFrame(columns=returns.columns),
                'cash': cash,
                'date': date,
            }
            
            if i == 0 or self._is_rebalance_date(date):
                weights = self._get_positions(date, context)
                
                # Calculate transaction costs
                if i > 0:  # Skip first day (no previous weights)
                    transaction_cost = self._calculate_transaction_costs(weights, prev_weights, cash)
                    total_transaction_costs += transaction_cost
                    cash -= transaction_cost
                    
                    # Count trades
                    trade_count = ((weights - prev_weights).abs() > self.min_trade_size).sum()
                    total_trades += trade_count
                
                # Track confidence metrics
                if self.regime_confidence is not None and date in self.regime_confidence.index:
                    avg_conf = self.regime_confidence.loc[date].mean()
                    confidence_metrics['avg_confidence'].append(avg_conf)
                    if avg_conf < self.confidence_threshold:
                        confidence_metrics['low_confidence_periods'] += 1
                    else:
                        confidence_metrics['high_confidence_periods'] += 1
                
                prev_weights = weights
            else:
                weights = prev_weights
                
            positions.loc[date] = weights
            
            # Compute daily return
            daily_ret = (weights * returns.loc[date]).sum()
            cash *= (1 + daily_ret)
            equity_curve.append(cash)
            daily_returns.append(daily_ret)
            
            # Update peak for drawdown calculation
            peak = max(peak, cash)
            
            # Risk management checks
            if self.stop_loss is not None and len(equity_curve) > 1:
                current_drawdown = (cash - peak) / peak
                if current_drawdown <= -self.stop_loss:
                    stop = True
                    stop_reason = f'stop_loss ({self.stop_loss})'
                    
            if self.take_profit is not None and len(equity_curve) > 1:
                total_return = (cash - self.initial_cash) / self.initial_cash
                if total_return >= self.take_profit:
                    stop = True
                    stop_reason = f'take_profit ({self.take_profit})'
                    
            if self.max_drawdown is not None and len(equity_curve) > 1:
                current_drawdown = (cash - peak) / peak
                if current_drawdown <= -self.max_drawdown:
                    stop = True
                    stop_reason = f'max_drawdown ({self.max_drawdown})'
                    
            if self.risk_management_func is not None:
                stop_custom, reason = self.risk_management_func(
                    pd.Series(equity_curve, index=dates[:i+1]), 
                    positions.iloc[:i+1], 
                    date, 
                    context
                )
                if stop_custom:
                    stop = True
                    stop_reason = reason
                    
            if stop:
                # Fill the rest of the positions with zeros
                for j in range(i+1, len(dates)):
                    positions.iloc[j] = 0
                    equity_curve.append(cash)
                break
                
        # FIXED: Ensure all DataFrames match the actual dates used
        actual_dates = dates[:len(equity_curve)]
        equity_curve = pd.Series(equity_curve, index=actual_dates)
        positions = positions.loc[actual_dates]
        trades = positions.diff().fillna(positions.iloc[0])
        
        # FIXED: Ensure daily_returns matches the actual dates used
        actual_dates = dates[:len(daily_returns)]
        daily_returns = pd.Series(daily_returns, index=actual_dates)
        
        # Calculate realistic metrics
        total_return = (equity_curve.iloc[-1] - self.initial_cash) / self.initial_cash
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / annualized_vol if annualized_vol > 0 else 0  # Using 2% risk-free rate
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        result = {
            'equity_curve': equity_curve,
            'positions': positions,
            'trades': trades,
            'final_value': equity_curve.iloc[-1],
            'returns': daily_returns,
            'stop_reason': stop_reason,
            'total_transaction_costs': total_transaction_costs,
            'total_trades': total_trades,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'turnover_ratio': trades.abs().sum().sum() / len(actual_dates),
        }
        
        # Add confidence metrics if available
        if confidence_metrics:
            result['confidence_metrics'] = confidence_metrics
        
        return result

    def _run_walk_forward(self):
        # Rolling walk-forward: train on window, test on next step, repeat
        results = []
        dates = self.returns.index.intersection(self.signals.index)
        n = len(dates)
        window = self.walk_forward_window
        step = self.walk_forward_step
        if window is None or step is None:
            raise ValueError("walk_forward_window and walk_forward_step must be set for walk-forward analysis.")
        for start in range(0, n - window, step):
            end = start + window
            test_end = min(end + step, n)
            window_dates = dates[start:end]
            test_dates = dates[end:test_end]
            if len(test_dates) == 0:
                break
            # Use only test_dates for out-of-sample
            test_returns = self.returns.loc[test_dates]
            test_signals = self.signals.loc[test_dates]
            res = self._run_backtest(test_returns, test_signals, self.initial_cash if len(results) == 0 else results[-1]['final_value'])
            results.append(res)
        # Concatenate equity curves and positions
        equity_curve = pd.concat([r['equity_curve'] for r in results])
        positions = pd.concat([r['positions'] for r in results])
        trades = pd.concat([r['trades'] for r in results])
        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'trades': trades,
            'final_value': equity_curve.iloc[-1],
            'returns': equity_curve.pct_change().fillna(0),
            'walk_forward_results': results,
        }

    def _is_rebalance_date(self, date):
        if self.rebalance_freq == 'D':
            return True
        elif self.rebalance_freq == 'M':
            return date.is_month_start or date.is_month_end
        elif self.rebalance_freq == 'W':
            return date.weekday() == 0  # Monday
        # Add more logic as needed
        return True

    @staticmethod
    def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.05):
        returns = equity_curve.pct_change().dropna()
        cum_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        annual_return = (1 + cum_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
        max_dd = (equity_curve / equity_curve.cummax() - 1).min()
        return {
            'Cumulative Return': cum_return,
            'Annualized Return': annual_return,
            'Annualized Vol': annual_vol,
            'Sharpe': sharpe,
            'Max Drawdown': max_dd,
        }

    def run_confidence_based_backtest(self, returns: pd.DataFrame, signals: pd.DataFrame, regime_confidence: pd.DataFrame, volatility: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run backtest with confidence-based position sizing
        
        Args:
            returns: Asset returns
            signals: Trading signals
            regime_confidence: Regime confidence scores
            volatility: Market volatility (optional)
            
        Returns:
            Dictionary with backtest results
        """
        # logger.info("Running confidence-based backtest...") # This line was not in the original file, so it's commented out.
        
        # Initialize portfolio
        portfolio_value = self.initial_cash
        positions = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        cash = pd.Series(self.initial_cash, index=returns.index)
        
        # Track confidence metrics
        confidence_metrics = {
            'avg_confidence': [],
            'low_confidence_periods': 0,
            'high_confidence_periods': 0,
            'position_size_changes': []
        }
        
        for i, date in enumerate(returns.index):
            if i == 0:
                continue
                
            # Get current confidence and volatility
            current_confidence = regime_confidence.loc[date] if date in regime_confidence.index else pd.Series(0.5, index=returns.columns)
            current_vol = volatility.loc[date] if volatility is not None and date in volatility.index else pd.Series(0.15, index=returns.columns)
            
            # Calculate confidence-based position sizing
            base_allocation = 0.10  # 10% base allocation
            
            # Confidence adjustment (reduce size when uncertain)
            confidence_adj = current_confidence.clip(0.2, 1.0)
            
            # Volatility adjustment (reduce size in high vol periods)
            vol_target = 0.12  # 12% annual volatility target
            vol_adj = vol_target / (current_vol + 1e-6)
            vol_adj = vol_adj.clip(0.3, 2.0)
            
            # Calculate position sizes
            position_sizes = signals.loc[date] * base_allocation * confidence_adj * vol_adj
            
            # Apply position bounds
            position_sizes = position_sizes.clip(-0.5, 0.5)
            
            # Update positions
            positions.loc[date] = position_sizes
            
            # Calculate portfolio returns
            asset_returns = returns.loc[date]
            portfolio_return = (position_sizes * asset_returns).sum()
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            cash.loc[date] = portfolio_value
            
            # Track confidence metrics
            avg_conf = current_confidence.mean()
            confidence_metrics['avg_confidence'].append(avg_conf)
            
            if avg_conf < 0.5:
                confidence_metrics['low_confidence_periods'] += 1
            elif avg_conf > 0.7:
                confidence_metrics['high_confidence_periods'] += 1
            
            # Track position size changes
            if i > 1:
                prev_positions = positions.iloc[i-1]
                pos_change = abs(position_sizes - prev_positions).mean()
                confidence_metrics['position_size_changes'].append(pos_change)
        
        # Calculate equity curve
        equity_curve = cash / self.initial_cash
        
        # Calculate metrics
        metrics = self.compute_metrics(equity_curve)
        
        # Add confidence metrics
        metrics['confidence_metrics'] = confidence_metrics
        
        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'metrics': metrics,
            'final_value': portfolio_value,
            'confidence_metrics': confidence_metrics
        }

    def run_regime_aware_backtest(self, returns: pd.DataFrame, signals: pd.DataFrame, regimes: pd.DataFrame, regime_confidence: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest with regime-aware risk controls
        
        Args:
            returns: Asset returns
            signals: Trading signals
            regimes: Regime predictions
            regime_confidence: Regime confidence scores
            
        Returns:
            Dictionary with backtest results
        """
        # logger.info("Running regime-aware backtest...") # This line was not in the original file, so it's commented out.
        
        # Initialize portfolio
        portfolio_value = self.initial_cash
        positions = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        cash = pd.Series(self.initial_cash, index=returns.index)
        
        # Risk control parameters
        risk_params = {
            'max_position_size': 0.10,  # Default 10% max per position
            'portfolio_heat': 0.20,      # Default 20% total portfolio risk
            'regime_confidence': 0.5     # Default confidence
        }
        
        for i, date in enumerate(returns.index):
            if i == 0:
                continue
                
            # Get current regime and confidence
            current_regime = regimes.loc[date].iloc[0] if date in regimes.index else 'Medium Vol'
            current_confidence = regime_confidence.loc[date].mean() if date in regime_confidence.index else 0.5
            
            # Apply regime-aware risk controls
            if current_regime == 'High Vol' or current_confidence < 0.7:
                # Reduce exposure during high volatility or uncertainty
                risk_params['max_position_size'] = 0.05  # 5% max per position
                risk_params['portfolio_heat'] = 0.15     # 15% total portfolio risk
            elif current_regime == 'Low Vol' and current_confidence > 0.8:
                # Increase exposure when confident in low volatility
                risk_params['max_position_size'] = 0.15  # 15% max per position
                risk_params['portfolio_heat'] = 0.25     # 25% total portfolio risk
            else:
                # Normal sizing
                risk_params['max_position_size'] = 0.10  # 10% max per position
                risk_params['portfolio_heat'] = 0.20     # 20% total portfolio risk
            
            risk_params['regime_confidence'] = current_confidence
            
            # Calculate position sizes with regime-aware limits
            position_sizes = signals.loc[date] * risk_params['max_position_size']
            
            # Apply confidence scaling
            confidence_adj = current_confidence.clip(0.2, 1.0)
            position_sizes *= confidence_adj
            
            # Apply position bounds
            position_sizes = position_sizes.clip(-risk_params['max_position_size'], risk_params['max_position_size'])
            
            # Apply portfolio heat limit
            total_exposure = abs(position_sizes).sum()
            if total_exposure > risk_params['portfolio_heat']:
                scale_factor = risk_params['portfolio_heat'] / total_exposure
                position_sizes *= scale_factor
            
            # Update positions
            positions.loc[date] = position_sizes
            
            # Calculate portfolio returns
            asset_returns = returns.loc[date]
            portfolio_return = (position_sizes * asset_returns).sum()
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            cash.loc[date] = portfolio_value
        
        # Calculate equity curve
        equity_curve = cash / self.initial_cash
        
        # Calculate metrics
        metrics = self.compute_metrics(equity_curve)
        
        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'metrics': metrics,
            'final_value': portfolio_value,
            'risk_params': risk_params
        }

# Example custom position sizing function
# def my_position_sizing(signals, returns, date, context):
#     # Example: risk parity weights
#     lookback = 21
#     if date not in returns.index:
#         return signals * 0
#     idx = returns.index.get_loc(date)
#     if idx < lookback:
#         return signals * 0
#     cov = returns.iloc[idx-lookback:idx].cov()
#     inv_vol = 1 / np.sqrt(np.diag(cov))
#     weights = inv_vol / inv_vol.sum()
#     return pd.Series(weights, index=signals.index) 