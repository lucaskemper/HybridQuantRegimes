# src/signals.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.regime import MarketRegimeDetector


class SignalGenerator:
    def __init__(
        self,
        lookback_fast: int = 10,  # Updated from 20 to 10 to align with optimal window size
        lookback_slow: int = 21,  # Updated from 50 to 21 for better alignment
        normalize: bool = True,
        use_regime: bool = True,
        regime_detector: Optional[MarketRegimeDetector] = None,
    ):
        self.lookback_fast = lookback_fast
        self.lookback_slow = lookback_slow
        self.normalize = normalize
        self.use_regime = use_regime
        self.regime_detector = regime_detector

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate combined alpha signal with regime-aware structure."""
        returns = market_data['returns']
        prices = market_data['prices']
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        macro = market_data.get('macro', None)

        # Detect regimes per asset (or use aggregate)
        regime_masks = {}
        regime_series_dict = {}
        if self.use_regime and self.regime_detector:
            for ticker in returns.columns:
                ticker_returns = returns[ticker]
                if not isinstance(ticker_returns, pd.Series):
                    ticker_returns = pd.Series(ticker_returns)
                regime_series = self.regime_detector.fit_predict(ticker_returns)
                regime_series_dict[ticker] = regime_series
                # FIXED: More inclusive regime filtering - allow trading in all regimes
                if regime_series.dtype == object or regime_series.dtype == 'string':
                    # Allow trading in all regimes, but adjust position size based on regime
                    regime_masks[ticker] = pd.Series(True, index=returns.index)
                else:
                    # For numeric regimes, allow trading in all regimes
                    regime_masks[ticker] = pd.Series(True, index=returns.index)
        else:
            for ticker in returns.columns:
                regime_masks[ticker] = pd.Series(True, index=returns.index)
                regime_series_dict[ticker] = None

        for ticker in returns.columns:
            ticker_price = prices[ticker]
            ticker_returns = returns[ticker]
            # Ensure pd.Series, not DataFrame
            if isinstance(ticker_price, pd.DataFrame):
                ticker_price = ticker_price.iloc[:, 0]
            if isinstance(ticker_returns, pd.DataFrame):
                ticker_returns = ticker_returns.iloc[:, 0]
            # Force to 1D Series
            ticker_price = pd.Series(np.asarray(ticker_price), index=ticker_price.index)
            ticker_returns = pd.Series(np.asarray(ticker_returns), index=ticker_returns.index)
            macro_df = None
            if macro is not None and ticker in macro:
                if isinstance(macro[ticker], pd.DataFrame):
                    macro_df = macro[ticker]
                else:
                    macro_df = None
            # Only pass macro_df if it's a DataFrame, else None
            regime_series = regime_series_dict.get(ticker, None)
            signal_df = self._build_signal_stack(ticker_price, ticker_returns, macro=macro_df if isinstance(macro_df, pd.DataFrame) else None, regime_series=regime_series)
            final_signal = self._combine_signals(signal_df)

            # FIXED: Apply regime adjustment instead of filtering
            final_signal = self._apply_regime_adjustment(final_signal, regime_series, regime_masks[ticker])

            # Optional normalization across time - less aggressive
            if self.normalize:
                final_signal = (final_signal - final_signal.rolling(252).mean()) / (final_signal.rolling(252).std() + 1e-6)
                final_signal = np.clip(final_signal, -2, 2)  # Allow larger positions

            signals[ticker] = final_signal.fillna(0)

        return signals

    def _build_signal_stack(self, price: pd.Series, returns: pd.Series, macro: Optional[pd.DataFrame] = None, regime_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compute multiple technical indicators and return as signal components, including mean reversion, volatility regime, and macro indicators if provided."""
        if not isinstance(price, pd.Series):
            price = pd.Series(price)
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        df = pd.DataFrame(index=returns.index)

        # Trend: Moving Average Crossover
        fast_ma = price.rolling(self.lookback_fast).mean()
        slow_ma = price.rolling(self.lookback_slow).mean()
        df["trend_signal"] = (fast_ma > slow_ma).astype(float) * 2 - 1

        # Momentum: past return over 20 days
        df["momentum"] = price.pct_change(20)

        # Mean Reversion: z-score of price vs. slow MA
        if isinstance(slow_ma, pd.Series):
            slow_ma_std = slow_ma.rolling(20).std()
            mean_rev = (price - slow_ma) / (slow_ma_std + 1e-6)
        else:
            slow_ma_std = pd.Series(slow_ma).rolling(20).std()
            mean_rev = (pd.Series(price) - pd.Series(slow_ma)) / (slow_ma_std + 1e-6)
        df["mean_reversion"] = -mean_rev.clip(-3, 3)  # Negative for reversion

        # RSI
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))

        # Volatility-Adjusted Return
        vol = returns.rolling(21).std()
        df["risk_adjusted_return"] = returns / (vol + 1e-6)
        df["volatility"] = vol

        # Volatility Regime (if provided)
        if regime_series is not None:
            if isinstance(regime_series, pd.Series):
                if regime_series.dtype == object or regime_series.dtype == 'string':
                    for label in regime_series.unique():
                        df[f"regime_{label}"] = (regime_series == label).astype(float)
                else:
                    df["regime_numeric"] = regime_series

        # MACD
        ema12 = price.ewm(span=12).mean()
        ema26 = price.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        df["macd_signal"] = np.sign(macd - signal_line)

        # Macro indicators (if provided)
        if macro is not None:
            for col in macro.columns:
                df[f"macro_{col}"] = macro[col].reindex(df.index).fillna(method='ffill').fillna(0)

        # Clean and clip
        df = df.fillna(0)
        return df

    def _combine_signals(self, signals: pd.DataFrame) -> pd.Series:
        """Combine signal components into a single composite."""
        # FIXED: More aggressive signal combination weights
        composite = (
            0.25 * signals["trend_signal"] +           # Reduced from 0.3
            0.25 * signals["momentum"].clip(-1, 1) +   # Increased from 0.2
            0.15 * signals["rsi_signal"] +             # Same
            0.20 * signals["risk_adjusted_return"].clip(-1, 1) +  # Same
            0.15 * signals["macd_signal"]              # Same
        )

        return composite.clip(-2, 2)  # Allow larger signal values

    def _apply_regime_adjustment(self, signal: pd.Series, regime_series: Optional[pd.Series], regime_mask: pd.Series) -> pd.Series:
        """Apply regime-based position size adjustment instead of filtering"""
        if regime_series is None:
            return signal
        
        adjusted_signal = signal.copy()
        
        # Define regime adjustments (more aggressive than filtering)
        regime_adjustments = {
            'Low Vol': 1.0,      # Full position in low volatility
            'Medium Vol': 0.8,    # 80% position in medium volatility  
            'High Vol': 0.6,      # 60% position in high volatility
            'Very Low Vol': 1.0,
            'Very High Vol': 0.5
        }
        
        # Apply adjustments based on regime
        for regime_label, adjustment in regime_adjustments.items():
            if regime_series.dtype == object or regime_series.dtype == 'string':
                mask = regime_series == regime_label
            else:
                # For numeric regimes, assume higher numbers = higher volatility
                if len(regime_adjustments) == 3:  # 3-regime case
                    if regime_label == 'Low Vol':
                        mask = regime_series == 0
                    elif regime_label == 'Medium Vol':
                        mask = regime_series == 1
                    elif regime_label == 'High Vol':
                        mask = regime_series == 2
                    else:
                        mask = pd.Series(False, index=signal.index)
                else:
                    mask = pd.Series(False, index=signal.index)
            
            adjusted_signal[mask] = signal[mask] * adjustment
        
        return adjusted_signal

    def explain_signal(self, ticker: str, date: pd.Timestamp, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Return a breakdown of component contributions to final signal."""
        returns = market_data["returns"][ticker]
        prices = market_data["prices"][ticker]
        # Runtime type checks
        if isinstance(prices, pd.DataFrame):
            raise TypeError(f"prices for {ticker} is a DataFrame, expected Series. Columns: {prices.columns}")
        if isinstance(returns, pd.DataFrame):
            raise TypeError(f"returns for {ticker} is a DataFrame, expected Series. Columns: {returns.columns}")
        components = self._build_signal_stack(prices, returns)
        combined = self._combine_signals(components)

        breakdown = {col: components.loc[date, col] for col in components.columns}
        breakdown["final_signal"] = combined.loc[date]
        return breakdown
