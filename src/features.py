import numpy as np
import pandas as pd
from typing import Optional, Dict

def calculate_rsi(returns: pd.Series, periods: int = 14) -> pd.Series:
    delta = returns.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(returns: pd.Series) -> pd.Series:
    ema12 = returns.ewm(span=12).mean()
    ema26 = returns.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return signal

def calculate_bollinger_position(returns: pd.Series) -> pd.Series:
    sma = returns.rolling(20).mean()
    std = returns.rolling(20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    position = (returns - lower) / (upper - lower)
    return position

def calculate_williams_r(returns: pd.Series) -> pd.Series:
    high = returns.rolling(14).max()
    low = returns.rolling(14).min()
    williams_r = ((high - returns) / (high - low)) * -100
    return williams_r

def calculate_obv(returns: pd.Series) -> pd.Series:
    obv = pd.Series(index=returns.index, dtype=float)
    obv.iloc[0] = 0
    for i in range(1, len(returns)):
        if returns.iloc[i] > returns.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + abs(returns.iloc[i])
        elif returns.iloc[i] < returns.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - abs(returns.iloc[i])
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def calculate_semiconductor_pmi(returns: pd.Series) -> pd.Series:
    momentum = returns.rolling(20).mean()
    volatility = returns.rolling(20).std()
    pmi = (momentum / volatility) * 50 + 50
    return pmi

def calculate_memory_logic_spread(returns: pd.Series) -> pd.Series:
    short_momentum = returns.rolling(5).mean()
    long_momentum = returns.rolling(20).mean()
    spread = short_momentum - long_momentum
    return spread

def calculate_equipment_design_ratio(returns: pd.Series) -> pd.Series:
    equipment_proxy = returns.rolling(10).std()
    design_proxy = returns.rolling(30).std()
    ratio = equipment_proxy / (design_proxy + 1e-6)
    return ratio

def calculate_enhanced_features(returns: pd.Series, macro_data: Optional[Dict] = None, include_semiconductor_features: bool = True) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    features["returns"] = returns
    features["log_returns"] = np.log(returns + 1).values
    features["volatility"] = returns.ewm(span=15).std().values
    features["realized_volatility"] = returns.rolling(15).std().values
    roll_mean = returns.rolling(15).mean()
    features["momentum"] = roll_mean.to_numpy() if hasattr(roll_mean, 'to_numpy') else np.asarray(roll_mean)
    for period in [5, 20, 60]:
        features[f"momentum_{period}d"] = returns.rolling(period).mean().values
        features[f"roc_{period}d"] = (returns / returns.shift(period) - 1).values
    features["rsi_14"] = calculate_rsi(returns, 14).values
    features["rsi_30"] = calculate_rsi(returns, 30).values
    features["macd_signal"] = calculate_macd(returns).values
    features["bollinger_position"] = calculate_bollinger_position(returns).values
    features["williams_r"] = calculate_williams_r(returns).values
    features["volume_ratio"] = (returns.rolling(5).std() / returns.rolling(21).std()).values
    features["volume_sma_ratio"] = (returns.rolling(5).std() / returns.rolling(21).std()).values
    features["on_balance_volume"] = calculate_obv(returns).values
    if macro_data:
        if 'VIX' in macro_data:
            pass  # Remove vix_level, vix_change, vix
        if 'TNX' in macro_data and 'TYX' in macro_data:
            features["yield_spread"] = (macro_data['TYX'] - macro_data['TNX']).values
            features["term_structure_slope"] = (macro_data['TYX'] - macro_data['TNX']).values
        if 'DXY' in macro_data:
            features["dollar_strength"] = macro_data['DXY'].pct_change().values
    else:
        features["yield_spread"] = returns.rolling(63).mean().values
        features["term_structure_slope"] = returns.rolling(63).mean().values
        features["dollar_strength"] = returns.rolling(21).std().values
    if include_semiconductor_features:
        features["semiconductor_pmi"] = calculate_semiconductor_pmi(returns).values
        features["memory_vs_logic_spread"] = calculate_memory_logic_spread(returns).values
        features["equipment_vs_design_ratio"] = calculate_equipment_design_ratio(returns).values
    features["skewness"] = returns.rolling(15).skew().values
    features["kurtosis"] = returns.rolling(15).kurt().values

    # --- New Feature Placeholders ---
    # Cross-asset flows (e.g., credit spreads, risk parity flows)
    features["credit_spread"] = 0.0  # TODO: Replace with real data
    features["risk_parity_flow"] = 0.0  # TODO: Replace with real data
    # Relative strength between semis and tech indexes
    features["semis_vs_tech_strength"] = 0.0  # TODO: Replace with real data
    # Sector ETF flows or volume anomalies
    features["sector_etf_flow"] = 0.0  # TODO: Replace with real data
    features["volume_anomaly"] = 0.0  # TODO: Replace with real data
    # Short interest, implied volatility skew
    features["short_interest"] = 0.0  # TODO: Replace with real data
    features["implied_vol_skew"] = 0.0  # TODO: Replace with real data
    # News-based sentiment (LLM-based scoring)
    features["news_sentiment"] = 0.0  # TODO: Replace with real data
    # Cross-sectional momentum ranking (placeholder, needs all tickers)
    features["momentum_rank"] = 0.0  # TODO: Replace with real cross-sectional momentum rank
    # NVDA/AMD dispersion (placeholder)
    features["nvda_amd_dispersion"] = 0.0  # TODO: Replace with real NVDA/AMD spread
    # ETF volume surge (placeholder)
    features["etf_volume_surge"] = 0.0  # TODO: Replace with real ETF volume surge
    # VIX/skew/IV percentile (placeholder)
    features["vix_percentile"] = 0.0  # TODO: Replace with real VIX percentile
    features["skew_percentile"] = 0.0  # TODO: Replace with real skew percentile
    features["iv_percentile"] = 0.0  # TODO: Replace with real IV percentile
    # Yield curve inversion (placeholder)
    features["yield_curve_inversion"] = 0.0  # TODO: Replace with real yield curve inversion
    # News-LLM sentiment (placeholder)
    features["news_llm_sentiment"] = 0.0  # TODO: Replace with real LLM sentiment

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features 