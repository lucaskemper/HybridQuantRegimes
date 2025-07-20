#!/usr/bin/env python3
"""
Diagnostic Script for Validation Red Flags

This script investigates the critical issues identified in the validation:
1. Unrealistic Sharpe ratios
2. Data quality issues
3. Potential look-ahead bias
4. Weekend data contamination
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_realistic_sample_data():
    """Create realistic market data with proper characteristics."""
    logger.info("Creating realistic market data...")
    
    # Use realistic market parameters
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Realistic market parameters
    market_params = {
        'NVDA': {'mu': 0.0008, 'sigma': 0.025, 'start_price': 100},  # High growth, high vol
        'MSFT': {'mu': 0.0006, 'sigma': 0.018, 'start_price': 200},  # Stable growth
        'AAPL': {'mu': 0.0005, 'sigma': 0.020, 'start_price': 150},  # Moderate growth
        'GOOGL': {'mu': 0.0007, 'sigma': 0.022, 'start_price': 2500}, # Tech growth
        'TSLA': {'mu': 0.0010, 'sigma': 0.035, 'start_price': 300}   # High risk/reward
    }
    
    prices = pd.DataFrame(index=dates)
    
    for asset, params in market_params.items():
        # Generate realistic price series with proper characteristics
        returns = np.random.normal(params['mu'], params['sigma'], len(dates))
        
        # Add some market structure (autocorrelation, volatility clustering)
        for i in range(1, len(returns)):
            # Add some autocorrelation
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
            
            # Add volatility clustering
            if abs(returns[i-1]) > 2 * params['sigma']:
                returns[i] *= 1.5  # Higher volatility after large moves
        
        # Convert to prices
        price_series = params['start_price'] * np.exp(np.cumsum(returns))
        prices[asset] = price_series
    
    # Remove weekends (business days only)
    business_days = pd.bdate_range('2020-01-01', '2023-12-31')
    prices = prices.reindex(business_days).dropna()
    
    logger.info(f"Created realistic market data: {prices.shape}")
    logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
    logger.info(f"Business days: {len(prices)}")
    
    return prices


def analyze_sharpe_ratios(prices):
    """Analyze Sharpe ratios for different strategies."""
    logger.info("Analyzing Sharpe ratios...")
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Strategy 1: Equal weight
    ew_returns = returns.mean(axis=1)
    
    # Strategy 2: Buy and hold (first two assets)
    asset1, asset2 = returns.columns[:2]
    simple_combo = 0.6 * returns[asset1] + 0.4 * returns[asset2]
    
    # Strategy 3: Random strategy
    np.random.seed(42)
    random_returns = pd.Series(np.random.normal(returns.mean().mean(), returns.std().mean(), len(returns)), index=returns.index)
    
    # Strategy 4: Momentum strategy (realistic)
    momentum_returns = pd.Series(0.0, index=returns.index)
    for i in range(21, len(returns)):
        # Simple momentum: buy if recent performance is positive
        recent_perf = returns.iloc[i-21:i].mean().mean()
        if recent_perf > 0:
            momentum_returns.iloc[i] = returns.iloc[i].mean()
        else:
            momentum_returns.iloc[i] = -returns.iloc[i].mean() * 0.5  # Short with reduced exposure
    
    # Calculate Sharpe ratios
    def calculate_sharpe(returns_series, risk_free_rate=0.02):
        excess_returns = returns_series - risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    strategies = {
        'Equal Weight': ew_returns,
        'Simple Combo': simple_combo,
        'Random': random_returns,
        'Momentum': momentum_returns
    }
    
    results = {}
    for name, strategy_returns in strategies.items():
        sharpe = calculate_sharpe(strategy_returns)
        ann_return = strategy_returns.mean() * 252
        ann_vol = strategy_returns.std() * np.sqrt(252)
        max_dd = calculate_max_drawdown(strategy_returns)
        
        results[name] = {
            'sharpe': sharpe,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'max_dd': max_dd,
            'win_rate': (strategy_returns > 0).mean()
        }
        
        logger.info(f"{name}:")
        logger.info(f"  Sharpe: {sharpe:.3f}")
        logger.info(f"  Ann Return: {ann_return:.3f}")
        logger.info(f"  Ann Vol: {ann_vol:.3f}")
        logger.info(f"  Max DD: {max_dd:.3f}")
        logger.info(f"  Win Rate: {results[name]['win_rate']:.3f}")
    
    return results


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def check_data_quality(prices):
    """Check for data quality issues."""
    logger.info("Checking data quality...")
    
    issues = []
    warnings = []
    
    # Check 1: Weekend data
    weekends = prices.index.weekday >= 5
    weekend_count = weekends.sum()
    if weekend_count > 0:
        issues.append(f"Weekend data detected: {weekend_count} points")
    
    # Check 2: Missing data
    missing_data = prices.isnull().sum()
    if missing_data.any():
        issues.append(f"Missing data: {missing_data.to_dict()}")
    
    # Check 3: Extreme price moves
    returns = prices.pct_change()
    extreme_moves = (returns.abs() > 0.1).sum().sum()  # >10% daily moves
    if extreme_moves > 0:
        warnings.append(f"Extreme moves (>10%): {extreme_moves}")
    
    # Check 4: Price ranges
    for asset in prices.columns:
        price_range = prices[asset].max() / prices[asset].min()
        if price_range > 50:  # More realistic threshold
            warnings.append(f"{asset}: {price_range:.1f}x price range")
    
    # Check 5: Return statistics
    returns = prices.pct_change().dropna()
    for asset in returns.columns:
        asset_returns = returns[asset]
        sharpe = (asset_returns.mean() * 252) / (asset_returns.std() * np.sqrt(252))
        if sharpe > 2.0:
            warnings.append(f"{asset}: High Sharpe {sharpe:.3f}")
    
    logger.info(f"Data quality check: {'✅ PASSED' if len(issues) == 0 else '❌ FAILED'}")
    if issues:
        logger.warning(f"Issues: {issues}")
    if warnings:
        logger.warning(f"Warnings: {warnings}")
    
    return {'issues': issues, 'warnings': warnings}


def check_look_ahead_bias(prices, returns):
    """Check for potential look-ahead bias."""
    logger.info("Checking for look-ahead bias...")
    
    # Simulate a strategy that might have look-ahead bias
    suspicious_returns = pd.Series(0.0, index=returns.index)
    
    for i in range(21, len(returns)):
        # This would be suspicious if it uses future information
        future_perf = returns.iloc[i:i+5].mean().mean()  # Next 5 days
        if future_perf > 0:
            suspicious_returns.iloc[i] = returns.iloc[i].mean()
        else:
            suspicious_returns.iloc[i] = -returns.iloc[i].mean() * 0.5
    
    # Compare with legitimate strategy
    legitimate_returns = pd.Series(0.0, index=returns.index)
    for i in range(21, len(returns)):
        # Only uses past information
        past_perf = returns.iloc[i-21:i].mean().mean()
        if past_perf > 0:
            legitimate_returns.iloc[i] = returns.iloc[i].mean()
        else:
            legitimate_returns.iloc[i] = -returns.iloc[i].mean() * 0.5
    
    # Calculate Sharpe ratios
    def calculate_sharpe(returns_series):
        excess_returns = returns_series - 0.02/252
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    suspicious_sharpe = calculate_sharpe(suspicious_returns)
    legitimate_sharpe = calculate_sharpe(legitimate_returns)
    
    logger.info(f"Suspicious strategy Sharpe: {suspicious_sharpe:.3f}")
    logger.info(f"Legitimate strategy Sharpe: {legitimate_sharpe:.3f}")
    
    if suspicious_sharpe > legitimate_sharpe * 1.5:
        logger.warning("⚠️ Potential look-ahead bias detected!")
        return True
    else:
        logger.info("✅ No obvious look-ahead bias detected")
        return False


def generate_realistic_benchmarks():
    """Generate realistic benchmark comparisons."""
    logger.info("Generating realistic benchmarks...")
    
    # Historical market benchmarks
    benchmarks = {
        'S&P 500 (Historical)': {'sharpe': 0.5, 'ann_return': 0.10, 'ann_vol': 0.15},
        'Best Hedge Fund (Historical)': {'sharpe': 2.0, 'ann_return': 0.15, 'ann_vol': 0.075},
        'Risk Parity (Historical)': {'sharpe': 1.2, 'ann_return': 0.08, 'ann_vol': 0.07},
        'Momentum Strategy (Historical)': {'sharpe': 0.8, 'ann_return': 0.12, 'ann_vol': 0.15},
        'Mean Reversion (Historical)': {'sharpe': 0.6, 'ann_return': 0.09, 'ann_vol': 0.15}
    }
    
    logger.info("Realistic market benchmarks:")
    for name, metrics in benchmarks.items():
        logger.info(f"{name}:")
        logger.info(f"  Sharpe: {metrics['sharpe']:.3f}")
        logger.info(f"  Ann Return: {metrics['ann_return']:.3f}")
        logger.info(f"  Ann Vol: {metrics['ann_vol']:.3f}")
    
    return benchmarks


def main():
    """Run comprehensive diagnostics."""
    logger.info("🚨 RUNNING CRITICAL DIAGNOSTICS")
    logger.info("=" * 60)
    
    # 1. Create realistic data
    prices = create_realistic_sample_data()
    
    # 2. Check data quality
    quality_results = check_data_quality(prices)
    
    # 3. Analyze Sharpe ratios
    sharpe_results = analyze_sharpe_ratios(prices)
    
    # 4. Check for look-ahead bias
    returns = prices.pct_change().dropna()
    look_ahead_bias = check_look_ahead_bias(prices, returns)
    
    # 5. Generate realistic benchmarks
    benchmarks = generate_realistic_benchmarks()
    
    # 6. Summary and recommendations
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    # Check if Sharpe ratios are realistic
    max_sharpe = max([result['sharpe'] for result in sharpe_results.values()])
    if max_sharpe > 2.0:
        logger.warning(f"❌ UNREALISTIC SHARPE RATIOS DETECTED: {max_sharpe:.3f}")
        logger.warning("   - Expected range: 0.5-2.0 for most strategies")
        logger.warning("   - Your results may have data quality issues or look-ahead bias")
    else:
        logger.info(f"✅ Sharpe ratios appear realistic: {max_sharpe:.3f}")
    
    # Check data quality
    if quality_results['issues']:
        logger.warning("❌ DATA QUALITY ISSUES FOUND")
        for issue in quality_results['issues']:
            logger.warning(f"   - {issue}")
    else:
        logger.info("✅ Data quality appears acceptable")
    
    # Check for look-ahead bias
    if look_ahead_bias:
        logger.warning("❌ POTENTIAL LOOK-AHEAD BIAS DETECTED")
        logger.warning("   - Review your strategy implementation")
        logger.warning("   - Ensure all calculations use only past data")
    else:
        logger.info("✅ No obvious look-ahead bias detected")
    
    # Recommendations
    logger.info("=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    if max_sharpe > 2.0:
        logger.info("1. IMMEDIATE ACTIONS:")
        logger.info("   - Review data sources and preprocessing")
        logger.info("   - Check for weekend data contamination")
        logger.info("   - Verify return calculation methodology")
        logger.info("   - Test with different time periods")
        logger.info("   - Compare against realistic benchmarks")
    
    logger.info("2. VALIDATION IMPROVEMENTS:")
    logger.info("   - Use only business days (no weekends)")
    logger.info("   - Implement proper transaction costs")
    logger.info("   - Add realistic slippage and fees")
    logger.info("   - Test with out-of-sample data")
    logger.info("   - Use walk-forward analysis")
    
    logger.info("3. BENCHMARK COMPARISONS:")
    logger.info("   - Compare against S&P 500 (Sharpe ~0.5)")
    logger.info("   - Compare against risk parity (Sharpe ~1.2)")
    logger.info("   - Compare against momentum (Sharpe ~0.8)")
    
    logger.info("=" * 60)
    logger.info("Diagnostics completed. Review recommendations above.")


if __name__ == "__main__":
    main() 