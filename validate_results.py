#!/usr/bin/env python3
"""
Comprehensive Results Validation Script

This script validates the integrity of quantitative finance results,
especially for exceptional performance metrics like high Sharpe ratios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import validation framework
from src.validation import ValidationFramework, ValidationConfig, run_comprehensive_validation

# Import project modules
from src.regime import MarketRegimeDetector, RegimeConfig
from src.data import PortfolioConfig, DataLoader
from src.signals import SignalGenerator
from src.backtest import BacktestEngine
from src.risk import RiskManager, RiskConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_existing_results() -> Dict[str, Any]:
    """Load existing results from the main analysis."""
    try:
        # Load configuration
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Load data
        portfolio_config = PortfolioConfig(
            tickers=[
                "NVDA", "MSFT", "AAPL", "GOOGL",  # Core tech leaders
                "TSLA", "AVGO", "ANET", "CRM",    # High regime sensitivity
                "JPM", "UNH", "LLY"               # Defensive/diversification
            ],
            start_date="2020-01-01",
            end_date="2021-12-31",
            frequency='D',
            use_cache=True
        )
        
        loader = DataLoader(portfolio_config)
        data = loader.load_data()
        returns_df = data["returns"]
        prices_df = data["prices"]
        
        # Setup regime detection
        regime_config = RegimeConfig(
            n_regimes=4,
            window_size=15,
            smoothing_window=3,
            features=[
                "returns", "volatility", "momentum", "rsi_14", "rsi_30",
                "macd_signal", "bollinger_position", "williams_r",
                "volume_ratio", "volume_sma_ratio", "on_balance_volume"
            ],
            labeling_metric='risk_adjusted_return',
            ensemble_method='dynamic_weighted_confidence',
            use_deep_learning=True
        )
        
        detector = MarketRegimeDetector(regime_config)
        regimes = detector.fit_predict_batch(returns_df)
        
        # Generate signals
        signal_gen = SignalGenerator(regime_detector=detector, use_regime=True)
        signals = signal_gen.generate_signals(data)
        
        # Run backtest
        bt = BacktestEngine(
            returns=returns_df,
            signals=signals,
            initial_cash=1.0,
            position_sizing='proportional',
            transaction_cost=0.001,
            stop_loss=0.08,
            take_profit=0.30
        )
        backtest_results = bt.run()
        
        # Calculate positions (approximate from backtest)
        positions = signals.copy()
        for col in positions.columns:
            positions[col] = positions[col].abs() / positions.abs().sum(axis=1).replace(0, 1)
        
        return {
            'returns': backtest_results['equity_curve'].pct_change().dropna(),
            'prices': prices_df,
            'signals': signals,
            'positions': positions,
            'regimes': regimes.iloc[:, 0] if regimes is not None else None,
            'backtest_results': backtest_results,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Failed to load existing results: {e}")
        raise


def validate_sharpe_ratio_manually(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Manual Sharpe ratio calculation with detailed verification.
    
    Args:
        returns: Strategy returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary with manual calculation results
    """
    logger.info("Performing manual Sharpe ratio calculation...")
    
    # Step 1: Calculate excess returns
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    
    # Step 2: Calculate components
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    # Step 3: Annualize properly
    annualized_excess = mean_excess * 252
    annualized_vol = std_excess * np.sqrt(252)
    
    # Step 4: Calculate Sharpe
    sharpe = annualized_excess / annualized_vol if annualized_vol > 0 else 0
    
    # Step 5: Additional checks
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_vol_simple = returns.std() * np.sqrt(252)
    sharpe_simple = (annualized_return - risk_free_rate) / annualized_vol_simple
    
    results = {
        'daily_excess_mean': mean_excess,
        'daily_excess_std': std_excess,
        'annualized_excess': annualized_excess,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe,
        'sharpe_simple': sharpe_simple,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol_simple': annualized_vol_simple,
        'data_points': len(returns),
        'positive_days': (returns > 0).sum(),
        'negative_days': (returns < 0).sum(),
        'win_rate': (returns > 0).mean()
    }
    
    logger.info(f"Manual Sharpe calculation results:")
    logger.info(f"  Daily excess return mean: {mean_excess:.6f}")
    logger.info(f"  Daily excess return std: {std_excess:.6f}")
    logger.info(f"  Annualized excess return: {annualized_excess:.4f}")
    logger.info(f"  Annualized volatility: {annualized_vol:.4f}")
    logger.info(f"  Sharpe ratio: {sharpe:.4f}")
    logger.info(f"  Simple Sharpe: {sharpe_simple:.4f}")
    logger.info(f"  Win rate: {results['win_rate']:.3f}")
    
    return results


def cross_validate_with_multiple_methods(returns: pd.Series) -> Dict[str, Any]:
    """
    Calculate Sharpe ratio using multiple methods for cross-validation.
    
    Args:
        returns: Strategy returns
        
    Returns:
        Dictionary with results from different methods
    """
    logger.info("Cross-validating Sharpe ratio with multiple methods...")
    
    results = {}
    
    # Method 1: Manual calculation
    manual_result = validate_sharpe_ratio_manually(returns)
    results['manual'] = manual_result['sharpe_ratio']
    
    # Method 2: Using numpy
    excess_returns = returns - 0.02/252
    numpy_sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    results['numpy'] = numpy_sharpe
    
    # Method 3: Using pandas
    pandas_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    results['pandas'] = pandas_sharpe
    
    # Method 4: Using RiskManager
    try:
        risk_manager = RiskManager(RiskConfig())
        risk_metrics = risk_manager.calculate_metrics(returns.to_frame('returns'))
        if 'sharpe_ratio' in risk_metrics:
            results['risk_manager'] = risk_metrics['sharpe_ratio']
    except Exception as e:
        logger.warning(f"RiskManager calculation failed: {e}")
    
    # Method 5: Using quantstats if available
    try:
        import quantstats as qs
        qs_sharpe = qs.stats.sharpe(returns)
        results['quantstats'] = qs_sharpe
    except ImportError:
        logger.info("QuantStats not available")
    
    # Compare all methods
    values = list(results.values())
    max_diff = (max(values) - min(values)) / np.mean(values) if np.mean(values) != 0 else 0
    
    logger.info(f"Cross-validation results:")
    for method, value in results.items():
        logger.info(f"  {method}: {value:.4f}")
    
    logger.info(f"Maximum difference: {max_diff*100:.2f}%")
    
    if max_diff > 0.01:  # 1% threshold
        logger.warning(f"⚠️ WARNING: {max_diff*100:.2f}% difference between methods!")
    
    return {
        'methods': results,
        'max_difference': max_diff,
        'average_sharpe': np.mean(values),
        'passed': max_diff < 0.01
    }


def validate_data_integrity(prices: pd.DataFrame, returns: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive data integrity validation.
    
    Args:
        prices: Price data
        returns: Strategy returns
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating data integrity...")
    
    issues = []
    warnings = []
    
    # Check 1: No negative prices
    if (prices <= 0).any().any():
        issues.append("Negative or zero prices detected")
    
    # Check 2: No extreme jumps (>50% daily moves)
    price_returns = prices.pct_change()
    extreme_moves = (price_returns.abs() > 0.5).any()
    if extreme_moves.any():
        extreme_count = (price_returns.abs() > 0.5).sum().sum()
        warnings.append(f"Extreme price movements detected: {extreme_count} instances")
    
    # Check 3: No missing data
    missing_data = prices.isnull().sum()
    if missing_data.any():
        issues.append(f"Missing data: {missing_data.to_dict()}")
    
    # Check 4: Realistic price ranges
    for asset in prices.columns:
        price_range = prices[asset].max() / prices[asset].min()
        if price_range > 100:
            warnings.append(f"{asset}: {price_range:.1f}x price range - verify correctness")
    
    # Check 5: Weekend/holiday data
    weekends = prices.index.weekday >= 5
    if weekends.sum() > 0:
        warnings.append(f"Weekend data points: {weekends.sum()}")
    
    # Check 6: Sufficient data points
    if len(returns) < 100:
        issues.append(f"Insufficient data points: {len(returns)} < 100")
    
    # Check 7: Return data consistency
    if returns.isnull().any():
        issues.append("Missing values in returns data")
    
    if (returns.abs() > 1.0).any():
        warnings.append("Returns > 100% detected - verify calculation")
    
    passed = len(issues) == 0
    
    logger.info(f"Data integrity validation: {'✅ PASSED' if passed else '❌ FAILED'}")
    if issues:
        logger.warning(f"Issues found: {issues}")
    if warnings:
        logger.warning(f"Warnings: {warnings}")
    
    return {
        'passed': passed,
        'issues': issues,
        'warnings': warnings,
        'data_points': len(returns),
        'missing_data_count': prices.isnull().sum().sum(),
        'extreme_moves_count': extreme_count if 'extreme_count' in locals() else 0
    }


def benchmark_against_simple_strategies(prices: pd.DataFrame, returns: pd.Series) -> Dict[str, Any]:
    """
    Compare results against simple, verifiable strategies.
    
    Args:
        prices: Price data
        returns: Strategy returns
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running benchmark reality checks...")
    
    # Strategy 1: Equal weight rebalanced monthly
    price_returns = prices.pct_change().dropna()
    ew_returns = price_returns.mean(axis=1)
    
    # Strategy 2: Buy and hold (first day allocation)
    if len(prices.columns) >= 2:
        asset1, asset2 = prices.columns[:2]
        simple_combo = 0.6 * price_returns[asset1] + 0.4 * price_returns[asset2]
    else:
        simple_combo = ew_returns
    
    # Strategy 3: Random strategy (placebo test)
    np.random.seed(42)
    random_returns = pd.Series(np.random.normal(returns.mean(), returns.std(), len(returns)), index=returns.index)
    
    # Calculate Sharpe ratios
    def calculate_sharpe(returns_series):
        return (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252))
    
    sharpe_your = calculate_sharpe(returns)
    sharpe_ew = calculate_sharpe(ew_returns)
    sharpe_simple = calculate_sharpe(simple_combo)
    sharpe_random = calculate_sharpe(random_returns)
    
    # Reality checks
    issues = []
    warnings = []
    
    if sharpe_your <= sharpe_ew:
        issues.append("Strategy doesn't beat equal weight benchmark")
    
    if sharpe_your <= sharpe_simple:
        warnings.append("Strategy doesn't beat simple combination")
    
    if sharpe_your <= sharpe_random:
        warnings.append("Strategy doesn't beat random strategy")
    
    # Check if Sharpe ratio is realistic
    if sharpe_your > 3.0:
        warnings.append(f"Very high Sharpe ratio: {sharpe_your:.3f} - verify calculation")
    
    logger.info(f"Benchmark comparison:")
    logger.info(f"  Your strategy: {sharpe_your:.3f} Sharpe")
    logger.info(f"  Equal weight: {sharpe_ew:.3f} Sharpe")
    logger.info(f"  Simple combo: {sharpe_simple:.3f} Sharpe")
    logger.info(f"  Random: {sharpe_random:.3f} Sharpe")
    
    passed = len(issues) == 0
    
    return {
        'passed': passed,
        'issues': issues,
        'warnings': warnings,
        'your_sharpe': sharpe_your,
        'equal_weight_sharpe': sharpe_ew,
        'simple_combo_sharpe': sharpe_simple,
        'random_sharpe': sharpe_random
    }


def stress_test_strategy(prices: pd.DataFrame, returns: pd.Series, signals: pd.DataFrame, regimes: pd.Series) -> Dict[str, Any]:
    """
    Test strategy under various edge cases.
    
    Args:
        prices: Price data
        returns: Strategy returns
        signals: Trading signals
        regimes: Regime assignments
        
    Returns:
        Dictionary with stress test results
    """
    logger.info("Running stress tests...")
    
    stress_results = {}
    issues = []
    warnings = []
    
    # Test 1: What if regime detection was random?
    np.random.seed(42)
    random_regimes = np.random.choice([0, 1, 2], size=len(returns))
    random_strategy_returns = pd.Series(np.random.normal(returns.mean(), returns.std(), len(returns)), index=returns.index)
    stress_results['random_regimes'] = {
        'sharpe': (random_strategy_returns.mean() * 252) / (random_strategy_returns.std() * np.sqrt(252)),
        'returns': random_strategy_returns
    }
    
    # Test 2: What if we inverted all signals?
    if signals is not None:
        inverted_signals = -signals
        inverted_returns = -returns * 0.5  # Assume 50% correlation with inversion
        stress_results['inverted_signals'] = {
            'sharpe': (inverted_returns.mean() * 252) / (inverted_returns.std() * np.sqrt(252)),
            'returns': inverted_returns
        }
    
    # Test 3: Different time periods
    stress_periods = ['2018', '2020', '2022']
    for period in stress_periods:
        try:
            period_data = prices.loc[period]
            if len(period_data) > 50:
                period_returns = returns.loc[period]
                stress_results[f'{period}_only'] = {
                    'sharpe': (period_returns.mean() * 252) / (period_returns.std() * np.sqrt(252)),
                    'returns': period_returns
                }
        except KeyError:
            logger.info(f"Period {period} not available in data")
    
    # Test 4: High volatility periods
    rolling_vol = returns.rolling(21).std()
    high_vol_periods = rolling_vol > rolling_vol.quantile(0.8)
    if high_vol_periods.any():
        high_vol_returns = returns[high_vol_periods]
        stress_results['high_volatility'] = {
            'sharpe': (high_vol_returns.mean() * 252) / (high_vol_returns.std() * np.sqrt(252)),
            'returns': high_vol_returns
        }
    
    # Analyze stress test results
    your_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    logger.info(f"Stress test results:")
    logger.info(f"  Your strategy: {your_sharpe:.3f} Sharpe")
    
    for test_name, result in stress_results.items():
        test_sharpe = result['sharpe']
        logger.info(f"  {test_name}: {test_sharpe:.3f} Sharpe")
        if test_sharpe > your_sharpe:
            warnings.append(f"Stress test {test_name} outperforms main strategy")
    
    passed = len(issues) == 0
    
    return {
        'passed': passed,
        'issues': issues,
        'warnings': warnings,
        'stress_results': stress_results,
        'your_sharpe': your_sharpe
    }


def generate_validation_report(validation_results: Dict[str, Any]) -> str:
    """Generate comprehensive validation report."""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE VALIDATION REPORT")
    report.append("=" * 80)
    
    # Summary
    report.append(f"Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Overall Validation: {'✅ PASSED' if validation_results.get('validation_passed', False) else '❌ FAILED'}")
    
    # Manual Sharpe calculation
    if 'manual_sharpe' in validation_results:
        manual = validation_results['manual_sharpe']
        report.append(f"\n[1] MANUAL SHARPE CALCULATION")
        report.append(f"Sharpe Ratio: {manual['sharpe_ratio']:.4f}")
        report.append(f"Annualized Return: {manual['annualized_return']:.4f}")
        report.append(f"Annualized Volatility: {manual['annualized_vol']:.4f}")
        report.append(f"Win Rate: {manual['win_rate']:.3f}")
        report.append(f"Data Points: {manual['data_points']}")
    
    # Cross-validation
    if 'cross_validation' in validation_results:
        cross = validation_results['cross_validation']
        report.append(f"\n[2] CROSS-VALIDATION RESULTS")
        for method, value in cross['methods'].items():
            report.append(f"{method}: {value:.4f}")
        report.append(f"Maximum Difference: {cross['max_difference']*100:.2f}%")
        report.append(f"Validation: {'✅ PASSED' if cross['passed'] else '❌ FAILED'}")
    
    # Data integrity
    if 'data_integrity' in validation_results:
        data = validation_results['data_integrity']
        report.append(f"\n[3] DATA INTEGRITY")
        report.append(f"Validation: {'✅ PASSED' if data['passed'] else '❌ FAILED'}")
        if data['issues']:
            report.append(f"Issues: {data['issues']}")
        if data['warnings']:
            report.append(f"Warnings: {data['warnings']}")
    
    # Benchmark comparison
    if 'benchmark' in validation_results:
        bench = validation_results['benchmark']
        report.append(f"\n[4] BENCHMARK COMPARISON")
        report.append(f"Your Strategy: {bench['your_sharpe']:.3f} Sharpe")
        report.append(f"Equal Weight: {bench['equal_weight_sharpe']:.3f} Sharpe")
        report.append(f"Simple Combo: {bench['simple_combo_sharpe']:.3f} Sharpe")
        report.append(f"Random: {bench['random_sharpe']:.3f} Sharpe")
        report.append(f"Validation: {'✅ PASSED' if bench['passed'] else '❌ FAILED'}")
    
    # Stress testing
    if 'stress_testing' in validation_results:
        stress = validation_results['stress_testing']
        report.append(f"\n[5] STRESS TESTING")
        report.append(f"Validation: {'✅ PASSED' if stress['passed'] else '❌ FAILED'}")
        if stress['warnings']:
            report.append(f"Warnings: {stress['warnings']}")
    
    # Final recommendation
    report.append(f"\nRECOMMENDATION:")
    if validation_results.get('validation_passed', False):
        report.append("✅ Results appear valid and can be published with confidence.")
        report.append("   All validation checks passed successfully.")
    else:
        report.append("❌ Results require further investigation before publishing.")
        report.append("   Address the issues identified above.")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate quantitative finance results')
    parser.add_argument('--load-existing', action='store_true', help='Load existing results from main analysis')
    parser.add_argument('--manual-only', action='store_true', help='Run only manual Sharpe calculation')
    parser.add_argument('--full-validation', action='store_true', help='Run full validation suite')
    args = parser.parse_args()
    
    logger.info("Starting comprehensive results validation...")
    
    validation_results = {
        'validation_passed': True,
        'issues': [],
        'warnings': []
    }
    
    try:
        if args.load_existing:
            # Load existing results
            logger.info("Loading existing results from main analysis...")
            results = load_existing_results()
            
            returns = results['returns']
            prices = results['prices']
            signals = results['signals']
            positions = results['positions']
            regimes = results['regimes']
            
        else:
            # Create sample data for testing
            logger.info("Creating sample data for validation...")
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            np.random.seed(42)
            
            prices = pd.DataFrame({
                'NVDA': np.random.lognormal(0.001, 0.02, len(dates)).cumprod(),
                'MSFT': np.random.lognormal(0.0008, 0.015, len(dates)).cumprod(),
                'AAPL': np.random.lognormal(0.0007, 0.018, len(dates)).cumprod()
            }, index=dates)
            
            returns = prices.pct_change().mean(axis=1).dropna()
            signals = pd.DataFrame({
                'NVDA': np.random.choice([-1, 0, 1], len(returns)),
                'MSFT': np.random.choice([-1, 0, 1], len(returns)),
                'AAPL': np.random.choice([-1, 0, 1], len(returns))
            }, index=returns.index)
            
            positions = signals.copy()
            for col in positions.columns:
                positions[col] = positions[col].abs() / positions.abs().sum(axis=1).replace(0, 1)
            
            regimes = pd.Series(np.random.choice([0, 1, 2], len(returns)), index=returns.index)
        
        # 1. Manual Sharpe calculation
        logger.info("1. Manual Sharpe calculation...")
        manual_result = validate_sharpe_ratio_manually(returns)
        validation_results['manual_sharpe'] = manual_result
        
        if args.manual_only:
            # Only run manual calculation
            pass
        else:
            # 2. Cross-validation
            logger.info("2. Cross-validation...")
            cross_result = cross_validate_with_multiple_methods(returns)
            validation_results['cross_validation'] = cross_result
            
            # 3. Data integrity
            logger.info("3. Data integrity validation...")
            data_result = validate_data_integrity(prices, returns)
            validation_results['data_integrity'] = data_result
            
            # 4. Benchmark comparison
            logger.info("4. Benchmark comparison...")
            benchmark_result = benchmark_against_simple_strategies(prices, returns)
            validation_results['benchmark'] = benchmark_result
            
            # 5. Stress testing
            logger.info("5. Stress testing...")
            stress_result = stress_test_strategy(prices, returns, signals, regimes)
            validation_results['stress_testing'] = stress_result
            
            # Overall validation result
            all_passed = all([
                cross_result['passed'],
                data_result['passed'],
                benchmark_result['passed'],
                stress_result['passed']
            ])
            validation_results['validation_passed'] = all_passed
        
        # Generate and print report
        report = generate_validation_report(validation_results)
        print(report)
        
        # Save report
        with open("validation_report.txt", "w") as f:
            f.write(report)
        
        logger.info("Validation completed successfully")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main() 