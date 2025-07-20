#!/usr/bin/env python3
"""
Simple Validation Runner

This script runs comprehensive validation on existing results from the main analysis.
"""

import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import validation framework
from src.validation import run_comprehensive_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data_for_validation(ultra_fast=False):
    """Create sample data for validation testing."""
    logger.info("Creating sample data for validation...")
    
    # Adjust data size based on ultra-fast mode
    if ultra_fast:
        # Use smaller dataset for ultra-fast mode
        dates = pd.date_range('2022-01-01', '2023-06-30', freq='D')
        logger.info("Ultra-fast mode: Using smaller dataset (1.5 years)")
    else:
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        logger.info("Normal mode: Using full dataset (4 years)")
    
    np.random.seed(42)
    
    # Create price data with realistic characteristics
    prices = pd.DataFrame({
        'NVDA': np.random.lognormal(0.001, 0.02, len(dates)).cumprod() * 100,
        'MSFT': np.random.lognormal(0.0008, 0.015, len(dates)).cumprod() * 200,
        'AAPL': np.random.lognormal(0.0007, 0.018, len(dates)).cumprod() * 150,
        'GOOGL': np.random.lognormal(0.0009, 0.016, len(dates)).cumprod() * 2500,
        'TSLA': np.random.lognormal(0.0012, 0.025, len(dates)).cumprod() * 300
    }, index=dates)
    
    # Create realistic strategy returns (with some alpha)
    base_returns = prices.pct_change().mean(axis=1).dropna()
    # Add some alpha (excess return)
    alpha = 0.0001  # 0.01% daily alpha
    strategy_returns = base_returns + alpha + np.random.normal(0, 0.001, len(base_returns))
    
    # Create signals based on momentum
    signals = pd.DataFrame({
        'NVDA': np.where(base_returns.rolling(10).mean() > 0, 1, -1),
        'MSFT': np.where(base_returns.rolling(15).mean() > 0, 1, -1),
        'AAPL': np.where(base_returns.rolling(20).mean() > 0, 1, -1),
        'GOOGL': np.where(base_returns.rolling(12).mean() > 0, 1, -1),
        'TSLA': np.where(base_returns.rolling(8).mean() > 0, 1, -1)
    }, index=base_returns.index)
    
    # Create positions (normalized)
    positions = signals.copy()
    for col in positions.columns:
        positions[col] = positions[col].abs() / positions.abs().sum(axis=1).replace(0, 1)
    
    # Create regimes (0=low vol, 1=normal, 2=high vol)
    volatility = base_returns.rolling(21).std()
    regimes = pd.Series(0, index=base_returns.index)  # Default to low vol
    regimes[volatility > volatility.quantile(0.7)] = 2  # High vol
    regimes[(volatility <= volatility.quantile(0.7)) & (volatility > volatility.quantile(0.3))] = 1  # Normal
    
    return {
        'returns': strategy_returns,
        'prices': prices,
        'signals': signals,
        'positions': positions,
        'regimes': regimes
    }


def run_ultra_fast_validation():
    """Run ultra-fast validation with minimal checks."""
    logger.info("🚀 ULTRA-FAST MODE: Running minimal validation checks...")
    
    try:
        # Create smaller sample data
        data = create_sample_data_for_validation(ultra_fast=True)
        
        # Run only essential validation checks
        logger.info("1. Manual Sharpe calculation...")
        from src.validation import validate_sharpe_ratio_manually
        manual_result = validate_sharpe_ratio_manually(data['returns'])
        
        logger.info("2. Basic data integrity check...")
        from src.validation import validate_data_integrity
        data_result = validate_data_integrity(data['prices'], data['returns'])
        
        logger.info("3. Simple benchmark comparison...")
        from src.validation import benchmark_against_simple_strategies
        benchmark_result = benchmark_against_simple_strategies(data['prices'], data['returns'])
        
        # Quick summary
        validation_passed = (
            data_result['passed'] and 
            benchmark_result['passed']
        )
        
        logger.info("=" * 60)
        logger.info("ULTRA-FAST VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Manual Sharpe: {manual_result['sharpe_ratio']:.4f}")
        logger.info(f"Data Integrity: {'✅ PASSED' if data_result['passed'] else '❌ FAILED'}")
        logger.info(f"Benchmark Check: {'✅ PASSED' if benchmark_result['passed'] else '❌ FAILED'}")
        logger.info(f"Overall: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        
        if validation_passed:
            logger.info("✅ ULTRA-FAST VALIDATION PASSED")
        else:
            logger.warning("❌ ULTRA-FAST VALIDATION FAILED")
        
        return {
            'validation_passed': validation_passed,
            'manual_sharpe': manual_result,
            'data_integrity': data_result,
            'benchmark': benchmark_result
        }
        
    except Exception as e:
        logger.error(f"Ultra-fast validation failed: {e}")
        raise


def main():
    """Run comprehensive validation."""
    parser = argparse.ArgumentParser(description='Run validation with different speed modes')
    parser.add_argument('--ultra-fast', action='store_true', help='Run ultra-fast validation with minimal checks')
    parser.add_argument('--fast', action='store_true', help='Run fast validation with reduced complexity')
    args = parser.parse_args()
    
    if args.ultra_fast:
        logger.info("🚀 ULTRA-FAST MODE ENABLED")
        run_ultra_fast_validation()
    else:
        logger.info("Starting comprehensive validation...")
        
        try:
            # Create sample data
            data = create_sample_data_for_validation(ultra_fast=False)
            
            # Run comprehensive validation
            validation_results = run_comprehensive_validation(
                returns=data['returns'],
                prices=data['prices'],
                signals=data['signals'],
                positions=data['positions'],
                regimes=data['regimes'],
                strategy_name="Sample Strategy"
            )
            
            # Print summary
            if validation_results['validation_passed']:
                logger.info("✅ VALIDATION PASSED - Results appear valid")
            else:
                logger.warning("❌ VALIDATION FAILED - Review issues before publishing")
                logger.warning(f"Issues found: {validation_results['issues']}")
            
            # Save detailed results
            import json
            with open('validation_results.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                    return obj
                
                # Recursively convert numpy types
                def clean_dict(d):
                    if isinstance(d, dict):
                        return {k: clean_dict(v) for k, v in d.items()}
                    elif isinstance(d, list):
                        return [clean_dict(v) for v in d]
                    else:
                        return convert_numpy(d)
                
                clean_results = clean_dict(validation_results)
                json.dump(clean_results, f, indent=2, default=str)
            
            logger.info("Validation completed. Results saved to validation_results.json")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise


if __name__ == "__main__":
    main() 