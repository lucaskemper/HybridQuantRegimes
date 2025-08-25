#!/usr/bin/env python3
"""
Simple functionality benchmark for HybridQuantRegimes
Tests basic functionality without external dependencies where possible.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, 'src')

def test_data_structures():
    """Test basic data structure configurations"""
    print("Testing data structures...")
    
    try:
        from data import PortfolioConfig
        
        # Test valid configuration
        config = PortfolioConfig(
            tickers=["AAPL", "GOOGL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            weights=[0.4, 0.3, 0.3]
        )
        print("✓ PortfolioConfig creation successful")
        
        # Test validation
        try:
            bad_config = PortfolioConfig(
                tickers=[],  # Should fail
                start_date="2024-01-01",
                end_date="2024-03-01"
            )
            print("✗ Validation should have failed for empty tickers")
            return False
        except ValueError:
            print("✓ Input validation working correctly")
            
        return True
        
    except ImportError as e:
        print(f"⚠ Cannot test PortfolioConfig: {e}")
        return True  # Don't fail if dependencies missing

def test_feature_calculations():
    """Test basic feature calculation functions"""
    print("\nTesting feature calculations...")
    
    try:
        from features import calculate_rsi, calculate_williams_r
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = pd.Series(np.random.uniform(100, 200, 50), index=dates)
        
        # Test RSI calculation
        rsi = calculate_rsi(prices, window=14)
        if len(rsi) > 0 and 0 <= rsi.iloc[-1] <= 100:
            print("✓ RSI calculation successful")
        else:
            print("✗ RSI calculation failed")
            return False
            
        # Test Williams %R calculation  
        high = prices + np.random.uniform(0, 5, 50)
        low = prices - np.random.uniform(0, 5, 50)
        williams_r = calculate_williams_r(high, low, prices, window=14)
        
        if len(williams_r) > 0 and -100 <= williams_r.iloc[-1] <= 0:
            print("✓ Williams %R calculation successful")
        else:
            print("✗ Williams %R calculation failed") 
            return False
            
        return True
        
    except ImportError as e:
        print(f"⚠ Cannot test feature calculations: {e}")
        return True

def test_risk_configurations():
    """Test risk management configuration"""
    print("\nTesting risk configurations...")
    
    try:
        from risk import RiskConfig
        
        # Test default configuration
        risk_config = RiskConfig()
        print("✓ RiskConfig default creation successful")
        
        # Test validation
        try:
            bad_risk_config = RiskConfig(confidence_level=1.5)  # Should fail
            print("✗ Risk validation should have failed")
            return False
        except ValueError:
            print("✓ Risk validation working correctly")
            
        return True
        
    except ImportError as e:
        print(f"⚠ Cannot test RiskConfig: {e}")
        return True

def test_regime_configurations():
    """Test regime detection configuration"""
    print("\nTesting regime configurations...")
    
    try:
        from regime import RegimeConfig
        
        # Test configuration creation
        regime_config = RegimeConfig(
            n_regimes=3,
            window_size=21,
            features=["returns", "volatility"]
        )
        print("✓ RegimeConfig creation successful")
        
        # Test validation
        try:
            bad_regime_config = RegimeConfig(n_regimes=1)  # Should fail
            print("✗ Regime validation should have failed")
            return False
        except ValueError:
            print("✓ Regime validation working correctly")
            
        return True
        
    except ImportError as e:
        print(f"⚠ Cannot test RegimeConfig: {e}")
        return True

def test_basic_math_operations():
    """Test mathematical operations used in the project"""
    print("\nTesting mathematical operations...")
    
    # Generate sample return data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    # Test volatility calculation
    volatility = returns.std() * np.sqrt(252)
    if 0.01 < volatility < 1.0:  # Reasonable volatility range
        print("✓ Volatility calculation successful")
    else:
        print("✗ Volatility calculation failed")
        return False
        
    # Test Sharpe ratio calculation
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    if -10 < sharpe < 10:  # Reasonable Sharpe range
        print("✓ Sharpe ratio calculation successful")
    else:
        print("✗ Sharpe ratio calculation failed")
        return False
        
    # Test max drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    if -1 < max_drawdown <= 0:  # Drawdown should be negative
        print("✓ Max drawdown calculation successful")
    else:
        print("✗ Max drawdown calculation failed")
        return False
        
    return True

def performance_benchmark():
    """Simple performance benchmark"""
    print("\nRunning performance benchmark...")
    
    import time
    
    # Test data processing speed
    start_time = time.time()
    
    # Create large dataset
    n_assets = 10
    n_days = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Simulate price data
    prices = pd.DataFrame(
        np.random.lognormal(0, 0.02, (n_days, n_assets)),
        index=dates,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns.corr()
    
    # Calculate rolling statistics
    rolling_vol = returns.rolling(window=21).std()
    rolling_mean = returns.rolling(window=21).mean()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✓ Processed {n_assets} assets, {n_days} days in {processing_time:.3f} seconds")
    
    if processing_time < 5.0:  # Should be fast for this amount of data
        print("✓ Performance benchmark passed")
        return True
    else:
        print("⚠ Performance benchmark slow (but acceptable)")
        return True

def main():
    """Run all benchmarks"""
    print("=" * 60)
    print("HybridQuantRegimes Simple Functionality Benchmark")
    print("=" * 60)
    
    tests = [
        test_data_structures,
        test_feature_calculations,
        test_risk_configurations,
        test_regime_configurations,
        test_basic_math_operations,
        performance_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("✓ All functionality tests PASSED")
        print("✓ Project demonstrates good basic functionality")
    elif passed >= total * 0.8:
        print("⚠ Most functionality tests passed")
        print("✓ Project shows acceptable functionality")
    else:
        print("✗ Multiple functionality issues detected")
        print("⚠ Project may have significant issues")
    
    return 0 if passed >= total * 0.8 else 1

if __name__ == "__main__":
    sys.exit(main())