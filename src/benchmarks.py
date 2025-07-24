import pandas as pd
import numpy as np
from typing import Dict
from src.backtest import BacktestEngine

class BenchmarkComparison:
    """Compare strategy against standard benchmarks"""
    
    def __init__(self, returns: pd.DataFrame, strategy_results: Dict):
        self.returns = returns
        self.strategy_results = strategy_results
        
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark comparisons"""
        
        benchmarks = {}
        
        # 1. Buy and Hold
        benchmarks['buy_hold'] = self._buy_and_hold_benchmark()
        
        # 2. Equal Weight
        benchmarks['equal_weight'] = self._equal_weight_benchmark()
        
        # 3. Simple Moving Average
        benchmarks['sma_strategy'] = self._sma_benchmark()
        
        # 4. Random Forest Regime (placeholder)
        benchmarks['random_forest'] = self._random_forest_benchmark()
        
        # 5. Volatility Targeting (placeholder)
        benchmarks['vol_target'] = self._volatility_targeting_benchmark()
        
        return benchmarks
    
    def _buy_and_hold_benchmark(self) -> Dict:
        """Simple buy-and-hold benchmark"""
        
        n_assets = len(self.returns.columns)
        weights = pd.DataFrame(1/n_assets, 
                              index=self.returns.index, 
                              columns=self.returns.columns)
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        equity_curve = (1 + portfolio_returns).cumprod()
        
        return {
            'equity_curve': equity_curve,
            'daily_returns': portfolio_returns,
            'metrics': BacktestEngine.compute_metrics(equity_curve)
        }
    
    def _equal_weight_benchmark(self) -> Dict:
        """Equal weight rebalanced daily benchmark"""
        n_assets = len(self.returns.columns)
        weights = pd.DataFrame(1/n_assets, 
                              index=self.returns.index, 
                              columns=self.returns.columns)
        portfolio_returns = (self.returns * weights).sum(axis=1)
        equity_curve = (1 + portfolio_returns).cumprod()
        return {
            'equity_curve': equity_curve,
            'daily_returns': portfolio_returns,
            'metrics': BacktestEngine.compute_metrics(equity_curve)
        }
    
    def _sma_benchmark(self, fast: int = 20, slow: int = 50) -> Dict:
        """Simple Moving Average crossover strategy"""
        signals = pd.DataFrame(0, index=self.returns.index, columns=self.returns.columns)
        for col in self.returns.columns:
            price = (1 + self.returns[col]).cumprod()
            sma_fast = price.rolling(fast).mean()
            sma_slow = price.rolling(slow).mean()
            signals[col] = np.where(sma_fast > sma_slow, 1, -1)
        engine = BacktestEngine(
            returns=self.returns,
            signals=signals,
            transaction_cost=0.001,
            position_sizing='proportional'
        )
        return engine.run()
    
    def _random_forest_benchmark(self) -> Dict:
        """Random Forest regime strategy placeholder"""
        # Placeholder: Implement actual random forest regime logic here
        return {'metrics': {}}
    
    def _volatility_targeting_benchmark(self) -> Dict:
        """Volatility targeting strategy placeholder"""
        # Placeholder: Implement actual volatility targeting logic here
        return {'metrics': {}}

def compare_with_benchmarks(strategy_results: Dict, returns: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table"""
    benchmark_comp = BenchmarkComparison(returns, strategy_results)
    benchmarks = benchmark_comp.run_all_benchmarks()
    comparison_data = []
    # Add strategy results
    comparison_data.append({
        'Strategy': 'HMM-LSTM Hybrid',
        **strategy_results['metrics']
    })
    # Add benchmark results
    for name, results in benchmarks.items():
        comparison_data.append({
            'Strategy': name.replace('_', ' ').title(),
            **results.get('metrics', {})
        })
    return pd.DataFrame(comparison_data) 