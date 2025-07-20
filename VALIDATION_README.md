# Comprehensive Code Validation Framework

This framework provides systematic validation methods to verify the integrity of quantitative finance results, especially for exceptional performance metrics like high Sharpe ratios.

## Overview

The validation framework implements 8 key validation categories:

1. **Data Integrity Validation** - Checks for data quality issues
2. **Manual Calculation Verification** - Step-by-step Sharpe ratio calculation
3. **Cross-Platform Validation** - Multiple calculation methods
4. **Logic and Implementation Audits** - Regime detection and position sizing
5. **Performance Attribution Analysis** - Return decomposition
6. **Benchmark Reality Checks** - Comparison against simple strategies
7. **Stress Testing** - Edge case testing
8. **Final Validation Checklist** - Comprehensive review

## Quick Start

### 1. Run Basic Validation

```bash
python run_validation.py
```

This will run comprehensive validation on sample data and generate a detailed report.

### 2. Validate Your Existing Results

```bash
python validate_results.py --load-existing --full-validation
```

This will load your existing results from the main analysis and run full validation.

### 3. Manual Sharpe Calculation Only

```bash
python validate_results.py --manual-only
```

This performs only the manual Sharpe ratio calculation with detailed verification.

## Validation Components

### 1. Data Integrity Validation

Checks for:
- Negative or zero prices
- Extreme daily price moves (>50%)
- Missing data in critical periods
- Unrealistic price ranges
- Weekend/holiday data
- Sufficient data points
- Return data consistency

### 2. Manual Sharpe Calculation

Performs step-by-step verification:
```python
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
sharpe = annualized_excess / annualized_vol
```

### 3. Cross-Platform Validation

Compares results using:
- Manual calculation
- NumPy implementation
- Pandas implementation
- RiskManager calculation
- QuantStats (if available)

### 4. Logic and Implementation Audits

Validates:
- Regime detection consistency
- Position sizing logic
- Signal-position correlation
- Position size constraints
- Regime switch rates

### 5. Performance Attribution Analysis

Decomposes returns by:
- Asset selection effect
- Market timing effect
- Win rate analysis
- Return distribution analysis
- Top/bottom day contribution

### 6. Benchmark Reality Checks

Compares against:
- Equal weight strategy
- Simple combination strategy
- Random strategy (placebo test)
- Realistic performance thresholds

### 7. Stress Testing

Tests under:
- Random regime detection
- Inverted signals
- Different time periods
- High volatility periods
- Edge cases

### 8. Final Validation Checklist

Comprehensive review of:
- Data quality verification
- Return calculation cross-validation
- Performance metrics manual calculation
- Regime detection audit
- Position sizing logic verification
- Transaction costs inclusion
- Look-ahead bias confirmation
- Benchmark comparison
- Stress test completion
- Code review status

## Configuration

The validation framework uses `ValidationConfig` with customizable thresholds:

```python
config = ValidationConfig(
    correlation_threshold=0.999,  # Minimum correlation for method agreement
    sharpe_discrepancy_threshold=0.01,  # Maximum Sharpe ratio difference
    position_sum_tolerance=0.1,  # Allowable deviation from 100% invested
    max_position_threshold=0.35,  # Maximum position size
    regime_switch_rate_threshold=0.1,  # Maximum regime switch rate
    max_daily_move=0.5,  # Maximum daily price move (50%)
    max_price_range=100.0,  # Maximum price range ratio
    min_data_points=100,  # Minimum data points for analysis
    min_sharpe_for_validation=1.5,  # Minimum Sharpe to trigger extra validation
    max_drawdown_threshold=0.3,  # Maximum acceptable drawdown
)
```

## Usage Examples

### Basic Validation

```python
from src.validation import run_comprehensive_validation

# Run validation on your results
validation_results = run_comprehensive_validation(
    returns=your_returns,
    prices=your_prices,
    signals=your_signals,
    positions=your_positions,
    regimes=your_regimes,
    strategy_name="Your Strategy"
)

# Check results
if validation_results['validation_passed']:
    print("✅ Results validated successfully")
else:
    print("❌ Validation failed - review issues")
```

### Custom Validation

```python
from src.validation import ValidationFramework, ValidationConfig

# Create custom configuration
config = ValidationConfig(
    sharpe_discrepancy_threshold=0.005,  # Stricter threshold
    min_data_points=200,  # More data required
)

# Create framework
framework = ValidationFramework(config)

# Run validation
results = framework.validate_all(
    returns=returns,
    prices=prices,
    signals=signals,
    positions=positions,
    regimes=regimes,
    strategy_name="Custom Strategy"
)
```

## Output Files

The validation framework generates several output files:

1. **validation_report.txt** - Comprehensive validation report
2. **validation_results.json** - Detailed results in JSON format
3. **validation.log** - Detailed logging information

## Report Interpretation

### ✅ PASSED Validation
- All checks passed successfully
- Results appear valid and can be published
- No significant issues detected

### ❌ FAILED Validation
- Some checks failed
- Review issues before publishing
- Address specific problems identified

### ⚠️ WARNINGS
- Potential issues that don't fail validation
- Recommendations for improvement
- Areas requiring attention

## Common Issues and Solutions

### 1. High Sharpe Ratio Discrepancy
**Issue**: Different calculation methods produce different Sharpe ratios
**Solution**: Verify annualization method and risk-free rate usage

### 2. Data Quality Issues
**Issue**: Missing data, extreme values, or unrealistic prices
**Solution**: Clean data, handle missing values, verify price sources

### 3. Regime Detection Problems
**Issue**: High regime switch rate or unrealistic regime distribution
**Solution**: Adjust regime detection parameters, verify feature engineering

### 4. Position Sizing Issues
**Issue**: Over-investment, under-investment, or constraint violations
**Solution**: Review position sizing logic, adjust constraints

### 5. Benchmark Comparison Failures
**Issue**: Strategy doesn't beat simple benchmarks
**Solution**: Review strategy logic, check for implementation errors

## Integration with Main Analysis

To integrate validation with your existing analysis:

1. **Add validation to main.py**:
```python
from src.validation import run_comprehensive_validation

# After running your analysis
validation_results = run_comprehensive_validation(
    returns=backtest_results['equity_curve'].pct_change().dropna(),
    prices=data['prices'],
    signals=signals,
    positions=positions,
    regimes=regimes,
    strategy_name="Your Strategy"
)
```

2. **Include validation in reporting**:
```python
if validation_results['validation_passed']:
    print("✅ Results validated successfully")
    # Continue with reporting
else:
    print("❌ Validation failed - review before publishing")
    print(f"Issues: {validation_results['issues']}")
```

## Best Practices

1. **Run validation before publishing results**
2. **Use multiple calculation methods**
3. **Compare against simple benchmarks**
4. **Test under different market conditions**
5. **Document all validation steps**
6. **Review warnings and address issues**
7. **Include validation results in reports**

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Missing Data
Ensure your data files are in the correct location and format.

### Calculation Discrepancies
Check that all methods use the same:
- Risk-free rate
- Annualization method
- Return calculation method

### Performance Issues
For large datasets, consider:
- Sampling for validation
- Parallel processing
- Caching intermediate results

## Support

For issues or questions about the validation framework:

1. Check the validation logs
2. Review the validation report
3. Examine the JSON results file
4. Verify data quality and format

The validation framework is designed to ensure the integrity of your quantitative finance results and provide confidence in your reported performance metrics. 