import pandas as pd
import numpy as np
import yaml
from src.data import DataLoader, PortfolioConfig
from src.risk import RiskManager, RiskConfig


def load_config(path="config.yml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_portfolio_data():
    config = load_config()
    pconf = config["portfolio"]
    portfolio_config = PortfolioConfig(
        tickers=pconf["tickers"],
        start_date=pconf["start_date"],
        end_date=pconf["end_date"],
        use_cache=pconf.get("use_cache", True),
        include_macro=False,
    )
    loader = DataLoader(portfolio_config)
    data = loader.load_data()
    return data["prices"], data["returns"]

def calc_metrics(returns):
    risk_config = RiskConfig()
    risk_mgr = RiskManager(risk_config, risk_free_rate=0.0)
    metrics = risk_mgr.calculate_metrics(returns)
    # Format for table
    return {
        "Return": f"{metrics['mean_return']*252*100:.1f}%",
        "Volatility": f"{metrics['portfolio_volatility']*100:.1f}%",
        "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
        "Max Drawdown": f"{metrics['max_drawdown']*100:.1f}%",
    }

def equal_weight_long_only(returns):
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    port_ret = returns.dot(weights)
    return port_ret

def naive_momentum_20d(returns):
    # Each day, long top half by 20-day return
    lookback = 20
    port_rets = []
    for i in range(lookback, len(returns)):
        window = returns.iloc[i-lookback:i]
        past_perf = window.add(1).prod() - 1
        top = past_perf.sort_values(ascending=False).index[:len(past_perf)//2]
        w = pd.Series(0, index=returns.columns)
        w[top] = 1/len(top)
        port_rets.append((returns.iloc[i] * w).sum())
    port_rets = pd.Series(port_rets, index=returns.index[lookback:])
    return port_rets

def main():
    prices, returns = get_portfolio_data()
    results = {}
    # Equal-weight long-only
    eq_ret = equal_weight_long_only(returns)
    results["Equal-weight long-only"] = calc_metrics(eq_ret)
    # Naive momentum (20-day)
    mom_ret = naive_momentum_20d(returns)
    results["Naive momentum (20-day)"] = calc_metrics(mom_ret)
    # Print table
    df = pd.DataFrame(results).T
    print("\nStrategy Performance (2019-2024):")
    print(df)

if __name__ == "__main__":
    main()



