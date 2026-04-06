import pandas as pd
import matplotlib.pyplot as plt
import os

def backtest(prices, sp500):
    # PnL per stock: Signal (True/False) * Future Return
    prices['pnl'] = prices['signal'] * prices['monthly_future_return']

    # Strategy return: sum of PnL / sum of signals (as per instructions)
    strategy_pnl  = prices.groupby('Date')['pnl'].sum()
    strategy_return = strategy_pnl / prices.groupby('Date')['signal'].sum()
    print(prices.groupby('Date')['signal'].sum())

    # SP500 benchmark: $20 invested each month
    sp500_pnl = 20 * sp500['sp500_return']

    # Cumulative PnL using cumsum (as per instructions)
    cum_strategy = strategy_pnl.cumsum()
    cum_sp500 = sp500_pnl.cumsum()

    # Align on common dates
    common_dates = cum_strategy.index.intersection(cum_sp500.index)
    cum_strategy = cum_strategy.loc[common_dates]
    cum_sp500 = cum_sp500.loc[common_dates]

    # Total returns
    total_return_strat  = strategy_return.sum()
    total_return_sp500  = (sp500_pnl.sum()) / (20 * len(sp500_pnl))

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/results.txt", "w") as f:
        f.write("Backtesting Performance Report\n")
        f.write("==============================\n")
        f.write(f"Strategy Total PnL:        ${cum_strategy.iloc[-1]:.2f}\n")
        f.write(f"S&P 500 Total PnL:         ${cum_sp500.iloc[-1]:.2f}\n")
        f.write(f"Strategy Total Return:     {total_return_strat*100:.2f}%\n")
        f.write(f"S&P 500 Total Return:      {total_return_sp500*100:.2f}%\n")

    # Plot
    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy, label='Stock Picking 20 Strategy', color='blue')
    plt.plot(cum_sp500,    label='S&P 500 Benchmark',         color='red', linestyle='--')
    plt.title("Cumulative Performance: Strategy vs S&P 500")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/performance_comparison.png")
    plt.close()

    print("Results saved in results/results.txt")
    print("Plot saved in results/plots/performance_comparison.png")