import pandas as pd
import matplotlib.pyplot as plt

def backtest(prices, sp500):
    # Calcul du PnL par action
    ## relation => Signal (True/False) * Rendement Futur
    prices['pnl'] = prices['signal'] * prices['monthly_future_return']

    # Rendement mensuel : Moyenne des 20 actions
    ### On divise par 20 car on a investi 1/20eme de notre capital dans chaque action
    strategy_perf = prices.groupby('Date')['pnl'].sum() / 20

    # C'est la somme de tous les gains/pertes mensuels
    total_pnl_value = strategy_perf.sum()
    print(total_pnl_value)

    # Calcul des rendements cumulés (Compound Returns)
    ## (1 + r).cumprod() => simuler la croissance de 1$
    cum_strategy = (1 + strategy_perf).cumprod()
    cum_sp500 = (1 + sp500['sp500_return']).cumprod()

    # pour la comparaison
    common_dates = cum_strategy.index.intersection(cum_sp500.index)
    cum_strategy = cum_strategy.loc[common_dates]
    cum_sp500 = cum_sp500.loc[common_dates]


    total_return_strat = (cum_strategy.iloc[-1] - 1) * 100
    total_return_sp500 = (cum_sp500.iloc[-1] - 1) * 100

    with open("results/results.txt", "w") as f:
        f.write(f"Strategy Total PnL (Sum): {total_pnl_value:.2f}%\n")
        f.write(f"Total Return Strategy: {total_return_strat:.2f}%\n")
        f.write(f"Total Return S&P 500: {total_return_sp500:.2f}%\n")

    # Graph
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy, label='Stock Picking 20 Strategy', color='blue')
    plt.plot(cum_sp500, label='S&P 500 Benchmark', color='red', linestyle='--')
    plt.title("Cumulative Performance: Strategy vs S&P 500")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Base 1.0)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/performance_comparison.png")
    plt.close()

    print("image saved in /results 😁")