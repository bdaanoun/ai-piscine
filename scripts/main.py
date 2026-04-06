from memory_reducer import memory_reducer
from preprocessing import preprocessing
from create_signal import create_signal
from backtester import backtest

# 1 => Load
prices = memory_reducer("../data/stock_prices.csv")
sp500 = memory_reducer("../data/sp500.csv")

# 2 => Clean
prices, sp500 = preprocessing(prices, sp500)

# 3 => Signal
prices = create_signal(prices)

# 4 => Result
backtest(prices, sp500)