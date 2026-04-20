# <span style="color:magenta">Backtesting-SP500</span>

This project implements a robust Momentum-based stock-picking strategy applied to the S&P 500 constituents. The engine handles memory optimization for large financial datasets, performs advanced data cleaning on noisy historical prices, and executes a modular backtesting pipeline to compare strategy performance against the market benchmark.

## <span style="color:pink">Technical Architecture</span>
#### <span style="color:yellow">1. Memory Optimization (memory_reducer.py)</span>

A core utility function reduces the memory footprint of stock_prices.csv from several hundred megabytes to under 8MB. This is achieved by:Iterating through numeric columns.Detecting the smallest possible NumPy datatype (e.g., np.float32, np.int32) that preserves precision.

#### <span style="color:yellow">2. Data Preprocessing (preprocessing.py)</span>
The pipeline transforms raw daily data into clean monthly intervals:Resampling: Monthly month-end (ME) frequency.Outlier Removal: Filtering prices outside the $\$0.1$ to $\$10,000$ range.Volatility Masking: Returns exceeding $100\%$ or dropping below $-50\%$ are treated as outliers and removed, except during the 2008-2009 Financial Crisis where high volatility is expected.Imputation: Missing data points are filled using forward-fill (ffill) logic grouped by ticker.

#### <span style="color:yellow">3. Signal Generation (create_signal.py)</span>
The strategy identifies the Top 20 stocks each month based on their 12-month rolling average return.Metric: average_return_1y.Ranking: Stocks are ranked monthly; the top 20 receive an active True signal.

#### <span style="color:yellow">4. Backtesting Framework (backtester.py)</span>
The framework computes the strategy’s performance by multiplying the signal by monthly_future_return.PnL Calculation: Vectorized multiplication for efficiency.Benchmark: Comparison against a passive $\$20$/month S&P 500 investment.Visualization: Cumulative return plots are generated and saved to the results/ directory.


## <span style="color:pink">Project Structure</span>
```
project/
│   README.md
│   requirements.txt
│
└───data/
│       sp500.csv             # Index benchmark data
│       stock_prices.csv       # Raw price data for constituents
│
└───notebook/
│       analysis.ipynb        # EDA, outlier detection, and visualization
│
└───scripts/
│       memory_reducer.py     # Optimization logic
│       preprocessing.py      # Cleaning and return calculations
│       create_signal.py      # Momentum ranking logic
│       backtester.py         # PnL and Plotting
│       main.py               # End-to-end pipeline execution
│
└───results/
    │   plots/                # PNG performance charts
    │   results.txt           # Final return metrics
    │   outliers.txt          # Documented historical price anomalies
```

## <span style="color:pink">How to Run</span>
```
python3 main.py
```

## <span style="color:pink">Collaborators</span>

[Bilal Daanouni](https://github.com/bdaanoun)           
[Hasnae Lamrani](https://github.com/Hasnaaaae)