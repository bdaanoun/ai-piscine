import pandas as pd
import numpy as np

def memory_reducer(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    
    for col in df.columns:
        
        # 2 - skip non-numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        col_data = df[col].dropna()
        
        # 3 - can it be an integer?
        is_integer = (col_data % 1 == 0).all()
        
        # 4 - min and max
        col_min = col_data.min()
        col_max = col_data.max()
        
        # 5 - smallest fitting dtype
        if is_integer:
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min <= col_min and col_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            for dtype in [np.float32, np.float64]:
                if np.finfo(dtype).min <= col_min and col_max <= np.finfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    # print( "----",df[col].dtype)
                    break
    
    return df


df_sp500 = memory_reducer("../data/sp500.csv")
mb = df_sp500.memory_usage(deep=True).sum() / 1024 ** 2
print(f"SP500: {mb:.4f} MB")

df_prices = memory_reducer("../data/stock_prices.csv")
mb = df_prices.memory_usage(deep=True).sum() / 1024 ** 2
print(f"Prices: {mb:.4f} MB")
