import pandas as pd
import numpy as np
import os

def preprocessing(prices, sp500):
    # Passage au format Long (Date & Ticker en index)
   
    prices = prices.set_index("Date")
    # print(prices.index.dtype)
    prices = prices.resample("ME").last()
    prices = prices.stack(future_stack=True).to_frame('Price')
    prices.index.names = ['Date', 'Ticker']
    prices = prices.sort_index(level=[ 'Date' , 'Ticker'])

    # Nettoyage des prix (Filtre $0.1 - $10,000)
    prices.loc[(prices['Price'] < 0.1) | (prices['Price'] > 10000), 'Price'] = np.nan

    # Calcul des rendements (par Ticker)
    ## Past return: (Prix actuel / Prix mois dernier) - 1
    ## Future return: (Prix mois prochain / Prix actuel) - 1
    group = prices.groupby('Ticker')['Price']
    prices['monthly_past_return'] = group.pct_change(fill_method=None)
    prices['monthly_future_return'] = group.shift(-1) / prices['Price'] - 1

    # Filtre des Outliers de rendements (sauf 2008-2009)
    years = prices.index.get_level_values('Date').year
    is_crisis = years.isin([2008, 2009])
    
    # Condition: rendement > 100% ou < -50%
    bad_return = (prices['monthly_past_return'] > 0.5) | (prices['monthly_past_return'] < -0.5)
    bad_future = (prices['monthly_future_return'] > 0.5) | (prices['monthly_future_return'] < -0.5)
   
    # On met en NaN si c'est un mauvais rendement ET que ce n'est pas la crise
    prices.loc[bad_return & ~is_crisis, 'monthly_past_return'] = np.nan
    prices.loc[bad_future & ~is_crisis, 'monthly_future_return'] = np.nan

    # Remplissage des vides (Forward Fill)
    ## On complete les trous avec la derniere valeur connue de l'entreprise
    prices['Price'] = prices.groupby('Ticker')['Price'].ffill()
    
    prices['monthly_past_return'] = prices.groupby('Ticker')['monthly_past_return'].ffill()
    prices['monthly_future_return'] = prices.groupby('Ticker')['monthly_future_return'].ffill()

    prices = prices.dropna()
    
    # Traitement
    sp500 = sp500.set_index("Date").resample("ME").last()
    sp500['sp500_return'] = sp500['Adjusted Close'].pct_change()
    sp500 = sp500.dropna()

    return prices, sp500