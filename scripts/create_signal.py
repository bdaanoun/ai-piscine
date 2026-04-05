import pandas as pd

def create_signal(prices):
    # Calcul de la moyenne mobile sur 12 mois (par Ticker)
    prices['average_return_1y'] = (
        prices.groupby('Ticker')['monthly_past_return']
        .transform(lambda x: x.rolling(window=12).mean())
    )

    # Creation du signal (Top 20 par mois)
    prices['signal'] = (
        prices.groupby('Date')['average_return_1y']
        .rank(ascending=False, method='first') <= 20
    )

    # Nettoyage
    prices = prices.dropna(subset=['average_return_1y'])

    return prices