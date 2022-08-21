from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from main import cryptocurrency_further, timeframe_further, start_date_further
import os
from os.path import exists
from main import api_url, api_key, secret_key
from alpaca_trade_api.rest import REST, TimeFrame
import datetime

# Alpaca API
rest_api = REST(api_key, secret_key, api_url)

crypto_bars = rest_api.get_crypto_bars(cryptocurrency_further, timeframe_further, start_date_further, exchanges='CBSE').df
crypto_bars = crypto_bars.drop('exchange', axis=1)

# Convert to .csv for future use
if exists('crypto_bars_future.csv'):
    os.remove('crypto_bars_future.csv')
crypto_bars.to_csv('crypto_bars_future.csv')