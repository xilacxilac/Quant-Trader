import requests
import json
from alpaca.data.timeframe import TimeFrame

# API and Keys
api_url = "https://paper-api.alpaca.markets"
api_key = "PKWG2GMBP2YW2XSFEZWO"
secret_key = "4SAlx3qpdmZuGZEleTVZx6xpqRqU3wBuYZ6d42Bq"

# Cryptocurrency Next Day Predictor
cryptocurrency_next = "ETH/USD"
start_date_next = "2020-08-17"

# Cryptocurrency Future Predictor
cryptocurrency_further = "BTCUSD"
timeframe_further = TimeFrame.Hour
start_date_further = "2019-08-22"

# Cryptocurrency Latest Quote
cryptocurrency_quote = "ETH/USD"

ACCOUNT_URL = "{}/v2/account".format(api_url)
ORDERS_URL = "{}/v2/orders".format(api_url)
HEADERS = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': secret_key}

def get_account():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    return json.loads(r.content)

def create_order(symbol, qty, side, type, time_in_force):
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }

    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    return json.loads(r.content)