import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from main import api_key, secret_key

stock_client = StockHistoricalDataClient(api_key, secret_key)
multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=["TQQQ"])
latest_multisymbol_quotes = stock_client.get_stock_latest_quote(multisymbol_request_params)
latest_ask_price = latest_multisymbol_quotes["TQQQ"].ask_price
print(latest_multisymbol_quotes)
print(latest_ask_price)