from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest
from main import cryptocurrency_quote

# Crypto clients with Alpaca API
crypto_client = CryptoHistoricalDataClient()
crypto_latest_request_params = CryptoLatestQuoteRequest(symbol_or_symbols=cryptocurrency_quote)

# Retrieve latest quote from Alpaca API
crypto_latest_quote = crypto_client.get_crypto_latest_quote(crypto_latest_request_params)
print(crypto_latest_quote[cryptocurrency_quote].ask_price)