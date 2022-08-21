import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import plotly.graph_objects as go
from main import cryptocurrency_next, start_date_next
import os

# Crypto clients with Alpaca API
crypto_client = CryptoHistoricalDataClient()

# Crypto request parameters from Alpaca API
crypto_request_params = CryptoBarsRequest(
                        symbol_or_symbols=cryptocurrency_next,
                        timeframe=TimeFrame.Day,
                        start=start_date_next)

# Remove crypto type in DataFrame
crypto_bars = crypto_client.get_crypto_bars(crypto_request_params).df.droplevel(0, axis=0)

# Create candlestick chart using plotly
fig = go.Figure(data=[go.Candlestick(
    x=crypto_bars.index._data,
    open=crypto_bars['open'],
    high=crypto_bars['high'],
    low=crypto_bars['low'],
    close=crypto_bars['close'])])

# Calculating Simple Moving Average (SMA) and Exponential Moving Average (EMA) using pandas rolling mean
sma5 = crypto_bars['close'].rolling(5).mean().dropna()
sma10 = crypto_bars['close'].rolling(10).mean().dropna()
sma20 = crypto_bars['close'].rolling(20).mean().dropna()
sma50 = crypto_bars['close'].rolling(50).mean().dropna()
sma100 = crypto_bars['close'].rolling(100).mean().dropna()
sma200 = crypto_bars['close'].rolling(200).mean().dropna()
ema13 = crypto_bars['close'].ewm(span=13).mean().dropna()
ema485 = crypto_bars['close'].ewm(span=48.5).mean().dropna()

# Adding SMA and EMA lines on the candlestick chart
fig.add_trace(go.Scatter(x=sma5.index, y=sma5, marker_color='darkred', name='5 Day SMA'))
fig.add_trace(go.Scatter(x=sma10.index, y=sma10, marker_color='red', name='10 Day SMA'))
fig.add_trace(go.Scatter(x=sma20.index, y=sma20, marker_color='pink', name='20 Day SMA'))
fig.add_trace(go.Scatter(x=sma50.index, y=sma50, marker_color='cyan', name='50 Day SMA'))
fig.add_trace(go.Scatter(x=sma100.index, y=sma100, marker_color='blue', name='100 Day SMA'))
fig.add_trace(go.Scatter(x=sma200.index, y=sma200, marker_color='darkblue', name='200 Day SMA'))
fig.add_trace(go.Scatter(x=ema13.index, y=ema13, marker_color='magenta', name='13 Day EMA'))
fig.add_trace(go.Scatter(x=ema485.index, y=ema485, marker_color='purple', name='48.5 Day EMA'))

# Bollinger Band
std = crypto_bars['close'].rolling(20).std(ddof=0)
upper20 = (sma20 + (std * 2)).dropna()
lower20 = (sma20 - (std * 2)).dropna()

# Upper Bound
fig.add_trace(go.Scatter(x=upper20.index, y=upper20, line_color='lightgray',
                         line={'dash': 'dash'}, name='Upper Band 20', opacity=0.5))

# Lower Bound
fig.add_trace(go.Scatter(x=lower20.index, y=lower20, line_color='lightgray',
                         line={'dash': 'dash'}, fill='tonexty', name='Lower Band 20', opacity=0.5))

# RSI
close_delta = crypto_bars['close'].diff()
up = close_delta.clip(lower=0)
down = -1 * close_delta.clip(upper=0)
ma_up = up.ewm(com=13, adjust=True, min_periods=14).mean()
ma_down = down.ewm(com=13, adjust=True, min_periods=14).mean()
rsi = 100 - (100 / (1 + (ma_up / ma_down)))

# MACD
exp1 = crypto_bars['close'].ewm(span=12, adjust=False).mean()
exp2 = crypto_bars['close'].ewm(span=16, adjust=False).mean()
macd = exp1 - exp2
signal_line = macd.ewm(span=9, adjust=False).mean()
macd_delta = macd - signal_line
fig.add_trace(go.Scatter(x=macd.index, y=macd, marker_color='green', name='MACD'))
fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, marker_color='red', name='Signal Line'))

# Stochastic Oscillator Indicator
high14 = crypto_bars['high'].rolling(14).max()
low14 = crypto_bars['low'].rolling(14).min()
percentK = (crypto_bars['close'] - low14) * 100 / (high14 - low14)
percentD = percentK.rolling(3).mean()
stochastic_delta = percentK - percentD
fig.add_trace(go.Scatter(x=percentK.index, y=percentK, marker_color='cyan', name='%K'))
fig.add_trace(go.Scatter(x=percentD.index, y=percentD, marker_color='yellow', name='%D'))

# Average Directional Movement
plus_dm = crypto_bars['high'].diff()
minus_dm = crypto_bars['low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm > 0] = 0

tr1 = crypto_bars['high'] - crypto_bars['low']
tr2 = abs(crypto_bars['high'] - crypto_bars['close'].shift(1))
tr3 = abs(crypto_bars['low'] - crypto_bars['close'].shift(1))
frames = [tr1, tr2, tr3]
tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
atr = tr.rolling(14).mean()

plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / atr)
minus_di = abs(100 * (minus_dm.ewm(alpha=1 / 14).mean() / atr))
dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
adx = ((dx.shift(1) * (13)) + dx) / 14
adx_smooth = adx.ewm(alpha=1 / 14).mean()
fig.add_trace(go.Scatter(x=plus_di.index, y=plus_di, marker_color='lime', name='+ DI 14'))
fig.add_trace(go.Scatter(x=minus_di.index, y=minus_di, marker_color='lightpink', name='- DI 14'))
fig.add_trace(go.Scatter(x=adx_smooth.index, y=adx_smooth, marker_color='lightblue', name='ADX 14'))

# Kijun-sen and Tenkan-sen (NOT GRAPHED)
max26 = crypto_bars['high'].rolling(26).max()
min26 = crypto_bars['low'].rolling(26).min()
max9 = crypto_bars['high'].rolling(9).max()
min9 = crypto_bars['low'].rolling(9).min()
kijun_sen = (max26 + min26) / 2
tenkan_sen = (max9 + min9) / 2

# Chikou-span (NOT GRAPHED)
chikou_span = crypto_bars['close'].shift(-26)

# Ichimoku Cloud (NOT GRAPHED)
max52 = crypto_bars['high'].rolling(52).max()
min52 = crypto_bars['low'].rolling(52).min()
senkou_span_a = (tenkan_sen + kijun_sen) / 2
senkou_span_b = (max52 + min52) / 2

# Parabolic SAR - to do (needs uptrend/downtrend)

# Fibonacci Retracement - to do (needs uptrend/downtrend)

# On-Balance Volume (NOT GRAPHED)
obv = (np.sign(crypto_bars['close'].diff()) * crypto_bars['volume']).fillna(0).cumsum()

# Adding a title and axes labels
fig.update_layout(
    title="Technical Analysis Chart of Cryptocurrency over Time",
    xaxis_title="Date (days)",
    yaxis_title="Price ($USD)",
)

# Displaying candlestick chart
fig.show()

# Adding all algorithmic/technical analysis values to the DataFrame
crypto_bars['sma5'] = sma5
crypto_bars['sma10'] = sma10
crypto_bars['sma20'] = sma20
crypto_bars['sma50'] = sma50
crypto_bars['sma100'] = sma100
crypto_bars['sma200'] = sma200
crypto_bars['ema13'] = ema13
crypto_bars['ema485'] = ema485
crypto_bars['std'] = std
crypto_bars['lower20'] = lower20
crypto_bars['upper20'] = upper20
crypto_bars['rsi'] = rsi
crypto_bars['macd_delta'] = macd_delta
crypto_bars['stochastic_delta'] = stochastic_delta
crypto_bars['plus_di'] = plus_di
crypto_bars['minus_di'] = minus_di
crypto_bars['adx_smooth'] = adx_smooth
crypto_bars['kijun_sen'] = kijun_sen
crypto_bars['tenkan_sen'] = tenkan_sen
crypto_bars['senkou_span_a'] = senkou_span_a
crypto_bars['senkou_span_b'] = senkou_span_b
crypto_bars['obv'] = obv

# Convert DateTime to Date
crypto_bars.index = crypto_bars.index.date

# Shifting values up
crypto_bars['volume'] = crypto_bars['volume'].shift(-1)
crypto_bars['trade_count'] = crypto_bars['trade_count'].shift(-1)
crypto_bars['vwap'] = crypto_bars['vwap'].shift(-1)
crypto_bars['sma5'] = crypto_bars['sma5'].shift(-1)
crypto_bars['sma10'] = crypto_bars['sma10'].shift(-1)
crypto_bars['sma20'] = crypto_bars['sma20'].shift(-1)
crypto_bars['sma50'] = crypto_bars['sma50'].shift(-1)
crypto_bars['sma100'] = crypto_bars['sma100'].shift(-1)
crypto_bars['sma200'] = crypto_bars['sma200'].shift(-1)
crypto_bars['ema13'] = crypto_bars['ema13'].shift(-1)
crypto_bars['ema485'] = crypto_bars['ema485'].shift(-1)
crypto_bars['std'] = crypto_bars['std'].shift(-1)
crypto_bars['lower20'] = crypto_bars['lower20'].shift(-1)
crypto_bars['upper20'] = crypto_bars['upper20'].shift(-1)
crypto_bars['rsi'] = crypto_bars['rsi'].shift(-1)
crypto_bars['macd_delta'] = crypto_bars['macd_delta'].shift(-1)
crypto_bars['stochastic_delta'] = crypto_bars['stochastic_delta'].shift(-1)
crypto_bars['plus_di'] = crypto_bars['plus_di'].shift(-1)
crypto_bars['minus_di'] = crypto_bars['minus_di'].shift(-1)
crypto_bars['adx_smooth'] = crypto_bars['adx_smooth'].shift(-1)
crypto_bars['kijun_sen'] = crypto_bars['kijun_sen'].shift(-1)
crypto_bars['tenkan_sen'] = crypto_bars['tenkan_sen'].shift(-1)
crypto_bars['senkou_span_a'] = crypto_bars['senkou_span_a'].shift(-1)
crypto_bars['senkou_span_b'] = crypto_bars['senkou_span_b'].shift(-1)
crypto_bars['obv'] = crypto_bars['obv'].shift(-1)

# Replace NaN with 0 and Validation
crypto_bars.fillna(0, inplace=True)
print(crypto_bars.isnull().sum())

# Convert to .csv for future use
os.remove('crypto_bars_next_day.csv')
crypto_bars.to_csv('crypto_bars_next_day.csv')
