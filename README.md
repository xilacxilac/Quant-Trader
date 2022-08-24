# Quant Trader

**Quant Trader** is a Python project which attempts to predict future stock market
    and cryptocurrency movement through the utilization of deep learning with long
    short-term memory layers. The project has three forms of predictions: closing
    price some time in the future, next day candlestick, and candlestick some
    time in the future.

****

### How It Works:

This program creates datasets from Alpaca API, modifies them, and then uses them to
    train a neural network to predict future close prices or candlesticks.

****

### Performance:

#### Cryptocurrency Closing Price Predictor

LSTM trained with hourly closing values of Bitcoin from 2019-08-22 to 2022-08-22.

<img src="performances/Closing Cryptocurrency/LSTM/2019-08-22-e150.PNG" />

Bi-LSTM trained with hourly closing values of Bitcoin from 2020-08-22 to 2022-08-22.

<img src="performances/Closing Cryptocurrency/Bidirectional LSTM/2020-08-22-e150.PNG" />

Bi-LSTM trained with hourly closing values of Bitcoin from 2019-08-22 to 2022-08-22.

<img src="performances/Closing Cryptocurrency/Bidirectional LSTM/2019-08-22-e200.PNG" />

****

### In Progress:
- Stock predictions
  - next day candlestick
  - closing price some time in the future
  - candlestick some time in the future
- Parabolic SAR and Fibonacci Retracement (needs uptrend/downtrend)
- Cryptocurrency prediction
  - next day candlestick
  - candlestick some time in the future
- Client to trade on paper account

### APIs and Libraries:
- Alpaca API
- pandas
- numpy
- TensorFlow
- Keras
- scikit-learn
- plotly
- matplotlib 
- requests
- json
- os