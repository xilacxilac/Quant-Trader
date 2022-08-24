import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.layers import Dropout, Bidirectional
import matplotlib.pyplot as plt
import plotly.express as px

# Read .csv
df = pd.read_csv("datasets/crypto_bars_future.csv")


def to_sequences(data, seq_len):
    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    return np.array(d)


def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]
    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]
    return X_train, y_train, X_test, y_test


ax = df.plot(x='timestamp', y='close');
ax.set_xlabel('Date')
ax.set_ylabel('Close Price (USD)')
plt.show()

# Convert closing prices into scalar on a 0 to 1 scale and into an ndarray
scaler = MinMaxScaler()
close_price = df['close'].values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)

# Count NaN
print('NaN count: ' + str(np.count_nonzero(np.isnan(scaled_close))))

# How many time periods prior to look at when predicting
SEQ_LEN = 100

# X_train and X_test are 2D arrays with cryptocurrency closing prices SEQ_LEN in the past
# y_train and y_test are arrays with cryptocurrency closing prices
X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split=0.80)

model = tf.keras.Sequential()
model.add(Bidirectional(layers.LSTM(units=32, return_sequences=True, input_shape=(SEQ_LEN - 1, 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(layers.LSTM(units=32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(layers.LSTM(units=32)))
model.add(Dropout(0.2))
model.add(layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

BATCH_SIZE = 64
history = model.fit(X_train, y_train, epochs=150, batch_size=BATCH_SIZE, shuffle=False, validation_split=0.1)
model.summary()
model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_hat = model.predict(X_test)
y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label='Actual Price', color='green')
plt.plot(y_hat_inverse, label='Predicted Price', color='red')

plt.title('Cryptocurrency Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show();
