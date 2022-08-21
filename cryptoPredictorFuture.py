import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import layers
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv("crypto_bars_future.csv")


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

SEQ_LEN = 100
scaler = MinMaxScaler()
close_price = df['close'].values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)
scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1)

X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split=0.80)

model = tf.keras.Sequential()
model.add(layers.LSTM(units=32, return_sequences=True, input_shape=(99, 1), dropout=0.2))
model.add(layers.LSTM(units=32, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(units=32, dropout=0.2))
model.add(layers.Dense(units=1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

BATCH_SIZE = 64
history = model.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE, shuffle=False, validation_split=0.1)
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
