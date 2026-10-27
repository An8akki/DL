"""
Experiment 3: Time Series Forecasting using RNN and LSTM
==========================================================

Dataset: International Airline Passengers (monthly totals, 1949-1960)
This is the classic Box-Jenkins "airline passengers" dataset, the standard
teaching dataset for time-series forecasting with RNN/LSTM. It is fetched
automatically using keras.utils.get_file(), so no manual download is needed
and it works the same way in Google Colab or a local Jupyter notebook.

Framework: TensorFlow / Keras (tensorflow>=2.x)
    pip install tensorflow pandas numpy matplotlib scikit-learn

A SimpleRNN model, an LSTM model, and a GRU model are all built, trained,
and compared on the same forecasting task so students can see the
difference between the three architectures. GRU (Gated Recurrent Unit)
is a lighter-weight gated alternative to LSTM (2 gates instead of 3, no
separate cell state) that often performs comparably.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------------------
# keras.utils.get_file downloads and caches the file (~/.keras/datasets/),
# so this only happens once.
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
csv_path = get_file("airline-passengers.csv", origin=DATA_URL)

df = pd.read_csv(csv_path)
# The CSV has columns: Month, Passengers
series = df.iloc[:, 1].values.astype("float32").reshape(-1, 1)

print("Dataset shape:", series.shape)
print(df.head())

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("International Airline Passengers (1949-1960)")
plt.xlabel("Month")
plt.ylabel("Passengers (thousands)")
plt.tight_layout()
plt.savefig("raw_series.png")
plt.close()

# -------------------------------------------------------------------
# 2. Preprocess: scale to [0, 1] and build sliding-window sequences
# -------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(series)

LOOK_BACK = 12  # use the previous 12 months to predict the next month


def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


X, y = create_dataset(scaled, LOOK_BACK)

# Train/test split (chronological, no shuffling for time series)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape to [samples, timesteps, features] as required by RNN/LSTM layers
X_train = X_train.reshape((X_train.shape[0], LOOK_BACK, 1))
X_test = X_test.reshape((X_test.shape[0], LOOK_BACK, 1))

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------------------------------------------------
# 3. Helper to train a model and report results
# -------------------------------------------------------------------


def train_and_evaluate(model, name, epochs=100):
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=8, verbose=0
    )

    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # Invert scaling to get back to real passenger numbers
    train_pred_inv = scaler.inverse_transform(train_pred)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_pred_inv = scaler.inverse_transform(test_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    print(f"\n[{name}] Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title(f"{name} - Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_loss.png")
    plt.close()

    return test_pred_inv, y_test_inv, test_rmse


# -------------------------------------------------------------------
# 4. Model A: Simple RNN
# -------------------------------------------------------------------
rnn_model = Sequential([
    SimpleRNN(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])
rnn_pred, y_test_inv, rnn_rmse = train_and_evaluate(rnn_model, "SimpleRNN")

# -------------------------------------------------------------------
# 5. Model B: LSTM
# -------------------------------------------------------------------
lstm_model = Sequential([
    LSTM(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])
lstm_pred, _, lstm_rmse = train_and_evaluate(lstm_model, "LSTM")

# -------------------------------------------------------------------
# 6. Model C: GRU (Gated Recurrent Unit)
# -------------------------------------------------------------------
gru_model = Sequential([
    GRU(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])
gru_pred, _, gru_rmse = train_and_evaluate(gru_model, "GRU")

# -------------------------------------------------------------------
# 7. Compare predictions on the test set
# -------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual", linewidth=2)
plt.plot(rnn_pred, label=f"SimpleRNN (RMSE={rnn_rmse:.1f})")
plt.plot(lstm_pred, label=f"LSTM (RMSE={lstm_rmse:.1f})")
plt.plot(gru_pred, label=f"GRU (RMSE={gru_rmse:.1f})")
plt.title("Airline Passengers Forecast: RNN vs LSTM vs GRU")
plt.xlabel("Time step (test set)")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_vs_lstm_forecast.png")
plt.close()

print("\nSaved plots: raw_series.png, simplernn_loss.png, lstm_loss.png, "
      "gru_loss.png, rnn_vs_lstm_forecast.png")
print(f"\nSummary -> SimpleRNN: {rnn_rmse:.2f} | LSTM: {lstm_rmse:.2f} | GRU: {gru_rmse:.2f}")