import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# METHOD 1: Download the dataset automatically.
# The file is downloaded once and cached locally by Keras.

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
csv_path = get_file("airline-passengers.csv", origin=url)

df = pd.read_csv(csv_path)


# METHOD 2: Load the CSV directly from your computer.
# If your teacher provides the CSV file, comment out METHOD 1 above
# and uncomment the following line. Change the path to your file.
#
# df = pd.read_csv(r"C:\Users\YourName\Downloads\airline-passengers.csv")


# Extract passenger counts as a numerical array.
series = df["Passengers"].values.astype("float32").reshape(-1, 1)

print(df.head())
print("Dataset shape:", series.shape)

# Convert Month to datetime for clean x-axis year labels
df["Month"] = pd.to_datetime(df["Month"])

# Plot the original time series.
plt.figure(figsize=(10, 4))
plt.plot(df["Month"], series)
plt.title("Monthly Airline Passengers (1949-1960)")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()

# Scale passenger counts between 0 and 1.
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

# Use the previous 12 months to predict the next month.
LOOK_BACK = 12

X = []
y = []

for i in range(len(scaled) - LOOK_BACK):
    X.append(scaled[i:i + LOOK_BACK, 0])
    y.append(scaled[i + LOOK_BACK, 0])

X = np.array(X)
y = np.array(y)

# Split chronologically: first 80% for training, remaining 20% for testing.
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# RNN input format: samples, time steps, features.
X_train = X_train.reshape(-1, LOOK_BACK, 1)
X_test = X_test.reshape(-1, LOOK_BACK, 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

rnn_model = Sequential([
    SimpleRNN(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])

rnn_model.compile(optimizer="adam", loss="mse")

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)

# Plot training and validation loss.
plt.figure(figsize=(7, 4))
plt.plot(rnn_history.history["loss"], label="Training Loss")
plt.plot(rnn_history.history["val_loss"], label="Validation Loss")
plt.title("SimpleRNN Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Predict passenger counts and convert them back to the original scale.
rnn_pred = scaler.inverse_transform(rnn_model.predict(X_test, verbose=0))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rnn_rmse = np.sqrt(mean_squared_error(y_test_actual, rnn_pred))
print("SimpleRNN Test RMSE:", round(rnn_rmse, 2))

lstm_model = Sequential([
    LSTM(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)

plt.figure(figsize=(7, 4))
plt.plot(lstm_history.history["loss"], label="Training Loss")
plt.plot(lstm_history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test, verbose=0))

lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
print("LSTM Test RMSE:", round(lstm_rmse, 2))

gru_model = Sequential([
    GRU(50, activation="tanh", input_shape=(LOOK_BACK, 1)),
    Dense(1)
])

gru_model.compile(optimizer="adam", loss="mse")

gru_history = gru_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)

plt.figure(figsize=(7, 4))
plt.plot(gru_history.history["loss"], label="Training Loss")
plt.plot(gru_history.history["val_loss"], label="Validation Loss")
plt.title("GRU Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

gru_pred = scaler.inverse_transform(gru_model.predict(X_test, verbose=0))

gru_rmse = np.sqrt(mean_squared_error(y_test_actual, gru_pred))
print("GRU Test RMSE:", round(gru_rmse, 2))

# Compare actual passenger counts with predictions from all models.

plt.figure(figsize=(10, 5))

plt.plot(y_test_actual, label="Actual", linewidth=2)
plt.plot(rnn_pred, label="SimpleRNN")
plt.plot(lstm_pred, label="LSTM")
plt.plot(gru_pred, label="GRU")

plt.title("Airline Passenger Forecast Comparison")
plt.xlabel("Test Time Step")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# Compare test RMSE values. Lower RMSE indicates lower prediction error.
models = ["SimpleRNN", "LSTM", "GRU"]
rmse_values = [rnn_rmse, lstm_rmse, gru_rmse]

for name, score in zip(models, rmse_values):
    print(name, "RMSE:", round(score, 2))

plt.figure(figsize=(7, 4))
plt.bar(models, rmse_values)
plt.title("Test RMSE Comparison")
plt.ylabel("RMSE")
plt.show()
