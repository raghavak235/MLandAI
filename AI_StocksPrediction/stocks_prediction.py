from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Preprocessing the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Define the number of timesteps and the number of features
timesteps = 60
n_features = data.shape[1]

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size,:], data[train_size:len(data),:]

# Create the training dataset
X_train, y_train = create_dataset(train_data, timesteps)

# Create the testing dataset
X_test, y_test = create_dataset(test_data, timesteps)

# Reshape the input data for the LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, n_features)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Invert the predictions to get the original scale
y_pred = scaler.inverse_transform(y_pred)
