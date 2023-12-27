# Recurrent-Neural-Network-RNN-with-TensorFlow
Example of creating a simple Recurrent Neural Network (RNN) using TensorFlow.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

# Generate sample data for time series prediction
np.random.seed(0)
timesteps = 10
X = np.random.rand(100, timesteps, 1)
y = np.sum(X, axis=1)

# Create RNN model
model = Sequential([
    SimpleRNN(10, input_shape=(timesteps, 1), activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, verbose=0)

# Make predictions
test_sequence = np.random.rand(1, timesteps, 1)
prediction = model.predict(test_sequence)
print(f"Predicted sum: {prediction[0][0]}")
