
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Create the output directory if it doesn't exist
if not os.path.exists('deep_learning/data'):
    os.makedirs('deep_learning/data')

# Load the dataset
df = pd.read_csv('eda/dataset/crime_aginest_women.csv')

# Drop non-numeric columns and columns that are not crime types
crime_columns = df.columns.drop(['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles'])

# Convert crime columns to numeric, coercing errors to 0
for col in crime_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# Calculate the total crimes
df['total_crimes'] = df[crime_columns].sum(axis=1)

# Group by year and sum the total crimes
yearly_crimes = df.groupby('year')['total_crimes'].sum().reset_index()

# Save the aggregated data
yearly_crimes.to_csv('deep_learning/data/yearly_crimes.csv', index=False)

print("Yearly crime data has been prepared and saved to deep_learning/data/yearly_crimes.csv")


# Load the dataset
data = pd.read_csv('deep_learning/data/yearly_crimes.csv')
crimes = data['total_crimes'].values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
crimes = scaler.fit_transform(crimes.reshape(-1, 1))

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(crimes, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])


# Forecast the next 5 years
last_data = crimes[len(crimes) - look_back:]
last_data = last_data.reshape(1, look_back, 1)

predictions = []
for _ in range(5):
    prediction = model.predict(last_data)
    predictions.append(prediction[0][0])
    # update last_data to include the new prediction and remove the oldest value
    last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['year'], scaler.inverse_transform(crimes), label='Historical Data')

# Plot the training predictions
train_predict_plot = np.empty_like(crimes)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(trainPredict)+look_back, :] = trainPredict
plt.plot(data['year'], train_predict_plot, label='Training Prediction')


forecast_years = np.arange(data['year'].max() + 1, data['year'].max() + 6)
plt.plot(forecast_years, future_predictions, label='Forecast', linestyle='--')

plt.title('Crime Rate Forecast using LSTM')
plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.legend()
plt.grid(True)
plt.savefig('deep_learning/forecast.png')
plt.show()

print("Forecast plot saved to deep_learning/forecast.png")
