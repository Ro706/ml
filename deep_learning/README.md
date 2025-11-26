
# Crime Rate Forecasting using Deep Learning

This part of the project uses a deep learning model to forecast crime rates based on historical data.

## 1. Data Preparation

The script `data_preparation.py` prepares the data for the model. It does the following:
- Loads the `crime_aginest_women.csv` dataset.
- Aggregates the crime data by year to get the total number of crimes for each year.
- Saves the aggregated data into `deep_learning/data/yearly_crimes.csv`.

## 2. Model Training and Forecasting

The script `model.py` builds, trains, and uses an LSTM (Long Short-Term Memory) model for forecasting. Here's a breakdown of the steps:
- **Load Data**: The script loads the `yearly_crimes.csv` data prepared in the previous step.
- **Data Normalization**: The crime data is normalized to a range of 0 to 1, which helps improve the performance of the neural network.
- **Model Building**: An LSTM model is built using TensorFlow/Keras. The model is designed to capture temporal dependencies in the time series data.
- **Training**: The model is trained on the historical crime data.
- **Forecasting**: Once trained, the model is used to predict the total number of crimes for the next 5 years.

## 3. Results

The forecast, along with the historical data and the model's predictions on the training set, is saved as a plot in `deep_learning/forecast.png`.

## How to Run

1. **Install the required libraries**:
   ```
   pip install pandas tensorflow scikit-learn matplotlib
   ```
2. **Run the data preparation script**:
   ```
   python data_preparation.py
   ```
3. **Run the model training and forecasting script**:
   ```
   python model.py
   ```

This will train the model and generate the `forecast.png` plot in the `deep_learning` directory, showing the forecasted crime rates.
