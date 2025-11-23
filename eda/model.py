
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
import os

# Create a directory to save the plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
df = pd.read_csv('dataset/crime_aginest_women.csv')

# Drop the 'id' column as it is just an index
df = df.drop('id', axis=1)

# Get numeric columns and calculate total crimes
numeric_df = df.select_dtypes(include=['number']).drop('year', axis=1)
df['total_crimes'] = numeric_df.sum(axis=1)

# Aggregate total crimes by year
yearly_crimes = df.groupby('year')['total_crimes'].sum()

# Create a time-series object
ts = pd.Series(yearly_crimes.values, index=pd.to_datetime(yearly_crimes.index, format='%Y'))

# Train the Holt's linear trend model
model = Holt(ts, initialization_method="estimated").fit()

# Forecast for the next 5 years
forecast = model.forecast(5)

# Create a dataframe for the forecast
forecast_years = pd.to_datetime([str(year) for year in range(ts.index[-1].year + 1, ts.index[-1].year + 6)])
forecast_df = pd.DataFrame({'year': forecast_years, 'predicted_crimes': forecast})

print("Forecast for the next 5 years:")
print(forecast_df)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Historical Data')
plt.plot(forecast_df['year'], forecast_df['predicted_crimes'], label='Forecast', linestyle='--')
plt.title('Total Crimes Against Women: Forecast')
plt.xlabel('Year')
plt.ylabel('Total Number of Cases')
plt.legend()
plt.grid(True)
plt.savefig('plots/forecast.png')
plt.close()

print("\nForecast plot has been saved in the 'plots' directory.")
