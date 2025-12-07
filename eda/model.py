import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
import os
import warnings

warnings.filterwarnings('ignore')

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '../dataset/crime_against_women.csv')
    df = pd.read_csv(file_path)
    return df

def forecast_crimes():
    df = load_data()
    
    # Preprocessing
    crime_cols = df.columns[7:]
    df[crime_cols] = df[crime_cols].fillna(0)
    for col in crime_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Total Crimes'] = df[crime_cols].sum(axis=1)
    
    # Prepare Time Series
    yearly_crimes = df.groupby('year')['Total Crimes'].sum()
    
    # Convert index to datetime for statsmodels
    yearly_crimes.index = pd.to_datetime(yearly_crimes.index, format='%Y')
    
    # Set frequency to Year Start
    ts_data = yearly_crimes.asfreq('YS')
    
    print("Training data:")
    print(ts_data)
    
    try:
        # Fit Holt's Linear Trend Model
        # Using simple exponential smoothing if Holt fails due to data size, but Holt should work with 3 points
        model = Holt(ts_data, initialization_method="estimated").fit()
        
        # Forecast 5 years
        forecast = model.forecast(5)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(ts_data.index, ts_data.values, label='Observed Data', marker='o', color='blue')
        plt.plot(forecast.index, forecast.values, label='Forecast (Holt\'s Method)', marker='o', color='red', linestyle='--')
        
        plt.title('Total Crime Forecasting (Next 5 Years)')
        plt.xlabel('Year')
        plt.ylabel('Total Crimes')
        plt.legend()
        plt.grid(True)
        plt.savefig('eda/plots/forecast.png')
        plt.close()
        
        print("Saved forecast.png")
        print("\nForecasted Values:")
        print(forecast)
        
    except Exception as e:
        print(f"Forecasting failed: {e}")
        print("Note: Forecasting requires sufficient data points. The dataset might be too small (3 years).")

if __name__ == "__main__":
    forecast_crimes()
