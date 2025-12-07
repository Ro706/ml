import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '../dataset/crime_against_women.csv')
    df = pd.read_csv(file_path)
    return df

def perform_regression():
    df = load_data()
    
    # Preprocessing
    crime_cols = df.columns[7:]
    df[crime_cols] = df[crime_cols].fillna(0)
    for col in crime_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Total Crimes'] = df[crime_cols].sum(axis=1)
    
    # Feature Selection
    # predicting Total Crimes based on a subset of major crimes
    # This simulates estimating total crime load based on key indicators
    features = [
        'rape_women_above_18', 
        'dowry_deaths', 
        'kidnapping_and_abduction', 
        'cruelty_by_husband_or_his_relatives',
        'assault_on_womenabove_18' # Note: Check spelling in CSV, assumed from context or matching
    ]
    
    # Filter features that actually exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    if not features:
        print("Error: No feature columns found.")
        return

    X = df[features]
    y = df['Total Crimes']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Regression Analysis Results:")
    print(f"Features used: {features}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Total Crimes')
    plt.xlabel('Actual Total Crimes')
    plt.ylabel('Predicted Total Crimes')
    plt.legend()
    plt.grid(True)
    plt.savefig('eda/plots/actual_vs_predicted.png')
    plt.close()
    print("Saved actual_vs_predicted.png")

if __name__ == "__main__":
    perform_regression()
