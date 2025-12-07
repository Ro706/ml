import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")

# Create plots directory if it doesn't exist (redundant but safe)
os.makedirs('eda/plots', exist_ok=True)

def load_data():
    # Load dataset
    file_path = os.path.join(os.path.dirname(__file__), '../dataset/crime_against_women.csv')
    df = pd.read_csv(file_path)
    return df

def perform_eda():
    df = load_data()
    
    # Preprocessing
    # Identify crime columns (assuming they start from the 8th column, index 7)
    crime_cols = df.columns[7:]
    
    # Fill missing values with 0 and ensure numeric
    df[crime_cols] = df[crime_cols].fillna(0)
    for col in crime_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Create Total Crimes column
    df['Total Crimes'] = df[crime_cols].sum(axis=1)
    
    # --- Plot 1: Crime Distribution ---
    plt.figure(figsize=(15, 10))
    crime_sums = df[crime_cols].sum().sort_values(ascending=False)
    sns.barplot(x=crime_sums.values, y=crime_sums.index, palette="viridis")
    plt.title('Distribution of Different Crimes Against Women (2017-2019)')
    plt.xlabel('Number of Cases')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.savefig('eda/plots/crime_distribution.png')
    plt.close()
    print("Saved crime_distribution.png")
    
    # --- Plot 2: Total Crimes Over Years ---
    plt.figure(figsize=(10, 6))
    yearly_crimes = df.groupby('year')['Total Crimes'].sum()
    sns.lineplot(x=yearly_crimes.index, y=yearly_crimes.values, marker='o', linewidth=2.5)
    plt.title('Total Crimes Against Women Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.xticks(yearly_crimes.index) # Ensure integers for years
    plt.grid(True)
    plt.savefig('eda/plots/total_crimes_over_years.png')
    plt.close()
    print("Saved total_crimes_over_years.png")

    # --- Plot 3: Severe Crimes Over Years ---
    # Selecting a few specific severe crimes for trend analysis
    severe_crimes = ['rape_women_above_18', 'dowry_deaths', 'acid_attack', 'kidnapping_and_abduction', 'cruelty_by_husband_or_his_relatives']
    # Filter only those that exist in columns (to be safe)
    severe_crimes = [c for c in severe_crimes if c in df.columns]
    
    if severe_crimes:
        plt.figure(figsize=(12, 6))
        yearly_severe = df.groupby('year')[severe_crimes].sum()
        
        # Plotting each line
        for col in severe_crimes:
            sns.lineplot(x=yearly_severe.index, y=yearly_severe[col], marker='o', label=col)
            
        plt.title('Trend of Severe Crimes Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Cases')
        plt.xticks(yearly_severe.index)
        plt.legend()
        plt.grid(True)
        plt.savefig('eda/plots/severe_crimes_over_years.png')
        plt.close()
        print("Saved severe_crimes_over_years.png")
        
    # Print Summary Stats
    print("\nSummary Statistics:")
    print(f"Total Crimes Recorded: {df['Total Crimes'].sum()}")
    print(f"Yearly Breakdown:\n{yearly_crimes}")

if __name__ == "__main__":
    perform_eda()
