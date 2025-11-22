import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save the plots
if not os.path.exists('eda/plots'):
    os.makedirs('eda/plots')

# Load the dataset
df = pd.read_csv('eda/dataset/crime_aginest_women.csv')

# Display the first few rows of the dataframe
print("First 5 rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary statistics of the dataset:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop the 'id' column as it is just an index
df = df.drop('id', axis=1)

# Plotting the distribution of different crimes
numeric_df = df.select_dtypes(include=['number']).drop('year', axis=1)
plt.figure(figsize=(12, 8))
sns.barplot(data=numeric_df, orient='h')
plt.title('Distribution of Crimes Against Women')
plt.xlabel('Number of Cases')
plt.ylabel('Crime')
plt.tight_layout()
plt.savefig('eda/plots/crime_distribution.png')
plt.close()

# Plotting the trend of total crimes over the years
df['total_crimes'] = numeric_df.sum(axis=1)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='total_crimes')
plt.title('Total Crimes Against Women Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Number of Cases')
plt.grid(True)
plt.savefig('eda/plots/total_crimes_over_years.png')
plt.close()

# Plotting the trend of some of the most severe crimes over the years
severe_crimes = ['murder_with_rape_gang_rape', 'dowry_deaths', 'rape_women_above_18', 'rape_girls_below_18']
plt.figure(figsize=(12, 8))
for crime in severe_crimes:
    sns.lineplot(data=df, x='year', y=crime, label=crime)
plt.title('Trend of Severe Crimes Against Women Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.savefig('eda/plots/severe_crimes_over_years.png')
plt.close()

print("\nEDA plots have been saved in the 'eda/plots' directory.")
