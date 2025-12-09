import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# 1. Load Dataset
print("Loading Diabetes dataset from sklearn...")
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print(f"Dataset Shape: {df.shape}")
print("Features:", diabetes.feature_names)
print("\nFirst 5 rows:")
print(df.head())

# 2. EDA
print("\nPerforming EDA...")

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('images/correlation_matrix.png')
plt.close()

# Distribution of Target
plt.figure(figsize=(8, 6))
sns.histplot(df['target'], kde=True)
plt.title('Distribution of Disease Progression (Target)')
plt.xlabel('Disease Progression')
plt.savefig('images/target_distribution.png')
plt.close()

# Scatter plot of BMI vs Target (BMI is usually highly correlated)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bmi', y='target', data=df)
plt.title('BMI vs Disease Progression')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.savefig('images/bmi_vs_target.png')
plt.close()

# 3. Model Training
print("\nTraining Models...")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.4f}")

    # Plot Actual vs Predicted for each model
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'images/pred_vs_actual_{name.lower().replace(" ", "_")}.png')
    plt.close()

# 4. Model Comparison Graph
metrics_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(metrics_df)

plt.figure(figsize=(10, 6))
metrics_df['R2'].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Model Comparison - R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/model_comparison.png')
plt.close()

best_model_name = metrics_df['R2'].idxmax()
print(f"\nBest Performing Model: {best_model_name} with R2: {metrics_df.loc[best_model_name, 'R2']:.4f}")
