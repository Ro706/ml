import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------
iris = load_iris()
x = iris.data
y = iris.target

df = pd.DataFrame(x, columns=iris.feature_names)
df['target'] = y

print("Dataset Head:\n", df.head())
print("\nShape:", df.shape)

# -------------------------------------------------------
# 2. CHECK FOR NULL VALUES
# -------------------------------------------------------
print("\nNull values:\n", df.isnull().sum())

# -------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# -------------------------------------------------------
print("\nStatistics:\n", df.describe())

# -------------------------------------------------------
# 4. CORRELATION MATRIX
# -------------------------------------------------------
print("\nCorrelation:\n", df.corr())

# (Optional plot)
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()

# -------------------------------------------------------
# 5. OUTLIER DETECTION USING IQR
# -------------------------------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).sum()
print("\nOutlier count in each column:\n", outliers)

# -------------------------------------------------------
# 6. FEATURE SCALING (STANDARDIZATION + NORMALIZATION)
# -------------------------------------------------------
scaler_std = StandardScaler()
x_std = scaler_std.fit_transform(x)

scaler_norm = MinMaxScaler()
x_norm = scaler_norm.fit_transform(x)

print("\nStandardized Sample:", x_std[0])
print("\nNormalized Sample:", x_norm[0])

# -------------------------------------------------------
# 7. OPTIONAL: FEATURE ENGINEERING (POLYNOMIAL FEATURES)
# -------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

print("\nShape after polynomial features:", x_poly.shape)

# -------------------------------------------------------
# 8. TRAIN–TEST SPLIT
# -------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x_std, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 9. TRAIN THE MODEL
# -------------------------------------------------------
model = LinearRegression()
model.fit(x_train, y_train)

# -------------------------------------------------------
# 10. MAKE PREDICTIONS
# -------------------------------------------------------
y_pred = model.predict(x_test)

# -------------------------------------------------------
# 11. EVALUATE THE MODEL
# -------------------------------------------------------
print("\nPredicted:", y_pred)
print("\nActual:", y_test)
print("\nR2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

