import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------
iris = load_iris()
x = iris.data
y = iris.target

# create dataframe early so descriptive checks work
df = pd.DataFrame(x, columns=iris.feature_names)
df['target'] = y

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

# Optional heatmap
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()

# -------------------------------------------------------
# 5. OUTLIER DETECTION USING IQR
# -------------------------------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("\nOutlier count in each column:\n", outliers)

# -------------------------------------------------------
# 6. FEATURE SCALING (STANDARDIZATION + NORMALIZATION)
# -------------------------------------------------------
scaler_std = StandardScaler()
x_std = scaler_std.fit_transform(x)

scaler_norm = MinMaxScaler()
x_norm = scaler_norm.fit_transform(x)

print("\nStandardized Sample (first row):", x_std[0])
print("\nNormalized Sample (first row):", x_norm[0])

# -------------------------------------------------------
# 7. OPTIONAL: FEATURE ENGINEERING (POLYNOMIAL FEATURES)
# -------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
print("\nShape after polynomial features:", x_poly.shape)

# If you want to use polynomial features for modelling, replace x_std with x_poly (and consider scaling x_poly)

# -------------------------------------------------------
# 8. TRAIN–TEST SPLIT
# Use stratify to maintain class proportions
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    x_std, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# 9. TRAIN THE MODEL (Classification)
# -------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------
# 10. MAKE PREDICTIONS
# -------------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------------
# 11. EVALUATE THE MODEL (Classification metrics)
# -------------------------------------------------------
print("\nPredicted:", y_pred)
print("\nActual:", y_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nDataset Head:\n", df.head())
print("\nShape:", df.shape)

# -------------------------------------------------------
# 12. HYPERPARAMETER SEARCH (GridSearchCV for RandomForest)
# -------------------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Instantiate the GridSearchCV object with correct estimator and n_jobs
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# view the best hyperparameters
print(f"Best Hyperparameters : {grid_search.best_params_}")
print(f"Best CV Score : {grid_search.best_score_:.4f}")

# You can evaluate best estimator on test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nTest Accuracy of Best Model:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report (best model):\n", classification_report(y_test, y_pred_best))

