from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

housing = datasets.fetch_california_housing()
x = housing.data
y = housing.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=432)  
LR = LinearRegression()
LR.fit(x_train, y_train)

y_pred = LR.predict(x_test)
print("R2 score:", r2_score(y_test, y_pred))

# Plotting Actual vs Predicted values
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")  
plt.title("Actual vs Predicted Values - Linear Regression")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.show()
# joblib.dump(LR, "LinearRegressionModel.joblib")