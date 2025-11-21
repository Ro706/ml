import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df['target'] = diabetes.target 

print("Dataframe head: \n",df.head())

X = df[['bmi','bp','s1','s2','s3','s4','s5','s5']]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("R^2 score:",r2_score(y_test,y_pred))


#scatter plot: Actual vs Predicted

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-')
plt.grid(True)
plt.show()
