import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
diabetes = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.keys())
# print(diabetes.data)  # (442, 10)
# print(diabetes.DESCR)  # (442,)

# This is a regression task, so we will use the linear regression model.
# The target is a continuous value, which is the disease progression one year after baseline.
diabetes_X = diabetes.data[:, np.newaxis, 2]  # (442, 1)
# diabetes_X = diabetes.data # All features (442, 10)
# diabetes_X = diabetes.data[:, np.newaxis, 0]  # (442, 1) # Age


diabetes_X_train = diabetes_X[:-30]  # (422, 1)
diabetes_X_test = diabetes_X[-30:]  # (20, ) this is the test set for the model
# The training set is used to train the model, while the test set is used to evaluate its performance.
diabetes_y_train = diabetes.target[:-30]  # (422,)  The target values for the training set (target is the disease progression one year after baseline)
diabetes_y_test = diabetes.target[-30:]  # (20,)  The target values for the test set
# The target values are the disease progression one year after baseline for the training and test sets.

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(diabetes_X_train, diabetes_y_train)
# fit() method is used to train the model. It takes the training data and the target values as input.
# The model learns the relationship between the input features and the target values during training.

# Predict the response for test dataset
diabetes_y_pred = model.predict(diabetes_X_test)
# The predict() method is used to make predictions on new data. It takes the input features as input and returns the predicted target values.

# mean squared error
print("Mean squared error: ", mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The mean_squared_error() function is used to evaluate the performance of the model. It calculates the average of the squared differences between the predicted and actual target values.

print("weights: ", model.coef_)
# The coef_ attribute contains the coefficients of the linear regression model. It represents the weights assigned to each feature in the model.
print("intercept: ", model.intercept_)
# The intercept_ attribute contains the intercept of the linear regression model. It represents the value of the target variable when all input features are zero.

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=3)
plt.show()

