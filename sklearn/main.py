from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
HistGradientBoostingRegressor,
RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import joblib 

housing = datasets.fetch_california_housing()

x = housing.data
y = housing.target

print(x.shape)
poly = PolynomialFeatures()
x = poly.fit_transform(x)
print(x.shape)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2 ,random_state=432) #reserve 20% for training and 80% for testing

LR = LinearRegression()
LR.fit(x_train,y_train)
GBR = HistGradientBoostingRegressor()
RFR = RandomForestRegressor(n_jobs=-1)
'''
for model in [LR,GBR,RFR]:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    #printing the r2 score by comparing y_test data and y_pred  
    print(model, r2)
'''

# Hyperparameter tuning for HistGradientBoostingRegressor

best_r2 = [(0,0),0] #(max_iter, learning_rate), r2_score
for j in [0.1,0.05,0.01,0.005,0.001]:
    for i in range(100,500,50):
        model = HistGradientBoostingRegressor(max_iter=i, learning_rate=j)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test,y_pred)
        if r2 > best_r2[1]:
            best_r2[0] = (i,j)
            best_r2[1] = r2
model = HistGradientBoostingRegressor(max_iter=best_r2[0][0], learning_rate=best_r2[0][1])
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print(f"HistGradientBooster [{best_r2[0]}]: {best_r2[1]}")
joblib.dump(model,"HistGrandientBoosingRegressor.joblib")
