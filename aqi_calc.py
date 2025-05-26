import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

train = pd.read_csv('AQI-and-Lat-Long-of-Countries.csv')
print(train.head())

#Create Model
m1 = RandomForestRegressor()

train1 = train.drop(['AQI Value'], axis=1)
target = train['AQI Value']

print(train1)
print(target)


m1.fit(train1 , target)
m1.score(train1, target)*100
# predicting the model with other values (testing the data)
prediction_result= m1.predict([[1, 10, 5, 11, 10, 5]])

print(prediction_result)


