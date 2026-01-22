import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import joblib #for saving the model

# Load dataset
data_path = os.path.join("..", "/home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/data/women_health_dataset.csv")
df = pd.read_csv(data_path)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)
x=df.drop('delay_days', axis=1)
y=df['delay_days']  

#train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #there are 50 samples in total in which 20% is 10 samples for testing and 40 samples for training and random state is set to 42 for reproducibility that means every time we run the code we will get the same split of data for training and testing because random state is fixed to 42 meaning of 42 is just a number chosen arbitrarily it could be any integer value so that the random processes involved in splitting the data can be reproduced exactly in future runs of the code


#model
model=RandomForestRegressor(n_estimators=100, random_state=42)#100 trees in the forest and random state is set to 42 for reproducibility which means every time we run the code we will get the same random processes involved in training the model and n estimators is set to 100 meaning the model will create 100 decision trees during training decision trees are the basic building blocks of random forest model that shows how the model makes predictions based on the input features
model.fit(X_train, y_train)     #training the model


#predictions
y_pred = model.predict(X_test)  #making predictions on the test set


#evaluation
mae = mean_absolute_error(y_test, y_pred)#calculating mean absolute error between actual and predicted values
mse = mean_squared_error(y_test, y_pred)#`calculating mean squared error between actual and predicted values
rmse = mse ** 0.5#calculating root mean squared error by taking square root of mean squared error
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")   

importances=model.feature_importances_
for feature ,imp in zip(x.columns, importances):#printing feature importance for each feature in the dataset which indicates how much each feature contributes to the model's predictions
    print(f"Feature: {feature}, Importance: {imp}")#importance value is rounded to 3 decimal places for better readability
    print(feature ,":" ,round(imp,3))#printing feature importance for each feature in the dataset which indicates how much each feature contributes to the model's predictions


# Save the model
model_path = os.path.join("..", "/home/ritirai/AI-Powered Women Health Insight System/Women Health Insight/model/delay_predictor_model.pkl")     
print(f"Saving model to {model_path}")
joblib.dump(model, model_path)
print("Model saved successfully.")  