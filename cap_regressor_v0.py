# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions_BTMPV import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
np.random.seed(42)

# dataset_ausgrid = pd.read_csv('input/Ausgrid/2012-2013 Solar home electricity input v2.csv', header=1)
# # print(dataset_ausgrid)
#
# df_capacity = pd.DataFrame()
# for customer in range(1, 301):
#     print(customer)
#     df_customer = select_customer(dataset_ausgrid, 'Customer', customer)
#     df_customer.reset_index(drop=True, inplace=True)
#     # Obtain real capacity and drop columns
#     PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')
#     # Calculate power load for each customer
#     df_power = preprocessing(df_customer)
#     # Calculate Total Consumption and Net Consumption - We'll use the net consumption
#     df_BTM = calculate_consumptions(df_power)
#     # Calculate C_min and C_max
#     features_dict = features(df_BTM)
#     features_dict['y'] = PV_capacity
#     features_dict['Customer'] = customer
#
#     # Function to obtain other features
#     df_features = pd.DataFrame(features_dict)
#     df_capacity = pd.concat([df_capacity, df_features], ignore_index=True)
#
# df_capacity.set_index('Customer', inplace=True)
# df_capacity.to_csv('input/Ausgrid/features_1213_v3.csv')

# for i in range(0, 5):
df_capacity = pd.read_csv('data/Ausgrid/features_1213_v3.csv')
df_capacity.set_index('Customer', inplace=True)
# Splitting the input into features and target variable
X = df_capacity.drop(columns=['y', 'Cmax_2', 'Cmax_3', 'Median_min', 'Mean_min', 'Min_75q'])
y = df_capacity['y']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating Lasso model
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)

# Creating and fitting the Random Forest Regressor
RF_reg = RandomForestRegressor(n_estimators=500, max_depth=50, max_features='sqrt')
RF_reg.fit(X_train_scaled, y_train)

# Creating and fitting the Support Vector Regressor
SVR = SVR(epsilon=0.01, C=20, kernel='linear')
SVR.fit(X_train_scaled, y_train)

# Making predictions
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_RF = RF_reg.predict(X_test_scaled)
y_pred_SVR = SVR.predict(X_test_scaled)

# Evaluating models
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
mape_lasso = mean_absolute_percentage_error(y_test, y_pred_lasso)
mse_RF = mean_squared_error(y_test, y_pred_RF)
rmse_RF = np.sqrt(mse_RF)
r2_RF = r2_score(y_test, y_pred_RF)
mape_RF = mean_absolute_percentage_error(y_test, y_pred_RF)
mse_SVR = mean_squared_error(y_test, y_pred_SVR)
rmse_SVR = np.sqrt(mse_SVR)
r2_SVR = r2_score(y_test, y_pred_SVR)
mape_SVR = mean_absolute_percentage_error(y_test, y_pred_SVR)

# print(f"Lasso MSE: {mse_lasso}")
print(f"Lasso RMSE: {rmse_lasso}")
print(f"Lasso R2: {r2_lasso}")
print(f"Lasso MAPE: {mape_lasso}")
# print(f"Random Forest MSE: {mse_RF}")
print(f"Random Forest RMSE: {rmse_RF}")
print(f"RF R2: {r2_RF}")
print(f"RF MAPE: {mape_RF}")
print(f"SVR RMSE: {rmse_SVR}")
print(f"SVR R2: {r2_SVR}")
print(f"SVR MAPE: {mape_SVR}")


# df_results = pd.concat([X_test, y_test], axis=1)
# df_results['y_pred'] = y_pred_RF
# df_results['error'] = abs(df_results['y_pred'] - df_results['y'])
# print(df_results[df_results['error'] == df_results['error'].max()])
# df_results['MAPE'] = abs((df_results['y']-df_results['y_pred'])/df_results['y'])
#
# # Creating scatter plot for RF
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_RF, alpha=0.7)
# plt.plot(y_test, y_test, color='red', label='Perfect Prediction')
# plt.title('RF: Actual vs Predicted')
# plt.xlabel('Actual values')
# plt.ylabel('Predicted values')
# plt.grid(True)
# plt.show()
#
# df_results.boxplot(column=['MAPE'])
# plt.show()
