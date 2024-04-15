# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from functions_BTMPV import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, pairwise_distances, mean_absolute_error, mean_absolute_percentage_error
np.random.seed(0)

# ===================== IDENTIFICATION OF STANDARD PV PROFILE =====================
dataset_ausgrid = pd.read_csv('data/Ausgrid/2012-2013 Solar home electricity data v2.csv', header=1)

# # Postcodes with more customers: 2259 (28), 2261 (21), 2290 (13).
# count_customers(dataset_ausgrid, 5)

selected_PC = 2259
df_PC = dataset_ausgrid[dataset_ausgrid['Postcode'] == selected_PC].reset_index(drop=True)
# df_PC = dataset_ausgrid.copy()
df_stGG_all = pd.DataFrame()

for customer in df_PC['Customer'].unique():
    df_customer = select_customer(df_PC, customer)
    df_customer.reset_index(drop=True, inplace=True)
    PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')
    df_customer.drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'], inplace=True)
    # Formatting dataframe to present consumption in columns
    df_format = format_df(df_customer)
    # Energy magnitudes to power
    df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
    # Calculate Total Consumption and Net Consumption
    df_BTM = calculate_consumptions(df_power)
    # Some customers have less samples, so we just exclude them from our calculations
    if len(df_BTM) < 17520:
        continue
    df_stGG = calculate_st_gen(df_BTM, PV_capacity)
    df_stGG.rename(columns={'stGG': 'stGG_' + str(customer)}, inplace=True)
    df_stGG_all = pd.concat([df_stGG_all, df_stGG['stGG_' + str(customer)]], axis=1)

# Calculate the average profile
average_stGG = df_stGG_all.mean(axis=1)

# Calculate the Euclidean distance between each profile and the average profile
distance = pairwise_distances(df_stGG_all.T, [average_stGG], metric='euclidean')

# Find the profile with the lowest distance to the average
most_similar_profile = df_stGG_all.columns[distance.argmin()]

standard_GG = df_stGG_all[most_similar_profile]

# ===================== CAPACITY PREDICTION =====================
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

# Creating and fitting the Random Forest Regressor
RF_reg = RandomForestRegressor(n_estimators=200, max_depth=50, max_features='sqrt')
RF_reg.fit(X_train_scaled, y_train)

# Making predictions
y_pred_RF = RF_reg.predict(X_test_scaled)

# RESULTS
df_results = pd.concat([X_test, y_test], axis=1)
df_results['y_pred'] = y_pred_RF
df_results.reset_index(inplace=True)
df_aggregated = pd.DataFrame()
mse_list = []
rel_mse_list = []
mae_list = []
mape_gg_list = []
mse_gg_list = []
mape_nc_list = []
rel_mse_PC = []
mae_PC = []
mape_gg_PC = []
rel_mse_noPC = []
mae_noPC = []
mape_gg_noPC = []
same_PC = 0
different_PC = 0
for customer in df_results['Customer'].unique():
    print(customer)
    pred_capacity = df_results[df_results['Customer'] == customer]['y_pred'].values
    df_customer_pred = select_customer(dataset_ausgrid, customer)
    df_customer_pred.reset_index(drop=True, inplace=True)
    real_capacity, postcode = obtain_capacity_postcode(df_customer_pred, 'Generator Capacity', 'Postcode')
    df_customer_pred.drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'], inplace=True)
    # Formatting dataframe to present consumption in columns
    df_format = format_df(df_customer_pred)
    # Energy magnitudes to power
    df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
    df_power = calculate_consumptions(df_power)
    df_power['GG_pred'] = pred_capacity[0] * standard_GG.values
    df_power['GG_error'] = df_power['GG'] - df_power['GG_pred']
    df_power['BL_pred'] = df_power['NC'] + df_power['GG_pred']
    if df_aggregated.empty:
        df_aggregated['GG_agg'] = df_power['GG']
        df_aggregated['GG_agg_pred'] = df_power['GG_pred']
        df_aggregated['NC_agg'] = df_power['NC']
        df_aggregated['NC_agg_pred'] = df_power['BL_pred']
    else:
        df_aggregated['GG_agg'] += df_power['GG']
        df_aggregated['GG_agg_pred'] += df_power['GG_pred']
        df_aggregated['NC_agg'] += df_power['NC']
        df_aggregated['NC_agg_pred'] += df_power['BL_pred']
    mse = mean_squared_error(df_power['GG'], df_power['GG_pred'])
    mae = mean_absolute_error(df_power['GG'], df_power['GG_pred'])
    # Need to change restriction to time between sunrise and sunset
    mape_gg = mean_absolute_percentage_error(df_power[df_power['GG'] > df_power['GG'].mean()]['GG'],
                                             df_power[df_power['GG'] > df_power['GG'].mean()]['GG_pred'])
    # mape_nc = mean_absolute_percentage_error(df_power[df_power['NC'] > df_power['NC'].mean()]['NC'],
    #                                          df_power[df_power['NC'] > df_power['NC'].mean()]['BL_pred'])
    rel_mse = mse/real_capacity
    mse_list.append(mse)
    mae_list.append(mae)
    mape_gg_list.append(mape_gg)
    # mape_nc_list.append(mape_nc)
    rel_mse_list.append(mse)
    # if postcode == selected_PC:
    #     rel_mse_PC.append(rel_mse)
    #     mae_PC.append(mae)
    #     mape_gg_PC.append(mape_gg)
    #     if same_PC == 0:
    #         same_PC = 1
    #         n = 100
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(df_power['GG'][48 * n:48 * (n + 7)], label='Real Gross Generation', alpha=1)
    #         plt.plot(df_power['GG_pred'][48 * n:48 * (n + 7)], label='Predicted Gross Generation', alpha=1)
    #         plt.plot(df_power['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
    #         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #         plt.xlabel('Time')
    #         plt.ylabel('PV Generation (kW)')
    #         plt.title('PV Generation prediction same Postal Code')
    #         plt.legend(loc='upper left')
    #         plt.show()
    # elif postcode != selected_PC:
    #     rel_mse_noPC.append(rel_mse)
    #     mae_noPC.append(mae)
    #     mape_gg_noPC.append(mape_gg)
    #     if (different_PC == 0) & (real_capacity >= 2):
    #         different_PC = 1
    #         n = 100
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(df_power['GG'][48 * n:48 * (n + 7)], label='Real Gross Generation', alpha=1)
    #         plt.plot(df_power['GG_pred'][48 * n:48 * (n + 7)], label='Predicted Gross Generation', alpha=1)
    #         plt.plot(df_power['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
    #         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #         plt.xlabel('Time')
    #         plt.ylabel('PV Generation (kW)')
    #         plt.title('PV Generation prediction different Postal Code')
    #         plt.legend(loc='upper left')
    #         plt.show()

print('MSE: ', np.mean(mse_list))
print('Relative MSE: ', np.mean(rel_mse_list))
print('Relative MSE same PC:', np.mean(rel_mse_PC))
print('Relative MSE different PC:', np.mean(rel_mse_noPC))
print('MAE: ', np.mean(mae_list))
print('MAE same PC:', np.mean(mae_PC))
print('MAE different PC:', np.mean(mae_noPC))
print('MAPE GG: ', np.mean(mape_gg_list))
print('MAPE GG same PC:', np.mean(mape_gg_PC))
print('MAPE GG different PC:', np.mean(mape_gg_noPC))
# print('MAPE Load: ', np.mean(mape_nc_list))
print('Aggregated GG MAPE: ', mean_absolute_percentage_error(df_aggregated[(df_aggregated.index.hour >= 7) & (df_aggregated.index.hour <= 17)]['GG_agg'],
                                                          df_aggregated[(df_aggregated.index.hour >= 7) & (df_aggregated.index.hour <= 17)]['GG_agg_pred']))
print('Aggregated NC MAPE: ', mean_absolute_percentage_error(df_aggregated['NC_agg'], df_aggregated['NC_agg_pred']))


# plt.figure(figsize=(10, 6))
# plt.plot(df_aggregated['GG_agg'][48 * n:48 * (n + 7)], label='Real Gross Generation', alpha=1)
# plt.plot(df_aggregated['GG_agg_pred'][48 * n:48 * (n + 7)], label='Predicted Gross Generation', alpha=1)
# # plt.plot(df_aggregated['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.xlabel('Time')
# plt.ylabel('PV Generation (kW)')
# plt.title('PV Generation Aggregated')
# plt.legend(loc='upper left')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.plot(df_aggregated['NC_agg'][48 * n:48 * (n + 7)], label='Real Net Consumpton', alpha=1)
# plt.plot(df_aggregated['NC_agg_pred'][48 * n:48 * (n + 7)], label='Predicted Net Consumpton', alpha=1)
# # plt.plot(df_aggregated['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.xlabel('Time')
# plt.ylabel('Net Consumption (kW)')
# plt.title('Net Consumption Aggregated')
# plt.legend(loc='upper left')
# plt.show()

n = 362
plt.figure(figsize=(10, 6))
plt.plot(df_power['GG'][48 * n:48 * (n + 2)], label='Real Gross Generation', alpha=1)
plt.plot(df_power['GG_pred'][48 * n:48 * (n + 2)], label='Predicted Gross Generation', alpha=1)
# plt.plot(df_aggregated['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.ylabel('PV Generation (kW)')
plt.title('PV Generation')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_power['BL'][48 * n:48 * (n + 2)], label='Real Baseline Load', alpha=1)
plt.plot(df_power['BL_pred'][48 * n:48 * (n + 2)], label='Predicted Baseline Load', alpha=1)
# plt.plot(df_aggregated['GG_error'][48 * n:48 * (n + 7)], label='Error', alpha=1)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.ylabel('Net Consumption (kW)')
plt.title('Net Consumption')
plt.legend(loc='upper left')
plt.show()


