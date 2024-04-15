# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ======================================== FUNCTIONS ========================================

def select_customer(dataframe, customer, id_column='Customer'):
    """
    Function to select a specific customer of the dataframe.

    :param dataframe: df with all the customers
    :param id_column: name of the column including the customer id
    :param customer: int or string of the customer to select
    :return: sliced dataframe including the customer selected
    """
    dataframe_customer = dataframe.copy()
    return dataframe_customer[dataframe_customer[id_column] == customer]


def obtain_capacity_postcode(dataframe_customer, capacity_column, pc_column):
    """
    Function to obtain capacity and postcode of a specific customer.

    :param dataframe_customer: df with all the customers
    :param capacity_column: name of the column including the capacity
    :param pc_column: name of the column including the postcode
    :return: capacity and postcode
    """
    return dataframe_customer[capacity_column][0], dataframe_customer[pc_column][0]


def subtract_half_hour(time_str):
    """
    Function to subtract 30 minutes from the 'hour' column (i.e. in the dataset, the column '00:30' indicates the
    consumption from 00:00 to 00:30. The column name should be changed to '00:00')

    :param time_str: string indicating the hour of the consumption (e.g. '00:30')
    :return: the time string with 30 minutes less to indicate when the consumption starts
    """
    hour, minute = map(int, time_str.split(':'))
    if minute >= 30:
        minute -= 30
    else:
        minute += 30
        hour -= 1
        if hour == -1:
            hour = 23
    return f"{hour}:{minute:02}"


def format_df(dataframe):
    """
    Function to apply all the formatting necessary to have time series values in a single column
    :param dataframe: dataframe obtained from the Ausgrid dataset
    :return: dataframe formatted with the time series values in a single column
    """
    df_melted = pd.melt(dataframe, id_vars=['Consumption Category', 'date'], var_name='hour',
                        value_name='Consumption')

    # Apply the function to the 'hour' column
    df_melted['hour'] = df_melted['hour'].apply(subtract_half_hour)  # Be aware that weather input is averaged at the end.
    # Concatenate date and hour to create the new 'datetime' column
    df_melted['datetime'] = df_melted['date'] + ' ' + df_melted['hour']
    # Convert the 'datetime' column to pandas datetime type
    df_melted['datetime'] = pd.to_datetime(df_melted['datetime'], format='%d/%m/%Y %H:%M')

    # Drop the original 'date' and 'hour' columns if needed
    df_melted.drop(columns=['date', 'hour'], inplace=True)
    # Sorting the DataFrame by 'datetime' if needed
    df_melted.sort_values(by='datetime', inplace=True)

    # Pivot the DataFrame to get 'Consumption Category' values as columns
    df_consumptions = df_melted.pivot(index='datetime', columns='Consumption Category', values='Consumption')
    df_consumptions.columns.name = None
    return df_consumptions


def energy_to_power(dataframe, columns, granularity):
    """
    Function to convert energy values (in kWh) to power (in kW).
    :param dataframe: dataframe to apply the conversions to
    :param columns: list of column names where electrical energy magnitudes have to be transformed to power
    :param granularity: granularity to calculate the power (in hours)
    :return: dataframe with magnitudes in kW instead of kWh
    """
    dataframe_power = dataframe.copy()
    if 'CL' not in dataframe_power.columns:
        dataframe_power['CL'] = 0
    dataframe_power[columns] = dataframe_power[columns] / granularity
    return dataframe_power


def calculate_consumptions(dataframe):
    """
    Function to calculate total consumption 'TL', which combines controlled loads and general consumption, and net
    consumption 'NC', which represents the readings at the Smart Meter level.
    :param dataframe: dataframe with columns 'CL', 'GC', and 'GG'
    :return: dataframe with columns 'TL' and 'NC', apart from 'CL', 'GC', and 'GG'
    """
    df_consumptions = dataframe.copy()

    # Controlled Load + General Consumption = Baseline Load
    df_consumptions['TL'] = df_consumptions['CL'] + df_consumptions['GC']
    # Net Consumption = Total Consumption - Gross Generation
    df_consumptions['NC'] = df_consumptions['TL'] - df_consumptions['GG']

    return df_consumptions


def calculate_st_gen(dataframe, capacity):
    df = dataframe.copy()
    df['stGG'] = df['GG']/capacity
    return df


def plot_nc_gg(dataframe, starting_day, number_of_days, GG=True):
    """
    Function to plot Net Consumption vs. Gross Generation.
    :param dataframe: Dataframe with NC and GG as time series
    :param starting_day: first day to plot (0 is 01-07-2012)
    :param number_of_days: number of days to plot
    :return: plot
    """
    # Plotting 'GC' and 'GG' columns superposed
    d = starting_day  # starting day to plot
    n = number_of_days  # number of days to plot
    if GG:
        plt.plot(dataframe.index[48 * d + 1:48 * (d + n)], dataframe['GG'][48 * d + 1:48 * (d + n)], label='GG',
                 color='orange')
        plt.plot(dataframe.index[48 * d + 1:48 * (d + n)], dataframe['TL'][48 * d + 1:48 * (d + n)], label='TL',
                 color='green')
    else:
        plt.plot(dataframe.index[48 * d + 1:48 * (d + n)], dataframe['NC'][48 * d + 1:48 * (d + n)], label='NC')

    plt.xlabel('Datetime')
    plt.ylabel('Power [kW]')
    if GG:
        plt.title('Gross Generation and Baseline Load')
    else:
        plt.title('Net Consumption')
    plt.legend()

    # Format the x-axis tick labels to show d-m H:M
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m %H:%M'))
    plt.xticks(rotation=45)  # Rotate the x-axis tick labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from getting cut off
    plt.show()


def combination_1(dataframe, id_column='Customer', customer=1):
    """
    Select one customer, print the capacity and postcode and plot net consumption vs. gross generation and some other
    interesting input.
    :param dataframe: Dataframe from Ausgrid dataset
    :param id_column: Column with customers IDs
    :param customer: Customer ID to be selected
    :return: Capacity, postcode, plot of NC and GG, maximum generation, minimum net consumption
    """
    df_customer = select_customer(dataframe, id_column, customer)
    df_customer.reset_index(drop=True, inplace=True)

    PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')

    df_customer = df_customer.copy().drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])

    # Formatting dataframe to present consumption in columns
    df_format = format_df(df_customer)

    # Energy magnitudes to power
    df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)

    # Calculate Total Consumption and Net Consumption
    df_BTM = calculate_consumptions(df_power)
    # print(df_BTM)

    # Plot Net Consumption vs. Gross Generation
    plot_nc_gg(df_BTM, 1, 7)

    print('PV Capacity: ', PV_capacity)
    print('Postcode: ', postcode)

    # Maximum generation
    print('Maximum generation: ', df_BTM['GG'].max(), 'kW. At', df_BTM['GG'].idxmax())
    print('Minimum net consumption: ', df_BTM['NC'].min(), 'kW. At', df_BTM['NC'].idxmin())
    print('GG when minimum NC: ', df_BTM.loc[pd.Timestamp(df_BTM['NC'].idxmin()), 'GG'])


def count_customers(df, n_appearances=1):
    """
    Return a list of postal codes with more than n customers in the dataset
    :param df: Dataframe from Ausgrid dataset
    :param n_appearances: Number of customers from every PC
    :return: dataframe with PCs and number of customers from that region in the dataframe
    """
    # Group by 'Postcode' and count unique 'Customer' values
    PC_count = df.groupby('Postcode')['Customer'].nunique().reset_index()
    # Rename the column to 'CustomerCount'
    PC_count = PC_count.rename(columns={'Customer': 'CustomerCount'})
    PC_filtered = PC_count[PC_count['CustomerCount'] > n_appearances]
    PC_sorted = PC_filtered.sort_values(by='CustomerCount',ascending=False).reset_index(drop=True)
    return PC_sorted


# def capacity_interval(df_customer):
#     """
#     Calculates the minimum capacity of the PV system. Based on article Li2019. The value should be divided by the
#     generation of the standard PV system - or somehow  corrected with meteorological values (TBD).
#     :param df_customer: Dataframe
#     :return: capacity interval
#     """
#     # Obtain real capacity and drop columns
#     PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')
#     df_customer = df_customer.copy().drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])
#     # Formatting dataframe to present consumption in columns
#     df_format = format_df(df_customer)
#     # Energy magnitudes to power
#     if 'CL' in df_format.columns:
#         df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
#     else:
#         df_power = energy_to_power(df_format, ['GC', 'GG'], 0.5)
#
#     # Calculate Total Consumption and Net Consumption - We'll use the net consumption
#     df_BTM = calculate_consumptions(df_power)
#
#     # Maximum generation is printed to use as benchmark
#     # print('Maximum generation: ', df_BTM['GG'].max(), 'kW. At', df_BTM['GG'].idxmax())
#
#     # Minimum capacity calculation
#     C_min = - df_BTM['NC'].min()
#     # Day with minimum net consumption
#     date_min = df_BTM['NC'].idxmin()
#     # print('Minimum net consumption at: ', date_min)
#
#     # Maximum capacity calculation - Option 1
#     # Base load calculated during night hours - Hypothesis: 0h to 6h
#     selected_data = df_BTM[(df_BTM.index.hour >= 0) | (df_BTM.index.hour <= 6)]
#     avg_night = selected_data['NC'].mean()
#     C_max = C_min + avg_night
#
#     return C_min, C_max, PV_capacity


def preprocessing(df_customer):
    df_customer = df_customer.copy().drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])
    # Formatting dataframe to present consumption in columns
    df_format = format_df(df_customer)
    # Energy magnitudes to power
    if 'CL' in df_format.columns:
        df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
    else:
        df_power = energy_to_power(df_format, ['GC', 'GG'], 0.5)
    return df_power


def features(df):
    # Extract array of daily min
    min_array = np.array([])
    diff_array = np.array([])
    for day in np.unique(df.index.date):
        df_day = df[df.index.date == day]
        daily_min = df_day['NC'].min()
        min_array = np.append(min_array, daily_min)
        diff = df_day['NC'].max() - daily_min
        diff_array = np.append(diff_array, diff)
    min_array = min_array[~np.isnan(min_array)]

    # Minimum capacity calculation (negative) - Update: done with quantile 1% to avoid possible outliers
    # C_min = df['NC'].min()
    C_min = np.quantile(min_array, 0.01)
    # # Day with minimum net consumption
    # date_min = df['NC'].idxmin()
    # print('Minimum net consumption at: ', date_min)

    # Maximum capacity calculation - Option 1
    # Base load calculated during night hours - Hypothesis: 0h to 6h
    selected_data = df[(df.index.hour >= 0) | (df.index.hour <= 6)]
    avg_night = selected_data['NC'].mean()
    C_max_1 = C_min - avg_night

    # Maximum capacity calculation - Option 2
    # Base load calculated with all the positive readings
    selected_data = df[df['NC'] >= 0]
    avg_positive = selected_data['NC'].mean()
    C_max_2 = C_min - avg_positive

    # Maximum capacity calculation - Option 3
    # Base load calculated with all the positive readings, but using median
    selected_data = df[df['NC'] >= 0]
    avg_positive = selected_data['NC'].median()
    C_max_3 = C_min - avg_positive

    diff_array = diff_array[~np.isnan(diff_array)]
    mean_min = np.mean(min_array)
    median_min = np.median(min_array)
    std_min = np.std(min_array)
    min_25q = np.quantile(min_array, 0.25)
    min_75q = np.quantile(min_array, 0.75)
    mean_diff = np.mean(diff_array)
    median_diff = np.median(diff_array)
    diff_75q = np.quantile(diff_array, 0.75)
    features_dict = {'Cmin': [C_min], 'Cmax_1': [C_max_1], 'Cmax_2': [C_max_2], 'Cmax_3': [C_max_3],
                     'Mean_min': [mean_min], 'Median_min': [median_min], 'Std_min': [std_min], 'Min_25q': [min_25q],
                     'Min_75q': [min_75q], 'Mean_diff': [mean_diff], 'Median_diff': [median_diff],
                     'Diff_75q': [diff_75q]}
    return features_dict


def cluster_generation(df, list_customers, id_column='Customer'):
    df = df.copy()
    df_day_type = pd.DataFrame()

    for customer in list_customers:
        print(customer)
        df_customer = select_customer(df, id_column, customer)
        df_customer.reset_index(drop=True, inplace=True)

        # Obtain real capacity and drop columns
        PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')
        df_customer = df_customer.copy().drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])

        # Formatting dataframe to present consumption in columns
        df_format = format_df(df_customer)
        # Energy magnitudes to power
        if 'CL' in df_format.columns:
            # In some customers, the 'CL' column disappears for some days
            df_format['CL'].fillna(0, inplace=True)
            df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
        else:
            df_power = energy_to_power(df_format, ['GC', 'GG'], 0.5)

        # Calculate Total Consumption and Net Consumption - We'll use the net consumption
        df_BTM = calculate_consumptions(df_power)
        df_BTM.reset_index(inplace=True)
        df_BTM['datetime'] = pd.to_datetime(df_BTM['datetime'])
        df_cluster = pd.DataFrame()

        if len(df_BTM['datetime'].dt.date.unique()) < 365:
            continue
        else:
            for day in df_BTM['datetime'].dt.date.unique():
                df_day = df_BTM[df_BTM['datetime'].dt.date == day]
                df_day = df_day[(df_day['datetime'].dt.hour >= 6) & (df_day['datetime'].dt.hour <= 20)]
                mean_day = df_day['GG'].mean()
                max_day = df_day['GG'].max()
                min_day = df_day['GG'].min()
                std_day = df_day['GG'].std()
                row_day = pd.DataFrame({'Day': [day], 'Mean': [mean_day], 'Max': [max_day], 'Min': [min_day],
                                        'Std': [std_day]})
                df_cluster = pd.concat([df_cluster, row_day], ignore_index=True)

        scaler = StandardScaler()

        X = df_cluster.set_index('Day')
        X_scaled = scaler.fit_transform(X)

        clusters_k = 4

        # Perform K-Means clustering with the chosen number of clusters
        kmeans = KMeans(n_clusters=clusters_k, random_state=1)
        kmeans.fit(X_scaled)

        # Add cluster labels to the DataFrame
        df_cluster['Cluster'] = kmeans.labels_

        cluster_sorted = df_cluster.groupby('Cluster')['Mean'].mean().sort_values(ascending=True).reset_index()

        type_day_dict = {}
        type_day_list = ['A', 'B', 'C', 'D']
        for i, row in cluster_sorted.iterrows():
            type_day_dict[row['Cluster']] = type_day_list[i]

        df_cluster['Cluster'] = df_cluster['Cluster'].replace(type_day_dict)
        df_cluster.rename({'Cluster': 'C' + str(customer)}, axis=1, inplace=True)
        df_cluster.drop(columns=['Mean', 'Max', 'Min', 'Std'], inplace=True)

        if customer == list_customers[0]:
            df_day_type = df_cluster.copy()
        else:
            df_day_type = pd.merge(df_day_type, df_cluster, on='Day', how='outer')

    df_day_type.set_index('Day', inplace=True)
    overall_days = df_day_type.T.apply(lambda x: x.value_counts())
    print(overall_days.T.sum())
    return overall_days
