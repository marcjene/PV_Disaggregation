# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt


# ======================================== FUNCTIONS ========================================

def select_customer(dataframe, id_column, customer):
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
    df_melted['hour'] = df_melted['hour'].apply(subtract_half_hour)
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
    dataframe_power[columns] = dataframe_power[columns] / granularity
    return dataframe_power


def calculate_consumptions(dataframe):
    """
    Function to calculate total consumption 'TC', which combines controlled loads and general consumption, and net
    consumption 'NC', which represents the readings at the Smart Meter level.
    :param dataframe: dataframe with columns 'CL', 'GC', and 'GG'
    :return: dataframe with columns 'TC' and 'NC', apart from 'CL', 'GC', and 'GG'
    """
    df_consumptions = dataframe.copy()

    # Controlled Load + General Consumption = Total Consumption
    df_consumptions['TC'] = df_consumptions['CL'] + df_consumptions['GC']
    # Net Consumption = Total Consumption - Gross Generation
    df_consumptions['NC'] = df_consumptions['TC'] - df_consumptions['GG']

    return df_consumptions


def plot_nc_gg(dataframe, starting_day, number_of_days):
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
    plt.plot(dataframe.index[48 * d + 1:48 * (d + n)], dataframe['NC'][48 * d + 1:48 * (d + n)], label='NC')
    plt.plot(dataframe.index[48 * d + 1:48 * (d + n)], dataframe['GG'][48 * d + 1:48 * (d + n)], label='GG')

    plt.xlabel('Datetime')
    plt.ylabel('Power [kW]')
    plt.title('Net Consumption and Gross Generation')
    plt.legend()
    # Format the x-axis tick labels to show d-m H:M
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m %H:%M'))
    plt.xticks(rotation=45)  # Rotate the x-axis tick labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from getting cut off
    plt.show()


def combination_1(dataframe, id_column='Customer', customer=1):
    """
    Select one customer, print the capacity and postcode and plot net consumption vs. gross generation and some other
    interesting data.
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

def count_customers(df, n_appearances):
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

    print(PC_count[PC_count['CustomerCount'] > n_appearances])

def capacity_interval(df, id_column='Customer', customer=1):
    """
    Calculates the minimum capacity of the PV system. Based on article Li2019. The value should be divided by the
    generation of the standard PV system - or somehow  corrected with meteorological values (TBD).
    :param df: Dataframe from Ausgrid dataset
    :param customer: Customer ID
    :return: minimum capacity
    """
    # Select customer to calculate the capacity
    df_customer = select_customer(df, id_column, customer)
    df_customer.reset_index(drop=True, inplace=True)

    # Obtain real capacity and drop columns
    PV_capacity, postcode = obtain_capacity_postcode(df_customer, 'Generator Capacity', 'Postcode')
    df_customer = df_customer.copy().drop(columns=['Customer', 'Generator Capacity', 'Postcode', 'Row Quality'])

    # Formatting dataframe to present consumption in columns
    df_format = format_df(df_customer)
    # Energy magnitudes to power
    df_power = energy_to_power(df_format, ['CL', 'GC', 'GG'], 0.5)
    # Calculate Total Consumption and Net Consumption - We'll use the net consumption
    df_BTM = calculate_consumptions(df_power)

    # Maximum generation is printed to use as benchmark
    print('Maximum generation: ', df_BTM['GG'].max(), 'kW. At', df_BTM['GG'].idxmax())

    # Minimum capacity calculation
    C_min = abs(df_BTM['NC'].min())
    # Day with minimum net consumption
    date_min = df_BTM['NC'].idxmin()
    print('Minimum net consumption at: ', date_min)

    # Maximum capacity calculation
    # Base load calculated during night hours - Hypothesis: 1 am to 6 am
    # Can it be calculated following a different approach? - Maybe find a daily load curve without generation
    start_time = date_min.replace(hour=1, minute=0, second=0)
    end_time = date_min.replace(hour=6, minute=0, second=0)
    selected_data = df_BTM[(df_BTM.index >= start_time) & (df_BTM.index <= end_time)]
    print(df_BTM['NC'].loc[start_time])
    print(df_BTM['NC'].loc[end_time])
    avg_night = selected_data['NC'].mean()

    C_max = abs(df_BTM['NC'].min()) + avg_night
    return C_min, C_max, PV_capacity


# ======================================== MAIN ========================================

dataset_ausgrid = pd.read_csv('data/Ausgrid/2012-2013 Solar home electricity data v2.csv', header=1)

# print(dataset_ausgrid)
# combination_1(dataset_ausgrid, 'Customer', 1)

# count_customers(dataset_ausgrid, 5)

C_min, C_max, PV_capacity = capacity_interval(dataset_ausgrid, customer=3)

print(C_min, C_max, PV_capacity)
