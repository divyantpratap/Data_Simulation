import configparser
import logging
import os
import csv
import json
import uuid
import time
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')
import boto3
import requests
import streamlit as st
import warnings
# Set the paths for the output folders
heat_folder = "./heatmap_data"
merged_older = "./created_user_raw_data"

# Display the output folder locations



def main():
    st.title("Simulation Configuration")

    # Set custom styles
    st.markdown("""
        <style>
            body {
                background-color: white;
            }
            input, select {
                background-color: #F0F0F0;
            }
            .stButton>button {
                width: 100%;
            }
            .btn-success {
                background-color: #28a745;
            }
            .btn-danger {
                background-color: #dc3545;
            }
            .big-gap {
                height: 2rem;
            }
            .col-gap {
                width: 10%;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, gap, col2 = st.columns([45, 20, 45])

    with col1:
        donor_appliance = st.selectbox("Donor appliance*:", ["EV", "Pool Pump", "Timed Water heater", "AC", "Space Heating"])
        hours_of_shift = st.number_input("Hours of shift*:", min_value=0, max_value=12, value=0)
        acceptor_uuid_file = st.text_input("Acceptor UUID list*:", "acceptor_uuid.csv")
        

    with col2:
        randomization = st.selectbox("Randomization variable*:", ["Fixed Hours Shift", "Normal Distribution Shift", "Uniform Distribution Shift"])
        donor_uuid_file = st.text_input("Donor UUID list (optional):", "")
        s3_disagg_output_location = st.text_input("S3 donor location (optional):", "")
        raw_data_file = ""
        heatmap_folder =  ""

    button_col1, button_col2 = st.columns(2)

    with button_col1:
        run_simulation_button = st.button("Run Simulation", key="run_simulation")

    with button_col2:
        close_button = st.button("Close", key="close")

    st.markdown("""
        <style>
            [data-testid="stButton"][aria-describedby="run_simulation"] {
                background-color: #28a745;
            }
            [data-testid="stButton"][aria-describedby="close"] {
                background-color: #dc3545;
            }
        </style>
    """, unsafe_allow_html=True)

    if run_simulation_button:
        donor_appliance_mapping = {
            'EV': 'ev',
            'Pool Pump': 'pp',
            'Timed Water heater': 'timed_wh',
            'AC': 'ac',
            'Space Heating': 'sh'
        }
        donor_appliance = donor_appliance_mapping[donor_appliance]
        randomization_mapping = {
                'Fixed Hours Shift': '0',
                'Normal Distribution Shift': '1',
                'Uniform Distribution Shift': '2'
            }
        randomization = randomization_mapping[randomization]
        
        # Call your simulation function with the user's input values
        result = run_simulation(donor_appliance, hours_of_shift, acceptor_uuid_file, randomization, donor_uuid_file, s3_disagg_output_location, raw_data_file, heatmap_folder)
        st.write("Simulation results:", result)
        st.write("Output folder locations:")
        st.write(f"UUID mapping file: {os.getcwd()}/uuid_mapping.csv")
        #st.write(f"Donor data folder: file://{os.getcwd()}/donor_data/")
        st.write(f"Simulated User Raw Data Location: {os.getcwd()}/created_user_raw_data")
        st.write(f"Simulated User Heatmap folder: {os.getcwd()}/heatmap_data")

def run_simulation(donor_appliance, hours_of_shift, acceptor_uuid_file, randomization, donor_uuid_file, s3_disagg_output_location, raw_data_file, heatmap_folder):
    # Configure the logger
    logging.basicConfig(filename='data_simulation.log', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    print(randomization)

    client_id = {
        'dev': 'admin',
        'ds': 'admin',
        'nonprodqa': 'admin',
        'prod-na': 'sahanam@bidgely.com',
        'prod-eu': 'sahanam@bidgely.com',
        'prod-jp': '',
        'prod-ca': 'nisha@bidgely.com',
        'prod-na-2': 'sahanam@bidgely.com',
        'preprod-na': 'admin',
        'qaperfenv': 'admin',
        'uat': 'admin',
    }

    client_key = {
        'dev': 'admin',
        'ds': 'admin',
        'nonprodqa': 'admin',
        'prod-na': 'L2FVrraL',
        'prod-eu': 'UYWiUGx8',
        'prod-jp': '',
        'prod-ca': 'pNJ1y3na',
        'prod-na-2': 'FHoD0RFq',
        'preprod-na': 'admin',
        'qaperfenv': 'admin',
        'uat': 'admin',
    }


    def get_env_properties(env):
        """
        Parameters:
            env             (str)               : Environment for which variables need to be extracted
        Returns:
            properties      (dict)              : Dictionary containing basic information
        """

        env_properties = {
            'dev': dict({
                'protocol': 'https://',
                'primary': 'devapi.bidgely.com',
                'aws_region': 'us-west-2'
            }),

            'ds': dict({
                'protocol': 'http://',
                'primary': 'dspyapi.bidgely.com',
                'aws_region': 'us-east-1'
            }),

            'nonprodqa': dict({
                'protocol': 'https://',
                'primary': 'nonprodqaapi.bidgely.com',
                'aws_region': 'us-west-2'
            }),
            'prod-na': dict({
                'protocol': 'https://',
                'primary': 'napyapi.bidgely.com',
                'aws_region': 'us-east-1'
            }),
            'prod-eu': dict({
                'protocol': 'https://',
                'primary': 'eupyapi.bidgely.com',
                'aws_region': 'eu-central-1'
            }),
            'prod-jp': dict({
                'protocol': 'https://',
                'primary': 'jppyapi.bidgely.com',
                'aws_region': 'ap-northeast-1'
            }),
            'prod-ca': dict({
                'protocol': 'https://',
                'primary': 'capyapi.bidgely.com',
                'aws_region': 'ca-central-1'
            }),
            'prod-na-2': dict({
                'protocol': 'https://',
                'primary': 'na2pyapi.bidgely.com',
                'aws_region': 'us-east-1'
            }),
            'preprod-na': dict({
                'protocol': 'https://',
                'primary': 'napreprodapi.bidgely.com',
                'aws_region': 'us-east-1'
            }),
            'qaperfenv': dict({
                'protocol': 'http://',
                'primary': 'awseb-e-i-awsebloa-1jk42nlshi8yb-2130246765.us-west-2.elb.amazonaws.com',
                'aws_region': 'us-west-2'
            }),
            'uat': dict({
                'protocol': 'https://',
                'primary': 'uatapi.bidgely.com',
                'aws_region': 'us-west-2'
            }),
        }

        env_prop = env_properties.get(str.lower(env))
        return env_prop


    def generating_access_token(env):
        """
        Generates an access token for a given environment.
        
        Returns:
            dict: A dictionary containing the generated access token under the key 'env_token'.
        """
        
        token_params = get_env_properties(env)
        protocol = token_params['protocol']
        primary_api = token_params['primary']
        client_id_value = client_id.get(env)
        client_secret_value = client_key.get(env)
        url = '{0}{1}:{2}@{3}/oauth/token?grant_type=client_credentials&scope=all'.format(protocol, client_id_value,
                                                                                        client_secret_value,
                                                                                        primary_api)
        response = requests.get(url)
        message_load = response.json()
        access_token = message_load.get('access_token')
        env_token = {'env_token': access_token}
        return env_token


    def compareSamplingRate(donor_timestamps, acceptor_timestamps):
        """
        Calculates the absolute difference between the sampling rates of two datasets.
        """
        return abs(donor_timestamps - acceptor_timestamps)


    def get_disagg_data(path, donor_appliance):
        """
        Reads in a disaggregation output file and extracts the data for a specified donor appliance.

        Args:
            path (str): The path to the disaggregation output file.
            donor_appliance (str): The name of the donor appliance to extract data for.

        Returns:
            pandas.DataFrame: A DataFrame containing the timestamp and output data for the specified donor appliance.
        """

        disagg_output_home1 = pd.read_csv(path)
        disagg_output_home1 = disagg_output_home1.rename(
            columns={'epoch': 'timestamp', donor_appliance: 'appliance_output'})
        return disagg_output_home1.loc[:, ['timestamp', 'appliance_output']]


    def check_folder(folder_path, default_folder_name):
        """
        Checks if a folder path exists, and creates it if it does not.
        If the input `folder_path` is empty or `None`, the function will use the `default_folder_name` instead.
        
        Args:
            folder_path (str): The path of the folder to check or create.
            default_folder_name (str): The default name of the folder to use if `folder_path` is empty or `None`.
        """
    
        if folder_path == '' or folder_path == None:
            folder_path = default_folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path


    def getrawData(env, uuid, start, end):
        """
        Retrieves raw data from the API for a specified time range.

        Args:
            env (str): The environment to use for authentication.
            uuid (str): The user UUID to retrieve data for.
            start (str): The start time of the data range in ISO format.
            end (str): The end time of the data range in ISO format.

        Returns:
            dict or None: A dictionary containing the raw data from the API, or None if there is no data available.
        """
        token_params = get_env_properties(env)
        protocol = token_params['protocol']
        primary_api = token_params['primary']
        raw_url = '{0}{1}/streams/users/{2}/homes/1/gws/2/gb.json?t0={3}&t1={4}'.format(
            protocol, primary_api, uuid, start, end)
        env_token = generating_access_token(env).get('env_token')
        header = {"Authorization": f"Bearer {env_token}"}
        response = requests.get(raw_url, headers=header)
        if len(response.text) > 0:
            raw_api = json.loads(response.text)
            return raw_api
        return None


    def appliance_detection_instances(df):
        """
        Detects instances of appliance activity in donor user dataframe.

        Args:
            df (pandas.DataFrame): The DataFrame to detect appliance instances in.

        Returns:
            list: A list of tuples representing the start and end indices of each appliance instance in the DataFrame.
        """
        
        instances = []
        in_region = False
        start_idx = None
        end_idx = None

        for idx, row in df.iterrows():
            if row['appliance_output'] != 0:
                if not in_region:
                    start_idx = idx
                    in_region = True
            else:
                if in_region:
                    end_idx = idx - 1
                    if instances and (start_idx - instances[-1][1] - 1) <= 8:
                        instances[-1] = (instances[-1][0], end_idx)
                    else:
                        instances.append((start_idx, end_idx))
                    in_region = False

        if in_region:
            end_idx = len(df) - 1
            instances.append((start_idx, end_idx))

        return instances

    def generate_shift_values(num_instances, randomization, points_to_shift, sigma):
        if randomization == '1':
            shift_values = truncnorm(a=-points_to_shift/sigma, b=points_to_shift/sigma, scale=sigma).rvs(size=num_instances)
        elif randomization == '2':
            shift_values = np.random.randint(-points_to_shift, points_to_shift+1, size=num_instances)
        else:
            raise ValueError("Wrong randomization method used")
        return shift_values

    def shifting_disagg_data(df, hours_of_shift, randomization, donor_sampling_rate):
        if randomization == '0':
            shifted_df = df.copy()
            shifted_df['timestamp'] = shifted_df['timestamp'] + (hours_of_shift * 60 * 60)
            shifted_df.rename(columns={'appliance_output': 'value'}, inplace=True)
        else:
            shifted_df = df.copy()
            shifted_df.rename(columns={'appliance_output': 'value'}, inplace=True)
            instances = appliance_detection_instances(df)
            sampling_factor = int(3600 // donor_sampling_rate)
            points_to_shift = hours_of_shift * sampling_factor
            sigma = points_to_shift // 3
            num_instances = len(instances)
            shift_values = generate_shift_values(num_instances, randomization, points_to_shift, sigma)
            shift_values = (shift_values / sampling_factor).tolist()

            non_overlapping_instances = []
            for i, (start_idx, end_idx) in enumerate(instances):
                if randomization == '1' or randomization == '2':
                    shift_value = shift_values[i]
                    shifted_df['timestamp'][start_idx:end_idx + 1] += shift_value * 60 * 60
                else:
                    print("Wrong randomization method used")

                # Check if the instance overlaps with the previous instance
                if i > 0 and shifted_df.loc[non_overlapping_instances[-1][1], 'timestamp'] >= shifted_df.loc[start_idx, 'timestamp']:
                    continue

                non_overlapping_instances.append((start_idx, end_idx))

            # Create a new DataFrame with non-overlapping instances only
            non_overlapping_df = pd.DataFrame()
            for start_idx, end_idx in non_overlapping_instances:
                non_overlapping_df = pd.concat([non_overlapping_df, shifted_df.iloc[start_idx:end_idx + 1]], ignore_index=True)

            shifted_df = non_overlapping_df

        shifted_df = pd.merge(shifted_df, df, on='timestamp', how='right')
        shifted_df = shifted_df.drop(columns='appliance_output')
        shifted_df = shifted_df.fillna(0)
        return shifted_df



    def create_new_user(disagg_output, raw_energy_data):
        """
        Creates a new user DataFrame by merging disaggregation output and raw energy data.

        Args:
            disagg_output (pandas.DataFrame): The DataFrame of disaggregation output data.
            raw_energy_data (pandas.DataFrame): The DataFrame of raw energy data.

        Returns:
            pandas.DataFrame: The merged DataFrame containing the timestamp and total output data.
        """
        merged_data = pd.merge(disagg_output, raw_energy_data,
                            on='timestamp', how='right')
        if 'appliance_output' in merged_data.columns:
            merged_data['total_output'] = merged_data['appliance_output'] + merged_data['value']
            merged_data.drop(['appliance_output', 'value', 'duration'],
                            axis=1, inplace=True)
            merged_data.rename(columns={'total_output': 'value'}, inplace=True)
        else:
            merged_data['total_output'] = merged_data['value_x'] + merged_data['value_y']
            columns_to_drop = ['value_x', 'value_y', 'datetimenew', 'date', 'time']
            for col in columns_to_drop:
                if col in merged_data.columns:
                    merged_data.drop(col, axis=1, inplace=True)
            merged_data.rename(columns={'total_output': 'value'}, inplace=True)

        return merged_data


    def create_heatmap(data, title, timezone, ax=None, min_value=None, max_value=None):
        """
        Creates a heatmap plot of time-series data.

        The input `data` DataFrame should have a 'timestamp' column and a 'value' column.

        Args:
            data (pandas.DataFrame): The DataFrame of time-series data to plot.
            title (str): The title of the heatmap plot.
            timezone (str): The timezone to use for the x-axis labels.
            ax (matplotlib.axes.Axes, optional): The Axes object to draw the plot onto. If not provided, a new figure will be created.
            min_value (float, optional): The minimum value for the color scale.
            max_value (float, optional): The maximum value for the color scale.
        """
        data["datetimenew"] = pd.to_datetime(data["timestamp"], unit='s')
        data['date'] = data.datetimenew.dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert(f'{timezone}')
        data['time'] = data['date'].dt.time
        data['date'] = data['date'].dt.date
        data = data.fillna(0)
        data = data.pivot_table(index='date', columns=['time'], values='value', aggfunc=sum)
        sns.heatmap(data, cmap='jet', cbar=True, xticklabels=4, yticklabels=30, ax=ax, vmin=min_value, vmax=max_value)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(data.columns))[::4])
        ax.set_xticklabels([data.columns[x].strftime('%H')
                        for x in np.arange(len(data.columns))[::4]])
        ax.set_yticks(np.arange(len(data.index))[::30])
        ax.set_yticklabels([data.index[x].strftime('%b -%Y')
                        for x in np.arange(len(data.index))[::30]])
        ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
        plt.xticks(rotation=90)


    def move_to_s3(local_file_path, bucket_name, s3_file_path):
        # create an S3 client
        s3 = boto3.client('s3')

        # upload file to S3
        s3.upload_file(local_file_path, bucket_name, s3_file_path)


    def check_timestamp_more_than_six_months(start_timestamp, last_timestamp):
        """
        Checks if the difference between two timestamps is within six months.

        The timestamps should be in Unix format.

        Args:
            start_timestamp (int): The starting timestamp.
            last_timestamp (int): The ending timestamp.

        Returns:
            bool: True if the difference is within six months, False otherwise.
        """
        # Calculate the time difference in seconds
        timestamp_difference = last_timestamp - start_timestamp

        # Calculate the number of seconds in six months (roughly)
        six_months_in_seconds = 6 * 30 * 24 * 60 * 60

        # Check if the difference is within six months
        if timestamp_difference >= six_months_in_seconds:
            return True
        else:
            return False


    def get_last_billing_cycle(uuid):
        """
        Retrieves the last billing cycle data for a user having no data for same year as of donor uuid.

        Args:
            uuid (str): The user ID to retrieve billing cycle data for.

        Returns:
            tuple: A tuple containing billing cycle value. If an error occurs or no billing cycles are found, None is returned.
        """
        endpoint = "https://dsapi.bidgely.com/billingdata/users/{}/homes/1/billingcycles"
        api = endpoint.format(uuid)
        headers = {"Authorization": "Bearer 5bf55250-c832-4f80-b0c0-0f5423e23732"}

        response = requests.get(api, headers=headers)

        if response.status_code == 200:
            try:
                billing = response.json()

                # Fetch the last billing cycle only
                if len(billing) > 0:
                    last_billing_cycle = billing[-1]
                    return uuid, last_billing_cycle["key"], last_billing_cycle["value"]
                else:
                    print("No billing cycles found.")
                    return None
            except:
                print("Error occurred while parsing the JSON response")
                return None
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    def get_metadata(env, uuid, access_token):
        endpoint = f"https://{env}api.bidgely.com/meta/users/{uuid}/homes/1"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            try:
                metadata = response.json()
                
                # Save metadata to CSV
                folder_path = 'metadata'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                metadata_df = pd.DataFrame([metadata], columns = metadata.keys())
                metadata_df.to_csv(os.path.join(folder_path, f"{uuid}_metadata.csv"), index=False)
                
                timezone = metadata.get('timezone')
                country = metadata.get('country')
                
                return timezone, country
            except Exception as e:
                print(e)
                print(f"Failed to parse metadata for uuid: {uuid}")
        return None, None
    
    def create_different_year_user(disagg_output, raw_energy_data):
        
        """
        Creates a new user DataFrame by merging disaggregation output and raw energy data based on day of the year.

        Args:
            disagg_output (pandas.DataFrame): The DataFrame of disaggregation output data.
            raw_energy_data (pandas.DataFrame): The DataFrame of raw energy data.

        Returns:
            pandas.DataFrame: The merged DataFrame containing the timestamp and total output data.

        """
        # Convert the timestamps to datetime objects
        disagg_output['timestamp'] = pd.to_datetime(disagg_output['timestamp'], unit='s')
        raw_energy_data['timestamp'] = pd.to_datetime(raw_energy_data['timestamp'], unit='s')

        # Add a new column to both dataframes to store the day of the year
        disagg_output['day_of_year'] = disagg_output['timestamp'].dt.dayofyear
        raw_energy_data['day_of_year'] = raw_energy_data['timestamp'].dt.dayofyear

        # Merge the dataframes on the 'day_of_year' column
        merged_data = pd.merge(disagg_output, raw_energy_data,on='day_of_year', how='right')

        if 'appliance_output' in merged_data.columns:
            merged_data['total_output'] = merged_data['appliance_output'] + merged_data['value']
            merged_data.drop(['appliance_output', 'value', 'duration'],
                            axis=1, inplace=True)
            merged_data.rename(columns={'total_output': 'value'}, inplace=True)
        else:
            merged_data['total_output'] = merged_data['value_x'] + merged_data['value_y']
            columns_to_drop = ['value_x', 'value_y', 'datetimenew', 'date', 'time']
            for col in columns_to_drop:
                if col in merged_data.columns:
                    merged_data.drop(col, axis=1, inplace=True)
            merged_data.rename(columns={'total_output': 'value'}, inplace=True)

        # Drop the 'day_of_year' and 'timestamp_x' columns from the merged data
        merged_data.drop(['day_of_year', 'timestamp_x'], axis=1, inplace=True)

        # Convert the 'timestamp_y' column to epoch format and rename it to 'timestamp'
        merged_data['timestamp'] = merged_data['timestamp_y'].astype(int) // 10**9
        merged_data.drop(['timestamp_y'], axis=1, inplace=True)
        merged_data.rename(columns={'timestamp': 'timestamp'}, inplace=True)
        # Reorder the columns in the desired format
        merged_data = merged_data[['timestamp', 'value']]
        return merged_data


    def downsample(donor_data, downsample_factor,donor_sampling_rate):
        """
        Downsamples the given raw_data by the downsample_factor.

        :param raw_data: pandas DataFrame containing the raw data
        :param downsample_factor: int, factor by which to downsample the raw_data
        :return: pandas DataFrame containing the downsampled data
        """

        # Convert the epoch timestamp to a datetime object
        donor_data['timestamp'] = pd.to_datetime(donor_data['timestamp'], unit='s')

        # Set the timestamp column as the index
        donor_data.set_index('timestamp', inplace=True)

        # Downsample the data using the specified downsample_factor
        resampling_frequency = f"{downsample_factor * donor_sampling_rate}S"
        downsampled_data = donor_data.resample(resampling_frequency).sum()

        # Reset the index and convert the datetime index back to epoch timestamp
        downsampled_data.reset_index(inplace=True)
        downsampled_data['timestamp'] = downsampled_data['timestamp'].astype('int64') // 10**9

        return downsampled_data


    def generate_heatmap(shifted_disagg_output, raw_energy_data, donor_uuid, uuid, acceptor_sampling_rate, timezone,heatmap_folder,year_change,donor_sampling_rate):
        """
        Creates and process donor_user.Generates a heatmap of the appliance usage for the donor user, shifted usage, acceptor user, and created user.

        """
        print(shifted_disagg_output.head(), raw_energy_data.head())
        
        if year_change == 0:
            merged_data = create_new_user(shifted_disagg_output, raw_energy_data)
        elif year_change ==1:
            merged_data = create_different_year_user(shifted_disagg_output,raw_energy_data)
        print(merged_data.head())
        fig, axs = plt.subplots(1, 4, figsize=(30, 15), gridspec_kw={'wspace': 0.5})
        # print(df_x.head())
        df_x.rename(columns={'appliance_output': 'value'}, inplace=True)
        create_heatmap(df_x, "Donor User Appliance Usage", timezone, ax=axs[0])
        new_disagg = shifted_disagg_output.rename(columns={'appliance_output': 'value'})

        create_heatmap(new_disagg, "Shifted Usage", timezone, ax=axs[1])

        min_value = min(raw_energy_data['value'].min(), merged_data['value'].min())
        max_value = max(raw_energy_data['value'].max(), merged_data['value'].max())

        create_heatmap(raw_energy_data, "Acceptor User Usage", timezone, ax=axs[2], min_value=min_value, max_value=max_value)

        create_heatmap(merged_data, "Created User", timezone, ax=axs[3], min_value=min_value, max_value=max_value)

        fig.suptitle('UUID: {}, Acceptor Sampling Rate: {} , Donor Sampling Rate: {}'.format(uuid + "  x  " + donor_uuid, acceptor_sampling_rate, donor_sampling_rate), fontsize=18)

        heatmap_folder = check_folder(heatmap_folder, "heatmap_data")
        heatmap_file = os.path.join(heatmap_folder, f"heatmap_{uuid}_x_{donor_uuid}.png")
        fig.savefig(heatmap_file)
        plt.close(fig)
        print(f"Heatmap File printed successfully at {heatmap_file} for {uuid}_x_{donor_uuid}")
        folder_path = 'created_user_raw_data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the merged_data dataframe to a CSV file
        merged_data.to_csv(os.path.join(folder_path, f"{uuid}_{donor_uuid}.csv"), index=False)
        print(f"Raw Data File saved successfully for {uuid}_x_{donor_uuid}")
        del new_disagg
        


    def process_raw_data(raw_file,raw_data, uuid, donor_uuid, disagg_output_home, shifted_disagg_output, timezone, env,donor_sampling_rate,year_change):
        """
        Processes the raw energy data and generates a heatmap for the acceptor user.
        """
        
        with open(raw_file, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "value", "duration"])
            for data in raw_data:
                writer.writerow([data["time"], data["value"], data["duration"]])    

            raw_energy_data = pd.read_csv(raw_file)
            print(raw_energy_data.head())
            acceptor_sampling_rate = raw_energy_data["duration"][1]
            print("Acceptor UUID Sampling Rate is", acceptor_sampling_rate)
            if compareSamplingRate(donor_sampling_rate, acceptor_sampling_rate) == 0:
                generate_heatmap(shifted_disagg_output, raw_energy_data, donor_uuid, uuid, acceptor_sampling_rate, timezone,heatmap_folder,year_change,donor_sampling_rate)

            elif donor_sampling_rate < acceptor_sampling_rate:
                downsample_factor = int(acceptor_sampling_rate // donor_sampling_rate)
                shifted_disagg_data = downsample(shifted_disagg_output, downsample_factor,donor_sampling_rate)
                generate_heatmap(shifted_disagg_output, raw_energy_data, donor_uuid, uuid, acceptor_sampling_rate, timezone,heatmap_folder,year_change,donor_sampling_rate)

            else:
                print(f"Sampling Rates are not matching for {uuid}")
                

    accepted_appliances = ["ev", "pp", "timed_wh", "ac", "sh"]


    if donor_appliance not in accepted_appliances:
        print("Error: Donor appliance is not in the list of accepted values.")


    def check_env(client_id, env):
        if env not in client_id:
            return f"{env} is given wrong client_id"


    if donor_uuid_file is None or donor_uuid_file == '':
        if donor_appliance == "ev":
            donor_uuid_file = "donor_uuid_ev.csv"
        elif donor_appliance == "pp":
            donor_uuid_file = "donor_uuid_pp.csv"
        elif donor_appliance == "timed_wh":
            donor_uuid_file = "donor_uuid_timed_wh.csv"
        elif donor_appliance == "ac":
            donor_uuid_file = "donor_uuid_ac.csv"
        elif donor_appliance == "sh":
            donor_uuid_file = "donor_uuid_sh.csv"


    if s3_disagg_output_location is None or s3_disagg_output_location == '':

        if donor_appliance == "ev":
            s3_disagg_output_location = "s3://bidgely-ds/divyant/data_simulation/test/ev/data_simulation_test_ev_2023-04-06_12:16:45"
        elif donor_appliance == "pp":
            s3_disagg_output_location = "s3://bidgely-ds/divyant/data_simulation/test/ev/data_simulation_test_ev_2023-04-06_12:16:45"
        elif donor_appliance == "timed_wh":
            s3_disagg_output_location = "s3://bidgely-ds/divyant/data_simulation/test/ev/data_simulation_test_ev_2023-04-06_12:16:45"
        elif donor_appliance == "ac":
            s3_disagg_output_location = "s3://bidgely-ds/divyant/data_simulation/test/ev/data_simulation_test_ev_2023-04-06_11:39:13"
        elif donor_appliance == "sh":
            s3_disagg_output_location = "s3://bidgely-ds/divyant/data_simulation/test/ev/data_simulation_test_ev_2023-04-06_11:39:13"



    print(donor_uuid_file)
    print(acceptor_uuid_file)
    print(donor_appliance)
    acceptor_uuids = pd.read_csv(acceptor_uuid_file)
    donor_uuids = pd.read_csv(donor_uuid_file, header=None)
    print(acceptor_uuids.head())
    
    
    total_donor_uuids = len(donor_uuids[0])
    total_acceptor_uuids = len(acceptor_uuids)
    total_uuids = total_donor_uuids * total_acceptor_uuids
    progress_bar = st.progress(0)
    
    total_processing_time = 0
    
    def update_progress(completed_uuids):
        progress = completed_uuids / total_uuids
        progress_bar.progress(progress)
        
        
    completed_uuids = 0

    for index, row in donor_uuids.iterrows():
        donor_uuid, donor_env = row[0], row[1]
        print(donor_uuid,donor_env)
        donor_access_token = generating_access_token(donor_env).get('env_token')
        
        timezone, donor_country = get_metadata(env, uuid, access_token) 
        print(donor_uuid, donor_timezone, donor_env,donor_country)
        
        try:
            if not os.path.exists('uuid_mapping.csv'):
                # Write header row to CSV file
                with open('uuid_mapping.csv', mode='w', newline='') as csv_file:
                    fieldnames = ['donor_uuid', 'acceptor_uuid', 'env', 'timezone','status']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                
            s3_path = f"{s3_disagg_output_location}/tou_disagg/{donor_uuid}_tou.csv"
            if not os.path.exists('./donor_data'):
                os.makedirs('./donor_data')

            os.system(f"aws s3 cp {s3_path} ./donor_data/")

            disagg_output_home = get_disagg_data(f"./donor_data/{donor_uuid}_tou.csv", donor_appliance)
            df_x = disagg_output_home
            shifted_disagg_output = shifting_disagg_data(df_x, hours_of_shift, randomization)
            # print(shifted_disagg_output.head())
            start = disagg_output_home['timestamp'].min()
            end = disagg_output_home['timestamp'].max()
            delta = end - start
            print(donor_uuid, start, end)
            donor_sampling_rate = disagg_output_home['timestamp'][0:2].diff().iloc[1]
            print("Donor UUID sampling rate is", donor_sampling_rate)
            for index, row in acceptor_uuids.iterrows():
                uuid = row['uuid']
                env = row['env']
                check_env(client_id, env)
                
                
                    
                access_token = generating_access_token(env).get('env_token')
                
                timezone, acceptor_country = get_metadata(env, uuid, access_token)
                print(uuid, timezone, env, acceptor_country)
                
                with open('uuid_mapping.csv', mode='a', newline='') as csv_file:
                    fieldnames = ['donor_uuid', 'acceptor_uuid', 'env', 'timezone']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({'donor_uuid': donor_uuid, 'acceptor_uuid': uuid, 'env': env, 'timezone': timezone}) 
                    
                
                raw_data = getrawData(env, uuid, int(start), int(end))
                raw_data_folder = check_folder(raw_data_file, "raw_data")
                raw_file = os.path.join(raw_data_folder, f"raw_data_{uuid}.csv")
                start_time = time.time()
                
                 
                #print(check_timestamp_within_six_months(start_timestamp, last_timestamp))
                if raw_data:
                    #print(f"Debug: raw_data for user {uuid}: {raw_data}")
                    last_timestamp = raw_data[-1]["time"]
                    start_timestamp = raw_data[0]["time"]
                    acceptor_delta = last_timestamp-start_timestamp
                    print(acceptor_delta) 
                    if check_timestamp_more_than_six_months(start_timestamp, last_timestamp):
                        year_change = 0 
                        process_raw_data(raw_file,raw_data, uuid, donor_uuid, disagg_output_home,
                                    shifted_disagg_output,  timezone, env,donor_sampling_rate,year_change=0)
                    
                    
                elif raw_data ==[] or raw_data is None:
                    #print("Changing different year data")
                    last_billing_cycle = get_last_billing_cycle(uuid)
                    epoch_timestamp = last_billing_cycle["value"]
                    acceptor_end_time = datetime.datetime.fromtimestamp(epoch_timestamp)
                    acceptor_end_date = acceptor_end_time.date()
                    formatted_date = "{}/{}".format(acceptor_end_date.month,
                                                    acceptor_end_date.day)

                    donor_end_time = datetime.datetime.fromtimestamp(end)
                    donor_end_date = donor_end_time.date()
                    year = donor_end_date.year

                    final_formatted_date = "{}/{}".format(formatted_date, year)
                    print(final_formatted_date)

                    dt = datetime.datetime.strptime(final_formatted_date, "%m/%d/%Y")
                    end_epoch_timestamp = int(dt.timestamp())
                    print(end_epoch_timestamp)

                    start_epoch_timestamp = end_epoch_timestamp - delta
                    print(start_epoch_timestamp, end_epoch_timestamp)

                    raw_data = getrawData(env, uuid, start_epoch_timestamp, end_epoch_timestamp)

                    
                    if raw_data:
                        year_change =1
                        process_raw_data(raw_file, raw_data, uuid, donor_uuid, disagg_output_home,
                                        shifted_disagg_output, timezone, env,donor_sampling_rate,year_change=1)
                    else:
                        print(f"Not enough data for uuid :{uuid}")
                    
                else:
                    print(f"Not enough data for uuid :{uuid}")

                
                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                print(
                    f'Processing time for UUID {uuid}: {processing_time:.2f} seconds')
                logging.info(
                    f'Processing time for UUID {uuid}: {processing_time:.2f} seconds')
                
                completed_uuids += 1
                percent_complete = update_progress(completed_uuids)
                
                
                
        except Exception as e:
            print(f"Error in simulating user at {uuid}: {e}")
            logging.error(f"Error in simulating user at {uuid}: {e}")
            completed_uuids += 1
            update_progress(completed_uuids)

    
    logging.info(f'Total processing time for all UUIDs: {total_processing_time:.2f} seconds')
    return print(f'Total processing time for all UUIDs: {total_processing_time:.2f} seconds')

if __name__ == "__main__":
    main()
