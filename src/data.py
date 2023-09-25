# Required imports
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


# Data Download & Loading Functions
# ---------------------------------

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Downloads a single file of raw taxi ride data from the NYC website for the specified year and month.
    The downloaded file is saved to the local storage directory specified in `RAW_DATA_DIR`.

    Args:
        year (int): The year of the data to download.
        month (int): The month of the data to download.

    Returns:
        Path: The path to the downloaded file in local storage.

    Raises:
        Exception: If the download fails or the requested file is not available.
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        with open(path, "wb") as file:
            file.write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')


def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Validates and filters a DataFrame of taxi rides to ensure all rows fall within the specified year and month.

    The function removes rows where the 'pickup_datetime' column falls outside the desired month and year.
    For example, if the year is 2022 and the month is 3 (March), all rows where the pickup_datetime 
    is not in March 2022 will be filtered out.

    Args:
        rides (pd.DataFrame): A DataFrame containing taxi ride data with a 'pickup_datetime' column.
        year (int): The desired year to filter for.
        month (int): The desired month (1-12) to filter for.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rides from the specified year and month.

    Raises:
        ValueError: If the 'pickup_datetime' column is not present in the rides DataFrame.
    """
    if 'pickup_datetime' not in rides.columns:
        raise ValueError("The 'pickup_datetime' column is missing from the rides DataFrame.")

    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


def fetch_ride_events_from_data_warehouse(
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """
    Simulates production data by fetching historical taxi ride events from a simulated data warehouse 
    based on the provided date range, and then adjusting the dates to match the desired production range.

    The function samples historical data from exactly 52 weeks (1 year) ago from the provided dates. 
    If the `from_date` and `to_date` are in the same month and year, only one file of data is downloaded. 
    Otherwise, two files, one for each month, are downloaded and merged.

    After fetching, the 'pickup_datetime' column of the resulting DataFrame is shifted forward by 52 weeks 
    to simulate production data based on the historical data.

    Args:
        from_date (datetime): The start date of the desired production data range.
        to_date (datetime): The end date of the desired production data range.

    Returns:
        pd.DataFrame: A DataFrame containing simulated production taxi ride data for the specified date range.

    Notes:
        This function assumes the existence of a `load_raw_data` function that can fetch raw data 
        based on year and month.
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # download 1 file of data only
        rides = load_raw_data(year=from_date_.year, month=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides = rides[rides.pickup_datetime < to_date_]

    else:
        # download 2 files from website
        rides = load_raw_data(year=from_date_.year, month=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, month=to_date_.month)
        rides_2 = rides_2[rides_2.pickup_datetime < to_date_]
        rides = pd.concat([rides, rides_2])

    # shift the pickup_datetime back 1 year ahead, to simulate production data
    # using its 7*52-days-ago value
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides


def load_raw_data(
    year: int,
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Loads taxi ride data either from local storage or downloads it from the NYC website, depending 
    on the availability. The downloaded or retrieved data is then loaded into a Pandas DataFrame, 
    with columns renamed and validated as needed.

    Data files are stored with a naming convention of 'rides_{year}-{month}.parquet'. The function 
    checks for the presence of these files in `RAW_DATA_DIR` and only downloads them if they're absent.

    Args:
        year (int): The year of the data to load.
        months (Optional[List[int]]): A list of months (integers 1-12) for which the data is to be loaded.
            If `None`, data for all months in the specified year will be loaded.
            If an integer is provided instead of a list, it will be converted to a list.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - pickup_datetime (datetime): Timestamp indicating when the taxi was hired.
            - pickup_location_id (int): Identifier for the location where the taxi was hired.

    Notes:
        This function assumes the existence of `download_one_file_of_raw_data` and `validate_raw_data` 
        functions. If data for a specific month is not available on the website or is corrupted, 
        it will be skipped.
    """
    rides = pd.DataFrame()

    if months is None:
        # download data for the entire year (all months)
        months = list(range(1, 13))
    elif isinstance(months, int):
        # download data only for the month specified by the int `month`
        months = [months]

    for month in months:

        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # download the file from the NYC website
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue
        else:
            print(f'File {year}-{month:02d} was already in local storage')

        # load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        }, inplace=True)

        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append to existing data
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Augments the input time-series data by adding missing hourly slots and pickup locations. 
    The function ensures a continuous hourly series for each pickup location within the given data's time frame.

    Specifically, if the 'ts_data' DataFrame has missing hours or pickup location IDs, this function fills 
    in those gaps, ensuring that every hour for each location has a corresponding entry. If an entry is missing, 
    it is filled with a count of zero rides.

    Args:
        ts_data (pd.DataFrame): The input time-series data. Expected columns:
            - pickup_hour (datetime): The hour when rides occurred.
            - pickup_location_id (int): Identifier for the location where rides occurred.
            - rides (int): Count of rides for the corresponding hour and location.

    Returns:
        pd.DataFrame: A DataFrame with the same structure as `ts_data`, but with any gaps filled in. 
            Every hour within the data's time span will have an entry for each location, and missing entries 
            are filled with zero rides.

    Notes:
        This function uses a method to quickly add missing dates from a StackOverflow post:
        https://stackoverflow.com/a/19324591
    """
    location_ids = range(1, ts_data['pickup_location_id'].max() + 1)
    full_range = pd.date_range(ts_data['pickup_hour'].min(),
                               ts_data['pickup_hour'].max(),
                               freq='H')
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']]

        if ts_data_i.empty:
            # add a dummy entry with a 0
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
            ])

        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)

        # add back `location_id` columns
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])

    # move the pickup_hour from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output


def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms raw taxi ride data into a time-series format by aggregating the number of rides per hour 
    for each pickup location. The returned DataFrame will have an entry for every hour and pickup location, 
    with missing entries filled with zero rides.

    Args:
        rides (pd.DataFrame): The input raw ride data. Expected columns:
            - pickup_datetime (datetime): The exact datetime when the ride occurred.
            - pickup_location_id (int): Identifier for the location where the ride was picked up.

    Returns:
        pd.DataFrame: A time-series formatted DataFrame with the following columns:
            - pickup_hour (datetime): The rounded hour when rides occurred.
            - pickup_location_id (int): Identifier for the pickup location.
            - rides (int): Aggregated count of rides for the corresponding hour and location.

    Notes:
        This function relies on `add_missing_slots` to ensure that every hour and location has an entry, 
        and it fills in any gaps in the time-series data with zero ride counts.
    """
    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms time-series data into a format suitable for supervised machine learning models. 
    The time series is sliced into sequences of length `input_seq_len` to serve as features, 
    and the next hour's data serves as the target.

    Args:
        ts_data (pd.DataFrame): Time-series data with the following columns:
            - pickup_hour (datetime): The rounded hour when rides occurred.
            - pickup_location_id (int): Identifier for the pickup location.
            - rides (int): Aggregated count of rides for the corresponding hour and location.
        input_seq_len (int): The length of the input sequence for the model, i.e., 
            the number of historical hours of data to use as features.
        step_size (int): The step size for sliding the window to create new examples.
            For example, if `step_size` is 1, every hour of data is used as a new starting point.
            If `step_size` is 2, every other hour is used, and so on.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - pd.DataFrame: Features for supervised learning. Columns correspond to 
              historical data (e.g., "rides_previous_1_hour", "rides_previous_2_hour", etc.), 
              'pickup_hour', and 'pickup_location_id'.
            - pd.Series: Target variable for each row of the features. It represents the 
              number of rides in the next hour.

    Notes:
        This function slices the data for each pickup location independently. 
        `get_cutoff_indices_features_and_target` is used to determine where the slices should be made.
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id,
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> list:
    """
    Calculates the indices for slicing the input data into (features, target) pairs for supervised learning.

    For a given sequence length (`input_seq_len`) and step size (`step_size`), the function computes the starting, 
    middle, and ending indices of each sub-sequence in the input data. The starting to middle range 
    is used as the input features, and the middle to ending range (which is just one step) 
    is used as the target.

    Args:
        data (pd.DataFrame): The input time-series data to be sliced.
        input_seq_len (int): The length of the input sequence for the model, i.e., 
            the number of historical data points to use as features.
        step_size (int): The step size for sliding the window to create new examples.
            A smaller step size results in more overlapping sequences, while a larger step 
            size results in fewer sequences with larger gaps between them.

    Returns:
        list: A list of tuples. Each tuple contains three integers:
            - The starting index of the sub-sequence (inclusive).
            - The middle index, where the target starts (inclusive).
            - The ending index of the sub-sequence (exclusive).

    Notes:
        For instance, for `input_seq_len` = 3 and `step_size` = 1, given a data of length 10, 
        one of the returned tuples might be (2, 5, 6), indicating a feature sequence from 
        index 2 to 4 (inclusive) and a target at index 5.
    """
    stop_position = len(data) - 1

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = input_seq_len
    subseq_last_idx = input_seq_len + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices
