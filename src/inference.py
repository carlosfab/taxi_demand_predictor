from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import hopsworks
from hsfs.feature_store import FeatureStore

import src.config as config
from src.feature_store_api import get_feature_store


def get_hopsworks_project():
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """    
    Load a batch of features for a specific date from the feature store.

    Given a target date, this function fetches the relevant time-series data from 
    a feature store for a period leading up to the current date. The time-series 
    data is then transformed into a structured format suitable for ML training/testing.

    Parameters:
    - current_date (datetime): The target date for which the feature batch is required.

    Returns:
    - pd.DataFrame: A dataframe containing the structured features. Each row corresponds
      to a specific pickup location and contains historical ride data as features.

    Notes:
    - The function fetches data for a period defined in the `config` (28 days by default).
    - It assumes the existence of a `config.N_FEATURES` parameter to denote the number 
      of historical data points to consider.
    - The function checks for data completeness and raises an assertion error if the 
      expected data volume is not met.
    - The fetched time-series data is then structured into a format where each row 
      represents a `pickup_location_id` and the columns represent historical ride data.

    # Raises:
    # - AssertionError: If the time-series data is not complete.
    """
    n_features = config.N_FEATURES

    feature_store = get_feature_store()

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print("FROM-TO DATA:\n", fetch_data_from, fetch_data_to, "\n\n\n")
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    print(len(ts_data), n_features*len(location_ids))
    print(ts_data.head())
    print(fetch_data_from, fetch_data_to)
    assert len(ts_data) == n_features*len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    # print(f'{ts_data=}')

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_model_from_registry(model_file='model.pkl'):
    """
    Load a machine learning model from the Hopsworks model registry.

    This function fetches a specific model from the Hopsworks model registry
    using predefined model name and version from the configuration. Once 
    downloaded, the model is then loaded into memory using joblib.

    Returns:
    - Trained Model: The trained machine learning model loaded from the registry.

    Notes:
    - The function relies on `config.MODEL_NAME` and `config.MODEL_VERSION` 
      to specify which model to retrieve from the registry.
    - The model is assumed to be saved as 'model.pkl' in the registry.

    Example:
    ```python
    model = load_model_from_registry()
    predictions = model.predict(X_test)
    ```

    Dependencies:
    - joblib: Required for model deserialization.
    - pathlib.Path: For path manipulations.

    """
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )

    model_dir = model.download()
    model = joblib.load(Path(model_dir) / model_file)

    return model


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict taxi demand based on historical ride data.

    Given a trained model and a set of features representing historical 
    taxi rides, this function predicts the taxi demand for the next hour 
    and returns the results in a structured format.

    Parameters:
    - model: A trained machine learning model capable of predicting taxi demand.
    - features (pd.DataFrame): A dataframe containing the historical taxi ride data.

    Returns:
    - pd.DataFrame: A dataframe with the 'pickup_location_id' and its corresponding 
      'predicted_demand' for the next hour.
    """
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results


def load_predictions_from_store(
        from_pickup_hour: datetime,
        to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Fetch model predictions from the feature store for a specified time range.

    This function connects to the feature store and retrieves taxi demand predictions 
    for each `pickup_location_id` within the time window defined by the arguments 
    `from_pickup_hour` and `to_pickup_hour`.

    Args:
        from_pickup_hour (datetime): Start time (inclusive) from which predictions are required.
        to_pickup_hour (datetime): End time (inclusive) up to which predictions are required.

    Returns:
        pd.DataFrame: A dataframe containing the following columns:
            - `pickup_location_id`: Identifier for pickup locations.
            - `predicted_demand`: Model's predicted taxi demand.
            - `pickup_hour`: The hour for which the prediction was made.

    Note:
        If the feature view does not exist in the feature store, this function attempts 
        to create it.
    """
    from src.feature_store_api import get_feature_store
    import src.config as config

    feature_store = get_feature_store()

    prediction_fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,
        version=1
    )

    # Try to create the feature view if it does not exist
    try:
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
            version=1,
            query=prediction_fg.select_all()
        )
    except:
        print(f'Feature view {config.FEATURE_VIEW_MODEL_PREDICTIONS} already exists. Creation skipped.')

    predictions_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
        version=1
    )

    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour} and {to_pickup_hour}')
    predictions = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1)
    )
    predictions = predictions[predictions.pickup_hour.between(
        from_pickup_hour, to_pickup_hour)]

    print(predictions.head())
    # Sort by `pickup_hour` and `pickup_location_id`
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    print(predictions.head())

    return predictions
