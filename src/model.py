# import libraries
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from src.paths import TRANSFORMED_DATA_DIR
from src.data_split import train_test_split


class PastWeeksHourlyAverage(BaseEstimator, TransformerMixin):
    """
    Returns the average rides of the same hour from the last n weeks.

    Args:
        n_weeks (int): Number of weeks to average. Default is 4.
    """

    def __init__(self, n_weeks: int = 4):
        self.n_weeks = n_weeks

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        average_n_weeks = np.average(
            [X_[f'rides_previous_{i*7*24}_hour'].values for i in range(1, self.n_weeks + 1)],
            axis=0
        )
        X_[f'average_rides_{self.n_weeks}_weeks'] = average_n_weeks
        return X_


class DatetimeComponentsExtractor(BaseEstimator, TransformerMixin):
    """Extractor for datetime components like hour and day of the week."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.dayofweek
        X_['is_weekend'] = X_['pickup_hour'].dt.weekday.isin([5, 6]).astype(int)

        X_.drop(columns=['pickup_hour'], inplace=True)

        return X_


def get_pipeline(**hyperparams) -> Pipeline:
    """
    Creates a pipeline with preprocessing steps and a LightGBM regressor.

    Args:
    - **hyperparams (dict): Hyperparameters for the pipeline's components.
                            Keys should match the component's hyperparameter names.

    Returns:
    - Pipeline: A scikit-learn pipeline with preprocessing steps and regressor.
    """
    datetime_extractor = DatetimeComponentsExtractor()
    past_weeks_averager = PastWeeksHourlyAverage(n_weeks=4)

    pipeline = make_pipeline(
        datetime_extractor,
        past_weeks_averager,
        lgb.LGBMRegressor(**hyperparams)
    )

    return pipeline
