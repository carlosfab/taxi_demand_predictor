import pandas as pd
from typing import Tuple, Optional
import datetime


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime.datetime,
    target_col_name: str,
    datetime_col: str = 'pickup_hour'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the input DataFrame into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        cutoff_date (datetime.datetime): The date to split the DataFrame on.
        target_col_name (str): The name of the target column to predict.
        datetime_col (str, optional): The name of the column containing datetime values. 
                                      Defaults to 'pickup_hour'.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
            - Training features DataFrame.
            - Training target Series.
            - Testing features DataFrame.
            - Testing target Series.

    Raises:
        ValueError: If the provided target_col_name or datetime_col doesn't exist in the DataFrame.
    """

    if target_col_name not in df.columns:
        raise ValueError(f"'{target_col_name}' does not exist in the provided DataFrame.")

    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' does not exist in the provided DataFrame.")

    # split the data into training and testing sets
    train = df.loc[df[datetime_col] < cutoff_date].reset_index(drop=True)
    test = df.loc[df[datetime_col] >= cutoff_date].reset_index(drop=True)

    # split the data into X and y
    X_train = train.drop(target_col_name, axis=1)
    y_train = train[target_col_name]
    X_test = test.drop(target_col_name, axis=1)
    y_test = test[target_col_name]

    return X_train, y_train, X_test, y_test
