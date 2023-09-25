# Required imports
from typing import Optional
from datetime import timedelta

import pandas as pd
import plotly.express as px


def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None
) -> px.Figure:
    """
    Plots a sample with historical rides, actual values, and predicted values.

    Args:
    - example_id: Index of the sample to be plotted.
    - features: DataFrame containing historical rides.
    - targets: Series with actual values.
    - predictions: Series with predicted values.

    Returns:
    - A Plotly Figure object.
    """
    # Extract data
    sample_features = features.iloc[example_id]
    target_value = targets.iloc[example_id] if targets is not None else None

    # Extract time series data
    ts_columns = [col for col in features.columns if col.startswith('rides_previous_')]
    ts_values = [sample_features[col] for col in ts_columns] + [target_value]

    ts_dates = pd.date_range(
        start=sample_features['pickup_hour'] - timedelta(hours=len(ts_columns)),
        end=sample_features['pickup_hour'],
        freq='H'
    )

    # Create the figure
    title = f'Pick up hour={sample_features["pickup_hour"]}, location_id={sample_features["pickup_location_id"]}'
    fig = px.line(x=ts_dates, y=ts_values, template='plotly_dark', markers=True, title=title)

    # Add actual and predicted values (if available)
    if targets is not None:
        fig.add_scatter(x=ts_dates[-1:], y=[target_value], line_color='green', mode='markers', marker_size=10, name='actual value')
    if predictions is not None:
        predicted_value = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[predicted_value], line_color='red', mode='markers', marker_symbol='x', marker_size=15, name='prediction')

    return fig
