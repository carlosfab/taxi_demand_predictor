# Imports
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import app_component as ac
from src.inference import load_batch_of_features_from_store, load_model_from_registry, get_model_predictions, load_predictions_from_store

# Page configuration
st.set_page_config(
    page_title="Taxi Demand Prediction",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=taxiDemandPrediction"
)

# Styles & Customizations
st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)


# Header Section
home_title = "Taxi Demand Prediction"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Web App</font></span>""", unsafe_allow_html=True)

st.markdown("""\n""")

loading_info = st.empty()

st.markdown("#### Greetings 🚖")
st.write(
    """
    Welcome to the Taxi Demand Predictor Hub, where cutting-edge Machine Learning meets urban mobility needs. For those interested in the technical aspects, the project repository offers comprehensive insights.
    """
)

# App Component
ac.robo_avatar_component()

# Sidebar
progress_bar = st.sidebar.header(":gear: Project Progress")
progress_bar.progress(0)

# constant for number of steps in progress bar
N_STEPS = 3


# STEP 1 - load shape data for NYC taxi zones
@st.cache_data
def wrapped_load_shape_data_file() -> gpd.geopandas.GeoDataFrame:
    return ac.load_shape_data_file()


with loading_info, st.spinner("Downloading data... this may take a while! \n Don't worry, this is a one-time thing. :wink:"):
    shape_data = wrapped_load_shape_data_file
    st.sidebar.write(":white_check_mark: Shape data download complete!")
    progress_bar.progress(1/N_STEPS)


# STEP 2 - Load Predictions from the store
@st.cache_data
def wrapped_load_predictions_from_store(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:
    print(f'{from_pickup_hour=}')
    print(f'{to_pickup_hour=}')
    return load_predictions_from_store(from_pickup_hour, to_pickup_hour)


with st.spinner(text="Fetching predictions from the store"):
    current_date = pd.to_datetime(datetime.utcnow()).floor('H')
    predictions_df = wrapped_load_predictions_from_store(
        from_pickup_hour=current_date - timedelta(hours=4),
        to_pickup_hour=current_date
    )
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(2/N_STEPS)
    print("\n\n\nSTEP 2:")
    print(f'{current_date=}')

    print(f'{predictions_df=}')
    print(predictions_df.head(10))


# STEP 3 - Fetching batch of features
def is_prediction_ready(df, target_date):
    """Check if the predictions for the given date are available."""
    return not df[df.pickup_hour == target_date].empty


def get_predictions_for_date(df, target_date):
    """Retrieve predictions for the given date."""
    return df[df.pickup_hour == target_date]


# Check availability of predictions

print(f'Current date: {current_date}')
print(predictions_df.head())

next_hour_predictions_ready = is_prediction_ready(predictions_df, current_date)
prev_hour_predictions_ready = is_prediction_ready(predictions_df, current_date - timedelta(hours=1))

# Retrieve the relevant predictions
if next_hour_predictions_ready:
    predictions_df = get_predictions_for_date(predictions_df, current_date)
elif prev_hour_predictions_ready:
    predictions_df = get_predictions_for_date(predictions_df, current_date - timedelta(hours=1))
    current_date -= timedelta(hours=1)
    st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')
else:
    raise Exception('Features are not available for the last 2 hours. Is your feature pipeline up and running? 🤔')


@st.cache_data
def wrapped_load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    return load_batch_of_features_from_store(current_date)


with loading_info, st.spinner("Fetching data from Feature Store..."):
    features = wrapped_load_batch_of_features_from_store(current_date)
    st.sidebar.write(":white_check_mark: Feature data fetched!")
    progress_bar.progress(2/N_STEPS)


ac.render_contact()


# Real-world Machine Learning Section
st.markdown("\n")
st.markdown("#### Real-World Machine Learning 🛠")
st.write("""
         Welcome to a real-world ML service predicting NYC taxi rides, crafted with MLOps best practices. Transitioning from raw data to a robust data pipeline, and from a model prototype to a fully-functional batch-scoring system, powered by a Feature Store and GitHub Actions.
         """)

# Ongoing Work Section
st.markdown("#### Ongoing Work 🚧")
st.markdown("""
            I'll be deploying and updating the Streamlit app and the repository in the coming days."""
            )
st.markdown("\n")
st.info("""
        The deployment of our Streamlit web app is in progress. \n\n**Stay updated:**  While waiting for the full app to be live, you can also [check out the repository here.](https://github.com/carlosfab/taxi_demand_predictor)
.\n\n- Connect with me on [LinkedIn](http://linkedin.com/in/carlos-melo-data-science/).\n- Read my articles on my [personal blog](https://sigmoidal.ai/en).
        """)

# Repository Link Button
st.link_button(":star: Star the Repository!", "https://github.com/carlosfab/taxi_demand_predictor", type='secondary', use_container_width=True)
