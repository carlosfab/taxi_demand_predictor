# Imports
from datetime import datetime
import pandas as pd
import streamlit as st
import app_component as ac
from src.inference import load_batch_of_features_from_store

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
progress_bar = st.sidebar.progress(0)

# constant for number of steps in progress bar
N_STEPS = 7

# STEP 1 - load shape data for NYC taxi zones
with loading_info, st.spinner("Downloading data... this may take a while! \n Don't worry, this is a one-time thing. :wink:"):
    shape_data = ac.load_shape_data_file()
    st.sidebar.write(":white_check_mark: Shape data download complete!")
    progress_bar.progress(1/N_STEPS)

# STEP 2 - Fetch batch of inference data
with loading_info, st.spinner("Fetching data from Feature Store..."):
    current_date = pd.to_datetime(datetime.utcnow()).floor('H')
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write(":white_check_mark: Inference data fetched!")
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
