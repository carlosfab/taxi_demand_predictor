# Imports
import streamlit as st
import app_component as ac

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
st.markdown("#### Greetings 🚖")
st.write(
    """
    Welcome to the Taxi Demand Predictor Hub, where cutting-edge Machine Learning meets urban mobility needs. This web application is an integral component of an end-to-end Machine Learning Project. For those interested in the technical aspects, the project repository offers comprehensive insights, including explanatory notebooks, source code, and automation utilities.
    """
)

# App Component
ac.robo_avatar_component()

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

# Sidebar
st.sidebar.image("header.gif", use_column_width=True)
st.sidebar.title("Contact")
st.sidebar.info(
    """
    Carlos at [sigmoidal.ai](https://sigmoidal.ai/en) | [GitHub](https://github.com/carlosfab) | [Twitter](https://twitter.com/carlos_melo_py) | [YouTube](https://www.youtube.com/@CarlosMeloSigmoidal) | [Instagram](http://instagram.com/carlos_melo.py) | [LinkedIn](http://linkedin.com/in/carlos-melo-data-science/)
    """
)
