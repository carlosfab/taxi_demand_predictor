# Standard Libraries
import random
import zipfile
import requests

# External Libraries
import streamlit as st
import geopandas as gpd
import streamlit.components.v1 as c
import src.config as config
from src.paths import DATA_DIR


def robo_avatar_component():
    """
    Render a series of robo avatars using dicebear API.
    """
    robo_avatar_seed = [0, 'aRoN', 'gptLAb', 180, 'nORa', 'dAVe', 'Julia', 'WEldO', 60]
    robo_html = "<div style='display: flex; flex-wrap: wrap; justify-content: left;'>"

    for seed in robo_avatar_seed:
        avatar_url = f"https://api.dicebear.com/5.x/bottts-neutral/svg?seed={seed}"
        robo_html += f"<img src='{avatar_url}' style='width: 50px; height: 50px; margin: 10px;'>"

    robo_html += "</div>"

    # Responsive style for avatars
    style = """
    <style>
        @media (max-width: 800px) {
            img {
                max-width: calc((100% - 60px) / 6);
                height: auto;
                margin: 0 10px 10px 0;
            }
        }
    </style>
    """
    c.html(style + robo_html, height=70)


def st_button(url, label, font_awesome_icon):
    """
    Render a button with a Font Awesome icon.
    """
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
    button_code = f'<a href="{url}" target="_blank"><i class="fa {font_awesome_icon}"></i> {label}</a>'
    return st.markdown(button_code, unsafe_allow_html=True)


def render_cta():
    """
    Render Call To Action buttons in the sidebar.
    """
    with st.sidebar:
        st.write("Let's connect!")
        st_button(url="https://twitter.com/carlos_melo_py", label="Twitter", font_awesome_icon="fa-twitter")
        st_button(url="http://linkedin.com/in/carlos-melo-data-science/", label="LinkedIn", font_awesome_icon="fa-linkedin")


def render_contact():
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    Carlos at [sigmoidal.ai](https://sigmoidal.ai/en) | [GitHub](https://github.com/carlosfab) | [Twitter](https://twitter.com/carlos_melo_py) | [YouTube](https://www.youtube.com/@CarlosMeloSigmoidal) | [Instagram](http://instagram.com/carlos_melo.py) | [LinkedIn](http://linkedin.com/in/carlos-melo-data-science/)
    """
    )


def load_shape_data_file() -> gpd.geopandas.GeoDataFrame:
    """
    Load shape data for NYC taxi zones.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the shape data for NYC taxi zones.
    """
    # download zip file
    url_path = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    path = DATA_DIR / 'taxi_zones.zip'
    response = requests.get(url_path)

    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Could not download data from {url_path}")

    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape data
    shape_data = gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp')
    return shape_data
