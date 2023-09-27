# Import libraries
import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# Load environment variables
load_dotenv(PARENT_DIR / ".env")

# Extract HOPSWORKS_API_KEY from environment
try:
    HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
except KeyError:
    raise EnvironmentError("HOPSWORKS_API_KEY not found in environment variables.")

# Constants
HOPSWORKS_PROJECT_NAME = 'taxi_demand_api'
FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
