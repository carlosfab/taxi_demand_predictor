# --- Import libraries ---
from typing import Optional
import hsfs
import hopsworks

# Import local configurations
import src.config as config

# --- Function Definitions ---


def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """
    Connects to Hopsworks and retrieves a pointer to the feature store.

    This function uses the Hopsworks project name and API key from the 
    configuration to connect to the Hopsworks instance.

    Returns:
        hsfs.feature_store.FeatureStore: Pointer to the feature store.
    """

    # Log in to Hopsworks using configuration details
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

    # Return feature store associated with the project
    return project.get_feature_store()


def get_feature_group(
    name: str,
    version: Optional[int] = 1
) -> hsfs.feature_group.FeatureGroup:
    """
    Connects to the feature store and retrieves a pointer to the specified 
    feature group by its name and version.

    Args:
        name (str): Name of the feature group.
        version (Optional[int], optional): Version of the feature group. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: Pointer to the feature group.
    """

    # Get feature store
    feature_store = get_feature_store()

    # Return the desired feature group from the feature store
    return feature_store.get_feature_group(
        name=name,
        version=version,
    )
