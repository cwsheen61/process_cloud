import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_dtype_from_config(config):
    """
    Constructs a NumPy dtype for the point cloud data based on the configuration.

    The configuration should contain:
      - "filtering.format_name": e.g., "pointcloud_sensor" (identifies which format to use)
      - "data_formats": A dictionary mapping format names to their field definitions,
         e.g., {"pointcloud_sensor": {"time": "float64", "x": "float64", ... }}

    Args:
        config (JSONRegistry or dict): The configuration object.
        
    Returns:
        np.dtype: A NumPy dtype constructed from the field mapping.
    """
    # Get the filtering format name (default to "pointcloud_sensor" if not set)
    filtering_format = config.get("filtering.format_name", "pointcloud_sensor")
    # Get the data_formats dictionary.
    data_formats = config.get("data_formats", {})
    if filtering_format not in data_formats:
        logger.error(f"Unknown point cloud format: {filtering_format}.")
        raise ValueError(f"Unknown point cloud format: {filtering_format}")
    field_mapping = data_formats[filtering_format]
    
    # Build a list of (field_name, dtype) tuples.
    dtype_list = []
    for field, type_str in field_mapping.items():
        try:
            dtype_list.append((field, np.dtype(type_str)))
        except TypeError as e:
            logger.error(f"Error converting type for field '{field}': {e}")
            raise
    return np.dtype(dtype_list)
