import numpy as np
import logging
from filters.apply_range_filter import apply_range_filter
from filters.apply_motion_filter import apply_motion_filter
from filters.apply_knn_filter import apply_knn_filter
from filters.apply_kd_tree_filter import apply_kd_tree_filter


# Mapping from filter key names (as defined in the config) to their implementation functions.

FILTER_FUNCTIONS = {
    "range_filter": apply_range_filter,
    "motion_filter": apply_motion_filter,
    "knn_filter": apply_knn_filter,
    "kd_tree_filter": apply_kd_tree_filter,  # Newly added KDTree filter.
}

logger = logging.getLogger(__name__)


def apply_filters(points, traj_data, config):
    """
    Applies a series of filters to a point cloud dataset based on configuration.

    The filtering configuration should reside under the "filtering" key in config.
    It is expected to include:
      - "format_name": e.g., "pointcloud_sensor" (used to look up the data format mapping).
      - One or more filter configurations (e.g., "range_filter", "motion_filter", etc.),
        each containing an "enabled" flag and additional parameters.

    For each enabled filter, the corresponding function is called to generate a boolean mask.
    The final global mask (the logical AND of all individual masks) is used to separate
    points that pass the filters from those that do not.

    Args:
        points (np.ndarray): The input point cloud data (structured array).
        traj_data (np.ndarray): The trajectory data (may be used by some filters).
        config (JSONRegistry or dict): The configuration object.

    Returns:
        tuple: (pass_points, fail_points)
    """
    # Ensure the config supports .get
    if not hasattr(config, "get"):
        logger.error("❌ Config object is not using JSONRegistry. Check initialization.")
        raise TypeError("Config must be an instance of JSONRegistry.")

    # Log the full filtering configuration for debugging.
    filtering_config = config.get("filtering", {})
    logger.info(f"🔍 Filtering configuration:\n{filtering_config}")

    # Retrieve the format name to determine the data format mapping.
    format_name = filtering_config.get("format_name")
    if not format_name:
        logger.error("❌ No 'format_name' specified in filtering config!")
        raise ValueError("Missing 'format_name' in filtering config.")

    logger.info(f"Filtering will use format: {format_name}")

    # Retrieve the data_formats mapping and get the field mapping for the specified format.
    data_formats = config.get("data_formats", {})
    if format_name not in data_formats:
        available = list(data_formats.keys())
        logger.error(f"❌ Unknown point cloud format: {format_name}.")
        raise ValueError(f"Unknown point cloud format: {format_name}. Available formats: {available}")
    field_mapping = data_formats[format_name]

    # Start with a global mask that passes all points.
    global_mask = np.ones(len(points), dtype=bool)

    # Iterate over each filter configuration (skip "format_name").
    for filter_key, filter_params in filtering_config.items():
        if filter_key == "format_name":
            continue
        if not isinstance(filter_params, dict):
            logger.warning(f"Skipping filter '{filter_key}' because its configuration is not a dict.")
            continue
        if not filter_params.get("enabled", False):
            logger.info(f"Filter '{filter_key}' is disabled; skipping.")
            continue

        filter_func = FILTER_FUNCTIONS.get(filter_key)
        if not filter_func:
            logger.warning(f"No implementation for filter '{filter_key}'; skipping.")
            continue

        try:
            # Apply the filter to get a boolean mask.
            mask = filter_func(points, filter_params, field_mapping)
            logger.info(f"Filter '{filter_key}' applied; {np.sum(mask)} of {len(points)} points pass.")
            global_mask &= mask
        except Exception as e:
            logger.error(f"Error applying filter '{filter_key}': {e}")
            raise

    pass_points = points[global_mask]
    fail_points = points[~global_mask]

    logger.info(f"✅ Filtering complete: {len(pass_points)} points kept, {len(fail_points)} points removed.")
    return pass_points, fail_points
