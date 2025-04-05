import numpy as np
import logging
from filters.apply_range_filter import apply_range_filter
from filters.apply_motion_filter import apply_motion_filter
from filters.apply_knn_filter import apply_knn_filter
from filters.apply_kd_tree_filter import apply_kd_tree_filter

# Mapping from filter key names to their implementation functions
FILTER_FUNCTIONS = {
    "range_filter": apply_range_filter,
    "motion_filter": apply_motion_filter,
    "knn_filter": apply_knn_filter,
    "kd_tree_filter": apply_kd_tree_filter,
}

logger = logging.getLogger(__name__)


def apply_filters(points, traj_data, config):
    """
    Applies filters to a point cloud chunk based on configuration.

    Args:
        points (np.ndarray): Point cloud chunk (structured NumPy array).
        traj_data (np.ndarray): Trajectory data.
        config (JSONRegistry or dict): Configuration object.

    Returns:
        tuple: (pass_points, fail_points)
    """
    # Accept either dict or JSONRegistry
    if hasattr(config, "get"):
        filtering_config = config.get("filtering", {})
    else:
        filtering_config = config.get("filtering", {})

    logger.info(f"🔍 Filtering using format: {filtering_config.get('format_name', '<unknown>')}")

    # Start with a global mask that passes all points
    global_mask = np.ones(len(points), dtype=bool)

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
            # The new model no longer uses config field mapping, just the point dtype
            mask = filter_func(points, filter_params)
            logger.info(f"✅ Filter '{filter_key}' passed {np.sum(mask)} / {len(points)}")
            global_mask &= mask
        except Exception as e:
            logger.error(f"❌ Error applying filter '{filter_key}': {e}")
            raise

    pass_points = points[global_mask]
    fail_points = points[~global_mask]

    logger.info(f"✅ Filtering complete: {len(pass_points)} points kept, {len(fail_points)} points removed.")
    return pass_points, fail_points
