import numpy as np
import logging
from filters.range_filter import apply_range_filter

logger = logging.getLogger(__name__)

def apply_filters(points, traj_data, config):
    """Applies the necessary filters to a given point cloud dataset."""

    # ✅ Ensure JSONRegistry instance
    if not hasattr(config, "get"):
        logger.error("❌ Config object is not using JSONRegistry. Check initialization.")
        raise TypeError("Config must be an instance of JSONRegistry.")

    # 🔍 Debugging: Print the full config structure
    logger.info(f"🔍 Full Config Contents:\n{config.config}")

    # ✅ Retrieve format_name safely
    format_name = config.get("format_name")
    if not format_name:
        logger.error("❌ No 'format_name' specified in config!")
        available_keys = list(config.config.keys())  # Extract dictionary keys from JSONRegistry
        raise ValueError(f"Missing 'format_name' in config. Available keys: {available_keys}")

    # ✅ Retrieve data_formats and validate format_name
    data_formats = config.get("data_formats", {})
    if format_name not in data_formats:
        logger.error(f"❌ Unknown point cloud format: {format_name}")
        available_formats = list(data_formats.keys())
        raise ValueError(f"Unknown point cloud format: {format_name}. Available formats: {available_formats}")

    # ✅ Retrieve field mapping from the selected format
    field_mapping = data_formats[format_name]

    # ✅ Initialize the global mask (all True by default)
    global_mask = np.ones(len(points), dtype=bool)

    # ✅ Apply range filtering if enabled
    range_config = config.get("filtering", {}).get("range_filter", {})
    if range_config.get("enabled", False):
        range_mask = apply_range_filter(points, range_config, field_mapping)
        global_mask &= range_mask

    # ✅ Apply the mask to separate pass/fail points
    pass_points = points[global_mask]
    fail_points = points[~global_mask]

    logger.info(f"✅ Filtering complete: {len(pass_points)} points kept, {len(fail_points)} points removed")

    return pass_points, fail_points
