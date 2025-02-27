import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_range_filter(points, range_config, field_mapping):
    """ Filters points based on range values dynamically. """

    # ✅ Find the column index for 'range' dynamically
    if "range" not in field_mapping:
        logger.error("❌ 'range' field not found in point cloud format!")
        raise ValueError("Field 'range' not found in format definition.")

    # Extract the column index dynamically
    field_names = list(field_mapping.keys())
    range_index = field_names.index("range")

    range_min = range_config.get("range_min", float('-inf'))
    range_max = range_config.get("range_max", float('inf'))

    logger.info(f"📏 Filtering points with range between {range_min}m and {range_max}m")

    # ✅ Create mask for range filtering
    mask = (points[:, range_index] >= range_min) & (points[:, range_index] <= range_max)

    logger.info("✅ Range filter applied: %d points kept, %d points removed",
                np.sum(mask), np.sum(~mask))

    return mask
