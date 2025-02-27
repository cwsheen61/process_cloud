import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_range_filter(points, range_config, field_mapping):
    """ Filters points based on range values dynamically. """

    # ✅ Ensure 'range' exists in the field mapping
    if "range" not in field_mapping:
        logger.error("❌ 'range' field not found in point cloud format!")
        raise ValueError("Field 'range' not found in format definition.")

    # ✅ Get the column index for range dynamically
    range_index = list(field_mapping.keys()).index("range")

    range_min = range_config.get("range_min", float('-inf'))
    range_max = range_config.get("range_max", float('inf'))

    logger.info(f"📏 Filtering points with range between {range_min}m and {range_max}m")

    # ✅ Create filter mask
    mask = (points[:, range_index] >= range_min) & (points[:, range_index] <= range_max)

    logger.info("✅ Range filter applied: %d points kept, %d points removed",
                np.sum(mask), np.sum(~mask))

    return mask
