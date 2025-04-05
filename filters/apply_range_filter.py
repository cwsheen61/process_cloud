import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_range_filter(points, filter_params):
    """
    Applies a range filter to the input points using the 'range' field.

    Args:
        points (np.ndarray): Structured array of point cloud data.
        filter_params (dict): Parameters including 'range_min' and 'range_max'.

    Returns:
        np.ndarray: Boolean mask indicating which points pass the filter.
    """
    if 'range' not in points.dtype.names:
        logger.error("❌ 'range' field not found in point cloud!")
        raise ValueError("Field 'range' not found in point cloud data.")

    range_min = filter_params.get("range_min", 0.0)
    range_max = filter_params.get("range_max", np.inf)

    mask = (points["range"] >= range_min) & (points["range"] <= range_max)

    logger.debug(f"🔎 Range filter [{range_min}, {range_max}] → {np.sum(mask)} / {len(points)} points kept.")
    return mask
