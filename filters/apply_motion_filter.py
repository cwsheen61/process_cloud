import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_motion_filter(points, motion_config, field_mapping):
    """
    Filters points based on their motion. The motion is computed as the Euclidean distance 
    between consecutive points' sensor coordinates (x, y, z). Points that show a difference 
    below the motion_threshold are filtered out.

    Args:
        points (np.ndarray): Structured array of point cloud sensor data.
        motion_config (dict): Contains key "motion_threshold" (float). 
                              Points with difference below this threshold are filtered out.
        field_mapping (dict): Mapping of field names; expected to contain "x", "y", and "z".

    Returns:
        np.ndarray: Boolean mask with True for points that pass the motion filter.
    """
    motion_threshold = motion_config.get("motion_threshold", 0.001)
    logger.info(f"📡 Applying motion filter with threshold {motion_threshold}")

    # Ensure the required fields exist.
    for field in ["x", "y", "z"]:
        if field not in field_mapping:
            logger.error(f"❌ '{field}' field not found in point cloud format!")
            raise ValueError(f"Field '{field}' not found in format definition.")

    # Extract coordinates from the structured array.
    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)

    # Compute differences between consecutive points.
    # For the first point, we can assume it always passes.
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    # Create a mask: True if the distance is greater than or equal to the threshold.
    mask = distances >= motion_threshold
    mask[0] = True  # Always include the first point.
    logger.info(f"Motion filter: {np.sum(mask)} of {len(points)} points pass.")
    return mask
