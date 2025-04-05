import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_motion_filter(points, filter_params):
    """
    Applies a motion filter that removes points with very little local variation,
    based on point-to-point distance.

    Args:
        points (np.ndarray): Structured array of point cloud data.
        filter_params (dict): Parameters including 'motion_threshold'.

    Returns:
        np.ndarray: Boolean mask indicating which points pass the filter.
    """
    required_fields = {"x", "y", "z"}
    if not required_fields.issubset(points.dtype.names):
        logger.error(f"❌ Motion filter requires fields: {required_fields}")
        raise ValueError(f"Missing required fields for motion filtering: {required_fields}")

    threshold = filter_params.get("motion_threshold", 0.01)

    # Calculate simple motion magnitude using finite difference
    coords = np.stack((points["x"], points["y"], points["z"]), axis=1)
    diffs = np.diff(coords, axis=0)
    motion_magnitude = np.linalg.norm(diffs, axis=1)

    # Shift to match original point array length
    motion_magnitude = np.insert(motion_magnitude, 0, motion_magnitude[0])
    mask = motion_magnitude > threshold

    logger.debug(f"📉 Motion filter threshold: {threshold} → {np.sum(mask)} / {len(points)} points kept.")
    return mask
