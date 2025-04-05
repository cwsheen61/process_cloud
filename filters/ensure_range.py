import numpy as np
import logging
from scipy.interpolate import interp1d
from numpy.linalg import norm

logger = logging.getLogger(__name__)

def ensure_range_from_trajectory(points: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """
    Ensures the point cloud has a 'range' field.
    If not present, calculates it using the trajectory data.

    Args:
        points (np.ndarray): Structured array of point data.
        trajectory (np.ndarray): Structured array of trajectory data.

    Returns:
        np.ndarray: Points with 'range' field added or preserved.
    """

    if "range" in points.dtype.names:
        logger.info("🟢 'range' already exists in point cloud — no calculation needed.")
        return points

    required_fields = ["GpsTime", "x", "y", "z"]
    if not all(field in points.dtype.names for field in required_fields):
        logger.error(f"❌ Chunk missing required fields: {', '.join(required_fields)}.")
        return points

    if not all(field in trajectory.dtype.names for field in ["GpsTime", "x", "y", "z"]):
        logger.error("❌ Trajectory missing required fields: GpsTime, x, y, z.")
        return points

    try:
        # Interpolate scanner position at each point's time
        interp_x = interp1d(trajectory["GpsTime"], trajectory["x"], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(trajectory["GpsTime"], trajectory["y"], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_z = interp1d(trajectory["GpsTime"], trajectory["z"], kind='linear', bounds_error=False, fill_value="extrapolate")

        scanner_x = interp_x(points["GpsTime"])
        scanner_y = interp_y(points["GpsTime"])
        scanner_z = interp_z(points["GpsTime"])

        vectors = np.stack([
            points["x"] - scanner_x,
            points["y"] - scanner_y,
            points["z"] - scanner_z
        ], axis=-1)

        ranges = norm(vectors, axis=1)

        # Add range field to structured array
        points = append_field(points, 'range', ranges.astype(np.float32))
        logger.info("📏 Computed and injected 'range' using trajectory.")

        return points

    except Exception as e:
        logger.error(f"❌ Failed to compute range: {e}")
        return points


def append_field(arr, name, data):
    """Add a new field to a structured numpy array."""
    from numpy.lib import recfunctions as rfn
    return rfn.append_fields(arr, name, data, usemask=False)
