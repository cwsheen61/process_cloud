from pyproj import Transformer, CRS
import numpy as np
import logging
from modules.crs_registry import crs_registry  # Ensure CRS tracking

logger = logging.getLogger(__name__)

def transform_to_epsg(points, target_epsg):
    """
    Transforms XY(Z) coordinates from their current CRS (tracked in crs_registry) to target_epsg.
    If the current CRS is unknown, it raises an error.

    Args:
        points (numpy.ndarray): Structured array with fields 'x', 'y', and optional 'z'.
        target_epsg (int): EPSG code of the target CRS.

    Returns:
        numpy.ndarray: Transformed points with updated 'x', 'y', and 'z' (if applicable).
    """
    global crs_registry  # Ensure global access

    # Retrieve Source EPSG from crs_registry
    source_epsg = crs_registry.get("points", None)
    if source_epsg is None:
        raise ValueError("Cannot transform: CRS of 'points' is unknown! Check crs_registry.")

    # If already in target CRS, return unchanged
    if source_epsg == target_epsg:
        logger.info(f"Skipping transformation: Already in EPSG:{target_epsg}")
        return points

    logger.info(f"Transforming from EPSG:{source_epsg} to EPSG:{target_epsg}")

    # Initialize Transformer
    transformer = Transformer.from_crs(CRS.from_epsg(source_epsg), CRS.from_epsg(target_epsg), always_xy=True)

    # Prepare X, Y, (Z) data for transformation
    xy = np.column_stack((points["x"], points["y"]))
    transformed_points = points.copy()

    if "z" in points.dtype.names:
        z = points["z"]
        x_new, y_new, z_new = transformer.transform(xy[:, 0], xy[:, 1], z)
        transformed_points["z"] = z_new
    else:
        x_new, y_new = transformer.transform(xy[:, 0], xy[:, 1])

    transformed_points["x"] = x_new
    transformed_points["y"] = y_new

    # Update crs_registry
    crs_registry["points"] = target_epsg
    logger.info(f"Transformation complete: EPSG:{target_epsg}")

    return transformed_points
