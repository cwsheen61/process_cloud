import numpy as np
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

def apply_kd_tree_filter(points, kd_tree_config, field_mapping):
    """
    Filters points based on the distance to their single nearest neighbor using a KDTree.
    
    This filter computes, for each point, the distance to its nearest neighbor (ignoring itself)
    and returns a boolean mask. A point passes if its nearest neighbor distance is between the
    specified minimum and maximum thresholds.
    
    The configuration (kd_tree_config) is expected to include:
      - "kd_tree_leaf_size": int, optional leaf size for building the KDTree (default: 10).
      - "kd_tree_min_dist": float, the minimum allowable distance (default: 0.005).
      - "kd_tree_max_dist": float, the maximum allowable distance (default: 1.0).
    
    Args:
        points (np.ndarray): Structured array of point cloud data.
        kd_tree_config (dict): Configuration for the KDTree filter.
        field_mapping (dict): Mapping of field names; expected to include "x", "y", and "z".
        
    Returns:
        np.ndarray: A boolean mask (of length equal to points) with True for points that pass.
    """
    # Ensure the required coordinate fields exist.
    for field in ["x", "y", "z"]:
        if field not in field_mapping:
            logger.error(f"❌ '{field}' field not found in point cloud format!")
            raise ValueError(f"Field '{field}' not found in format definition.")
    
    # Extract coordinate values from the structured array.
    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)
    coords = np.column_stack((x, y, z))
    
    # Retrieve configuration parameters.
    leaf_size = kd_tree_config.get("kd_tree_leaf_size", 10)
    min_dist = kd_tree_config.get("kd_tree_min_dist", 0.005)
    max_dist = kd_tree_config.get("kd_tree_max_dist", 1.0)
    
    # Build the KDTree with the specified leaf size.
    tree = cKDTree(coords, leafsize=leaf_size)
    
    # Query each point's 2 nearest neighbors (the first is the point itself, with distance 0).
    distances, _ = tree.query(coords, k=2)
    # The nearest neighbor distance is in the second column.
    nearest_neighbor_dist = distances[:, 1]
    
    # Create a mask: True if the nearest neighbor distance is between min_dist and max_dist.
    mask = (nearest_neighbor_dist >= min_dist) & (nearest_neighbor_dist <= max_dist)
    
    logger.info(f"KDTree filter: {np.sum(mask)} of {len(points)} points pass (min: {min_dist}, max: {max_dist})")
    return mask
