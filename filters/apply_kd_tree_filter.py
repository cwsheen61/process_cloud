import numpy as np
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger(__name__)

def apply_kd_tree_filter(points, params):
    """
    Filters points using a KD-tree-based local density check.
    Retains points with at least one neighbor within a given distance range.

    Args:
        points (np.ndarray): Input point cloud with 'X', 'Y', 'Z'
        params (dict): Config with:
            - 'kd_tree_leaf_size': leaf size
            - 'kd_tree_min_dist': min dist to keep
            - 'kd_tree_max_dist': max dist to keep

    Returns:
        np.ndarray: Boolean mask of valid points
    """
    required = ("x", "y", "z")
    for dim in required:
        if dim not in points.dtype.names:
            logger.error(f"❌ Missing required field '{dim}' for KD-tree filter.")
            raise ValueError(f"Missing required field: '{dim}'")

    x = points["x"]
    y = points["y"]
    z = points["z"]
    coords = np.stack([x, y, z], axis=1)

    leaf_size = params.get("kd_tree_leaf_size", 10)
    min_dist = params.get("kd_tree_min_dist", 0.005)
    max_dist = params.get("kd_tree_max_dist", 1.0)

    try:
        tree = KDTree(coords, leaf_size=leaf_size)
        ind = tree.query_radius(coords, r=max_dist)

        # Count how many neighbors each point has in the range
        mask = np.array([
            any(min_dist < np.linalg.norm(coords[i] - coords[j]) < max_dist for j in neighbors if j != i)
            for i, neighbors in enumerate(ind)
        ])

        return mask

    except Exception as e:
        logger.error(f"❌ KD-tree filter failed: {e}")
        raise
