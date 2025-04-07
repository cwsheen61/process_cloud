import numpy as np
import logging
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)

def apply_kd_tree_filter(points, params, header=None):
    """
    Filters points based on the number of neighbors found within a radius using KD-Tree.

    Args:
        points (np.ndarray): Structured NumPy array from LAS data.
        params (dict): Config dictionary with:
            - 'kd_tree_min_dist' (optional): minimum distance threshold (in meters)
            - 'kd_tree_max_dist': maximum distance threshold (in meters)
            - 'kd_tree_min_neighbors': minimum required neighbors to keep point
        header (laspy.LasHeader, optional): Required to unscale X/Y/Z

    Returns:
        np.ndarray: Boolean mask, True = keep
    """
    if header is None:
        raise ValueError("Missing LAS header: required to compute scaled XYZ positions.")

    min_dist = params.get("kd_tree_min_dist", 0.0)
    max_dist = params.get("kd_tree_max_dist", 1.0)
    min_neighbors = params.get("kd_tree_min_neighbors", 1)

    coords = np.stack([
        points["X"] * header.scales[0] + header.offsets[0],
        points["Y"] * header.scales[1] + header.offsets[1],
        points["Z"] * header.scales[2] + header.offsets[2],
    ], axis=1)

    try:
        tree = KDTree(coords)
        ind = tree.query_radius(coords, r=max_dist, count_only=False)

        mask = np.full(len(points), False)

        for i, neighbors in enumerate(ind):
            if min_dist > 0:
                # filter neighbors inside the min_dist band
                dists = np.linalg.norm(coords[neighbors] - coords[i], axis=1)
                count = np.sum((dists >= min_dist) & (dists <= max_dist))
            else:
                count = len(neighbors) - 1  # exclude self

            if count >= min_neighbors:
                mask[i] = True

        logger.debug(f"🌲 KD-Tree: min_dist={min_dist}, max_dist={max_dist}, min_neighbors={min_neighbors} → kept {np.sum(mask)} / {len(points)}")
        return mask

    except Exception as e:
        logger.error(f"❌ KD-tree filter failed: {e}")
        raise
