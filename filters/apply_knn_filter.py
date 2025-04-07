import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def apply_knn_filter(points, params, header=None):
    """
    Filters out points whose average distance to k nearest neighbors exceeds the threshold.

    Args:
        points (np.ndarray): Point cloud chunk from LAS (structured array).
        params (dict): Config with keys:
            - 'knn_neighbors': Number of neighbors to consider (default: 1)
            - 'knn_distance_threshold': Max average distance (in meters)
        header (laspy.LasHeader, optional): Needed to unscale X/Y/Z into real-world coordinates.

    Returns:
        np.ndarray: Boolean mask where True = keep, False = discard
    """
    if header is None:
        raise ValueError("Missing LAS header: required to compute scaled XYZ positions.")

    for dim in ("X", "Y", "Z"):
        if dim not in points.dtype.names:
            logger.error(f"❌ Missing required field '{dim}' in point cloud.")
            raise ValueError(f"Missing required field: {dim}")

    # Unscale X/Y/Z to real coordinates (meters)
    coords = np.stack([
        points["X"] * header.scales[0] + header.offsets[0],
        points["Y"] * header.scales[1] + header.offsets[1],
        points["Z"] * header.scales[2] + header.offsets[2]
    ], axis=1)

    k = int(params.get("knn_neighbors", 1))
    threshold = float(params.get("knn_distance_threshold", 1.0))

    if k < 1:
        raise ValueError("knn_neighbors must be ≥ 1")

    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        # Exclude distance to self (first column)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        mask = avg_distances <= threshold

        logger.debug(f"🔍 KNN filter: k={k}, threshold={threshold} → kept {np.sum(mask)} / {len(points)} points.")
        return mask

    except Exception as e:
        logger.error(f"❌ KNN filter failed: {e}")
        raise
