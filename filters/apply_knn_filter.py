import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

def apply_knn_filter(points, params):
    """
    Filters points using a k-nearest neighbors distance criterion.
    Points are kept if their mean distance to K neighbors is below a threshold.

    Args:
        points (np.ndarray): Input point cloud
        params (dict): Configuration with keys:
            - 'knn_neighbors': number of neighbors
            - 'knn_distance_threshold': distance threshold

    Returns:
        np.ndarray: Boolean mask of valid points
    """
    for dim in ("x", "y", "z"):
        if dim not in points.dtype.names:
            logger.error(f"❌ Missing required field '{dim}' in point cloud data.")
            raise ValueError(f"Missing required field: '{dim}'")

    coords = np.stack([points["x"], points["y"], points["z"]], axis=1)

    k = params.get("knn_neighbors", 10)
    threshold = params.get("knn_distance_threshold", 0.5)

    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        mask = mean_distances < threshold
        logger.debug(f"🔍 KNN filter applied with k={k}, threshold={threshold} → {np.sum(mask)} / {len(points)} points kept.")
        return mask
    except Exception as e:
        logger.error(f"❌ KNN filter failed: {e}")
        raise
