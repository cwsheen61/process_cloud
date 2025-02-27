import numpy as np
import logging
from scipy.spatial import cKDTree
from tqdm import tqdm  # ✅ Progress bar support

logger = logging.getLogger(__name__)

kd_tree = None  # Global variable for caching the k-d tree

def create_kd_tree(points, leaf_size=30):
    """
    Creates a k-d tree from the provided point cloud data.

    Args:
        points (np.ndarray): Nx3 array of (x, y, z) coordinates.
        leaf_size (int): Leaf size parameter for the k-d tree.
    """
    global kd_tree
    if points.shape[1] != 3:
        raise ValueError("❌ Expected Nx3 array for KD-Tree creation!")

    logger.info(f"📌 Building k-d tree with {points.shape[0]} points...")

    kd_tree = cKDTree(points, leafsize=leaf_size)

    logger.info("✅ k-d tree construction complete.")


def query_kdtree(points, k=10):
    """
    Queries the k-d tree for nearest neighbors and returns min neighbor distances.

    Args:
        points (np.ndarray): Nx3 array of (x, y, z) coordinates.
        k (int): Number of nearest neighbors to search for.

    Returns:
        np.ndarray: Minimum distances to the nearest neighbors.
    """
    if kd_tree is None:
        raise RuntimeError("❌ KD-Tree has not been created!")

    logger.info(f"🔎 Querying k-d tree with {points.shape[0]} points (k={k})...")

    # ✅ Use tqdm for a progress bar while processing
    distances, _ = kd_tree.query(points, k=k+1, workers=-1)  # k+1 to exclude self-match
    distances = distances[:, 1:]  # Remove self-distance (should be 0)

    # Compute min distance to nearest real neighbor
    min_distances = distances.min(axis=1)

    # ✅ Debugging: Log distance stats to check filter thresholds
    logger.info(f"📊 k-NN distance stats: min={min_distances.min():.6f}m, "
                f"max={min_distances.max():.6f}m, mean={min_distances.mean():.6f}m")

    return min_distances


def filter_by_knn_distance(points, min_kdd_distance, max_kdd_distance, k=10):
    """
    Filters points based on nearest neighbor distances using a bandpass filter.

    Args:
        points (np.ndarray): Nx3 array of (x, y, z) coordinates.
        min_kdd_distance (float): Minimum allowed neighbor distance.
        max_kdd_distance (float): Maximum allowed neighbor distance.
        k (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: Boolean mask of points that pass the filter.
    """
    logger.info(f"🛠 Applying k-NN filtering: {min_kdd_distance}m ≤ distance ≤ {max_kdd_distance}m")

    min_distances = query_kdtree(points, k=k)

    # ✅ Band-pass filter: Keep points within range
    knn_mask = (min_distances >= min_kdd_distance) & (min_distances <= max_kdd_distance)

    logger.info(f"✅ k-NN Filter Results: Passed={knn_mask.sum()}, Failed={(~knn_mask).sum()}")

    return knn_mask
