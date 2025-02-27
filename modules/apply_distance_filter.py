import numpy as np
import logging
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

def apply_distance_filter(points, config):
    """
    Filters out points that are too close (< min_dist) or too far (> max_dist) from their nearest neighbor.
    """
    min_dist = config["filtering"].get("kd_tree_min_dist", 0.005)
    max_dist = config["filtering"].get("kd_tree_max_dist", 1.0)

    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("Input must be an Nx3 numpy array.")

    logger.info(f"🔍 Analyzing {len(points)} points before filtering...")
    
    # Build KD-tree
    kdtree = KDTree(points)

    # Query nearest neighbor distances
    distances, _ = kdtree.query(points, k=2)  # k=2 because first result is self-distance (0)
    neighbor_distances = distances[:, 1]  # Extract second closest (true nearest neighbor)

    # Debugging Info
    logger.info(f"📊 Nearest Neighbor Distance Stats:")
    logger.info(f"   ➤ Min Distance: {np.min(neighbor_distances):.6f}")
    logger.info(f"   ➤ Max Distance: {np.max(neighbor_distances):.6f}")
    logger.info(f"   ➤ Mean Distance: {np.mean(neighbor_distances):.6f}")
    logger.info(f"   ➤ Median Distance: {np.median(neighbor_distances):.6f}")

    # Apply filtering
    valid_mask = (neighbor_distances >= min_dist) & (neighbor_distances <= max_dist)
    filtered_points = points[valid_mask]

    logger.info(f"✅ Points Kept: {len(filtered_points)} / {len(points)} ({len(filtered_points)/len(points)*100:.2f}%)")
    return filtered_points
