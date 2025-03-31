import numpy as np
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

def apply_knn_filter(points, knn_config, field_mapping):
    """
    Filters points based on the distances to their k nearest neighbors.
    
    For each point, the k-nearest neighbors are found using a KDTree.
    Then, the average distance (excluding the point itself) is computed.
    A point passes the filter if its average neighbor distance is less than or equal to
    the specified knn_distance_threshold.
    
    Args:
        points (np.ndarray): Structured array of point cloud data.
        knn_config (dict): Configuration for the KNN filter, expected to include:
            - "knn_neighbors": (int) number of neighbors to consider.
            - "knn_distance_threshold": (float) maximum allowable average distance.
        field_mapping (dict): Mapping of field names; expected to include "x", "y", and "z".
        
    Returns:
        np.ndarray: Boolean mask (of the same length as points) indicating which points pass.
    """
    # Ensure required fields are present.
    for field in ["x", "y", "z"]:
        if field not in field_mapping:
            logger.error(f"❌ '{field}' field not found in point cloud format!")
            raise ValueError(f"Field '{field}' not found in format definition.")
    
    # Extract coordinates from the structured array using field names.
    # Assumes that the structured array's field names correspond to the ones in field_mapping.
    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)
    coords = np.column_stack((x, y, z))
    
    # Build a KDTree for fast neighbor queries.
    tree = cKDTree(coords)
    
    # Get parameters from the configuration.
    k = knn_config.get("knn_neighbors", 10)
    threshold = knn_config.get("knn_distance_threshold", 0.05)
    
    # Query the k+1 nearest neighbors (the first neighbor is the point itself, distance zero).
    distances, _ = tree.query(coords, k=k+1)
    # Remove the zero distance (first column).
    distances = distances[:, 1:]
    
    # Compute the average distance for each point.
    mean_distances = np.mean(distances, axis=1)
    
    # Create a boolean mask: True if the average distance is less than or equal to the threshold.
    mask = mean_distances <= threshold
    logger.info(f"KNN filter: {np.sum(mask)} of {len(points)} points pass with threshold {threshold}")
    
    return mask
