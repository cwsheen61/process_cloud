import numpy as np
import logging
from modules.crs_registry import crs_registry
from modules.transform_to_epsg import transform_to_epsg  # Import function for EPSG transformation

logger = logging.getLogger(__name__)  # ✅ Use logging instead of print()

# Define expected field names for each dataset type
SENSOR_TRAJECTORY_FIELDS = ("traj_x", "traj_y", "traj_z")  # Sensor trajectory (original coordinate system)
LOCAL_TRAJECTORY_FIELDS = ("local_x", "local_y", "local_z")  # GNSS-aligned local trajectory
POINT_CLOUD_FIELDS = ("x", "y", "z")  # Point cloud data (already transformed)
NORMAL_FIELDS = ("nx", "ny", "nz")  # Pseudo-normal vectors (if present)

def apply_transformation(chunk, R, t, target_epsg=None):
    """
    Applies a 3D transformation (rotation around Z, full translation) to the dataset
    while preserving all attributes, rotating normals if present, and updating the CRS.

    Args:
        chunk (np.ndarray): Structured array representing a chunk of trajectory or point cloud data.
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 1x3 translation vector.
        target_epsg (int, optional): EPSG code of the transformed coordinate system.

    Returns:
        np.ndarray: Transformed dataset (including rotated normals if present).
    """
    # Retrieve target EPSG from CRS registry if not provided
    if target_epsg is None:
        target_epsg = crs_registry.get("trajectory", None)  # Use computed UTM EPSG from GNSS data
        if target_epsg is None:
            raise ValueError("❌ ERROR: Target EPSG is not defined in crs_registry!")

    crs_registry["points"] = target_epsg  # ✅ Store updated CRS

    transformed_chunk = np.empty_like(chunk)

    # Identify dataset type by checking field names
    if set(SENSOR_TRAJECTORY_FIELDS).issubset(chunk.dtype.names):
        x_field, y_field, z_field = SENSOR_TRAJECTORY_FIELDS
        dataset_type = "SENSOR_TRAJECTORY"
    elif set(LOCAL_TRAJECTORY_FIELDS).issubset(chunk.dtype.names):
        x_field, y_field, z_field = LOCAL_TRAJECTORY_FIELDS
        dataset_type = "LOCAL_TRAJECTORY"
    elif set(POINT_CLOUD_FIELDS).issubset(chunk.dtype.names):
        x_field, y_field, z_field = POINT_CLOUD_FIELDS
        dataset_type = "POINT_CLOUD"
    else:
        raise ValueError("❌ apply_transformation(): Could not determine dataset type. Missing expected field names.")

    logger.info(f"🔄 Applying transformation to {dataset_type} data...")

    # --- Apply Transformation to Position Data ---
    xyz = np.column_stack((chunk[x_field], chunk[y_field], chunk[z_field]))
    xyz_transformed = np.dot(R, xyz.T).T + t  # ✅ Apply rotation & translation

    # Assign transformed values
    transformed_chunk[x_field] = xyz_transformed[:, 0]
    transformed_chunk[y_field] = xyz_transformed[:, 1]
    transformed_chunk[z_field] = xyz_transformed[:, 2]

    # --- Handle Normals if Present ---
    if set(NORMAL_FIELDS).issubset(chunk.dtype.names):
        logger.info("🔄 Rotating Normals with Rotation Matrix R...")

        normal_vectors = np.column_stack((chunk["nx"], chunk["ny"], chunk["nz"]))
        normals_transformed = np.dot(R, normal_vectors.T).T  # ✅ Rotate normals only (no translation)

        transformed_chunk["nx"] = normals_transformed[:, 0]
        transformed_chunk["ny"] = normals_transformed[:, 1]
        transformed_chunk["nz"] = normals_transformed[:, 2]

        # ✅ Update CRS registry to mark that normals exist
        crs_registry["has_pseudo_normals"] = 1  
        logger.info("✅ Pseudo-normals detected & rotated.")

    # --- Preserve all other attributes ---
    for attr in chunk.dtype.names:
        if attr not in (x_field, y_field, z_field, "nx", "ny", "nz"):  # Exclude transformed fields
            transformed_chunk[attr] = chunk[attr]

    # --- Ensure CRS transformation if necessary ---
    if crs_registry.get("points") != target_epsg:
        logger.info(f"🔄 Transforming {dataset_type} to EPSG:{target_epsg}...")
        transformed_chunk = transform_to_epsg(transformed_chunk, target_epsg)

    return transformed_chunk























# import numpy as np
# import logging
# from crs_registry import crs_registry
# from transform_to_epsg import transform_to_epsg  # Import function for EPSG transformation

# logger = logging.getLogger(__name__)  # ✅ Use logging instead of print()

# # Define expected field names for each dataset type
# SENSOR_TRAJECTORY_FIELDS = ("traj_x", "traj_y", "traj_z")  # Sensor trajectory (original coordinate system)
# LOCAL_TRAJECTORY_FIELDS = ("local_x", "local_y", "local_z")  # GNSS-aligned local trajectory
# POINT_CLOUD_FIELDS = ("x", "y", "z")  # Point cloud data (already transformed)

# def apply_transformation(chunk, R, t, target_epsg=None):
#     """
#     Applies a 3D transformation (rotation around Z, full translation) to the dataset
#     while preserving all attributes and updating the CRS.

#     Args:
#         chunk (np.ndarray): Structured array representing a chunk of trajectory or point cloud data.
#         R (np.ndarray): 3x3 rotation matrix (ensuring only Z-axis rotation).
#         t (np.ndarray): 1x3 translation vector.
#         target_epsg (int, optional): EPSG code of the transformed coordinate system.
#                                      If None, it retrieves from `crs_registry`.

#     Returns:
#         np.ndarray: Transformed dataset (sensor trajectory, local trajectory, or point cloud).
#     """
#     global crs_registry  # Ensure CRS tracking

#     # Retrieve target EPSG from CRS registry if not provided
#     if target_epsg is None:
#         target_epsg = crs_registry.get("trajectory", None)  # Use computed UTM EPSG from GNSS data
#         if target_epsg is None:
#             raise ValueError("ERROR: Target EPSG is not defined in crs_registry!")

#     crs_registry["points"] = target_epsg  # Store updated CRS

#     transformed_chunk = np.empty_like(chunk)

#     # Identify dataset type by checking field names
#     if set(SENSOR_TRAJECTORY_FIELDS).issubset(chunk.dtype.names):
#         x_field, y_field, z_field = SENSOR_TRAJECTORY_FIELDS
#         dataset_type = "SENSOR_TRAJECTORY"
#     elif set(LOCAL_TRAJECTORY_FIELDS).issubset(chunk.dtype.names):
#         x_field, y_field, z_field = LOCAL_TRAJECTORY_FIELDS
#         dataset_type = "LOCAL_TRAJECTORY"
#     elif set(POINT_CLOUD_FIELDS).issubset(chunk.dtype.names):
#         x_field, y_field, z_field = POINT_CLOUD_FIELDS
#         dataset_type = "POINT_CLOUD"
#     else:
#         raise ValueError("apply_transformation(): Could not determine dataset type. Missing expected field names.")

#     logger.info(f"🔄 Applying transformation to {dataset_type} data...")

#     # Apply transformation
#     xyz = np.column_stack((chunk[x_field], chunk[y_field], chunk[z_field]))
#     xyz_transformed = np.dot(R, xyz.T).T + t  # Apply rotation & translation

#     # Assign transformed values
#     transformed_chunk[x_field] = xyz_transformed[:, 0]
#     transformed_chunk[y_field] = xyz_transformed[:, 1]
#     transformed_chunk[z_field] = xyz_transformed[:, 2]

#     # Preserve all other attributes
#     for attr in chunk.dtype.names:
#         if attr not in (x_field, y_field, z_field):  # Preserve non-coordinate fields
#             transformed_chunk[attr] = chunk[attr]

#     # Ensure CRS transformation to correct UTM zone if necessary
#     if crs_registry.get("points") != target_epsg:
#         logger.info(f"🔄 Transforming {dataset_type} to EPSG:{target_epsg}...")
#         transformed_chunk = transform_to_epsg(transformed_chunk, target_epsg)

#     return transformed_chunk
