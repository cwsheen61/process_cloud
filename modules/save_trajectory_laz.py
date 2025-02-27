import laspy
import numpy as np
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.crs_registry import crs_registry  # Ensure global CRS tracking

logger = logging.getLogger(__name__)

def save_trajectory_laz(sensor_traj_data, R_global, t_global, output_path):
    """
    Saves transformed sensor trajectory to LAZ format while preserving quaternion data.

    Args:
        sensor_traj_data (numpy.ndarray): Sensor trajectory structured array.
        R_global (numpy.ndarray): 3x3 rotation matrix.
        t_global (numpy.ndarray): 1x3 translation vector.
        output_path (str): Output LAZ file path.
    """

    logger.info(f"Saving transformed sensor trajectory to {output_path}...")

    # Extract structured data from the sensor trajectory
    time = sensor_traj_data["time"]
    xyz = np.column_stack((sensor_traj_data["x"], sensor_traj_data["y"], sensor_traj_data["z"]))  # (N,3)
    quaternions = np.column_stack((sensor_traj_data["qw"], sensor_traj_data["qx"], 
                                   sensor_traj_data["qy"], sensor_traj_data["qz"]))  # (N,4)

    # Apply transformation (GNSS correction)
    transformed_xyz = (R_global @ xyz.T).T + t_global  # Full 3D transformation applied
    transformed_x, transformed_y, transformed_z = transformed_xyz[:, 0], transformed_xyz[:, 1], transformed_xyz[:, 2]

    # Create fresh LAS 1.4 header
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])  # Millimeter precision
    header.offsets = np.array([t_global[0], t_global[1], t_global[2]])  # Full 3D offsets

    # Fetch and apply CRS from crs_registry
    crs_wkt = crs_registry.get("trajectory", None)  # Use "trajectory" key for consistency
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))
    else:
        logger.warning("No CRS found in crs_registry for trajectory data.")

    # Create LAS data object
    las = laspy.LasData(header)
    las.x = transformed_x
    las.y = transformed_y
    las.z = transformed_z
    las.gps_time = time  # Preserve timestamps

    # Store quaternion data as extra fields
    quaternion_fields = ["qw", "qx", "qy", "qz"]
    for idx, field in enumerate(quaternion_fields):
        las.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.float64))
        setattr(las, field, quaternions[:, idx])

    # Write to LAZ file
    with laspy.open(output_path, mode="w", header=header) as writer:
        writer.write_points(las.points)

    logger.info(f"Transformed sensor trajectory saved to {output_path}.")
