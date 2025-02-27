import numpy as np
import laspy
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.data_types import trajectory_sensor_dtype  # Ensure dtype consistency
from modules.crs_registry import crs_registry, epsg_to_wkt, get_crs  # Ensure CRS tracking is consistent

logger = logging.getLogger(__name__)

def append_trajectory_to_laz(file_path, trajectory_points):
    """
    Writes transformed sensor trajectory data to a LAZ file.

    - Uses structured NumPy arrays (dtype: trajectory_sensor_dtype).
    - Ensures only valid (finite) points are written.
    - Lets laspy handle offsets and scaling.
    - Writes GPS time if available.

    Args:
        file_path (str): Output LAZ file path.
        trajectory_points (numpy.ndarray): Sensor trajectory data (structured array).
    """

    if trajectory_points.size == 0:
        logger.warning(f"Skipping {file_path}: No valid trajectory points.")
        return

    structured = isinstance(trajectory_points, np.ndarray) and trajectory_points.dtype.names is not None

    # Validate presence of required fields
    required_fields = {"traj_x", "traj_y", "traj_z"}
    missing_fields = required_fields - set(trajectory_points.dtype.names) if structured else set()
    if missing_fields:
        logger.error(f"ERROR: Trajectory dataset is missing required fields: {missing_fields}")
        return

    # Apply valid mask (filter out NaN or infinite values)
    valid_mask = np.isfinite(trajectory_points["traj_x"]) & np.isfinite(trajectory_points["traj_y"]) & np.isfinite(trajectory_points["traj_z"])
    valid_points = trajectory_points[valid_mask]

    if valid_points.size == 0:
        logger.warning(f"Skipping {file_path}: All trajectory points are NaN or invalid.")
        return

    # Retrieve EPSG from crs_registry (JIT)
    epsg_code = get_crs("trajectory")  # Fetch EPSG from registry
    crs_wkt = epsg_to_wkt(epsg_code) if epsg_code > 0 else None

    # Let laspy handle scales and offsets automatically
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])  # Default precision
    header.offsets = np.array([
        np.min(valid_points["traj_x"]),
        np.min(valid_points["traj_y"]),
        np.min(valid_points["traj_z"])
    ])

    # Convert EPSG to WKT and attach to header
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))

    # Create a new LAS data object
    las = laspy.LasData(header)

    # Store in correct integer LAS format
    las.X = np.round((valid_points["traj_x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
    las.Y = np.round((valid_points["traj_y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
    las.Z = np.round((valid_points["traj_z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

    # Set GPS Time if available
    las.gps_time = valid_points["time"].astype(np.float64) if "time" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.float64)

    # Default values for classification, scan angle, and user data
    las.classification = np.zeros(valid_points.shape[0], dtype=np.uint8)
    las.scan_angle_rank = np.zeros(valid_points.shape[0], dtype=np.int8)
    las.user_data = np.zeros(valid_points.shape[0], dtype=np.uint8)

    # Store CRS in registry
    crs_registry[file_path] = epsg_code

    # Write to LAZ file
    try:
        with laspy.open(file_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        logger.info(f"Successfully wrote {valid_points.shape[0]} trajectory points to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to write trajectory points to {file_path}: {e}")
