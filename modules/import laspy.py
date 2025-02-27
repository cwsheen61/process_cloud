import laspy
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.data_types import ply_dtype  # Import structured dtype

def append_to_laz(file_path, points, crs_wkt):
    """
    Writes valid points to a LAZ file with LAS 1.4 encoding.
    
    - Ensures only valid (finite) points are written.
    - Uses structured NumPy arrays for consistent handling.
    - Attaches CRS WKT metadata if available.
    """
    global crs_registry  # Ensure CRS tracking is global

    if points.size == 0:
        print(f"Skipping {file_path}: No valid points.")
        return

    # Validate presence of required fields
    required_fields = ["x", "y", "z"]
    if not all(field in points.dtype.names for field in required_fields):
        print(f"ERROR: Point dataset is missing required fields: {required_fields}")
        return

    # Apply valid mask (filter out NaN or infinite values)
    valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    valid_points = points[valid_mask]

    if valid_points.size == 0:
        print(f"Skipping {file_path}: All points are NaN or invalid.")
        return

    # Create a fresh LAS header for each chunk
    header = laspy.LasHeader(point_format=3, version="1.4")

    # Let laspy handle offsets and scales automatically
    header.scales = np.array([0.001, 0.001, 0.001])  # Default precision
    header.offsets = np.array([0, 0, 0])  # Allow laspy to auto-adjust offsets

    # Apply WKT CRS before writing
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))
        crs_registry[file_path] = crs_wkt  # Store CRS in registry

    # Create LAS data object
    las = laspy.LasData(header)

    try:
        # Assign coordinate data
        las.X = np.round(valid_points["x"] / header.scales[0]).astype(np.int32)
        las.Y = np.round(valid_points["y"] / header.scales[1]).astype(np.int32)
        las.Z = np.round(valid_points["z"] / header.scales[2]).astype(np.int32)
        
        # Assign other fields safely
        las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
        las.return_number = np.ones(valid_points.shape[0], dtype=np.uint8)
        las.number_of_returns = np.ones(valid_points.shape[0], dtype=np.uint8)
        las.point_source_id = valid_points["ring"] if "ring" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
        las.gps_time = valid_points["time"] if "time" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.float64)
        las.classification = np.zeros(valid_points.shape[0], dtype=np.uint8)
        las.scan_angle_rank = np.zeros(valid_points.shape[0], dtype=np.int8)
        las.user_data = np.zeros(valid_points.shape[0], dtype=np.uint8)

        # Ensure "range" is included if available
        if "range" in valid_points.dtype.names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
            las["range"] = valid_points["range"].astype(np.float32)

    except Exception as e:
        print(f"ERROR in writing {file_path}: {e}")
        return

    # Write data to LAZ file
    with laspy.open(file_path, mode="w", header=header) as writer:
        writer.write_points(las.points)

    print(f"Successfully wrote {valid_points.shape[0]} points to {file_path}.")
