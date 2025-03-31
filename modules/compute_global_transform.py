import numpy as np
import os
import logging
from pyproj import Transformer, CRS
from modules.json_registry import JSONRegistry
from modules.crs_registry import crs_registry

logger = logging.getLogger(__name__)

def compute_global_transform(config_path):
    """Computes the global transformation from GNSS trajectory data with UTM conversion."""
    # Load configuration
    config = JSONRegistry(config_path, config_path)
    gnss_file = config.get("files.gnss_trajectory")
    
    if not os.path.exists(gnss_file):
        logger.error(f"❌ GNSS trajectory file not found: {gnss_file}")
        raise FileNotFoundError(f"GNSS trajectory file not found: {gnss_file}")
    
    logger.info(f"🔄 Computing global transformation using GNSS file: {gnss_file} ({__name__}:{__import__('inspect').currentframe().f_lineno})")
    
    # Load GNSS trajectory data; skip the header row
    try:
        gnss_data = np.loadtxt(gnss_file, dtype=np.float64, skiprows=1)
    except Exception as e:
        logger.error(f"❌ Failed to load GNSS trajectory data: {e}")
        raise
    
    # Extract sensor (local) coordinates from columns 0,1,2
    try:
        traj_x = gnss_data[:, 0]
        traj_y = gnss_data[:, 1]
        traj_z = gnss_data[:, 2]
        # Extract longitude, latitude, and elevation from columns 8,9,10 (0-indexed)
        gps_lon = gnss_data[:, 8]
        gps_lat = gnss_data[:, 9]
        gps_alt = gnss_data[:, 10]
    except IndexError:
        logger.error("❌ Unexpected GNSS data format: insufficient columns.")
        raise ValueError("Unexpected GNSS data format: insufficient columns.")
    
    # Determine UTM zone using mean GNSS longitude and latitude.
    lon_center = np.mean(gps_lon)
    lat_center = np.mean(gps_lat)
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = "north" if lat_center >= 0 else "south"
    utm_epsg = f"326{utm_zone}" if hemisphere == "north" else f"327{utm_zone}"
    logger.info(f"✅ GNSS Data UTM Zone: {utm_zone} ({hemisphere}) → EPSG:{utm_epsg}")
    
    # Convert GNSS longitude/latitude to UTM coordinates.
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    utm_e, utm_n = transformer.transform(gps_lon, gps_lat)
    
    # Construct GNSS positions in UTM coordinates (using gps_alt for elevation)
    gnss_utm_positions = np.column_stack((utm_e, utm_n, gps_alt))
    
    # For now, assume all points are high quality.
    high_quality_mask = np.ones(len(gnss_utm_positions), dtype=bool)
    if np.sum(high_quality_mask) < 3:
        raise ValueError("❌ Not enough high-quality GNSS points for a stable transformation!")
    
    gnss_filtered = gnss_utm_positions[high_quality_mask]
    traj_filtered = np.column_stack((traj_x, traj_y, traj_z))[high_quality_mask]
    
    # Compute transformation (R and t) using SVD.
    gnss_center = np.mean(gnss_filtered, axis=0)
    traj_center = np.mean(traj_filtered, axis=0)
    gnss_shifted = gnss_filtered - gnss_center
    traj_shifted = traj_filtered - traj_center
    
    U, _, Vt = np.linalg.svd(np.dot(gnss_shifted.T, traj_shifted))
    R = np.dot(Vt.T, U.T)
    # Enforce Z-axis remains unchanged (identity for Z)
    R = np.array([
        [R[0, 0], R[0, 1], 0],
        [R[1, 0], R[1, 1], 0],
        [0,       0,       1]
    ])
    
    # Compute translation vector
    t = gnss_center - np.dot(R, traj_center)
    
    # Get the CRS in WKT format.
    crs = CRS.from_epsg(int(utm_epsg))
    crs_wkt = crs.to_wkt()
    
    # Store computed transformation in the configuration.
    config.set("crs.epsg", int(utm_epsg))
    config.set("transformation.R", R.tolist())
    config.set("transformation.t", t.tolist())
    config.set("files.gnss_used", gnss_file)
    crs_registry["trajectory"] = int(utm_epsg)
    config.save()
    
    logger.info(f"✅ Computed Global Transformation:\nR =\n{R}\nt = {t}")
    logger.info(f"✅ Computed WKT CRS: {crs_wkt}")
    logger.info("✅ Updated config.json with transformation parameters.")



# import numpy as np
# import os
# import logging
# from modules.json_registry import JSONRegistry

# def compute_global_transform(config_path):
#     """Computes the global transformation from GNSS trajectory data."""
#     logger = logging.getLogger(__name__)
    
#     # Load config
#     config = JSONRegistry(config_path, config_path)
#     gnss_file = config.get("files.gnss_trajectory")
    
#     if not os.path.exists(gnss_file):
#         logger.error(f"❌ GNSS trajectory file not found: {gnss_file}")
#         raise FileNotFoundError(f"GNSS trajectory file not found: {gnss_file}")
    
#     logger.info(f"🔄 Computing global transformation using GNSS file: {gnss_file} ({__name__}:{__import__('inspect').currentframe().f_lineno})")
    
#     # Load GNSS trajectory data
#     try:
#         gnss_data = np.loadtxt(gnss_file, dtype=np.float64, skiprows=1)
#     except Exception as e:
#         logger.error(f"❌ Failed to load GNSS trajectory data: {e}")
#         raise
    
#     # Extract coordinates from GNSS data
#     try:
#         traj_x, traj_y, traj_z = gnss_data[:, 0], gnss_data[:, 1], gnss_data[:, 2]
#         gps_x, gps_y, gps_z = gnss_data[:, 3], gnss_data[:, 4], gnss_data[:, 5]
#     except IndexError:
#         logger.error("❌ Unexpected GNSS data format: insufficient columns.")
#         raise ValueError("Unexpected GNSS data format: insufficient columns.")
    
#     # Compute translation
#     t_global = np.mean([gps_x - traj_x, gps_y - traj_y, gps_z - traj_z], axis=1)
    
#     # Compute rotation: identity matrix for now (placeholder for refinement)
#     R_global = np.eye(3)
    
#     # Store results in config.json
#     config.set("transformation.R", R_global.tolist())
#     config.set("transformation.t", t_global.tolist())
#     config.save()
    
#     logger.info(f"✅ Computed transformation stored in config.json ({__name__}:{__import__('inspect').currentframe().f_lineno})")


# import os
# import numpy as np
# from scipy.interpolate import interp1d
# from pyproj import CRS, Transformer
# from modules.data_types import gnss_trajectory_dtype  # Ensure dtype consistency
# from modules.crs_registry import crs_registry  # Import CRS registry for consistency
# import logging

# logger = logging.getLogger(__name__)

# def compute_global_transform(config):
#     """
#     Computes a 3D transformation (rotation R and translation t) to align the sensor trajectory
#     with the sparse GNSS coordinates from the file specified in the config.

#     - Retrieves the GNSS file path from the config.
#     - Loads GNSS trajectory data and computes transformation.
#     - Stores computed parameters in the JSON registry for later use.

#     Args:
#         config (JSONRegistry): Configuration object storing file paths and settings.

#     Returns:
#         R (3x3 ndarray): Rotation matrix (Z-axis only)
#         t (1x3 ndarray): Translation vector (including Z)
#         crs_wkt (str): WKT representation of the GNSS CRS
#     """

#     # ✅ Extract GNSS file path from `config.json`
#     gnss_file = os.path.join(config.get("pathname"), config.get("files.gnss"))

#     if not os.path.exists(gnss_file):
#         logger.error(f"❌ GNSS trajectory file not found: {gnss_file}")
#         return None, None, None

#     try:
#         logger.info(f"📡 Loading GNSS trajectory for transformation from {gnss_file}...")

#         # Load GNSS data using the structured dtype
#         gnss_data = np.loadtxt(gnss_file, dtype=gnss_trajectory_dtype, delimiter=None, comments="%", skiprows=1)

#         # Extract required fields
#         local_x, local_y, local_z = gnss_data["local_x"], gnss_data["local_y"], gnss_data["local_z"]
#         gps_x, gps_y, gps_z = gnss_data["gps_x"], gnss_data["gps_y"], gnss_data["gps_z"]
#         longitude, latitude = gnss_data["longitude"], gnss_data["latitude"]
#         fix_quality = gnss_data["fix_quality"]

#         logger.info(f"✅ GNSS Fields Loaded: {gnss_data.dtype.names}")

#         # Determine UTM Zone from GNSS mean longitude/latitude
#         lon_center, lat_center = np.mean(longitude), np.mean(latitude)
#         utm_zone = int((lon_center + 180) / 6) + 1
#         hemisphere = "north" if lat_center >= 0 else "south"
#         utm_epsg = f"326{utm_zone}" if hemisphere == "north" else f"327{utm_zone}"

#         logger.info(f"✅ GNSS Data UTM Zone: {utm_zone} ({hemisphere}) → EPSG:{utm_epsg}")

#         # Convert GNSS Lon/Lat to UTM
#         transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
#         utm_e, utm_n = transformer.transform(longitude, latitude)

#         # Store GNSS positions in UTM coordinates
#         gnss_utm_positions = np.column_stack((utm_e, utm_n, gps_z))

#         # Filter for only high-quality GNSS fixes (Fix 4 or Fix 5)
#         high_quality_mask = (fix_quality == 4) | (fix_quality == 5)
#         if np.sum(high_quality_mask) < 3:
#             raise ValueError("❌ Not enough high-quality GNSS points for a stable R, T matrix!")

#         gnss_filtered = gnss_utm_positions[high_quality_mask]
#         traj_filtered = np.column_stack((local_x, local_y, local_z))[high_quality_mask]  # ✅ Use GNSS-aligned local trajectory

#         # Compute transformation (R, t)
#         gnss_center = np.mean(gnss_filtered, axis=0)
#         traj_center = np.mean(traj_filtered, axis=0)
#         gnss_shifted = gnss_filtered - gnss_center
#         traj_shifted = traj_filtered - traj_center

#         U, _, Vt = np.linalg.svd(np.dot(gnss_shifted.T, traj_shifted))
#         R = np.dot(Vt.T, U.T)

#         # Ensure a full 3D transformation matrix (Z-axis remains identity)
#         R = np.array([
#             [R[0, 0], R[0, 1], 0],
#             [R[1, 0], R[1, 1], 0],
#             [0, 0, 1]  # Z remains fixed
#         ])

#         # Compute full 3D translation
#         t = gnss_center - np.dot(R, traj_center)

#         # Get CRS in WKT format
#         crs = CRS.from_epsg(int(utm_epsg))
#         crs_wkt = crs.to_wkt()

#         # ✅ Store in `config.json`
#         config.set("crs.epsg", int(utm_epsg))
#         config.set("transformation.R", R.tolist())  # Convert NumPy array to list for JSON compatibility
#         config.set("transformation.t", t.tolist())  # Convert NumPy array to list for JSON compatibility
#         config.set("files.gnss_used", gnss_file)  # Store file reference for debugging

#         # ✅ Store EPSG code in `crs_registry` for consistency
#         crs_registry["trajectory"] = int(utm_epsg)

#         logger.info(f"✅ Computed Global Transformation:\nR =\n{R},\nt = {t}")
#         logger.info(f"✅ Computed WKT CRS: {crs_wkt}")
#         logger.info(f"✅ Updated config.json with transformation parameters.")

#         return R, t, crs_wkt

#     except ValueError as e:
#         logger.error(f"❌ Data format issue in '{gnss_file}': {e}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error loading GNSS trajectory from '{gnss_file}': {e}")

#     return None, None, None
