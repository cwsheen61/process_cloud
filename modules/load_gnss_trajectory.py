import os
import numpy as np
import logging
from modules.json_registry import JSONRegistry

logger = logging.getLogger(__name__)

def load_gnss_trajectory(config_path):
    """ Loads the GNSS trajectory data from current_config.json and ensures correct file path resolution. """

    # Load current config
    config = JSONRegistry(config_path, config_path)  # Correct initialization with json_path

    # Get GNSS file path
    gnss_filename = config.get("files.gnss")
    base_path = config.get("pathname", "")

    if not gnss_filename:
        logger.error("❌ GNSS trajectory file path is missing from config!")
        return None
    
    # Handle relative vs absolute path
    if not os.path.isabs(gnss_filename):
        logger.warning(f"⚠️ GNSS file path is relative: {gnss_filename}, appending base path: {base_path}")
        gnss_file = os.path.join(base_path, gnss_filename)
    else:
        gnss_file = gnss_filename

    # Verify existence AFTER forming full path
    if not os.path.exists(gnss_file):
        logger.error(f"❌ GNSS trajectory file not found: {gnss_file}")
        return None

    logger.info(f"✅ Using GNSS trajectory file: {gnss_file}")

    # Define dtype from config
    dtype_fields = config.get("data_formats.gnss_trajectory")
    if not dtype_fields:
        logger.error("❌ GNSS trajectory format missing in config!")
        return None
    
    dtype = np.dtype([(field, np.dtype(dtype_fields[field])) for field in dtype_fields])

    # Load the GNSS trajectory data
    try:
        with open(gnss_file, "r") as f:
            first_line = f.readline().strip()
            skip_rows = 1 if first_line.lower().startswith("traj_x") else 0

        gnss_data = np.loadtxt(gnss_file, dtype=dtype, skiprows=skip_rows)  # Skip header if needed
        logger.info(f"✅ Loaded GNSS trajectory: {gnss_file}, {gnss_data.shape[0]} entries.")
    except Exception as e:
        logger.error(f"❌ Error loading GNSS trajectory: {e}")
        return None
    
    # Update state in config
    config.set("current_trajectory_state", "gnss_loaded")
    config.save()

    return gnss_data



# import os
# import numpy as np
# import logging
# from modules.data_types import gnss_trajectory_dtype  # Import globally defined dtype
# from modules.crs_registry import crs_registry  # Ensure CRS tracking

# logger = logging.getLogger(__name__)

# def load_gnss_trajectory(config):
#     """
#     Loads GNSS trajectory data from a file specified in the config.

#     - Uses globally defined `gnss_trajectory_dtype`.
#     - Ignores comment lines starting with '%'.
#     - Stores its reference CRS in `crs_registry`.

#     Args:
#         config (JSONRegistry): Configuration object storing file paths and settings.

#     Returns:
#         np.ndarray or None: Loaded GNSS trajectory structured array, or None on failure.
#     """
#     global crs_registry  # Explicitly declare global

#     # ✅ Extract GNSS file path from config
#     file_path = os.path.join(config.get("pathname"), config.get("files.gnss"))

#     try:
#         logger.info(f"📡 Loading GNSS trajectory from {file_path}...")

#         # Ensure the file exists before attempting to load
#         if not os.path.exists(file_path):
#             logger.error(f"❌ GNSS trajectory file not found: {file_path}")
#             return None

#         # Determine whether to skip a header row
#         with open(file_path, "r") as f:
#             first_line = f.readline().strip()
#             skip_rows = 1 if first_line.lower().startswith("traj_x") else 0

#         # ✅ Load GNSS data from file
#         gnss_data = np.loadtxt(file_path, delimiter=None, dtype=gnss_trajectory_dtype, comments="%", skiprows=skip_rows)

#         # ✅ Register CRS in crs_registry (defaulting to WGS84 EPSG:4326)
#         crs_registry["gnss_traj_data"] = 4326

#         logger.info(f"✅ Successfully loaded GNSS trajectory from {file_path}. Assigned CRS: EPSG:4326")
#         return gnss_data

#     except ValueError as e:
#         logger.error(f"❌ Data format issue in '{file_path}': {e}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error loading GNSS trajectory from '{file_path}': {e}")

#     return None  # Return None on failure
