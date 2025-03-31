import numpy as np
from modules.json_registry import JSONRegistry
import os
import sys
import logging


logger = logging.getLogger(__name__)

def load_trajectory(config_path):
    """Loads the trajectory data from a file and updates the current trajectory state in config."""
    
    # ✅ Load current_config.json
    with open(config_path, "r") as f:
        config = JSONRegistry(config_path, config_path)

    # # Ensure trajectory format exists in config
    if not config.get("data_formats.trajectory_sensor"):
        logger.error("❌ 'trajectory_sensor' format missing in config!")
        raise ValueError("Missing trajectory_sensor format in config.json")

# Retrieve trajectory file path from the config

    trajectory_file = config.get("files.trajectory")

    if not trajectory_file:
        logger.error("❌ Trajectory file path is missing in config!")
        raise ValueError("Missing trajectory file path in config.json")

    # Load trajectory data
    try:
        dtype = [
            ("time", np.float64),
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("qw", np.float64),
            ("qx", np.float64),
            ("qy", np.float64),
            ("qz", np.float64),
        ]

        # Read first row to check if it is a header
        with open(trajectory_file, "r") as f:
            first_line = f.readline().strip().split()
        
        # Check if the first row is non-numeric (i.e., likely a header)
        try:
            [float(value) for value in first_line]  # Try converting to float
            skip_rows = 0  # First row is data
        except ValueError:
            skip_rows = 1  # First row is a header

        # Load trajectory, skipping the header row if detected
        trajectory = np.loadtxt(trajectory_file, dtype=dtype, skiprows=skip_rows)

        logger.info(f"✅ Loaded trajectory file: {trajectory_file}, {len(trajectory)} entries.")

        # Update current trajectory state
        config.set("current_trajectory_state", "sensor_loaded")

        # ✅ Save updated config
        config.save()

        logger.info("📌 Updated current trajectory state: sensor_loaded")

        return trajectory

    except Exception as e:
        logger.error(f"❌ Error loading trajectory: {str(e)}")
        raise




# import numpy as np
# import json
# import os
# import logging

# logger = logging.getLogger(__name__)

# def load_trajectory(trajectory_file, config_path):
#     """Loads the trajectory data from a file and updates the current trajectory state in config."""
    
#     # ✅ Load current_config.json
#     with open(config_path, "r") as f:
#         config = json.load(f)

#     # Ensure trajectory format exists in config
#     if "trajectory_sensor" not in config["data_formats"]:
#         logger.error("❌ 'trajectory_sensor' format missing in config!")
#         raise ValueError("Missing trajectory_sensor format in config.json")

#     # Load trajectory data
#     try:
#         dtype = [
#             ("time", np.float64),
#             ("x", np.float64),
#             ("y", np.float64),
#             ("z", np.float64),
#             ("qw", np.float64),
#             ("qx", np.float64),
#             ("qy", np.float64),
#             ("qz", np.float64),
#         ]

#         trajectory = np.loadtxt(trajectory_file, dtype=dtype)

#         logger.info(f"✅ Loaded trajectory file: {trajectory_file}, {len(trajectory)} entries.")

#         # Update current trajectory state
#         config["current_trajectory_state"] = "sensor_loaded"

#         # ✅ Save updated config
#         with open(config_path, "w") as f:
#             json.dump(config, f, indent=2)

#         logger.info("📌 Updated current trajectory state: sensor_loaded")

#         return trajectory

#     except Exception as e:
#         logger.error(f"❌ Error loading trajectory: {str(e)}")
#         raise

