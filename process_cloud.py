#!/usr/bin/env python3

"""
Processes a large PLY point cloud using a dense sensor trajectory and a sparse GNSS trajectory
to compute a 3D transformation (rotation about Z and translation). The transformed data is
optionally converted to a target CRS, streamed in chunks, and stored in LAZ format.
"""

# --- Global Imports ---
import sys
import os
import numpy as np
import laspy
import multiprocessing
from multiprocessing import Pool
import logging

# --- Ensure the modules directory is in Python's path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, "modules")
sys.path.append(module_dir)

# --- Module Imports ---
from modules.json_registry import JSONRegistry
from modules.process_chunk import process_chunk
from modules.load_trajectory import load_trajectory
from modules.compute_global_transform import compute_global_transform
from modules.apply_transformation import apply_transformation
from modules.load_ply_chunks import load_ply_chunks
from modules.merge_laz_files import merge_laz_files
from modules.append_trajectory_to_laz import append_trajectory_to_laz
from modules.crs_registry import get_crs, set_crs, epsg_to_wkt, print_crs_registry
from modules.prepare_temp_directory import prepare_temp_directory

# --- Setup Logging ---
def setup_logging(log_dir):
    """ Configures logging to write logs to a file in the PLY working directory. """
    log_file = os.path.join(log_dir, "process.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """ Main entry point for processing the PLY point cloud. """

    # --- Validate Command-Line Arguments ---
    if len(sys.argv) != 3:
        print("❌ Usage: process_cloud.py <input_ply> <config_json>")
        sys.exit(1)

    ply_path = sys.argv[1]
    json_config_path = sys.argv[2]
    ply_dir = os.path.dirname(ply_path)

    # --- Setup Logging in PLY Directory ---
    global logger
    logger = setup_logging(ply_dir)

    # --- Load Configuration & Initialize Current Config ---
    logger.info("🔍 Loading JSON configuration...")
    config = JSONRegistry(ply_path, json_config_path)

    # Create a working copy of the config in the same directory as the PLY file
    current_config_path = os.path.join(ply_dir, "current_config.json")
    config.save_as(current_config_path)
    logger.info(f"✅ Initialized current config: {current_config_path}")

    # --- Load Trajectory File ---
    trajectory_file = os.path.join(ply_dir, config.get("files.trajectory"))

    if not os.path.exists(trajectory_file):
        logger.error(f"❌ Trajectory file not found: {trajectory_file}")
        sys.exit(1)

    traj_data = load_trajectory(trajectory_file, current_config_path)
    logger.info(f"✅ Loaded trajectory: {trajectory_file}")

    # # --- Load GNSS Trajectory File ---
    # gnss_file = os.path.join(ply_dir, config.get("files.gnss"))

    # if not os.path.exists(gnss_file):
    #     logger.warning(f"⚠️ GNSS file not found: {gnss_file}. Transformation may not be possible.")
    #     gnss_data = None
    # else:
    #     gnss_data = load_gnss_trajectory(current_config_path)
    #     logger.info(f"✅ Loaded GNSS trajectory: {gnss_file}")

    # --- Compute Global Transformation ---
    compute_global_transform(current_config_path)


    config = JSONRegistry(current_config_path, current_config_path)  # Reinitialize to get updated values

    R_global = np.array(config.get("transformation.R"))
    t_global = np.array(config.get("transformation.t"))
    crs_epsg = config.get("crs.epsg_code")

    logger.info(f"✅ Computed global transform: R={R_global.tolist()}, t={t_global.tolist()}, EPSG={crs_epsg} (process_cloud.py:96)")

    # --- TEMPORARY EXIT FOR DEBUGGING AFTER COMPUTE_GLOBAL_TRANSFORM ---
    logger.info("🚧 Debugging exit after computing global transform.")
    sys.exit(0)

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3

# import sys
# import os
# import logging
# import multiprocessing
# import shutil
# from modules.json_registry import JSONRegistry
# from modules.process_chunk import process_chunk
# from modules.load_trajectory import load_trajectory
# from modules.compute_global_transformations import compute_global_transform
# from modules.apply_transformation import apply_transformation
# from modules.point_cloud_io import load_ply_chunks, merge_laz_files, append_trajectory_to_laz
# from modules.crs_registry import get_crs, print_crs_registry
# from modules.temp_manager import prepare_temp_directory

# # Initialize logger
# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger(__name__)

# def main():
#     """ Main function for processing GNSS and sensor trajectory data. """

#     # --- Validate Arguments ---
#     if len(sys.argv) != 3:
#         logger.error("❌ Usage: ./process_cloud.py <path_to_ply> <path_to_config.json>")
#         sys.exit(1)

#     ply_path = sys.argv[1]
#     json_config_path = sys.argv[2]

#     # ✅ Load JSON Configuration & Update Path
#     config = JSONRegistry(ply_path, json_config_path)

#     # ✅ Initialize and create `current_config.json` for state tracking
#     current_config_path = os.path.join(os.path.dirname(json_config_path), "current_config.json")
#     if os.path.exists(current_config_path):
#         logger.warning("⚠️ Overwriting existing current_config.json")
#     shutil.copy(json_config_path, current_config_path)  # Copy initial config

#     # ✅ Load `current_config.json`
#     current_config = JSONRegistry(ply_path, current_config_path)

#     # ✅ Ensure `valid_states` exist in config
#     if "valid_states" not in current_config.config:
#         logger.error("❌ No 'valid_states' defined in config.json! Add this section to track processing states.")
#         sys.exit(1)

#     # ✅ Ensure `current_state` tracking exists
#     if "current_state" not in current_config.config:
#         current_config.set("current_state", [])

#     # ✅ Add initial state if empty
#     if not current_config.get("current_state"):
#         initial_state = "initial_loaded"
#         current_config.set("current_state", [initial_state])
#         current_config.save()
#         logger.info(f"✅ Initial processing state set: {initial_state}")

#     # ✅ Construct filenames dynamically from JSON
#     ply_dir = config.get("pathname")
#     trajectory_file = os.path.join(ply_dir, config.get("files.trajectory"))
#     gnss_file = os.path.join(ply_dir, config.get("files.gnss"))
#     output_pass_path = os.path.join(ply_dir, config.get("files.output_pass"))
#     output_fail_path = os.path.join(ply_dir, config.get("files.output_fail"))

#     # ✅ Ensure required files exist
#     for file_path, label in [(trajectory_file, "Trajectory"), (gnss_file, "GNSS")]:
#         if not os.path.exists(file_path):
#             logger.error(f"❌ {label} file not found: {file_path}")
#         else:
#             logger.info(f"📂 {label} file found: {file_path}")

#     # ✅ `--test` Handling: Limit Total Points Processed
#     test_mode = config.get("processing.test_mode", False)
#     test_points_limit = config.get("processing.test_limit", 50_000_000) if test_mode else None
#     launched_points = 0  

#     # 🔹 Prepare TEMP directory (clean and recreate)
#     temp_dir = prepare_temp_directory(ply_path)
#     if temp_dir is None:
#         logger.error("❌ Failed to create TEMP directory. Exiting.")
#         sys.exit(1)

#     logger.info(f"📂 TEMP directory ready: {temp_dir}")

#     # --- Print Initial CRS Registry ---
#     logger.info("🔹 Initial CRS Registry State:")
#     print_crs_registry()

#     # --- Load Sensor Trajectory Data ---
#     logger.info("📡 Loading sensor trajectory data...")
#     traj_data = load_trajectory(trajectory_file)
#     if traj_data is None:
#         logger.error(f"❌ Failed to load sensor trajectory: {trajectory_file}")
#         sys.exit(1)
#     logger.info(f"✅ Sensor trajectory points loaded: {len(traj_data)}")

#     # ✅ Debugging Exit
#     logger.info("🛑 Debug Exit after load_trajectory()")
#     sys.exit(0)

#     # --- Compute Global Transformation (Rotation & Translation) ---
#     logger.info("🛰️ Computing Global Transformation...")
#     R_global, t_global, crs_epsg = compute_global_transform(config)

#     # --- Transform and Save the Sensor Trajectory ---
#     transformed_traj = apply_transformation(traj_data, R_global, t_global, get_crs("trajectory"))
#     transformed_traj_path = os.path.join(ply_dir, "sensor_trajectory.laz")
#     append_trajectory_to_laz(transformed_traj_path, transformed_traj)
#     logger.info(f"📌 Transformed sensor trajectory saved: {transformed_traj_path}")

#     # --- Process Point Cloud in Chunks Using Streaming + Multiprocessing ---
#     logger.info(f"🌍 Processing Point Cloud from {ply_path} in Chunks...")

#     num_workers = config.get("processing.num_workers", max(1, multiprocessing.cpu_count() - 1))
#     pass_files, fail_files = [], []

#     # ✅ DEBUG: Manually process ONE chunk before multiprocessing
#     for i, chunk in enumerate(load_ply_chunks(ply_path, config.get("processing.chunk_size", 5000000))):
#         if i > 0:  
#             break  # ✅ Process only ONE chunk for debugging

#         chunk_size = config.get("processing.chunk_size", 5000000)
#         logger.info(f"🔹 DEBUG: Processing Chunk {i} | Size={chunk_size}")

#         # Run process_chunk() directly
#         res = process_chunk((i, chunk, traj_data, R_global, t_global, crs_epsg, temp_dir, current_config))

#         if res:
#             logger.info(f"✅ DEBUG: Chunk Processed. Results: {res}")
#         else:
#             logger.error(f"❌ DEBUG: process_chunk() failed for Chunk {i}")

#     # ✅ Now proceed with multiprocessing
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         for i, chunk in enumerate(load_ply_chunks(ply_path, config.get("processing.chunk_size", 5000000))):
#             chunk_size = config.get("processing.chunk_size", 5000000)
#             logger.info(f"🔹 DEBUG: Launching Chunk {i} | Size={chunk_size}")

#             if test_mode and launched_points + chunk_size > test_points_limit:
#                 logger.warning(f"🛑 Test mode limit reached: {test_points_limit} points. Stopping new chunk launches.")
#                 break  

#             pool.apply_async(
#                 process_chunk,
#                 args=((i, chunk, traj_data, R_global, t_global, crs_epsg, temp_dir, current_config),),
#                 callback=lambda res: (
#                     pass_files.append(res[0]) if res and res[0] else None,
#                     fail_files.append(res[1]) if res and res[1] else None
#                 )
#             )

#             launched_points += chunk_size  

#         pool.close()
#         pool.join()

#     logger.info("✅ Processing complete!")

#     # ✅ Merge Processed Chunks
#     logger.info("🔄 Merging processed files...")
#     merge_laz_files(pass_files, output_pass_path)
#     merge_laz_files(fail_files, output_fail_path)

#     logger.info("✅ Merging complete. Output files ready!")


# if __name__ == "__main__":
#     main()
