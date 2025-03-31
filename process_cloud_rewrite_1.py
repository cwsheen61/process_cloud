#!/usr/bin/env python3

"""
Processes a large PLY point cloud using a dense sensor trajectory and a sparse GNSS trajectory
to compute a 3D transformation (rotation about Z and translation). The transformed data is
optionally converted to a target CRS, streamed in chunks, and stored in LAZ format.
"""

# --- Global Imports ---
import sys
import os
import json
import logging
import shutil
import inspect
import multiprocessing
import numpy as np
from modules.json_registry import JSONRegistry
from modules.load_trajectory import load_trajectory
from modules.load_gnss_trajectory import load_gnss_trajectory
from modules.compute_global_transform import compute_global_transform
from modules.prepare_temp_directory import prepare_temp_directory
from modules.load_ply_chunks import load_ply_chunks
from modules.process_chunk import process_chunk
from modules.merge_laz_files import merge_laz_files

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)")
logger = logging.getLogger(__name__)

# --- Helper Function for Line Numbers ---
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

def main():
    """
    Main execution pipeline for processing point cloud data using GNSS trajectories.
    """

    # ✅ 1. Keep all pre-processing from `process_cloud_test.py`
    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())

    # ✅ 2. Dynamically survey `sys.argv` to find the correct file paths
    ply_file_path = None
    json_file_path = None

    for arg in sys.argv[1:]:  # Ignore sys.argv[0] (script name)
        if arg.lower().endswith(".ply"):
            ply_file_path = arg
        elif arg.lower().endswith(".json"):
            json_file_path = arg

    # Validate that we found both required files
    if not ply_file_path or not json_file_path:
        logger.error("❌ Missing required file arguments. Provide both a .ply and a .json file. (%s:%d)", __file__, lineno())
        sys.exit(1)

    # Extract base directory and filenames dynamically
    ply_path, ply_file = os.path.split(ply_file_path)
    config_path, config_file = os.path.split(json_file_path)

    logger.info(f"📂 Detected ply_path: {ply_path}, ply_file: {ply_file}")
    logger.info(f"📂 Detected config_path: {config_path}, config_file: {config_file}")

    # --------------------- HARD STOP CHECKPOINT 1 ---------------------
    # ✅ Success to here at checkpoint 1
    # Commenting out the hard stop to proceed with `current_config.json`
    # sys.exit(0)

    # ✅ Checkpoint 2: Copy seed config to current_config.json and stop execution for verification.
    current_config_path = os.path.join(ply_path, "current_config.json")

    try:
        shutil.copy(json_file_path, current_config_path)
        logger.info("✅ Created `current_config.json` from seed config. (%s:%d)", __file__, lineno())
    except Exception as e:
        logger.error("❌ Failed to create `current_config.json`: %s (%s:%d)", e, __file__, lineno())
        sys.exit(1)

    # # 🔴 HARD STOP for verification
    # logger.info("🚧 Debug stop after copying `config.json` to `current_config.json`. Exiting early for verification. (%s:%d)", __file__, lineno())
    # sys.exit(0)  # Exit before proceeding


    # ===================================================
    # Checkpoint 3: Populate current_config.json with Paths
    # ===================================================

    working_config_file = os.path.join(ply_path, "current_config.json")
    logger.info(f"📄 Using working configuration file: {working_config_file} ({__file__}:{inspect.currentframe().f_lineno})")

    # Load the working configuration using JSONRegistry
    config = JSONRegistry(working_config_file, working_config_file)
    logger.info(f"✅ Reloaded config from updated file: {working_config_file}")

    logger.info("🔄 Populating `current_config.json` with runtime paths...")

    try:
        # Update file paths dynamically
        config.set("pathname", ply_path)  # Ensure it's the absolute path
        config.set("files.ply", os.path.join(ply_path, ply_file_path))
        config.set("files.trajectory", os.path.join(ply_path, config.get("files.trajectory", "trajectory.txt")) if not os.path.isabs(config.get("files.trajectory", "")) else config.get("files.trajectory", ""))
        config.set("files.gnss_trajectory", os.path.join(ply_path, config.get("files.gnss_trajectory", "gnss.txt")) if not os.path.isabs(config.get("files.gnss_trajectory", "")) else config.get("files.gnss_trajectory", ""))
        config.set("files.transformed_trajectory", os.path.join(ply_path, config.get("files.transformed_trajectory", "transformed_traj.txt")) if not os.path.isabs(config.get("files.transformed_trajectory", "")) else config.get("files.transformed_trajectory", ""))
        config.set("files.output_pass", os.path.join(ply_path, config.get("files.output_pass", "output_pass.laz")) if not os.path.isabs(config.get("files.output_pass", "")) else config.get("files.output_pass", ""))
        config.set("files.output_fail", os.path.join(ply_path, config.get("files.output_fail", "output_fail.laz")) if not os.path.isabs(config.get("files.output_fail", "")) else config.get("files.output_fail", ""))

        # Save updated configuration
        config.save()

        logger.info(f"✅ Updated `current_config.json` with dynamic paths: {working_config_file}")

    except Exception as e:
        logger.error(f"❌ Error updating `current_config.json`: {e}")
        sys.exit(1)

    # ✅ Checkpoint 3 Complete
    # Hard stop to verify correctness before proceeding
    

    # ✅ 4. Retain all previous logic unchanged

    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())
    config = JSONRegistry(current_config_path, current_config_path)

    logger.info("✅ Initialized current config: %s (%s:%d)", current_config_path, __file__, lineno())

    # Prepare working directory
    temp_dir = prepare_temp_directory(ply_path)

    logger.info("📂 Temp directory initialized: %s (%s:%d)", temp_dir, __file__, lineno())

    # Load trajectory
    trajectory_data = load_trajectory(current_config_path)

    trajectory_file = config.get("files.trajectory")
    logger.info("✅ Loaded trajectory: %s (%s:%d)", trajectory_file, __file__, lineno())

    # Load GNSS trajectory
    gnss_file = config.get("files.gnss_trajectory")
    if gnss_file is None:
        logger.error("❌ GNSS trajectory file is missing from config! (%s:%d)", __file__, lineno())
        sys.exit(1)

    gnss_data = load_gnss_trajectory(current_config_path)

    logger.info("✅ Loaded GNSS trajectory from %s (%s:%d)", gnss_file, __file__, lineno())

    # Compute global transformation
    compute_global_transform(current_config_path)

    logger.info("✅ Global transformation computed and stored in config (%s:%d)", __file__, lineno())

    # 4. Checkpoint:
    # print(f"✅  At Checkpoint 4")
    # sys.exit(0)

    # 5. Checkpoint Start

    # Reload updated config
    # config.reload()
    # logger.info("✅ Reloaded updated config. (%s:%d)", __file__, lineno())

    # Load and process point cloud in chunks
    logger.info("🔄 Loading point cloud chunks from %s (%s:%d)", ply_file_path, __file__, lineno())

    chunk_size = config.get("processing.chunk_size", 5000000)
    num_workers = config.get("processing.num_workers", multiprocessing.cpu_count())

    chunks = load_ply_chunks(current_config_path)

    logger.info("🚀 Processing %d chunks using %d workers. (%s:%d)", len(chunks), num_workers, __file__, lineno())

    args = []
    for idx, chunk in enumerate(chunks):
        args.append((idx, chunk, trajectory_data, R_global, t_global, crs_epsg, temp_dir, config))
        print(f"{idx}, {R_global},{t_global}, {crs_epsg}")


    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, args)

    pass_chunks, fail_chunks = zip(*results)

    logger.info("✅ Processing complete. Passed: %d, Failed: %d (%s:%d)", len(pass_chunks), len(fail_chunks), __file__, lineno())

    # At Checkpoint #5
    print(f"✅  at checkpoint 5.")
    # Merge LAZ files
    merge_laz_files(ply_path, config)

    logger.info("✅ LAZ merging completed (%s:%d)", __file__, lineno())

if __name__ == "__main__":
    main()
