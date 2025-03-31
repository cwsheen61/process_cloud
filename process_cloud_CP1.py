#!/usr/bin/env python3

"""
Processes a large PLY point cloud using a dense sensor trajectory and a sparse GNSS trajectory
to compute a 3D transformation (rotation about Z and translation). The transformed data is
optionally converted to a target CRS, streamed in chunks, and stored in LAZ format.
"""

# --- Global Imports ---
import sys
import os
import logging
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

    # 🔴 HARD STOP for debugging before moving forward
    logger.info("🚧 Debug stop after argument parsing. Exiting early for verification. (%s:%d)", __file__, lineno())
    sys.exit(0)  # Exit before proceeding

    # ✅ 3. Append all of `process_cloud.py` as is, unchanged.
    # --- Reinserted Code from process_cloud.py ---

    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())
    config = JSONRegistry(config_path, config_path)

    logger.info("✅ Initialized current config: %s (%s:%d)", config_path, __file__, lineno())

    # Prepare working directory
    temp_dir = prepare_temp_directory(ply_path)

    logger.info("📂 Temp directory initialized: %s (%s:%d)", temp_dir, __file__, lineno())

    # Load trajectory
    trajectory_file = os.path.join(ply_path, config.get("files.trajectory"))
    trajectory_data = load_trajectory(trajectory_file)

    logger.info("✅ Loaded trajectory: %s (%s:%d)", trajectory_file, __file__, lineno())

    # Load GNSS trajectory
    gnss_file = config.get("files.gnss_trajectory")
    if gnss_file is None:
        logger.error("❌ GNSS trajectory file is missing from config! (%s:%d)", __file__, lineno())
        sys.exit(1)

    gnss_data = load_gnss_trajectory(gnss_file)

    logger.info("✅ Loaded GNSS trajectory from %s (%s:%d)", gnss_file, __file__, lineno())

    # Compute global transformation
    compute_global_transform(config_path)

    logger.info("✅ Global transformation computed and stored in config (%s:%d)", __file__, lineno())

    # Reload updated config
    config.reload()
    logger.info("✅ Reloaded updated config. (%s:%d)", __file__, lineno())

    # Load and process point cloud in chunks
    logger.info("🔄 Loading point cloud chunks from %s (%s:%d)", ply_file_path, __file__, lineno())

    chunk_size = config.get("processing.chunk_size", 5000000)
    num_workers = config.get("processing.num_workers", multiprocessing.cpu_count())

    chunks = load_ply_chunks(ply_file_path, chunk_size)

    logger.info("🚀 Processing %d chunks using %d workers. (%s:%d)", len(chunks), num_workers, __file__, lineno())

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    pass_chunks, fail_chunks = zip(*results)

    logger.info("✅ Processing complete. Passed: %d, Failed: %d (%s:%d)", len(pass_chunks), len(fail_chunks), __file__, lineno())

    # Merge LAZ files
    merge_laz_files(ply_path, config)

    logger.info("✅ LAZ merging completed (%s:%d)", __file__, lineno())

if __name__ == "__main__":
    main()
