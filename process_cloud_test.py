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
import multiprocessing
from modules.json_registry import JSONRegistry
from modules.load_trajectory import load_trajectory
from modules.load_gnss_trajectory import load_gnss_trajectory
from modules.compute_global_transform import compute_global_transform
from modules.prepare_temp_directory import prepare_temp_directory
from modules.load_ply_chunks import load_ply_chunks
from modules.process_chunk import process_chunk
from modules.merge_laz_files import merge_laz_files
import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)")
logger = logging.getLogger(__name__)

def main():
    """
    Main execution pipeline for processing point cloud data using GNSS trajectories.
    """
    # Load configuration dynamically based on runtime PLY path
    ply_file = sys.argv[1]
    config_path = sys.argv[2]

    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())
    config = JSONRegistry(config_path, config_path)

    # Extract base directory from PLY file path

    output_dir, ply_file_name = os.path.split(ply_file)

    logger.info(f"output_dir: {output_dir}")

    # Initialize the local config copy
    current_config_path = os.path.join(output_dir, "current_config.json")
    logger.info(f"current_config_path: {current_config_path}")
    config.save_as(current_config_path)
    logger.info("✅ Initialized current config: %s (%s:%d)", current_config_path, __file__, lineno())

    # Prepare temporary directory
    temp_dir = prepare_temp_directory(output_dir+"/TEMP")
    logger.info("📂 Temp directory initialized: %s (%s:%d)", temp_dir, __file__, lineno())

    # Dynamically update file paths in config
    config.set("pathname", output_dir)
    config.set("files.ply", ply_file)
    config.set("files.trajectory", os.path.join(output_dir, config.get("files.trajectory")))
    config.set("files.gnss", os.path.join(output_dir, config.get("files.gnss")))

    # Fix missing keys for output paths
    config.set("files.output_pass", os.path.join(temp_dir, "pass_chunks"))
    config.set("files.output_fail", os.path.join(temp_dir, "fail_chunks"))
    config.set("files.transformed_trajectory", os.path.join(output_dir, "transformed_trajectory.txt"))

    # Ensure correct pathname value
    config.set("pathname", ply_file)

    config.save()

    # 🔴 DEBUG STOP HERE
    logger.info("🚧 Debug stop after TEMP directory creation. Exiting early for verification. (%s:%d)", __file__, lineno())
    sys.exit(0)  # Exit before proceeding to trajectory loading

    # Load trajectory data
    trajectory_file = config.get("files.trajectory")
    traj_data = load_trajectory(trajectory_file, current_config_path)
    logger.info("✅ Loaded trajectory: %s (%s:%d)", trajectory_file, __file__, lineno())

    # Load GNSS trajectory
    gnss_file = config.get("files.gnss")

    if not gnss_file:
        logger.error("❌ GNSS trajectory file is missing from config! (%s:%d)", __file__, lineno())
        sys.exit(1)

    gnss_data = load_gnss_trajectory(gnss_file, current_config_path)
    logger.info("✅ Loaded GNSS trajectory: %s (%s:%d)", gnss_file, __file__, lineno())

    # Compute global transformation
    compute_global_transform(current_config_path)
    config.reload()
    logger.info("✅ Reloaded updated config. (%s:%d)", __file__, lineno())

    # Load point cloud chunks
    chunk_size = config.get("processing.chunk_size")
    logger.info("🔄 Loading point cloud chunks from %s with chunk size %d... (%s:%d)",
                ply_file, chunk_size, __file__, lineno())

    chunks = load_ply_chunks(ply_file, chunk_size)

    # 🔴 STOP AFTER 1 CHUNK FOR TESTING
    if chunks:
        chunks = [chunks[0]]  # Keep only the first chunk

    num_workers = config.get("processing.num_workers")
    logger.info("🚀 Processing %d chunk(s) using %d workers. (%s:%d)",
                len(chunks), num_workers, __file__, lineno())

    # Process chunks
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Unpack results
    if results:
        pass_chunks, fail_chunks = zip(*results)
    else:
        pass_chunks, fail_chunks = [], []

    logger.info("✅ Processed %d chunks successfully, %d failed. (%s:%d)",
                len(pass_chunks), len(fail_chunks), __file__, lineno())

    # Merge LAZ files
    merge_laz_files(output_dir, config)
    logger.info("✅ Merged LAZ files. (%s:%d)", __file__, lineno())


if __name__ == "__main__":
    main()
