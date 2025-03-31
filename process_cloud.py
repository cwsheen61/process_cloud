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
from modules.yield_ply_chunks import yield_ply_chunks
from modules.process_chunk import process_chunk
from modules.merge_laz_files import merge_laz_files

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)")
logger = logging.getLogger(__name__)

def lineno():
    """Returns the current line number in our program."""
    import inspect
    return inspect.currentframe().f_back.f_lineno

def main():
    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())

    # --- Survey sys.argv for file paths ---
    ply_file_path = None
    json_file_path = None
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".ply"):
            ply_file_path = arg
        elif arg.lower().endswith(".json"):
            json_file_path = arg
    if not ply_file_path or not json_file_path:
        logger.error("❌ Missing required file arguments. Provide both a .ply and a .json file. (%s:%d)",
                     __file__, lineno())
        sys.exit(1)

    # --- Extract base paths ---
    ply_path, ply_file = os.path.split(ply_file_path)
    config_path, config_file = os.path.split(json_file_path)
    logger.info(f"📂 Detected ply_path: {ply_path}, ply_file: {ply_file}")
    logger.info(f"📂 Detected config_path: {config_path}, config_file: {config_file}")

    # --- Copy seed config to current_config.json ---
    current_config_path = os.path.join(ply_path, "current_config.json")
    try:
        shutil.copy(json_file_path, current_config_path)
        logger.info("✅ Created `current_config.json` from seed config. (%s:%d)", __file__, lineno())
    except Exception as e:
        logger.error("❌ Failed to create `current_config.json`: %s (%s:%d)", e, __file__, lineno())
        sys.exit(1)

    # --- Populate current_config.json with dynamic paths ---
    working_config_file = os.path.join(ply_path, "current_config.json")
    logger.info(f"📄 Using working configuration file: {working_config_file} ({__file__}:{lineno()})")
    config = JSONRegistry(working_config_file, working_config_file)
    logger.info(f"✅ Reloaded config from updated file: {working_config_file}")

    try:
        config.set("pathname", ply_path)
        config.set("files.ply", os.path.join(ply_path, ply_file))
        for key, default in [
            ("trajectory", "trajectory.txt"),
            ("gnss_trajectory", "gnss.txt"),
            ("transformed_trajectory", "transformed_traj.txt"),
            ("output_pass", "output_pass.laz"),
            ("output_fail", "output_fail.laz")]:
            full = config.get(f"files.{key}", default)
            if not os.path.isabs(full):
                full = os.path.join(ply_path, full)
            config.set(f"files.{key}", full)
        config.save()
        logger.info(f"✅ Updated `current_config.json` with dynamic paths: {working_config_file}")
    except Exception as e:
        logger.error(f"❌ Error updating `current_config.json`: {e}")
        sys.exit(1)

    # --- Reload updated config ---
    config = JSONRegistry(current_config_path, current_config_path)
    logger.info("✅ Initialized current config: %s (%s:%d)", current_config_path, __file__, lineno())

    # --- Prepare working directory ---
    temp_dir = prepare_temp_directory(ply_path)
    logger.info("📂 Temp directory initialized: %s (%s:%d)", temp_dir, __file__, lineno())

    # --- Load sensor trajectory ---
    trajectory_data = load_trajectory(current_config_path)
    traj_file = config.get("files.trajectory")
    logger.info("✅ Loaded trajectory: %s (%s:%d)", traj_file, __file__, lineno())

    # --- Load GNSS trajectory ---
    gnss_file = config.get("files.gnss_trajectory")
    if not gnss_file:
        logger.error("❌ GNSS trajectory file is missing from config! (%s:%d)", __file__, lineno())
        sys.exit(1)
    gnss_data = load_gnss_trajectory(current_config_path)
    logger.info("✅ Loaded GNSS trajectory from %s (%s:%d)", gnss_file, __file__, lineno())

    # --- Compute transformation ---
    compute_global_transform(current_config_path)
    logger.info("✅ Global transformation computed and stored in config (%s:%d)", __file__, lineno())

    config = JSONRegistry(current_config_path, current_config_path)
    R_global = np.array(config.get("transformation.R"))
    t_global = np.array(config.get("transformation.t"))
    crs_epsg = config.get("crs.epsg")
    logger.info("✅ Transformation parameters: R=%s, t=%s, EPSG=%s", R_global, t_global, crs_epsg)

    # --- Yield and process chunks with streaming dispatch ---
    test_mode = config.get("processing.test_mode", False)
    test_chunks = config.get("processing.test_chunks", 0)
    num_workers = config.get("processing.num_workers", multiprocessing.cpu_count())

    logger.info(f"Current config path: {current_config_path}")

    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for chunk_idx, points in yield_ply_chunks(config):
            if test_mode and chunk_idx >= test_chunks:
                logger.info(f"🧪 Test mode halted after {test_chunks} chunks.")
                break

            logger.info(f"🔍 Chunk {chunk_idx}: {len(points)} points, dtype: {points.dtype}")
            args_tuple = (chunk_idx, points, trajectory_data, R_global, t_global, crs_epsg, temp_dir, config.config)
            result = pool.apply_async(process_chunk, args=(args_tuple,))
            results.append(result)

        final_results = [r.get() for r in results]

    try:
        pass_chunks, fail_chunks = zip(*final_results)
    except ValueError:
        pass_chunks, fail_chunks = (), ()
    logger.info("✅ Processing complete. Passed: %d, Failed: %d (%s:%d)",
                len(pass_chunks), len(fail_chunks), __file__, lineno())

    merge_laz_files(current_config_path)
    logger.info("✅ LAZ merging completed (%s:%d)", __file__, lineno())

if __name__ == "__main__":
    main()
