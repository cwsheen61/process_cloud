#!/usr/bin/env python3
"""
Processes a large PLY point cloud using a dense sensor trajectory and a sparse GNSS trajectory
to compute a 3D transformation (rotation about Z and translation). The transformed data is
optionally converted to a target CRS, streamed in chunks, and stored in LAZ format.
"""
# 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

# --- Global Imports ---
import sys
import os
import json
import logging
import shutil
import inspect
import multiprocessing
import numpy as np
from functools import partial

from modules.json_registry import JSONRegistry
from modules.load_trajectory import load_trajectory
from modules.load_gnss_trajectory import load_gnss_trajectory
from modules.compute_global_transform import compute_global_transform
from modules.prepare_temp_directory import prepare_temp_directory
from modules.yield_ply_chunks import yield_ply_chunks
from modules.process_chunk import process_chunk
from modules.merge_laz_files import merge_laz_files
from modules.tiling import tiling

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)")
logger = logging.getLogger(__name__)

# === Field Normalization for Input PLY Files ===
FIELD_NAME_MAP = {
    "x": "x",
    "y": "y",
    "z": "z",
    "time": "GpsTime",
    "intensity": "intensity",
    "nx": "NormalX",
    "ny": "NormalY",
    "nz": "NormalZ",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "alpha": "alpha",
    "ring": "pt_source_id",
    "retNumber": "return_number",
    "range": "range"
}

def throttle_num_workers(config):
    def walk_and_throttle(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key == "num_workers" and isinstance(value, int):
                    available_cpus = multiprocessing.cpu_count()
                    if value > available_cpus:
                        logger.warning(f"⚠️ '{current_path}' has {value} workers; throttling to {available_cpus}")
                        config.set(current_path, available_cpus)
                    else:
                        logger.info(f"✅ '{current_path}' is within CPU limits: {value}")
                else:
                    walk_and_throttle(value, current_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk_and_throttle(item, f"{path}[{idx}]")

    full_config_dict = config.config
    walk_and_throttle(full_config_dict)
    config.save()
    logger.info("✅ Finished throttling all 'num_workers' settings in current_config.json.")




def lineno():
    import inspect
    return inspect.currentframe().f_back.f_lineno

 # --- Clean up stale directories ---
def clean_dirs(ply_path):
    for dirname in ["TEMP", "SORTED", "COL_LAS", "TEMP_VOXELS"]:
        full_path = os.path.join(ply_path, dirname)
        if os.path.exists(full_path):
            try:
                shutil.rmtree(full_path)
                logger.info(f"🧹 Removed stale directory: {full_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove {full_path}: {e}")


def main():
    logger.info("🔍 Loading JSON configuration... (%s:%d)", __file__, lineno())

   
    # Parse .ply and .json from sys.argv
    ply_file_path, json_file_path = None, None
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".ply"):
            ply_file_path = arg
        elif arg.lower().endswith(".json"):
            json_file_path = arg

    if not ply_file_path or not json_file_path:
        logger.error("❌ Missing required file arguments. Provide both a .ply and a .json file. (%s:%d)",
                     __file__, lineno())
        sys.exit(1)

    ply_path, ply_file = os.path.split(ply_file_path)
    clean_dirs(ply_path)  # Clean up BEFORE doing anything

    config_path, config_file = os.path.split(json_file_path)
    logger.info(f"📂 Detected ply_path: {ply_path}, ply_file: {ply_file}")
    logger.info(f"📂 Detected config_path: {config_path}, config_file: {config_file}")

    current_config_path = os.path.join(ply_path, "current_config.json")
    try:
        shutil.copy(json_file_path, current_config_path)
        logger.info("✅ Created `current_config.json` from seed config. (%s:%d)", __file__, lineno())
    except Exception as e:
        logger.error("❌ Failed to create `current_config.json`: %s (%s:%d)", e, __file__, lineno())
        sys.exit(1)

    working_config_file = current_config_path
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
            ("output_fail", "output_fail.laz"),
            ("sorted_las", "sorted_final.laz"),
            ("voxel_las", "voxel_final.laz")]:

            full = config.get(f"files.{key}", default)
            if not os.path.isabs(full):
                full = os.path.join(ply_path, full)
            config.set(f"files.{key}", full)
        config.save()
        logger.info(f"✅ Updated `current_config.json` with dynamic paths: {working_config_file}")
    except Exception as e:
        logger.error(f"❌ Error updating `current_config.json`: {e}")
        sys.exit(1)

    throttle_num_workers(config)


    config = JSONRegistry(current_config_path, current_config_path)
    logger.info("✅ Initialized current config: %s (%s:%d)", current_config_path, __file__, lineno())

    temp_dir = prepare_temp_directory(ply_path)
    logger.info("📂 Temp directory initialized: %s (%s:%d)", temp_dir, __file__, lineno())

    trajectory_data = load_trajectory(config)
    traj_file = config.get("files.trajectory")
    logger.info("✅ Loaded trajectory: %s (%s:%d)", traj_file, __file__, lineno())

    gnss_file = config.get("files.gnss_trajectory")
    if gnss_file and os.path.exists(gnss_file) and os.path.getsize(gnss_file) > 0:
        try:
            gnss_data = load_gnss_trajectory(current_config_path)
            logger.info("✅ Loaded GNSS trajectory from %s (%s:%d)", gnss_file, __file__, lineno())
        except Exception as e:
            logger.warning("⚠️ GNSS trajectory exists but failed to load: %s (%s:%d)", e, __file__, lineno())
            gnss_data = None
    else:
        logger.info("ℹ️ No GNSS trajectory provided or required for this data type (%s:%d)", __file__, lineno())
        gnss_data = None

    if gnss_data is not None:
        compute_global_transform(current_config_path)
        logger.info("✅ Global transformation computed and stored in config (%s:%d)", __file__, lineno())
        config = JSONRegistry(current_config_path, current_config_path)
        R_global = np.array(config.get("transformation.R"))
        t_global = np.array(config.get("transformation.t"))
        crs_epsg = config.get("crs.epsg")
        logger.info("✅ Transformation parameters: R=%s, t=%s, EPSG=%s", R_global, t_global, crs_epsg)
    else:
        R_global = np.eye(3)
        t_global = np.zeros(3)
        crs_epsg = 0
        logger.info("ℹ️ Skipping global transformation due to lack of GNSS fit metrics. Using identity R and zero t.")

    test_mode = config.get("processing.test_mode", False)
    test_chunks = config.get("processing.test_chunks", 0)
    num_workers = config.get("processing.num_workers", multiprocessing.cpu_count())

    final_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        chunk_generator = yield_ply_chunks(config, config.get("data_formats"), FIELD_NAME_MAP)

        def chunk_arg_generator():
            for chunk_idx, points in chunk_generator:
                if test_mode and chunk_idx >= test_chunks:
                    logger.info(f"🪺 Test mode halted after {test_chunks} chunks.")
                    break
                logger.info(f"🔍 Chunk {chunk_idx}: {len(points)} points, dtype: {points.dtype}")
                yield (chunk_idx, points, trajectory_data, R_global, t_global, crs_epsg, temp_dir, config.config)

        for result in pool.imap_unordered(process_chunk, chunk_arg_generator(), chunksize=1):
            final_results.append(result)

    try:
        pass_chunks, fail_chunks = zip(*final_results)
    except ValueError:
        pass_chunks, fail_chunks = (), ()

    logger.info("✅ Processing complete. Passed: %d, Failed: %d (%s:%d)",
                len(pass_chunks), len(fail_chunks), __file__, lineno())

    merge_laz_files(current_config_path)
    logger.info("✅ LAZ merging completed (%s:%d)", __file__, lineno())

    ret = tiling(config)


    # === Optional Post-Sort Filtering ===
    from filters.post_sort_filters import sorted_knn_filter, sorted_kd_filter, sorted_voxel_grid_filter, merge_voxel_chunks

    sorted_file = config.get("files.sorted_las")
    if sorted_file and os.path.exists(sorted_file):
        filter_cfg = config.get("filtering", {})

        if filter_cfg.get("knn_filter", {}).get("enabled", False):
            logger.info("🧪 Running post-sort KNN filter...")
            sorted_file = sorted_knn_filter(config)

        if filter_cfg.get("kd_tree_filter", {}).get("enabled", False):
            logger.info("🧪 Running post-sort KD-tree filter...")
            sorted_file = sorted_kd_filter(config)

        if filter_cfg.get("voxel_filter", {}).get("enabled", False):
            logger.info("🧪 Running post-sort Voxel Grid filter...")
            voxel_dir = sorted_voxel_grid_filter(config)
            if voxel_dir:
                merged_voxel_output = config.get("files.voxel_las")
                merge_voxel_chunks(config)

        config.set("files.sorted_las", sorted_file)
        config.save()
    else:
        logger.warning("⚠️ No sorted LAS file found; skipping post-sort filters.")


    # --- Final cleanup ---

    if not test_mode:
        clean_dirs(ply_path)




if __name__ == "__main__":
    main()
