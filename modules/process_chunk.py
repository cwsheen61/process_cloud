import os
import sys
import numpy as np
import laspy
import logging
from modules.apply_filters import apply_filters
from modules.apply_transformation import apply_transformation
from modules.save_partial_laz import save_partial_laz  # ✅ Writes filtered chunks to LAZ

logger = logging.getLogger(__name__)  # ✅ Use logging instead of print()

def process_chunk(args_tuple):
    """
    Processes a single point cloud chunk:
    1. Applies filtering based on trajectory data.
    2. Transforms coordinates using global R (rotation) and T (translation).
    3. Ensures CRS tracking and prevents double application of transformation.
    4. Writes pass/fail points to LAZ files.

    Args:
        args_tuple (tuple): Contains:
            - chunk_idx (int): Index of the chunk being processed.
            - chunk (np.ndarray): Point cloud data in sensor coordinates.
            - traj_data (np.ndarray): Sensor trajectory data.
            - R_global (np.ndarray): 3x3 Rotation matrix.
            - t_global (np.ndarray): 3D Translation vector.
            - crs_epsg (int): EPSG code for transformation.
            - temp_dir (str): Directory to save temporary LAZ files.
            - config (JSONRegistry): Configuration settings from JSON file.

    Returns:
        tuple: (pass_filename, fail_filename)
    """
    # Unpack arguments correctly
    chunk_idx, chunk, traj_data, R_global, t_global, crs_epsg, temp_dir, config = args_tuple

    logger.info(f"🔹 Processing Chunk {chunk_idx}...")

    # --- Apply filtering using JSON config ---
    pass_chunk, fail_chunk = apply_filters(chunk, traj_data, config)

    logger.info(f"✅ Chunk {chunk_idx} Filtered: Pass={len(pass_chunk)}, Fail={len(fail_chunk)}")

    # --- Ensure points are in the correct CRS before transformation ---
    if len(pass_chunk) > 0:
        pass_chunk = apply_transformation(pass_chunk, R_global, t_global, crs_epsg)
    if len(fail_chunk) > 0:
        fail_chunk = apply_transformation(fail_chunk, R_global, t_global, crs_epsg)

    logger.info(f"🔄 Chunk {chunk_idx} Transformed: Pass={len(pass_chunk)}, Fail={len(fail_chunk)}")

    # --- Define output file paths ---
    pass_filename = os.path.join(temp_dir, f"pass_{chunk_idx}.las") if len(pass_chunk) > 0 else None
    fail_filename = os.path.join(temp_dir, f"fail_{chunk_idx}.las") if len(fail_chunk) > 0 else None

    # --- Save the pass_chunk ---
    if pass_filename:
        try:
            save_partial_laz(pass_filename, pass_chunk)
            logger.info(f"📂 Writing Chunk {chunk_idx}: Pass='{pass_filename}'")
        except Exception as e:
            logger.error(f"❌ ERROR in writing {pass_filename}: {e}")
            pass_filename = None

    # --- Save the fail_chunk ---
    if fail_filename:
        try:
            save_partial_laz(fail_filename, fail_chunk)
            logger.info(f"📂 Writing Chunk {chunk_idx}: Fail='{fail_filename}'")
            sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing
        except Exception as e:
            logger.error(f"❌ ERROR in writing {fail_filename}: {e}")
            fail_filename = None
            sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing

    # --- Final check: Ensure files were actually written ---
    if pass_filename and not os.path.exists(pass_filename):
        logger.error(f"❌ Pass LAZ missing: {pass_filename}")
        pass_filename = None
    if fail_filename and not os.path.exists(fail_filename):
        logger.error(f"❌ Fail LAZ missing: {fail_filename}")
        fail_filename = None

    return pass_filename, fail_filename
