import os
import logging
import numpy as np
from modules.apply_filters import apply_filters
from modules.save_partial_laz import save_partial_laz

from modules.json_registry import JSONRegistry  # ✅ Dot-access config support
from modules.compute_pseudo_normals import compute_pseudo_normals
from modules.build_dtype import build_dtype_from_config

logger = logging.getLogger(__name__)

def process_chunk(args_tuple):
    chunk_idx, points, traj_data, R, t, crs_epsg, temp_dir, config = args_tuple

    print("============> top of process_chunk.py", flush=True)

    logger.info(f"[DEBUG] config type: {type(config)}")
    logger.info(f"[DEBUG] config dir: {dir(config)}")
    logger.info(f"[DEBUG] config repr: {repr(config)}")


    logger.info("🔹 Processing Chunk %d...", chunk_idx)

    if not isinstance(points, np.ndarray):
        logger.error(f"[ERROR] Chunk {chunk_idx} is not a NumPy array.")
        return None, None

    if points.size == 0:
        logger.warning(f"[DEBUG] Chunk {chunk_idx} contained no points.")
        return None, None

    logger.info(f"[DEBUG] Chunk {chunk_idx} has {len(points)} points, dtype: {points.dtype}")
    logger.info(f"[DEBUG] Chunk {chunk_idx} first row: {points[0]}")

    try:
        pass_chunk, fail_chunk = apply_filters(points, traj_data, config)
    except Exception as e:
        logger.error(f"Error applying filters on chunk {chunk_idx}: {e}")
        return None, None

    # Safely retrieve pseudo_normals flag
    normals_flag = config.get("processing", {}).get("pseudo_normals", True)
    print(f"===================> Processing Pseudo_Normals: {normals_flag}")

    if normals_flag:
        try:
            pass_chunk = compute_pseudo_normals(pass_chunk, traj_data)
        except Exception as e:
            logger.error(f"Error computing pseudo-normals on chunk {chunk_idx}: {e}")

    try:
        coords = np.stack([pass_chunk['x'], pass_chunk['y'], pass_chunk['z']], axis=-1)
        transformed = coords @ R.T + t
        pass_chunk['x'], pass_chunk['y'], pass_chunk['z'] = transformed[:, 0], transformed[:, 1], transformed[:, 2]

        if normals_flag:
            normals = np.stack([pass_chunk['nx'], pass_chunk['ny'], pass_chunk['nz']], axis=-1)
            rotated_normals = normals @ R.T
            pass_chunk['nx'], pass_chunk['ny'], pass_chunk['nz'] = rotated_normals[:, 0], rotated_normals[:, 1], rotated_normals[:, 2]

    except Exception as e:
        logger.error(f"Error transforming chunk {chunk_idx}: {e}")
        return None, None

    try:
        pass_filename = os.path.join(temp_dir, f"pass_chunk_{chunk_idx}.laz")
        fail_filename = os.path.join(temp_dir, f"fail_chunk_{chunk_idx}.laz")

        # Fix: access data_formats with dict syntax
        save_partial_laz(pass_filename, pass_chunk, config)
        save_partial_laz(fail_filename, fail_chunk, config)

        return pass_filename, fail_filename

    except Exception as e:
        logger.error(f"Error saving filtered chunk {chunk_idx}: {e}")
        return None, None
