import os
import logging
import numpy as np
from modules.apply_filters import apply_filters
from modules.save_partial_laz import save_partial_laz
from modules.json_registry import JSONRegistry
from modules.compute_pseudo_normals import compute_pseudo_normals
from filters.ensure_range import ensure_range_from_trajectory

logger = logging.getLogger(__name__)

def process_chunk(args_tuple):
    chunk_idx, points, traj_data, R, t, crs_epsg, temp_dir, config = args_tuple

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
        points = ensure_range_from_trajectory(points, traj_data)
        print(f"========================>>>  added range, probably")

        pass_chunk, fail_chunk = apply_filters(points, traj_data, config)
    except Exception as e:
        logger.error(f"Error applying filters on chunk {chunk_idx}: {e}")
        return None, None

    # Handle both dict and JSONRegistry for pseudo_normals flag
    try:
        config_data = getattr(config, "config", config)
        normals_flag = config_data.get("processing", {}).get("pseudo_normals", False)
    except Exception as e:
        logger.warning(f"⚠️ Could not access config for pseudo_normals flag: {e}")
        normals_flag = False

    print(f"===================> Processing Pseudo_Normals: {normals_flag}")

    if normals_flag:
        try:
            # Remove previous normals, if any
            existing_fields = list(pass_chunk.dtype.names)
            keep_fields = [f for f in existing_fields if f not in ('NormalX', 'NormalY', 'NormalZ')]
            trimmed_chunk = pass_chunk[keep_fields].copy()

            # Compute new normals
            new_normals = compute_pseudo_normals(trimmed_chunk, traj_data, config)

            # Reconstruct full dtype with existing + new normals
            new_fields = trimmed_chunk.dtype.descr + [
                ('NormalX', '<f8'),
                ('NormalY', '<f8'),
                ('NormalZ', '<f8'),
            ]
            enriched_chunk = np.empty(trimmed_chunk.shape, dtype=new_fields)
            for name in trimmed_chunk.dtype.names:
                enriched_chunk[name] = trimmed_chunk[name]

            enriched_chunk['NormalX'] = new_normals['NormalX']
            enriched_chunk['NormalY'] = new_normals['NormalY']
            enriched_chunk['NormalZ'] = new_normals['NormalZ']

            pass_chunk = enriched_chunk

        except Exception as e:
            logger.error(f"Error computing pseudo-normals on chunk {chunk_idx}: {e}")

    try:
        coords = np.stack([pass_chunk['x'], pass_chunk['y'], pass_chunk['z']], axis=-1)
        transformed = coords @ R.T + t
        pass_chunk['x'], pass_chunk['y'], pass_chunk['z'] = transformed[:, 0], transformed[:, 1], transformed[:, 2]

        if normals_flag:
            normals = np.stack([pass_chunk['NormalX'], pass_chunk['NormalY'], pass_chunk['NormalZ']], axis=-1)
            rotated_normals = normals @ R.T
            pass_chunk['NormalX'], pass_chunk['NormalY'], pass_chunk['NormalZ'] = (
                rotated_normals[:, 0], rotated_normals[:, 1], rotated_normals[:, 2]
            )

    except Exception as e:
        logger.error(f"Error transforming chunk {chunk_idx}: {e}")
        return 1, 1

    try:
        pass_filename = os.path.join(temp_dir, f"pass_chunk_{chunk_idx}.laz")
        fail_filename = os.path.join(temp_dir, f"fail_chunk_{chunk_idx}.laz")

        save_partial_laz(pass_filename, pass_chunk, config)
        save_partial_laz(fail_filename, fail_chunk, config)

        del points, pass_chunk, fail_chunk
        import gc
        gc.collect()

        return pass_filename, fail_filename

    except Exception as e:
        logger.error(f"Error saving filtered chunk {chunk_idx}: {e}")
        return 1, 1
