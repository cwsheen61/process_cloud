import numpy as np
from scipy.interpolate import interp1d
import logging
from numpy.lib import recfunctions as rfn

logger = logging.getLogger(__name__)

def compute_pseudo_normals(chunk, full_traj, config=None):
    """
    Compute and optionally overwrite normals based on trajectory direction.
    Controlled by `processing.pseudo_normals` in the config.

    Parameters:
    - chunk: numpy structured array with 'GpsTime', 'x', 'y', 'z'
    - full_traj: numpy structured array with 'GpsTime', 'x', 'y', 'z'
    - config: config object (may be JSONRegistry) or plain dict

    Returns:
    - chunk: updated structured array with NormalX/Y/Z
    """
    # Handle both raw dict and JSONRegistry
    if hasattr(config, "config"):
        pseudo_normals_enabled = config.config.get("processing", {}).get("pseudo_normals", False)
    else:
        pseudo_normals_enabled = config.get("processing", {}).get("pseudo_normals", False)

    if not pseudo_normals_enabled:
        logger.info("ℹ️ Pseudo-normal computation skipped (flag is false).")
        return chunk

    if len(chunk) == 0:
        logger.warning("⚠️ Chunk is empty. Skipping pseudo-normal computation.")
        return chunk

    required_fields = ('GpsTime', 'x', 'y', 'z')
    if not all(name in chunk.dtype.names for name in required_fields):
        logger.error("❌ Chunk missing required fields for normal calculation.")
        return chunk

    if not all(name in full_traj.dtype.names for name in required_fields):
        logger.error("❌ Trajectory missing required fields for interpolation.")
        return chunk

    try:
        logger.info("🔄 Recomputing pseudo-normals using trajectory...")

        fx = interp1d(full_traj['GpsTime'], full_traj['x'], bounds_error=False, fill_value="extrapolate")
        fy = interp1d(full_traj['GpsTime'], full_traj['y'], bounds_error=False, fill_value="extrapolate")
        fz = interp1d(full_traj['GpsTime'], full_traj['z'], bounds_error=False, fill_value="extrapolate")

        sensor_x = fx(chunk['GpsTime'])
        sensor_y = fy(chunk['GpsTime'])
        sensor_z = fz(chunk['GpsTime'])

        vectors = np.stack([
            sensor_x - chunk['x'],
            sensor_y - chunk['y'],
            sensor_z - chunk['z']
        ], axis=-1)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6

        normals = vectors / norms

        # Overwrite or append normals
        for i, name in enumerate(["NormalX", "NormalY", "NormalZ"]):
            if name in chunk.dtype.names:
                chunk[name] = normals[:, i]
            else:
                chunk = rfn.append_fields(chunk, name, normals[:, i], usemask=False)

        logger.info(f"✅ Pseudo-normals updated: dtype now {chunk.dtype.names}")

    except Exception as e:
        logger.error(f"❌ Error computing pseudo-normals: {e}")

    return chunk
