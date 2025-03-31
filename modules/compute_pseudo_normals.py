import numpy as np
from scipy.interpolate import interp1d
import logging
from numpy.lib import recfunctions as rfn

logger = logging.getLogger(__name__)

def compute_pseudo_normals(chunk, full_traj):
    """
    Compute pseudo-normals for each point in the chunk based on direction to sensor location.
    Interpolates sensor trajectory to get accurate positions at point times.

    Parameters:
    - chunk: numpy structured array with a 'time' field.
    - full_traj: numpy structured array with 'time', 'x', 'y', 'z' fields.

    Returns:
    - chunk: same as input, but with additional 'nx', 'ny', 'nz' fields for normals.
    """
    if len(chunk) == 0:
        logger.warning("Chunk is empty. Skipping pseudo-normal computation.")
        return chunk

    if not all(name in chunk.dtype.names for name in ('time', 'x', 'y', 'z')):
        logger.error("Chunk is missing required fields for normal calculation.")
        return chunk

    if not all(name in full_traj.dtype.names for name in ('time', 'x', 'y', 'z')):
        logger.error("Trajectory missing required fields for interpolation.")
        return chunk

    try:
        # Interpolate sensor positions at each point time
        fx = interp1d(full_traj['time'], full_traj['x'], bounds_error=False, fill_value="extrapolate")
        fy = interp1d(full_traj['time'], full_traj['y'], bounds_error=False, fill_value="extrapolate")
        fz = interp1d(full_traj['time'], full_traj['z'], bounds_error=False, fill_value="extrapolate")

        sensor_x = fx(chunk['time'])
        sensor_y = fy(chunk['time'])
        sensor_z = fz(chunk['time'])

        vectors = np.stack([
            sensor_x - chunk['x'],
            sensor_y - chunk['y'],
            sensor_z - chunk['z']
        ], axis=-1)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6  # Avoid division by zero

        normals = vectors / norms

        # Append the normals to the chunk using recfunctions
        chunk = rfn.append_fields(chunk, ['nx', 'ny', 'nz'],
                                  [normals[:, 0], normals[:, 1], normals[:, 2]],
                                  usemask=False)

        logger.info(f"✅ Pseudo-normals added: dtype now {chunk.dtype.names}")

    except Exception as e:
        logger.error(f"❌ Error computing pseudo-normals: {e}")

    return chunk
