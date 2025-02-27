import numpy as np
import logging
from modules.data_types import pointcloud_sensor_dtype   # Import globally defined dtype
from modules.crs_registry import crs_registry  # Ensure CRS tracking

logger = logging.getLogger(__name__)

def load_ply_chunks(ply_file, chunk_size):
    """
    Streams chunks of binary PLY data efficiently.

    Args:
        ply_file (str): Path to the binary PLY file.
        chunk_size (int): Number of points to read per chunk.

    Yields:
        np.ndarray: A structured NumPy array containing a chunk of PLY data.
    """
    global crs_registry  # Explicitly declare global

    try:
        crs_registry["points"] = -1  # Using -1 instead of text for SENSOR_COORDS consistency

        logger.info(f"Opening PLY file: {ply_file}")

        with open(ply_file, 'rb') as f:
            # Skip ASCII header and find binary data start
            while True:
                line = f.readline().decode('utf-8').strip()
                if line.startswith("end_header"):
                    break

            binary_start = f.tell()
            point_size = np.dtype(pointcloud_sensor_dtype ).itemsize
            chunk_num = 0

            while True:
                f.seek(binary_start + chunk_num * chunk_size * point_size)
                chunk_data = np.fromfile(f, dtype=ppointcloud_sensor_dtype , count=chunk_size)
                
                if chunk_data.size == 0:
                    break

                yield chunk_data
                chunk_num += 1

            logger.info(f"Finished streaming PLY file: {ply_file}")

    except FileNotFoundError:
        logger.error(f"PLY file not found: {ply_file}")
    except Exception as e:
        logger.error(f"Error reading PLY file '{ply_file}': {e}")
