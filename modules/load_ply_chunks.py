import os
import logging
import numpy as np
from modules.json_registry import JSONRegistry
from modules.build_dtype import build_dtype_from_config

logger = logging.getLogger(__name__)

def yield_ply_chunks(config_path):
    """
    Generator that reads a PLY file in chunks without loading the entire file into memory.
    
    Reads the header (assumed to end with "end_header") and then yields a tuple (chunk_idx, chunk)
    for each chunk of point data read using np.fromfile.
    
    The configuration should include:
      - "files.ply": Path to the PLY file.
      - "processing.chunk_size": Number of points to load per chunk.
      - "pathname": Base directory for relative paths.
      - "filtering.format_name": The key for the desired point cloud format.
      - "data_formats": Contains a mapping for the format.
    
    Yields:
        tuple: (chunk_idx, chunk) where chunk is a NumPy array of point records.
    """
    config = JSONRegistry(config_path, config_path)
    
    # Get and resolve the PLY file path.
    ply_file = config.get("files.ply")
    base_path = config.get("pathname")
    if not os.path.isabs(ply_file):
        ply_file = os.path.join(base_path, ply_file)
    
    chunk_size = config.get("processing.chunk_size", 5000000)
    
    dtype = build_dtype_from_config(config)

    try:
        with open(ply_file, "rb") as f:
            # Read header: assume header ends with "end_header"
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            logger.info(f"PLY header read ({len(header)} bytes). Starting to yield chunks.")
            
            chunk_idx = 0
            while True:
                chunk = np.fromfile(f, dtype=dtype, count=chunk_size)
                if chunk.size == 0:
                    break
                yield (chunk_idx, chunk)
                chunk_idx += 1
    except Exception as e:
        logger.error(f"Error reading PLY file in chunks: {e}")
        raise
