#!/usr/bin/env python3
"""
Yields fixed-size chunks of binary PLY data aligned to full point records.
"""

import os
import sys
import logging
import struct
import numpy as np
from modules.build_dtype import build_dtype_from_config

logger = logging.getLogger(__name__)

def yield_ply_chunks(config):
    ply_file_path = config.get("ply_path") or config.get("files.ply")
    if not ply_file_path or not os.path.isfile(ply_file_path):
        logger.error("❌ PLY file path is not specified or does not exist in config.")
        sys.exit(1)

    chunk_size = config.get("processing.chunk_size", 5_000_000)
    dtype = build_dtype_from_config(config)
    point_size = dtype.itemsize

    print(f"========================{point_size}")

    header_lines = []
    header_size = 0

    with open(ply_file_path, "rb") as f:
        # Read header line by line, byte-wise until 'end_header' is reached
        while True:
            line = b""
            while True:
                char = f.read(1)
                if not char:
                    break  # EOF
                line += char
                if char == b"\n":
                    break
            decoded_line = line.decode("utf-8").strip()
            header_lines.append(decoded_line)
            header_size += len(line)
            if decoded_line == "end_header": break

        
        chunk_size_points = config.get("processing.chunk_size", 5_000_000)
        

        logger.info("PLY header read (%d bytes). Starting to yield chunks.", header_size)
        logger.info(f"Chunk size: {chunk_size_points}")
        logger.info(f"Point size: {point_size}")
        
        chunk_idx = 0
        while True:
            points = np.fromfile(f, dtype=dtype, count=chunk_size_points)

            yield (chunk_idx, points)
            chunk_idx += 1

            if config.get("processing.test_mode", False):
                if chunk_idx >= config.get("processing.test_chunks", 10):
                    logger.info("🧪 Test mode halted after %d chunks.", chunk_idx)
                    break
