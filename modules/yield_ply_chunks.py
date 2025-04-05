#!/usr/bin/env python3
"""
Yields fixed-size chunks of binary PLY data aligned to full point records.
"""

import os
import struct
import numpy as np
import logging

logger = logging.getLogger(__name__)

GPS_EPOCH_OFFSET = 315964800  # seconds between Unix and GPS epoch


def load_ply_header(ply_path):
    """Parses the PLY header to extract element count and field types."""
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b'end_header':
                break

    format_map = {
        "char": "i1", "uchar": "u1",
        "short": "i2", "ushort": "u2",
        "int": "i4", "uint": "u4",
        "float": "f4", "double": "f8"
    }

    field_names = []
    field_types = []
    num_points = 0

    for line in header_lines:
        tokens = line.decode("ascii").strip().split()
        if not tokens:
            continue
        if tokens[0] == "element" and tokens[1] == "vertex":
            num_points = int(tokens[2])
        elif tokens[0] == "property":
            dtype = format_map[tokens[1]]
            name = tokens[2]
            field_names.append(name)
            field_types.append(dtype)

    header_length = sum(len(line) for line in header_lines)
    dtype = np.dtype(list(zip(field_names, field_types)))
    return num_points, header_length, dtype


def normalize_fields(data, field_map):
    """Applies field renaming and GPS time conversion."""
    new_dtype = []
    rename_dict = {}

    for name in data.dtype.names:
        if name not in field_map:
            logger.debug(f"⚠️ Field '{name}' not in map, skipping.")
            continue
        new_name = field_map[name]
        rename_dict[name] = new_name
        new_dtype.append((new_name, data.dtype[name]))

    # Build new structured array
    normalized = np.empty(data.shape, dtype=new_dtype)
    for old_name, new_name in rename_dict.items():
        normalized[new_name] = data[old_name]

    # Apply GPS time conversion
    if "GpsTime" in normalized.dtype.names:
        normalized["GpsTime"] -= GPS_EPOCH_OFFSET
        logger.debug("🕒 Converted POSIX time to GPS time.")

    return normalized


def yield_ply_chunks(config, data_formats, field_map, chunk_size=5_000_000):
    """
    Streams chunks of structured point data from a binary PLY file.
    Renames fields based on `field_map`, and converts POSIX time to GPS time.

    Args:
        config (JSONRegistry or dict): Loaded configuration.
        data_formats (dict): Mapping of known point formats (from config).
        field_map (dict): Dictionary mapping raw PLY names to LAS field names.
        chunk_size (int): Number of points per chunk.

    Yields:
        Tuple[int, np.ndarray]: (chunk index, normalized point data)
    """
    ply_path = config.get("files.ply")
    num_points, header_len, raw_dtype = load_ply_header(ply_path)

    logger.info(f"📄 Loaded header from {os.path.basename(ply_path)}: {num_points} points, dtype={raw_dtype}")

    chunk_idx = 0
    with open(ply_path, "rb") as f:
        f.seek(header_len)
        remaining = num_points

        while remaining > 0:
            to_read = min(chunk_size, remaining)
            buf = f.read(raw_dtype.itemsize * to_read)
            chunk = np.frombuffer(buf, dtype=raw_dtype)

            # Normalize field names and convert times
            norm_chunk = normalize_fields(chunk, field_map)

            yield chunk_idx, norm_chunk
            chunk_idx += 1
            remaining -= to_read
