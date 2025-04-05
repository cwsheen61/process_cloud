import os
import logging
import numpy as np
from modules.json_registry import JSONRegistry

logger = logging.getLogger(__name__)


def load_trajectory(config: JSONRegistry):
    traj_file = config.get("files.trajectory")
    base_path = config.get("pathname")
    full_path = os.path.join(base_path, traj_file)

    logger.info(f"📄 Loading trajectory from: {full_path}")

    # Skip the header row (e.g., "time x y z qx qy qz qw")
    skip_rows = 1

    # Read the trajectory file
    dtype = [
        ('time', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('z', 'f8'),
        ('qx', 'f8'),
        ('qy', 'f8'),
        ('qz', 'f8'),
        ('qw', 'f8'),
        ('dummy', 'f8'),  # in case there’s a 9th column; if not, remove this line
    ]

    try:
        data = np.loadtxt(full_path, dtype=dtype, skiprows=skip_rows)
    except ValueError as e:
        # Try fallback without 9th column
        dtype = [
            ('time', 'f8'),
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8'),
            ('qx', 'f8'),
            ('qy', 'f8'),
            ('qz', 'f8'),
            ('qw', 'f8'),
        ]
        data = np.loadtxt(full_path, dtype=dtype, skiprows=skip_rows)

    # Convert POSIX to approximate GpsTime offset if needed
    # (you could refine this, but here's a placeholder)
    gps_time = data['time'] - 315964800  # 1980-01-06 epoch offset in seconds

    # Construct new structured array with GpsTime
    new_dtype = [('GpsTime', 'f8')] + [(name, data.dtype[name]) for name in data.dtype.names if name != 'time']
    converted = np.empty(data.shape, dtype=new_dtype)
    converted['GpsTime'] = gps_time
    for name in data.dtype.names:
        if name != 'time':
            converted[name] = data[name]

    logger.info(f"✅ Loaded trajectory file: {full_path}, {len(converted)} entries.")
    config.set("current_trajectory_state", "sensor_loaded")
    return converted
