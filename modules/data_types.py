import numpy as np

# --- Sensor Point Cloud Data Type ---
pointcloud_sensor_dtype = np.dtype([
    ('time', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('intensity', np.float32),
    ('return', np.uint8),
    ('ring', np.uint8),
    ('range', np.float32)
])

# --- Sensor Trajectory Data Type ---
trajectory_sensor_dtype = np.dtype([
    ('time', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('qw', np.float64),
    ('qx', np.float64),
    ('qy', np.float64),
    ('qz', np.float64)
])

# --- Point Cloud with Normals Data Type ---
point_cloud_with_normals_dtype = np.dtype([
    ('time', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('nx', np.float64),
    ('ny', np.float64),
    ('nz', np.float64),
    ('intensity', np.float32),
    ('return', np.uint8),
    ('laser', np.uint8),
    ('range', np.float32)
])

# --- GNSS Trajectory Data Type ---
gnss_trajectory_dtype = np.dtype([
    ('traj_x', np.float64),
    ('traj_y', np.float64),
    ('traj_z', np.float64),
    ('gps_x', np.float64),
    ('gps_y', np.float64),
    ('gps_z', np.float64),
    ('gps_reported_error_norm', np.float64),
    ('gps_reported_horizontal_error_norm', np.float64),
    ('longitude', np.float64),
    ('latitude', np.float64),
    ('height', np.float64),
    ('error_norm', np.float32),
    ('horizontal_error_norm', np.float32),
    ('scaled_error_norm', np.float32),
    ('horizontal_scaled_error_norm', np.float32),
    ('fix_quality', np.uint8)
])

# --- Paint Cloud Sensor Data Type ---
paintcloud_sensor_dtype = np.dtype([
    ('time', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('nx', np.float64),
    ('ny', np.float64),
    ('nz', np.float64),
    ('r', np.uint8),
    ('g', np.uint8),
    ('b', np.uint8)
])

# --- Paint Cloud with Range Data Type ---
paintcloud_with_range_dtype = np.dtype([
    ('time', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('nx', np.float64),
    ('ny', np.float64),
    ('nz', np.float64),
    ('r', np.uint8),
    ('g', np.uint8),
    ('b', np.uint8),
    ('range', np.float32)
])
