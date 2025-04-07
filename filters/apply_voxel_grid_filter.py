import numpy as np

def apply_voxel_grid_filter(points, voxel_config, header):
    voxel_size = voxel_config.get("filtering.voxel_filter.voxel_size", 0.1)
    x_scale, y_scale, z_scale = header.scales
    x_offset, y_offset, z_offset = header.offsets

    # Ensure X/Y/Z are scaled to float if they're integers
    if np.issubdtype(points["X"].dtype, np.integer):
        x = points["X"] * x_scale + x_offset
        y = points["Y"] * y_scale + y_offset
        z = points["Z"] * z_scale + z_offset
    else:
        x = points["X"]
        y = points["Y"]
        z = points["Z"]

    # Compute voxel grid indices
    voxel_indices = (
        np.floor(x / voxel_size).astype(np.int64),
        np.floor(y / voxel_size).astype(np.int64),
        np.floor(z / voxel_size).astype(np.int64),
    )
    keys = np.stack(voxel_indices, axis=-1)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    num_voxels = len(uniq)

    voxel_data = {}

    # Basic averaging fields
    voxel_data["X"] = np.bincount(inverse, weights=x, minlength=num_voxels) / counts
    voxel_data["Y"] = np.bincount(inverse, weights=y, minlength=num_voxels) / counts
    voxel_data["Z"] = np.bincount(inverse, weights=z, minlength=num_voxels) / counts

    if "intensity" in points.dtype.names:
        voxel_data["intensity"] = np.bincount(inverse, weights=points["intensity"], minlength=num_voxels) / counts
    if "gps_time" in points.dtype.names:
        voxel_data["gps_time"] = np.bincount(inverse, weights=points["gps_time"], minlength=num_voxels) / counts

    # Set zeroed categorical values
    voxel_data["return_number"] = np.zeros(num_voxels, dtype=np.uint8)
    voxel_data["point_source_id"] = np.zeros(num_voxels, dtype=np.uint16)

    # Normals: average then normalize
    if "NormalX" in points.dtype.names and "NormalY" in points.dtype.names and "NormalZ" in points.dtype.names:
        nx = points["NormalX"].astype(np.float64)
        ny = points["NormalY"].astype(np.float64)
        nz = points["NormalZ"].astype(np.float64)

        nx_sum = np.bincount(inverse, weights=nx, minlength=num_voxels)
        ny_sum = np.bincount(inverse, weights=ny, minlength=num_voxels)
        nz_sum = np.bincount(inverse, weights=nz, minlength=num_voxels)

        norm = np.sqrt(nx_sum**2 + ny_sum**2 + nz_sum**2)
        norm[norm == 0] = 1  # Prevent division by zero

        voxel_data["NormalX"] = nx_sum / norm
        voxel_data["NormalY"] = ny_sum / norm
        voxel_data["NormalZ"] = nz_sum / norm

    # Range: simple average
    if "range" in points.dtype.names:
        voxel_data["range"] = np.bincount(inverse, weights=points["range"], minlength=num_voxels) / counts

    # Color channels
    for color in ("red", "green", "blue"):
        if color in points.dtype.names:
            voxel_data[color] = np.bincount(inverse, weights=points[color], minlength=num_voxels) / counts

    # Density
    voxel_volume = voxel_size ** 3
    voxel_data["Density"] = counts / voxel_volume

    # Build structured array output
    dtype = [(k, np.float32 if k not in ("return_number", "point_source_id") else np.uint16 if k == "point_source_id" else np.uint8)
             for k in voxel_data.keys()]
    output = np.empty(num_voxels, dtype=dtype)
    for k in voxel_data:
        output[k] = voxel_data[k]

    return output
