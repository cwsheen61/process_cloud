import numpy as np
from scipy.interpolate import interp1d
from pyproj import CRS, Transformer
from data_types import gnss_traj_dtype  # ✅ Explicitly import gnss_traj_dtype

def compute_global_transform(gnss_file):
    """
    Computes a 3D transformation (rotation R and translation t) to align the sensor trajectory
    with the sparse GNSS coordinates from gnss_file.
    
    Assumes gnss_file has a header and follows the gnss_traj_dtype format.

    Returns:
        R (3x3 ndarray): Rotation matrix (Z-axis only)
        t (1x3 ndarray): Translation vector (with Z=0)
        crs_wkt (str): WKT representation of the GNSS CRS
    """
    
    # ✅ Load GNSS data using the correct structured dtype
    gnss_data = np.loadtxt(gnss_file, dtype=gnss_traj_dtype, delimiter=None, comments="%")

    # Extract necessary GNSS fields
    gnss_times = gnss_data["time"]
    gnss_positions = np.column_stack((gnss_data["gps_x"], gnss_data["gps_y"]))  # Only X, Y

    # Determine UTM Zone from GNSS mean longitude/latitude
    lon_center, lat_center = np.mean(gnss_data["longitude"]), np.mean(gnss_data["latitude"])
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = "north" if lat_center >= 0 else "south"
    utm_epsg = f"326{utm_zone}" if hemisphere == "north" else f"327{utm_zone}"
    print(f"🛰️ GNSS Data UTM Zone: {utm_zone} ({hemisphere}) → EPSG:{utm_epsg}")

    # Convert GNSS Lon/Lat to UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    utm_e, utm_n = transformer.transform(gnss_data["longitude"], gnss_data["latitude"])
    gnss_utm_positions = np.column_stack((utm_e, utm_n))

    # Filter for only high-quality GNSS fixes (Fix 4 or Fix 5)
    high_quality_mask = (gnss_data["fix_quality"] == 4) | (gnss_data["fix_quality"] == 5)
    if np.sum(high_quality_mask) < 3:
        raise ValueError("⚠️ Not enough high-quality GNSS points for a stable R,T matrix!")

    gnss_filtered = gnss_utm_positions[high_quality_mask]
    traj_filtered = np.column_stack((gnss_data["traj_x"], gnss_data["traj_y"]))[high_quality_mask]

    # Compute transformation (R, t)
    gnss_center = np.mean(gnss_filtered, axis=0)
    traj_center = np.mean(traj_filtered, axis=0)
    gnss_shifted = gnss_filtered - gnss_center
    traj_shifted = traj_filtered - traj_center

    U, _, Vt = np.linalg.svd(np.dot(gnss_shifted.T, traj_shifted))
    R = np.dot(Vt.T, U.T)

    # Ensure only rotation about Z-axis
    R = np.array([
        [R[0, 0], R[0, 1], 0],
        [R[1, 0], R[1, 1], 0],
        [0, 0, 1]
    ])

    t = np.array([gnss_center[0] - traj_center[0], gnss_center[1] - traj_center[1], 0])  # Set Z=0

    # Get CRS in WKT format
    crs = CRS.from_epsg(int(utm_epsg))
    crs_wkt = crs.to_wkt()

    print(f"✅ Computed Global Transformation: R =\n{R},\n t = {t}")
    print(f"🌍 Computed WKT CRS: {crs_wkt}")

    return R, t, crs_wkt
