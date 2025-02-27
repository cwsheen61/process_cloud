import laspy
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.crs_registry import crs_registry, epsg_to_wkt  # Ensure CRS tracking

def append_to_laz(file_path, points):
    """
    Writes valid points to a LAZ file with LAS 1.4 encoding.
    
    - Ensures only valid (finite) points are written.
    - Uses structured NumPy arrays for consistent handling.
    - Fixes LAS scale at 0.001 for stability.
    - Sets offsets to the center of the dataset.
    - Attaches CRS WKT metadata if available.
    - Debugging: Prints min/max values after transformation.
    """

    print(f"\n📂 Writing to LAZ: {file_path}")

    if points.size == 0:
        print(f"⚠️ Skipping {file_path}: No valid points.")
        return

    # Validate required fields
    required_fields = ["x", "y", "z"]
    if not all(field in points.dtype.names for field in required_fields):
        print(f"❌ ERROR: Point dataset is missing required fields: {required_fields}")
        return

    # Filter out invalid (NaN/infinite) points
    valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    valid_points = points[valid_mask] if valid_mask.any() else np.array([], dtype=points.dtype)

    if valid_points.size == 0:
        print(f"⚠️ Skipping {file_path}: All points are NaN or invalid.")
        return

    # ✅ GNSS Time Conversion
    LEAP_SECONDS = 18  # Current leap second offset (2024)
    POSIX_TO_GNSS_OFFSET = 315964800  # Jan 6, 1980 GPS Epoch
    if "time" in valid_points.dtype.names:
        gnss_time = valid_points["time"] - POSIX_TO_GNSS_OFFSET + LEAP_SECONDS
    else:
        gnss_time = np.zeros(valid_points.shape[0], dtype=np.float64)

    # Retrieve EPSG and convert to WKT
    epsg_code = crs_registry.get("points", -1)  # Default to -1 if unknown
    crs_wkt = epsg_to_wkt(epsg_code) if epsg_code > 0 else None

    # Create LAS header with correct point format
    header = laspy.LasHeader(point_format=3, version="1.4")

    # ✅ Fixed Scale for Consistency
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

    # ✅ Set Offsets to the **Center** of the Dataset
    header.offsets = np.array([
        (valid_points["x"].max() + valid_points["x"].min()) / 2.0,
        (valid_points["y"].max() + valid_points["y"].min()) / 2.0,
        0,
    ])

    print(f"📏 LAS Scale: {header.scales}")
    print(f"📌 LAS Offsets: X={header.offsets[0]}, Y={header.offsets[1]}, Z={header.offsets[2]}")

    # Apply WKT CRS before writing
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))
        crs_registry[file_path] = crs_wkt  # Store CRS in registry

    # Create LAS data object
    print(f"Creating las header")
    las = laspy.LasData(header)

    print(f"Checking for fit")

    # Compute transformed values
    temp_x = np.round((valid_points["x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
    temp_y = np.round((valid_points["y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
    temp_z = np.round((valid_points["z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

    print(f"Created temp_x, temp_y, temp_z")

    # 🛑 Check if arrays are empty before calling min/max
    if temp_x.size > 0 and temp_y.size > 0 and temp_z.size > 0:
        print(f"Expected range las.x {temp_x.min()} to {temp_x.max()}")
        print(f"Expected range las.y {temp_y.min()} to {temp_y.max()}")
        print(f"Expected range las.z {temp_z.min()} to {temp_z.max()}")
    else:
        print("⚠️ WARNING: One or more coordinate arrays are empty! Check valid_points filtering.")


    try:
        # ✅ Convert coordinates to scaled integers (ensuring int32 range)
        print(f"\n\n✅  in the first try section")

        las.x = valid_points["x"] #np.round((valid_points["x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
        las.y = valid_points["y"] #np.round((valid_points["y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
        las.z = valid_points["z"] #np.round((valid_points["z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

        print(f"\n\n✅  got through the las.x las.y las.z assignment")

        # ✅ Print min/max values after transformation
        print(f"🔎 Transformed X Range: {las.x.min()} to {las.x.max()}")
        print(f"🔎 Transformed Y Range: {las.y.min()} to {las.y.max()}")
        print(f"🔎 Transformed Z Range: {las.z.min()} to {las.z.max()}")

        # ✅ Assign other LAS fields safely
        # las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
        # las.return_number = np.ones(valid_points.shape[0], dtype=np.uint8)
        # las.number_of_returns = np.ones(valid_points.shape[0], dtype=np.uint8)
        # las.point_source_id = valid_points["ring"] if "ring" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
        # las.gps_time = gnss_time
        # las.classification = np.zeros(valid_points.shape[0], dtype=np.uint8)
        # las.scan_angle_rank = np.zeros(valid_points.shape[0], dtype=np.int8)
        # las.user_data = np.zeros(valid_points.shape[0], dtype=np.uint8)

        print(f"\n\n✅ got through other assigments ")

        # ✅ Ensure "range" field is included if available
        if "range" in valid_points.dtype.names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
            las["range"] = valid_points["range"].astype(np.float32)

    except Exception as e:
        print(f"\n\n❌ ERROR in preparing LAS data: {e}")
        return

    # ✅ Write data to LAZ file with exception handling
    try:
        with laspy.open(file_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        print(f"✅ Successfully wrote {valid_points.shape[0]} points to {file_path}.")
    except Exception as e:
        print(f"❌ ERROR in writing {file_path}: {e}")

    # 🚨 Exit after first run for easy debugging 🚨
    print("\n🚨 Exiting after first append_to_laz() for debugging. Remove exit() after verification.")