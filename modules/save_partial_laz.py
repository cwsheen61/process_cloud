import sys
import laspy
import numpy as np
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.crs_registry import crs_registry, epsg_to_wkt  # Ensure CRS tracking

# --- Configure Logging ---
logger = logging.getLogger(__name__)

def save_partial_laz(file_path, points):
    """ Saves a transformed point cloud chunk to a LAZ file with correct scaling and CRS. """

    # 🕒 GPS TIME CONVERSION (POSIX → GPS Time)
    LEAP_SECONDS = 18  
    POSIX_TO_GNSS_OFFSET = 315964800  

    if points.size == 0:
        logger.warning(f"⚠️ Skipping {file_path}: No valid points.")
        return

    # ✅ Ensure required fields exist
    required_fields = ["x", "y", "z", "range"]
    missing_fields = [field for field in required_fields if field not in points.dtype.names]

    if missing_fields:
        logger.error(f"❌ ERROR: Missing required fields: {missing_fields}")
        return

    # 🔍 Validate finite values (remove NaN/inf)
    valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    valid_points = points[valid_mask]

    if valid_points.size == 0:
        logger.warning(f"⚠️ Skipping {file_path}: All points are NaN or invalid.")
        return

    # ✅ Debug: Check if 'range' exists
    has_range = "range" in valid_points.dtype.names
    logger.info(f"🔍 Checking 'range' field in point chunk: {'✅ Present' if has_range else '❌ Missing'}")

    # 🔄 CRS Handling
    epsg_code = crs_registry.get("points_with_normals", crs_registry.get("points", -1))  
    crs_wkt = epsg_to_wkt(epsg_code) if epsg_code > 0 else None

    # 🏗️ Create LAS Header
    header = laspy.LasHeader(point_format=1, version="1.4")

    # 📐 Compute scale factors dynamically
    range_x = np.max(valid_points["x"]) - np.min(valid_points["x"])
    range_y = np.max(valid_points["y"]) - np.min(valid_points["y"])
    range_z = np.max(valid_points["z"]) - np.min(valid_points["z"])

    header.scales = np.array([
        max(0.001, range_x / (2**31 - 1)),  
        max(0.001, range_y / (2**31 - 1)),  
        max(0.001, range_z / (2**31 - 1))   
    ])

    # 📌 Set offsets dynamically based on min values
    header.offsets = np.array([
        np.min(valid_points["x"]),
        np.min(valid_points["y"]),
        np.min(valid_points["z"])  
    ])

    logger.info(f"📏 Scale factors: {header.scales}")
    logger.info(f"📌 Offset values: {header.offsets}")

    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))  

    # 📂 Create LAS data object
    las = laspy.LasData(header)

    # ✅ Apply scaling & convert to int32
    las.points['X'] = np.round((valid_points["x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
    las.points['Y'] = np.round((valid_points["y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
    las.points['Z'] = np.round((valid_points["z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

    # 🕒 Convert Time to GPS_TIME
    las.gps_time = valid_points["time"] - POSIX_TO_GNSS_OFFSET + LEAP_SECONDS

    # 📊 Assign additional attributes
    las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
    las.return_number = valid_points["returnNum"] if "returnNum" in valid_points.dtype.names else np.ones(valid_points.shape[0], dtype=np.uint8)
    las.point_source_id = valid_points["ring"] if "ring" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)

    # 🔍 Handle nx, ny, nz if available (🚨 FIXED HERE 🚨)
    if all(dim in valid_points.dtype.names for dim in ["nx", "ny", "nz"]):
        logger.info("🔄 Storing Pseudo-Normals as FLOAT32 for CloudCompare compatibility")

        # ✅ Extract normal values and ensure float32 type
        normal_f32 = np.vstack([
            valid_points["nx"].astype(np.float32),
            valid_points["ny"].astype(np.float32),
            valid_points["nz"].astype(np.float32)
        ]).T  # ✅ Stack properly for laspy

        # ✅ Add extra dimensions for normals
        las.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
        las.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))

        # ✅ Assign normals back correctly
        las["nx"] = normal_f32[:, 0]
        las["ny"] = normal_f32[:, 1]
        las["nz"] = normal_f32[:, 2]

        logger.info("✅ Pseudo-normals successfully stored in LAS file")

    # ✅ Ensure 'range' is saved if present
    if has_range:
        logger.info("🔄 Storing 'range' as FLOAT32 in LAS file")

        # ✅ Ensure correct type for range
        range_f32 = valid_points["range"].astype(np.float32)

        # ✅ Add extra dimension for range
        las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
        las["range"] = range_f32

        logger.info(f"✅ 'range' successfully added to LAS file.")

    # 🚨 Check LAS Data Before Writing
    logger.info(f"🛠 Writing LAS file: {file_path}")
    logger.info(f"  ➤ Number of points: {valid_points.shape[0]}")
    logger.info(f"  ➤ las.x range: {las.x.min()} to {las.x.max()}")
    logger.info(f"  ➤ las.y range: {las.y.min()} to {las.y.max()}")
    logger.info(f"  ➤ las.z range: {las.z.min()} to {las.z.max()}")

    # 📂 Write to file
    file_path = file_path.replace(".laz", ".las")
    try:
        with laspy.open(file_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        logger.info(f"✅ Successfully wrote {valid_points.shape[0]} points to {file_path}.")
        sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing
    except Exception as e:
        logger.error(f"❌ ERROR writing {file_path}: {e}")
        sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing









# import sys
# import laspy
# import numpy as np
# import logging
# from laspy.vlrs.known import WktCoordinateSystemVlr
# from modules.crs_registry import crs_registry, epsg_to_wkt  # Ensure CRS tracking

# # --- Configure Logging ---
# logger = logging.getLogger(__name__)

# def save_partial_laz(file_path, points):
#     """ Saves a transformed point cloud chunk to a LAZ file with correct scaling and CRS. """

#     # 🕒 GPS TIME CONVERSION (POSIX → GPS Time)
#     LEAP_SECONDS = 18  
#     POSIX_TO_GNSS_OFFSET = 315964800  

#     if points.size == 0:
#         logger.warning(f"⚠️ Skipping {file_path}: No valid points.")
#         return

#     # ✅ Ensure required fields exist
#     required_fields = ["x", "y", "z"]
#     if not all(field in points.dtype.names for field in required_fields):
#         logger.error(f"❌ ERROR: Missing required fields: {required_fields}")
#         return

#     # 🔍 Validate finite values (remove NaN/inf)
#     valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
#     valid_points = points[valid_mask]

#     if valid_points.size == 0:
#         logger.warning(f"⚠️ Skipping {file_path}: All points are NaN or invalid.")
#         return

#     # 🔄 CRS Handling
#     epsg_code = crs_registry.get("points_with_normals", crs_registry.get("points", -1))  
#     crs_wkt = epsg_to_wkt(epsg_code) if epsg_code > 0 else None

#     # 🏗️ Create LAS Header
#     header = laspy.LasHeader(point_format=1, version="1.4")

#     # 📐 Compute scale factors dynamically
#     range_x = np.max(valid_points["x"]) - np.min(valid_points["x"])
#     range_y = np.max(valid_points["y"]) - np.min(valid_points["y"])
#     range_z = np.max(valid_points["z"]) - np.min(valid_points["z"])

#     header.scales = np.array([
#         max(0.001, range_x / (2**31 - 1)),  
#         max(0.001, range_y / (2**31 - 1)),  
#         max(0.001, range_z / (2**31 - 1))   
#     ])

#     # 📌 Set offsets dynamically based on min values
#     header.offsets = np.array([
#         np.min(valid_points["x"]),
#         np.min(valid_points["y"]),
#         np.min(valid_points["z"])  
#     ])

#     logger.info(f"📏 Scale factors: {header.scales}")
#     logger.info(f"📌 Offset values: {header.offsets}")

#     if crs_wkt:
#         header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))  

#     # 📂 Create LAS data object
#     las = laspy.LasData(header)

#     # ✅ Apply scaling & convert to int32
#     las.points['X'] = np.round((valid_points["x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
#     las.points['Y'] = np.round((valid_points["y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
#     las.points['Z'] = np.round((valid_points["z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

#     # 🕒 Convert Time to GPS_TIME
#     las.gps_time = valid_points["time"] - POSIX_TO_GNSS_OFFSET + LEAP_SECONDS

#     # 📊 Assign additional attributes
#     las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
#     las.return_number = valid_points["return_number"] if "return_number" in valid_points.dtype.names else np.ones(valid_points.shape[0], dtype=np.uint8)
#     las.point_source_id = valid_points["ring_number"] if "ring_number" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)

#     # 🔍 Handle nx, ny, nz if available
#     for dim in ["nx", "ny", "nz"]:
#         if dim in valid_points.dtype.names:
#             las.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
#             las[dim] = valid_points[dim].astype(np.float32)

#     # 📡 Store range if available
#     if "range" in valid_points.dtype.names:
#         las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
#         las["range"] = valid_points["range"].astype(np.float32)

#     # 🚨 Check LAS Data Before Writing
#     logger.info(f"🛠 Writing LAS file: {file_path}")
#     logger.info(f"  ➤ Number of points: {valid_points.shape[0]}")
#     logger.info(f"  ➤ las.x range: {las.x.min()} to {las.x.max()}")
#     logger.info(f"  ➤ las.y range: {las.y.min()} to {las.y.max()}")
#     logger.info(f"  ➤ las.z range: {las.z.min()} to {las.z.max()}")

#     # 📂 Write to file
#     file_path = file_path.replace(".laz", ".las")
#     try:
#         with laspy.open(file_path, mode="w", header=header) as writer:
#             writer.write_points(las.points)
#         logger.info(f"✅ Successfully wrote {valid_points.shape[0]} points to {file_path}.")
#         sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing
#     except Exception as e:
#         logger.error(f"❌ ERROR writing {file_path}: {e}")
#         sys.stdout.flush()  # ✅ Prevents buffering in multiprocessing
