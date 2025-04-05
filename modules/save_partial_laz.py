import sys
import laspy
import numpy as np
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.crs_registry import crs_registry, epsg_to_wkt

logger = logging.getLogger(__name__)

def save_partial_laz(filename, points, config):
    logger.info(f"📦 Saving chunk to LAZ. NumPy dtype: {points.dtype}")
    logger.info(f"🧠 Detected fields: {points.dtype.names}")

    if points.size == 0:
        logger.warning(f"⚠️ Skipping {filename}: No valid points.")
        return

    has_normals = all(dim in points.dtype.names for dim in ["NormalX", "NormalY", "NormalZ"])
    has_color = all(dim in points.dtype.names for dim in ["red", "green", "blue"])

    # Validate
    valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        logger.warning(f"⚠️ Skipping {filename}: All points are NaN or invalid.")
        return

    # Header & CRS
    header = laspy.LasHeader(point_format=3, version="1.4")  # Format 3 supports GPS + color

    # ✅ Fix min/max computation for structured arrays
    x_vals = valid_points["x"]
    y_vals = valid_points["y"]
    z_vals = valid_points["z"]
    min_xyz = np.array([np.min(x_vals), np.min(y_vals), np.min(z_vals)])
    max_xyz = np.array([np.max(x_vals), np.max(y_vals), np.max(z_vals)])
    range_xyz = max_xyz - min_xyz

    header.scales = np.maximum(range_xyz / (2**31 - 1), 0.001)
    header.offsets = min_xyz

    epsg_code = crs_registry.get("pointcloud_sensor", -1)
    if epsg_code > 0:
        wkt = epsg_to_wkt(epsg_code)
        header.vlrs.append(WktCoordinateSystemVlr(wkt))

    las = laspy.LasData(header)

    # Required fields
    las.X = np.round((x_vals - header.offsets[0]) / header.scales[0]).astype(np.int32)
    las.Y = np.round((y_vals - header.offsets[1]) / header.scales[1]).astype(np.int32)
    las.Z = np.round((z_vals - header.offsets[2]) / header.scales[2]).astype(np.int32)
    if "GpsTime" in valid_points.dtype.names:
        las.gps_time = valid_points["GpsTime"]
    else:
        logger.warning("⚠️ 'GpsTime' not found. GPS time will not be written.")

    # Optional standard fields
    las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
    las.return_number = valid_points["return"] if "return" in valid_points.dtype.names else np.ones(valid_points.shape[0], dtype=np.uint8)
    las.point_source_id = valid_points["pt_source_id"] if "pt_source_id" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)

    # Optional: Color
    if has_color:
        las.red = valid_points["red"].astype(np.uint16)
        las.green = valid_points["green"].astype(np.uint16)
        las.blue = valid_points["blue"].astype(np.uint16)
        logger.info("🎨 Color channels (RGB) included in output.")

    # Optional: Normals
    if has_normals:
        for dim in ["NormalX", "NormalY", "NormalZ"]:
            if dim not in las.point_format.extra_dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
            las[dim] = valid_points[dim].astype(np.float32)

    # Optional: Range
    if "range" in valid_points.dtype.names:
        if "range" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
        las["range"] = valid_points["range"].astype(np.float32)

    # Save to .las
    try:
        output_path = filename.replace(".laz", ".las")
        with laspy.open(output_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        logger.info(f"✅ Saved {valid_points.shape[0]} points to {output_path}")
    except Exception as e:
        logger.error(f"❌ ERROR writing {filename}: {e}")
        sys.stdout.flush()
