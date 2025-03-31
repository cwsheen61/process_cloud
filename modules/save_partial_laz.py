import sys
import laspy
import numpy as np
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr
from modules.crs_registry import crs_registry, epsg_to_wkt

logger = logging.getLogger(__name__)

def save_partial_laz(filename, points, config):
    """Save a chunk of filtered points to a LAZ file using config-driven format logic."""

    logger.info(f"📦 Saving chunk to LAZ. NumPy dtype: {points.dtype}")
    logger.info(f"🧠 Detected fields: {points.dtype.names}")

    if points.size == 0:
        logger.warning(f"⚠️ Skipping {filename}: No valid points.")
        return

    # Determine which format to use based on presence of normals
    has_normals = all(dim in points.dtype.names for dim in ["nx", "ny", "nz"])
    format_name = "point_cloud_with_normals" if has_normals else "pointcloud_sensor"

    # Load format from config
    point_format_def = config.get("data_formats", {}).get(format_name)
    if not point_format_def:
        logger.error(f"❌ Config missing '{format_name}' definition.")
        return

    # Clean invalid values
    valid_mask = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    valid_points = points[valid_mask]

    if valid_points.size == 0:
        logger.warning(f"⚠️ Skipping {filename}: All points are NaN or invalid.")
        return

    # Coordinate system
    epsg_code = crs_registry.get(format_name, -1)
    crs_wkt = epsg_to_wkt(epsg_code) if epsg_code > 0 else None

    # Build LAS header
    header = laspy.LasHeader(point_format=1, version="1.4")

    # Calculate dynamic scale and offset
    min_xyz = np.array([
        np.min(valid_points["x"]),
        np.min(valid_points["y"]),
        np.min(valid_points["z"]),
    ])
    max_xyz = np.array([
        np.max(valid_points["x"]),
        np.max(valid_points["y"]),
        np.max(valid_points["z"]),
    ])
    range_xyz = max_xyz - min_xyz

    header.scales = np.maximum(range_xyz / (2**31 - 1), 0.001)
    header.offsets = min_xyz

    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))

    # Create LAS data structure
    las = laspy.LasData(header)

    # Required XYZ conversion
    las.X = np.round((valid_points["x"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
    las.Y = np.round((valid_points["y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
    las.Z = np.round((valid_points["z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)

    # GPS time conversion
    LEAP_SECONDS = 18
    POSIX_TO_GNSS_OFFSET = 315964800
    las.gps_time = valid_points["time"] - POSIX_TO_GNSS_OFFSET + LEAP_SECONDS

    # Basic attributes
    las.intensity = valid_points["intensity"] if "intensity" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)
    las.return_number = valid_points["returnNum"] if "returnNum" in valid_points.dtype.names else np.ones(valid_points.shape[0], dtype=np.uint8)
    las.point_source_id = valid_points["ring"] if "ring" in valid_points.dtype.names else np.zeros(valid_points.shape[0], dtype=np.uint16)

    # Optional: Normals
    if has_normals:
        for dim in ["nx", "ny", "nz"]:
            if dim not in las.point_format.extra_dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=np.float32))
            las[dim] = valid_points[dim].astype(np.float32)

    # Optional: Range
    if "range" in valid_points.dtype.names:
        if "range" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="range", type=np.float32))
        las["range"] = valid_points["range"].astype(np.float32)

    try:
        output_path = filename.replace(".laz", ".las")
        with laspy.open(output_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        logger.info(f"✅ Saved {valid_points.shape[0]} points to {output_path}")
    except Exception as e:
        logger.error(f"❌ ERROR writing {filename}: {e}")
        sys.stdout.flush()
