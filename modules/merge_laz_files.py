import laspy
import os
import logging
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from tqdm import tqdm  # For progress tracking
from modules.crs_registry import crs_registry  # Ensure CRS consistency
from modules.json_registry import JSONRegistry

logger = logging.getLogger(__name__)

def merge_single_group(laz_files, merged_output_path):
    """
    Merges multiple LAZ files into a single output file.
    Returns True on success.
    """
    if not laz_files:
        logger.error(f"No valid LAZ files provided for merging into {merged_output_path}.")
        return False

    logger.info(f"🔄 Starting merge of {len(laz_files)} LAZ files into {merged_output_path}...")

    # Compute global minimum offsets from all files for correct UTM preservation.
    global_min_x, global_min_y, global_min_z = np.inf, np.inf, np.inf
    merged_crs_wkt = None

    for laz_file in laz_files:
        with laspy.open(laz_file) as las:
            if las.header.point_count == 0:
                tqdm.write(f"⚠️ Skipping empty file: {laz_file}")
                continue
            global_min_x = min(global_min_x, las.header.offsets[0])
            global_min_y = min(global_min_y, las.header.offsets[1])
            global_min_z = min(global_min_z, las.header.offsets[2])
            if merged_crs_wkt is None:
                for vlr in las.header.vlrs:
                    if isinstance(vlr, WktCoordinateSystemVlr):
                        merged_crs_wkt = vlr.string
                        break

    logger.info(f"📌 Global UTM Offsets: X={global_min_x}, Y={global_min_y}, Z={global_min_z}")

    # Use the first valid file as a reference to build the merged header.
    with laspy.open(laz_files[0]) as first_file:
        header = first_file.header
        merged_header = laspy.LasHeader(point_format=header.point_format, version=header.version)
        merged_header.offsets = np.array([global_min_x, global_min_y, global_min_z])
        merged_header.scales = header.scales

        if merged_crs_wkt:
            merged_header.vlrs.append(WktCoordinateSystemVlr(merged_crs_wkt))
            logger.info("📌 Using CRS from first valid file.")
        else:
            crs_epsg = crs_registry.get("points", -1)
            if crs_epsg > 0:
                logger.warning(f"⚠️ No CRS found in input files. Using EPSG:{crs_epsg} from crs_registry.")
                merged_header.vlrs.append(WktCoordinateSystemVlr(f"EPSG:{crs_epsg}"))

    # Write the merged points with proper UTM adjustments.
    with laspy.open(merged_output_path, mode="w", header=merged_header) as merged_file:
        total_written = 0
        for laz_file in tqdm(laz_files, desc="📂 Merging Progress", unit="file", dynamic_ncols=True):
            with laspy.open(laz_file) as las:
                points = las.read().points
                if len(points) == 0:
                    tqdm.write(f"⚠️ Skipping empty file: {laz_file}")
                    continue

                # Convert raw points to UTM coordinates.
                adjusted_x = (points.X * las.header.scales[0]) + las.header.offsets[0]
                adjusted_y = (points.Y * las.header.scales[1]) + las.header.offsets[1]
                adjusted_z = (points.Z * las.header.scales[2]) + las.header.offsets[2]

                # Re-scale to integers using the merged offsets.
                points.X = np.round((adjusted_x - global_min_x) / merged_header.scales[0]).astype(np.int32)
                points.Y = np.round((adjusted_y - global_min_y) / merged_header.scales[1]).astype(np.int32)
                points.Z = np.round((adjusted_z - global_min_z) / merged_header.scales[2]).astype(np.int32)

                merged_file.write_points(points)
                total_written += len(points)
        tqdm.write(f"✅ Total points written: {total_written} to {merged_output_path}")

    with laspy.open(merged_output_path) as las_check:
        logger.info(f"📌 Final LAS Offsets: X={las_check.header.offsets[0]}, Y={las_check.header.offsets[1]}, Z={las_check.header.offsets[2]}")
        logger.info(f"📏 Final LAS Scales: {las_check.header.scales}")
        logger.info(f"📊 Final LAS Point Count: {las_check.header.point_count}")

    if merged_crs_wkt:
        crs_registry["points"] = merged_crs_wkt
        logger.info("✅ Updated crs_registry['points'] with merged CRS.")

    logger.info(f"✅ Merge complete! Output: {merged_output_path}")
    return True

def merge_laz_files(config_path):
    """
    Merges both 'pass' and 'fail' LAZ files from the TEMP directory into their respective outputs.
    
    Assumes:
      - Base output directory is defined in config.get("pathname")
      - TEMP directory is at os.path.join(base_path, "TEMP")
      - Merged pass file is defined in config.get("files.output_pass")
      - Merged fail file is defined in config.get("files.output_fail")
    """
    config = JSONRegistry(config_path, config_path)
    base_path = config.get("pathname")
    if not base_path:
        logger.error("❌ 'pathname' is not defined in config.")
        raise ValueError("Missing 'pathname' in config.")

    temp_dir = os.path.join(base_path, "TEMP")
    if not os.path.isdir(temp_dir):
        logger.error(f"TEMP directory does not exist: {temp_dir}")
        raise FileNotFoundError(f"TEMP directory not found: {temp_dir}")

    # List LAZ files by prefix.
    all_files = os.listdir(temp_dir)
    pass_files = [
        os.path.join(temp_dir, f)
        for f in all_files
        if f.lower().startswith("pass_") and f.lower().endswith((".las", ".laz"))
    ]
    fail_files = [
        os.path.join(temp_dir, f)
        for f in all_files
        if f.lower().startswith("fail_") and f.lower().endswith((".las", ".laz"))
    ]
    logger.info(f"Found {len(pass_files)} pass files and {len(fail_files)} fail files in TEMP directory.")

    # Merge pass files.
    output_pass = config.get("files.output_pass")
    if not output_pass:
        logger.error("❌ 'files.output_pass' is not defined in config.")
        raise ValueError("Missing 'files.output_pass' in config.")
    merge_single_group(pass_files, output_pass)

    # Merge fail files if an output path is provided.
    output_fail = config.get("files.output_fail")
    if output_fail:
        merge_single_group(fail_files, output_fail)
    else:
        logger.warning("No output path specified for failed points (files.output_fail); skipping merge for failed points.")
