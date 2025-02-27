import laspy
import os
import logging
import numpy as np
from laspy.vlrs.known import WktCoordinateSystemVlr
from tqdm import tqdm  # ✅ For progress tracking
from modules.crs_registry import crs_registry  # ✅ Ensure CRS consistency

logger = logging.getLogger(__name__)

def merge_laz_files(laz_files, merged_output_path):
    """
    Merges multiple LAZ files into a single output file while preserving correct UTM alignment.

    Args:
        laz_files (list): List of LAZ file paths to merge.
        merged_output_path (str): Output path for merged points.
    """

    if not laz_files:
        logger.error(f"No valid LAZ files provided for merging into {merged_output_path}.")
        return

    logger.info(f"🔄 Starting merge of {len(laz_files)} LAZ files into {merged_output_path}...")

    # 🔹 Compute Global Min Offsets for UTM preservation
    global_min_x, global_min_y, global_min_z = np.inf, np.inf, np.inf
    merged_crs_wkt = None

    for laz_file in laz_files:
        with laspy.open(laz_file) as las:
            if las.header.point_count == 0:
                tqdm.write(f"⚠️ Skipping empty file: {laz_file}")  # ✅ Avoid log interference
                continue

            global_min_x = min(global_min_x, las.header.offsets[0])
            global_min_y = min(global_min_y, las.header.offsets[1])
            global_min_z = min(global_min_z, las.header.offsets[2])

            # ✅ Capture CRS from first valid file
            if merged_crs_wkt is None:
                for vlr in las.header.vlrs:
                    if isinstance(vlr, WktCoordinateSystemVlr):
                        merged_crs_wkt = vlr.string
                        break

    logger.info(f"📌 Global UTM Offsets: X={global_min_x}, Y={global_min_y}, Z={global_min_z}")

    # 🔹 Use the first valid file to create merged header
    with laspy.open(laz_files[0]) as first_file:
        header = first_file.header

        # ✅ Keep original UTM offsets
        merged_header = laspy.LasHeader(point_format=header.point_format, version=header.version)
        merged_header.offsets = np.array([global_min_x, global_min_y, global_min_z])
        merged_header.scales = header.scales  # Use the same scale as the input files

        # ✅ Ensure CRS tracking in `crs_registry`
        if merged_crs_wkt:
            merged_header.vlrs.append(WktCoordinateSystemVlr(merged_crs_wkt))
            logger.info(f"📌 Using CRS from first valid file.")
        else:
            # ✅ If no CRS found, fallback to `crs_registry["points"]`
            crs_epsg = crs_registry.get("points", -1)
            if crs_epsg > 0:
                logger.warning(f"⚠️ No CRS found in input files. Using EPSG:{crs_epsg} from crs_registry.")
                merged_header.vlrs.append(WktCoordinateSystemVlr(f"EPSG:{crs_epsg}"))

    # 🔹 Write merged points with Correct UTM Offsets
    with laspy.open(merged_output_path, mode="w", header=merged_header) as merged_file:
        total_written = 0  # ✅ Track number of points written

        # ✅ Smooth progress bar without log interference
        for laz_file in tqdm(laz_files, desc="📂 Merging Progress", unit="file", dynamic_ncols=True):
            with laspy.open(laz_file) as las:
                points = las.read().points
                if len(points) == 0:
                    tqdm.write(f"⚠️ Skipping empty file: {laz_file}")  # ✅ Avoid log interference
                    continue

                # ✅ Convert points to UTM coordinates before writing
                adjusted_x = (points.X * las.header.scales[0]) + las.header.offsets[0]
                adjusted_y = (points.Y * las.header.scales[1]) + las.header.offsets[1]
                adjusted_z = (points.Z * las.header.scales[2]) + las.header.offsets[2]

                # ✅ Convert back to scaled int32 using the correct merged offsets
                points.X = np.round((adjusted_x - global_min_x) / merged_header.scales[0]).astype(np.int32)
                points.Y = np.round((adjusted_y - global_min_y) / merged_header.scales[1]).astype(np.int32)
                points.Z = np.round((adjusted_z - global_min_z) / merged_header.scales[2]).astype(np.int32)

                merged_file.write_points(points)
                total_written += len(points)

        tqdm.write(f"✅ Total points written: {total_written} to {merged_output_path}")  # ✅ Final update

    # 🔹 Final Debugging Checks (Log AFTER tqdm to avoid interruptions)
    with laspy.open(merged_output_path) as las_check:
        logger.info(f"📌 Final LAS Offsets: X={las_check.header.offsets[0]}, Y={las_check.header.offsets[1]}, Z={las_check.header.offsets[2]}")
        logger.info(f"📏 Final LAS Scales: {las_check.header.scales}")
        logger.info(f"📊 Final LAS Point Count: {las_check.header.point_count}")

    # ✅ Ensure `crs_registry` is updated with merged file CRS
    if merged_crs_wkt:
        crs_registry["points"] = merged_crs_wkt
        logger.info(f"✅ Updated crs_registry['points'] with merged CRS.")

    logger.info(f"✅ Merge complete! Output: {merged_output_path}")











# import laspy
# import os
# import logging
# import numpy as np
# from laspy.vlrs.known import WktCoordinateSystemVlr
# from tqdm import tqdm  # ✅ For progress tracking

# logger = logging.getLogger(__name__)

# def merge_laz_files(laz_files, merged_output_path):
#     """
#     Merges multiple LAZ files into a single output file while preserving correct UTM alignment.

#     Args:
#         laz_files (list): List of LAZ file paths to merge.
#         merged_output_path (str): Output path for merged points.
#     """

#     if not laz_files:
#         logger.error(f"No valid LAZ files provided for merging into {merged_output_path}.")
#         return

#     logger.info(f"🔄 Starting merge of {len(laz_files)} LAZ files into {merged_output_path}...")

#     # 🔹 Compute Global Min Offsets for UTM preservation
#     global_min_x, global_min_y, global_min_z = np.inf, np.inf, np.inf

#     for laz_file in laz_files:
#         with laspy.open(laz_file) as las:
#             if las.header.point_count == 0:
#                 tqdm.write(f"⚠️ Skipping empty file: {laz_file}")  # ✅ Avoid log interference
#                 continue

#             global_min_x = min(global_min_x, las.header.offsets[0])
#             global_min_y = min(global_min_y, las.header.offsets[1])
#             global_min_z = min(global_min_z, las.header.offsets[2])

#     logger.info(f"📌 Global UTM Offsets: X={global_min_x}, Y={global_min_y}, Z={global_min_z}")

#     # 🔹 Create a merged header using the first file as a reference
#     with laspy.open(laz_files[0]) as first_file:
#         header = first_file.header

#         # ✅ Keep original UTM offsets
#         merged_header = laspy.LasHeader(point_format=header.point_format, version=header.version)
#         merged_header.offsets = np.array([global_min_x, global_min_y, global_min_z])
#         merged_header.scales = header.scales  # Use the same scale as the input files

#         # Preserve CRS if available
#         for vlr in header.vlrs:
#             if isinstance(vlr, WktCoordinateSystemVlr):
#                 merged_header.vlrs.append(vlr)

#     # 🔹 Write merged points with Correct UTM Offsets
#     with laspy.open(merged_output_path, mode="w", header=merged_header) as merged_file:
#         total_written = 0  # ✅ Track number of points written

#         # ✅ Smooth progress bar without log interference
#         for laz_file in tqdm(laz_files, desc="📂 Merging Progress", unit="file", dynamic_ncols=True):
#             with laspy.open(laz_file) as las:
#                 points = las.read().points
#                 if len(points) == 0:
#                     tqdm.write(f"⚠️ Skipping empty file: {laz_file}")  # ✅ Avoid log interference
#                     continue

#                 # ✅ Convert points to UTM coordinates before writing
#                 adjusted_x = (points.X * las.header.scales[0]) + las.header.offsets[0]
#                 adjusted_y = (points.Y * las.header.scales[1]) + las.header.offsets[1]
#                 adjusted_z = (points.Z * las.header.scales[2]) + las.header.offsets[2]

#                 # ✅ Convert back to scaled int32 using the correct merged offsets
#                 points.X = np.round((adjusted_x - global_min_x) / merged_header.scales[0]).astype(np.int32)
#                 points.Y = np.round((adjusted_y - global_min_y) / merged_header.scales[1]).astype(np.int32)
#                 points.Z = np.round((adjusted_z - global_min_z) / merged_header.scales[2]).astype(np.int32)

#                 merged_file.write_points(points)
#                 total_written += len(points)

#         tqdm.write(f"✅ Total points written: {total_written} to {merged_output_path}")  # ✅ Final update

#     # 🔹 Final Debugging Checks (Log AFTER tqdm to avoid interruptions)
#     with laspy.open(merged_output_path) as las_check:
#         logger.info(f"📌 Final LAS Offsets: X={las_check.header.offsets[0]}, Y={las_check.header.offsets[1]}, Z={las_check.header.offsets[2]}")
#         logger.info(f"📏 Final LAS Scales: {las_check.header.scales}")
#         logger.info(f"📊 Final LAS Point Count: {las_check.header.point_count}")

#     logger.info(f"✅ Merge complete! Output: {merged_output_path}")
