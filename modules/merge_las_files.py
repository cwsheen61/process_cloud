import laspy
import os
import logging
from laspy.vlrs.known import WktCoordinateSystemVlr

logger = logging.getLogger(__name__)

def merge_laz_files(las_files, merged_laz_path, chunk_size=100_000):
    """
    Merges multiple LAS files into a single compressed LAZ file using a streaming approach.

    Args:
        las_files (list): List of LAS file paths to merge.
        merged_laz_path (str): Output path for the merged LAZ file.
        chunk_size (int): Number of points to read and write at a time.
    """

    if not las_files:
        logger.error("❌ No valid LAS files provided for merging.")
        return

    logger.info(f"📂 Merging {len(las_files)} LAS files into {merged_laz_path} (streaming mode)...")

    # Open the first file to use as reference for the header
    with laspy.open(las_files[0]) as first_file:
        header = first_file.header

        # Create new LAS header for output
        merged_header = laspy.LasHeader(point_format=header.point_format, version=header.version)

        # Preserve CRS if available
        for vlr in header.vlrs:
            if isinstance(vlr, WktCoordinateSystemVlr):
                merged_header.vlrs.append(vlr)

    # ✅ Create the output LAZ file in write mode
    with laspy.open(merged_laz_path, mode="w", header=merged_header) as merged_laz:
        total_points = 0

        # ✅ Stream points from input LAS files
        for las_file in las_files:
            logger.info(f"📖 Processing {las_file}...")

            with laspy.open(las_file) as las_reader:
                for points in las_reader.chunk_iterator(chunk_size):  # ✅ Stream in chunks
                    merged_laz.write_points(points)
                    total_points += points.size

                    # Optional debug logging for progress
                    if total_points % (10 * chunk_size) < chunk_size:
                        logger.info(f"✅ Merged {total_points:,} points so far...")

    logger.info(f"✅ Merging complete: {total_points:,} total points written to {merged_laz_path}")

