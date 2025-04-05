import os
import re
import laspy
import shutil
import struct
import subprocess
import json
from pathlib import Path
import psutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from laspy import LasHeader, PointFormat, PackedPointRecord
from modules.json_registry import JSONRegistry
from modules.merge_laz_files import merge_laz_files, merge_single_group
from laspy import ScaleAwarePointRecord


def extract_column_name(grid_name):
    match = re.match(r"([A-Z]+)[0-9]*$", grid_name)
    return match.group(1) if match else None

def natural_sort(files):
    def sort_key(name):
        match = re.match(r"([A-Z]+)([0-9]+)", name)
        col, row = match.groups() if match else ("", "0")
        return (col, int(row))
    return sorted(files, key=sort_key)

import logging

logger = logging.getLogger(__name__)

MEMORY_THRESHOLD = 0.5

def get_memory_usage():
    return psutil.virtual_memory().percent / 100.0

def pdal_sort_xyz(input_las_path, output_las_path, point_format_id):
    pipeline = {
        "pipeline": [
            input_las_path,
            {"type": "filters.sort", "dimension": "Z"},
            {"type": "filters.sort", "dimension": "Y"},
            {"type": "filters.sort", "dimension": "X"},
            {
                "type": "writers.las",
                "filename": output_las_path,
                "minor_version": 4,
                "extra_dims": "all",
                "dataformat_id": point_format_id
            }
        ]
    }
    try:
        subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=json.dumps(pipeline).encode(),
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ PDAL sort failed for {input_las_path}: {e.stderr.decode().strip()}")
        return False

def merge_column_las_files(col_paths, merged_path):
    """
    Merge a list of LAS files (all from the same column) into a single output LAS file.
    Skips empty or missing files and preserves the point format.
    """
    import laspy
    from tqdm import tqdm
    import os

    # Find the first non-empty, valid LAS file to use as the header template
    for first_path in col_paths:
        if not os.path.exists(first_path) or os.path.getsize(first_path) == 0:
            continue
        with laspy.open(first_path) as first_file:
            if first_file.header.point_count == 0:
                continue
            header = first_file.header
            break
    else:
        logger.error(f"❌ No valid LAS files to merge for {merged_path}")
        return

    logger.info(f"📦 Using header from {first_path} for merge → {merged_path}")

    with laspy.open(merged_path, mode="w", header=header) as writer:
        for path in tqdm(col_paths, desc=f"🔗 Merging column to {os.path.basename(merged_path)}"):
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                logger.warning(f"⚠️ Skipping missing or empty file: {path}")
                continue
            with laspy.open(path) as part:
                if part.header.point_count == 0:
                    logger.warning(f"⚠️ Skipping zero-point LAS file: {path}")
                    continue
                for chunk in part.chunk_iterator(500_000):
                    writer.write_points(chunk)

    logger.info(f"✅ Merged {len(col_paths)} files to {merged_path}")


def convert_bin_columns_to_las(working_dir, las_dir, header):
    logger.info("🚀 Converting binary columns to LAS files...")

    if not os.path.exists(las_dir):
        os.makedirs(las_dir)

    dtype_file = os.path.join(working_dir, "_global_dtype.json")
    if not os.path.exists(dtype_file):
        logger.error("❌ Missing _global_dtype.json. Cannot convert binary tiles.")
        return

    with open(dtype_file, 'r') as f:
        dtype_descr = json.load(f)
        tupled_descr = [tuple(item) for item in dtype_descr]
        dtype = np.dtype(tupled_descr)

    for fname in os.listdir(working_dir):
        if not fname.endswith(".bin"):
            continue

        bin_path = os.path.join(working_dir, fname)
        grid_name = fname.replace(".bin", "")
        las_path = os.path.join(las_dir, f"{grid_name}.las")

        logger.info(f"📄 Converting {fname} → {grid_name}.las")

        try:
            structured = np.fromfile(bin_path, dtype=dtype)

            new_header = laspy.LasHeader()
            new_header.point_format = header.point_format
            new_header.scales = header.scales
            new_header.offsets = header.offsets
            new_header.version = header.version
            new_header.file_source_id = header.file_source_id
            new_header.global_encoding = header.global_encoding
            new_header.system_identifier = header.system_identifier
            new_header.generating_software = header.generating_software
            new_header.creation_date = header.creation_date

            las = laspy.LasData(header=new_header)

            # ✅ Set points
            las.points = laspy.ScaleAwarePointRecord(
                structured,
                new_header.point_format,
                new_header.scales,
                new_header.offsets
            )

            # ✅ Save LAS
            with laspy.open(las_path, mode="w", header=new_header) as writer:
                writer.write_points(las.points)

            logger.info(f"✅ Saved {las_path}")


        except Exception as e:
            logger.error(f"❌ Failed to convert {fname}: {e}")

    logger.info("✅ All binary tiles converted.")


def process_column(col, files, grid_dir, output_dir, point_format_id):
    input_las_path = os.path.join(grid_dir, f"{col}.las")
    sorted_output = os.path.join(output_dir, f"sorted_{col}.las")
    if pdal_sort_xyz(input_las_path, sorted_output, point_format_id):
        logger.info(f"✅ Sorted column {col} -> {sorted_output}")
        return sorted_output
    else:
        logger.error(f"❌ Failed to sort column {col}")
        return None

def sort_columns(grid_dir, output_dir, point_format_id, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    files_by_column = defaultdict(list)
    for fname in os.listdir(grid_dir):
        if not fname.endswith(".las") or fname.startswith("sorted_"):
            continue
        col = extract_column_name(fname.split('.')[0])
        if not col:
            continue
        files_by_column[col].append(fname)

    logger.info(f"🛯️ Starting PDAL sort of {len(files_by_column)} columns...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_column, col, files, grid_dir, output_dir, point_format_id): col
            for col, files in files_by_column.items()
        }
        for future in as_completed(futures):
            future.result()

def tile_las_file(input_file, grid_size, working_dir, chunk_size):
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    with laspy.open(input_file) as reader:
        header = reader.header
        X_min = header.mins[0]
        Y_min = header.mins[1]
        scale = header.scales
        offset = header.offsets

        total_points = header.point_count
        pbar = tqdm(total=total_points, desc="🗾 Tiling points")

        dtype_saved = False

        for points in reader.chunk_iterator(chunk_size):
            # Save dtype from first chunk
            if not dtype_saved:
                dtype_descr = points.array.dtype.descr
                with open(os.path.join(working_dir, "_global_dtype.json"), "w") as f:
                    json.dump(dtype_descr, f)
                dtype_saved = True

            # Real-world coordinates
            X = points.X * scale[0] + offset[0]
            Y = points.Y * scale[1] + offset[1]

            bin_buffers = defaultdict(list)

            for i in range(len(points)):
                col = int((X[i] - X_min) // grid_size)
                row = int((Y[i] - Y_min) // grid_size)
                col = max(0, min(col, 25))  # Clamp to A-Z
                grid_name = f"{chr(65 + col)}{row}"
                bin_buffers[grid_name].append(points.array[i].tobytes())

            for grid_name, point_data in bin_buffers.items():
                bin_path = os.path.join(working_dir, f"{grid_name}.bin")
                with open(bin_path, "ab") as f:
                    f.write(b"".join(point_data))

            pbar.update(len(points))
        pbar.close()

    return os.listdir(working_dir)


def inject_crs_with_pdal(input_path, output_path, epsg_code):
    logger.info(f"🌐 Injecting CRS using EPSG:{epsg_code}")
    pipeline = {
        "pipeline": [
            input_path,
            {
                "type": "filters.reprojection",
                "in_srs": f"EPSG:{epsg_code}",
                "out_srs": f"EPSG:{epsg_code}"
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "a_srs": f"EPSG:{epsg_code}",
                "extra_dims": "all",  # preserve color, etc.
                "minor_version": 4
                # Don't include dataformat_id!
            }
        ]
    }
    try:
        subprocess.run(
            ["pdal", "pipeline", "--stdin"],
            input=json.dumps(pipeline).encode(),
            capture_output=True,
            check=True
        )
        logger.info(f"✅ Reprojected final LAS with EPSG:{epsg_code} → {output_path}")

        pdal_info = subprocess.run(
            ["pdal", "info", "--metadata", output_path],
            capture_output=True,
            check=True
        )
        metadata = json.loads(pdal_info.stdout)
        wkt = metadata.get("metadata", {}).get("srs", {}).get("wkt", "")
        if wkt:
            logger.info(f"📡 Final file WKT starts with: {wkt[:80]}...")
        else:
            logger.warning("⚠️ No WKT found in final file metadata.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ PDAL reprojection failed: {e.stderr.decode().strip()}")
        return False


def tiling(config):
    base_path = config.get("pathname")
    input_file = config.get("files.output_pass")
    working_dir = os.path.join(base_path, "SORTED")
    grid_size = config.get("processing.grid_size", 250)
    chunk_size = config.get("processing.chunk_size", 500_000)

    logger.info("🚀 Starting tiling and column sort process...")
    grid_cells = tile_las_file(input_file, grid_size, working_dir, chunk_size)
    logger.info(f"🔖 Tiled into {len(grid_cells)} grid cells")

    with laspy.open(input_file) as reader:
        template_header = reader.header

    las_dir = os.path.join(base_path, "COL_LAS")
    logger.info("🚀 Converting binary columns to LAS files...")
    convert_bin_columns_to_las(working_dir, las_dir, template_header)
    logger.info("✅ Finished binary-to-LAS conversion.")

    # 🔗 Merge grid tiles into column LAS files (e.g., A0.las + A1.las → A.las)
    files_by_column = defaultdict(list)
    for fname in os.listdir(las_dir):
        if fname.endswith(".las") and not fname.startswith("sorted_"):
            col = extract_column_name(fname.split('.')[0])
            if col:
                files_by_column[col].append(os.path.join(las_dir, fname))

    for col, paths in files_by_column.items():
        merged_path = os.path.join(las_dir, f"{col}.las")
        merge_column_las_files(paths, merged_path)


    sort_columns(las_dir, las_dir, point_format_id=template_header.point_format.id, max_workers=4)

    # ✅ FIX: Accept both .las and .laz for sorted files
    sorted_files = [
        os.path.join(las_dir, f) for f in os.listdir(las_dir)
        if f.startswith("sorted_") and (f.endswith(".las") or f.endswith(".laz"))
    ]

    if not sorted_files:
        logger.error("❌ No valid sorted LAS files found for merging. Aborting tiling.")
        return False

    sorted_files = sorted(
        sorted_files,
        key=lambda x: extract_column_name(os.path.basename(x).replace("sorted_", "").replace(".las", "").replace(".laz", ""))
    )

    base_name = Path(input_file).stem.replace(".las", "").replace(".laz", "")
    temp_merged_path = os.path.join(base_path, f"{base_name}_sorted_temp.laz")
    final_output_path = os.path.join(base_path, f"{base_name}_sorted.laz")

    merge_single_group(sorted_files, temp_merged_path)

    epsg = config.get("crs", {}).get("epsg", 0)
    if epsg and os.path.exists(temp_merged_path):
        logger.info(f"🌐 Injecting CRS using EPSG:{epsg}")
        inject_crs_with_pdal(temp_merged_path, final_output_path, epsg)
        # try:
        #     os.remove(temp_merged_path)
        # except FileNotFoundError:
        #     pass
    else:
        logger.warning(f"⚠️ Skipping CRS injection: EPSG={epsg}, File exists? {os.path.exists(temp_merged_path)}")

    return True
