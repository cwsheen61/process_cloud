import os
import re
import laspy
import shutil
import struct
import subprocess
import json
import logging
import psutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from laspy import LasHeader, PointFormat, PackedPointRecord
from modules.json_registry import JSONRegistry
from modules.merge_laz_files import merge_laz_files, merge_single_group
from utils import natural_sort, extract_column_name, get_logger

logger = get_logger(__name__)

MEMORY_THRESHOLD = 0.5

def get_memory_usage():
    return psutil.virtual_memory().percent / 100.0

def pdal_sort_xyz(input_las_path, output_las_path):
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
                "extra_dims": "all"
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
    with open(merged_path, 'wb') as merged_file:
        for path in tqdm(col_paths, desc="🔗 Merging sorted columns"):
            with open(path, 'rb') as f:
                merged_file.write(f.read())

def convert_bin_columns_to_las(bin_dir, las_dir, template_header):
    if os.path.exists(las_dir):
        shutil.rmtree(las_dir)
    os.makedirs(las_dir, exist_ok=True)

    files_by_column = defaultdict(list)
    for fname in os.listdir(bin_dir):
        if fname.endswith(".bin"):
            col = extract_column_name(fname.replace(".bin", ""))
            if col:
                files_by_column[col].append(fname)

    logger.info(f"📂 Found {len(files_by_column)} columns in {bin_dir}")
    for col, files in files_by_column.items():
        files = natural_sort(files)
        raw_data = b""
        for fname in files:
            with open(os.path.join(bin_dir, fname), "rb") as f:
                raw_data += f.read()

        count = len(raw_data) // template_header.point_format.size
        out_path = os.path.join(las_dir, f"{col}.las")

        pf = PointFormat(6)
        header = LasHeader(point_format=pf, version="1.4")
        header.scales = template_header.scales
        header.offsets = template_header.offsets
        header.point_count = count

        header.vlrs.extend(template_header.vlrs)
        if getattr(template_header, "evlrs", None):
            header.evlrs.extend(template_header.evlrs)

        crs = template_header.parse_crs()
        if crs:
            header.assign_crs(crs)

        header.global_encoding._set_bit(0, True)

        with laspy.open(out_path, mode="w", header=header) as writer:
            if raw_data:
                point_record = PackedPointRecord.from_buffer(
                    raw_data,
                    point_format=header.point_format,
                    count=count
                )
                writer.write_points(point_record)

        logger.info(f"📘 Merged column {col}: {len(files)} files → {out_path} with {count} points")

def process_column(col, files, grid_dir, output_dir):
    input_las_path = os.path.join(grid_dir, f"{col}.las")
    sorted_output = os.path.join(output_dir, f"sorted_{col}.las")

    if pdal_sort_xyz(input_las_path, sorted_output):
        logger.info(f"✅ Sorted column {col} -> {sorted_output}")
        return sorted_output
    else:
        logger.error(f"❌ Failed to sort column {col}")
        return None

def sort_columns(grid_dir, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    files_by_column = defaultdict(list)
    for fname in os.listdir(grid_dir):
        if not fname.endswith(".las") or fname.startswith("sorted_"):
            continue
        col = extract_column_name(fname.split('.')[0])
        if not col:
            continue
        files_by_column[col].append(fname)

    logger.info(f"🧯 Starting PDAL sort of {len(files_by_column)} columns...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_column, col, files, grid_dir, output_dir): col
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
        for points in reader.chunk_iterator(chunk_size):
            X = points.X * scale[0] + offset[0]
            Y = points.Y * scale[1] + offset[1]

            bin_buffers = defaultdict(list)
            for i in range(len(points)):
                col = int((X[i] - X_min) // grid_size)
                row = int((Y[i] - Y_min) // grid_size)
                col = max(0, min(col, 25))
                grid_name = f"{chr(65 + col)}{row}"
                bin_buffers[grid_name].append(points.array[i].tobytes())

            for grid_name, point_data in bin_buffers.items():
                path = os.path.join(working_dir, f"{grid_name}.bin")
                with open(path, "ab") as f:
                    f.write(b"".join(point_data))

            pbar.update(len(points))
        pbar.close()

    return os.listdir(working_dir)

def tiling(config):
    base_path = config.get("pathname")
    input_file = config.get("files.output_pass")
    working_dir = os.path.join(base_path, "SORTED")
    grid_size = config.get("processing.grid_size", 250)
    chunk_size = config.get("processing.chunk_size", 500_000)

    logger.info("🚀 Starting tiling and column sort process...")
    grid_cells = tile_las_file(input_file, grid_size, working_dir, chunk_size)
    logger.info(f"🗖 Tiled into {len(grid_cells)} grid cells")

    with laspy.open(input_file) as reader:
        template_header = reader.header

    las_dir = os.path.join(base_path, "COL_LAS")
    convert_bin_columns_to_las(working_dir, las_dir, template_header)
    sort_columns(las_dir, las_dir, max_workers=4)

    sorted_files = [
        os.path.join(las_dir, f) for f in os.listdir(las_dir)
        if f.startswith("sorted_") and f.endswith(".las")
    ]
    sorted_files = sorted(
        sorted_files,
        key=lambda x: extract_column_name(os.path.basename(x).replace("sorted_", "").replace(".las", ""))
    )
    output_path = os.path.join(base_path, "merged_final.las")
    merge_single_group(sorted_files, output_path)

    return True




# import os
# import subprocess
# import json
# import logging
# import psutil
# import numpy as np
# import laspy
# import re
# import shutil
# from tqdm import tqdm
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from laspy import LasHeader, PointFormat, PackedPointRecord
# from modules.json_registry import JSONRegistry
# from modules.merge_laz_files import merge_single_group

# logger = logging.getLogger(__name__)

# MEMORY_THRESHOLD = 0.5  # 50% usage cap

# def get_memory_usage():
#     return psutil.virtual_memory().percent / 100.0

# def pdal_sort_xyz(input_las_path, output_las_path):
#     pipeline = {
#         "pipeline": [
#             input_las_path,
#             {"type": "filters.sort", "dimension": "Z"},
#             {"type": "filters.sort", "dimension": "Y"},
#             {"type": "filters.sort", "dimension": "X"},
#             {
#                 "type": "writers.las",
#                 "filename": output_las_path,
#                 "minor_version": 4,
#                 "extra_dims": "all"
#             }
#         ]
#     }
#     try:
#         subprocess.run(
#             ["pdal", "pipeline", "--stdin"],
#             input=json.dumps(pipeline).encode(),
#             capture_output=True,
#             check=True
#         )
#         return True
#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ PDAL sort failed for {input_las_path}: {e.stderr.decode().strip()}")
#         return False

# def merge_column_las_files(col_paths, merged_path):
#     with open(merged_path, 'wb') as merged_file:
#         for path in tqdm(col_paths, desc="🔗 Merging sorted columns"):
#             with open(path, 'rb') as f:
#                 merged_file.write(f.read())

# def extract_column_name(grid_name):
#     match = re.match(r"([A-Z]+)[0-9]*$", grid_name)
#     return match.group(1) if match else None

# def natural_sort(files):
#     def sort_key(name):
#         match = re.match(r"([A-Z]+)([0-9]+)", name)
#         col, row = match.groups() if match else ("", "0")
#         return (col, int(row))
#     return sorted(files, key=sort_key)

# def process_column(col, files, grid_dir, output_dir):
#     input_las_path = os.path.join(grid_dir, f"{col}.las")
#     sorted_output = os.path.join(output_dir, f"sorted_{col}.las")

#     if pdal_sort_xyz(input_las_path, sorted_output):
#         logger.info(f"✅ Sorted column {col} -> {sorted_output}")
#         return sorted_output
#     else:
#         logger.error(f"❌ Failed to sort column {col}")
#         return None

# def sort_columns(grid_dir, output_dir, max_workers=4):
#     os.makedirs(output_dir, exist_ok=True)
#     files_by_column = defaultdict(list)
#     for fname in os.listdir(grid_dir):
#         if not fname.endswith(".las") or fname.startswith("sorted_"):
#             continue
#         col = extract_column_name(fname.split('.')[0])
#         if not col:
#             continue
#         files_by_column[col].append(fname)

#     logger.info(f"🧯 Starting PDAL sort of {len(files_by_column)} columns...")
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(process_column, col, files, grid_dir, output_dir): col
#             for col, files in files_by_column.items()
#         }
#         for future in as_completed(futures):
#             future.result()

# def tile_las_file(input_file, grid_size, working_dir, chunk_size):
#     os.makedirs(working_dir, exist_ok=True)
#     with laspy.open(input_file) as reader:
#         header = reader.header
#         X_min = header.mins[0]
#         Y_min = header.mins[1]
#         scale = header.scales
#         offset = header.offsets

#         total_points = header.point_count
#         chunks = (total_points + chunk_size - 1) // chunk_size

#         pbar = tqdm(total=total_points, desc="🏾 Tiling points")
#         for points in reader.chunk_iterator(chunk_size):
#             X = points.X * scale[0] + offset[0]
#             Y = points.Y * scale[1] + offset[1]

#             bin_buffers = defaultdict(list)
#             for i in range(len(points)):
#                 col = int((X[i] - X_min) // grid_size)
#                 row = int((Y[i] - Y_min) // grid_size)
#                 col = max(0, min(col, 25))
#                 grid_name = f"{chr(65 + col)}{row}"
#                 bin_buffers[grid_name].append(points.array[i].tobytes())

#             for grid_name, point_data in bin_buffers.items():
#                 path = os.path.join(working_dir, f"{grid_name}.bin")
#                 with open(path, "ab") as f:
#                     f.write(b"".join(point_data))

#             pbar.update(len(points))
#         pbar.close()

#     return os.listdir(working_dir)

# def convert_bin_columns_to_las(bin_dir, las_dir, template_header):
#     if os.path.exists(las_dir):
#         shutil.rmtree(las_dir)
#     os.makedirs(las_dir, exist_ok=True)

#     files_by_column = defaultdict(list)
#     for fname in os.listdir(bin_dir):
#         if fname.endswith(".bin"):
#             col = extract_column_name(fname.replace(".bin", ""))
#             if col:
#                 files_by_column[col].append(fname)

#     logger.info(f"📂 Found {len(files_by_column)} columns in {bin_dir}")
#     for col, files in files_by_column.items():
#         files = natural_sort(files)
#         raw_data = b""
#         for fname in files:
#             with open(os.path.join(bin_dir, fname), "rb") as f:
#                 raw_data += f.read()

#         count = len(raw_data) // template_header.point_format.size
#         out_path = os.path.join(las_dir, f"{col}.las")

#         pf = PointFormat(6)
#         header = LasHeader(point_format=pf, version="1.4")
#         header.scales = template_header.scales
#         header.offsets = template_header.offsets
#         header.point_count = count

#         header.vlrs.extend(template_header.vlrs)
#         if getattr(template_header, "evlrs", None):
#             header.evlrs.extend(template_header.evlrs)
#         crs = template_header.parse_crs()
#         if crs:
#             header.assign_crs(crs)

#         # Set WKT present bit in global encoding for point formats 6–10
#         header.global_encoding = header.global_encoding.set_bit(0, True)

#         with laspy.open(out_path, mode="w", header=header) as writer:
#             if raw_data:
#                 point_record = PackedPointRecord.from_buffer(
#                     raw_data,
#                     point_format=header.point_format,
#                     count=count
#                 )
#                 writer.write_points(point_record)

#         logger.info(f"📘 Merged column {col}: {len(files)} files → {out_path} with {count} points")

# def tiling(config):
#     base_path = config.get("pathname")
#     input_file = config.get("files.output_pass")
#     working_dir = os.path.join(base_path, "SORTED")
#     if os.path.exists(working_dir):
#         for f in os.listdir(working_dir):
#             os.remove(os.path.join(working_dir, f))
#     else:
#         os.makedirs(working_dir)
#     grid_size = config.get("processing.grid_size", 250)
#     chunk_size = config.get("processing.chunk_size", 500_000)

#     logger.info("🚀 Starting tiling and column sort process...")
#     grid_cells = tile_las_file(input_file, grid_size, working_dir, chunk_size)
#     logger.info(f"🗖 Tiled into {len(grid_cells)} grid cells")

#     with laspy.open(input_file) as reader:
#         template_header = reader.header
#     convert_bin_columns_to_las(working_dir, os.path.join(base_path, "COL_LAS"), template_header)
#     sort_columns(os.path.join(base_path, "COL_LAS"), os.path.join(base_path, "COL_LAS"), max_workers=4)

#     sorted_las_dir = os.path.join(base_path, "COL_LAS")
#     sorted_files = [
#         f for f in os.listdir(sorted_las_dir)
#         if f.startswith("sorted_") and f.endswith(".las")
#     ]
#     sorted_files = sorted(
#         sorted_files,
#         key=lambda x: extract_column_name(x.replace("sorted_", "").replace(".las", ""))
#     )
#     sorted_files_full_paths = [os.path.join(sorted_las_dir, f) for f in sorted_files]
#     output_path = os.path.join(base_path, "merged_final.las")
#     merge_single_group(sorted_files_full_paths, output_path)

#     return True
