import os
import logging
import laspy
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from filters.apply_knn_filter import apply_knn_filter
from filters.apply_kd_tree_filter import apply_kd_tree_filter
from filters.apply_voxel_grid_filter import apply_voxel_grid_filter


def update_sorted_las(config, suffix):
    """
    Update the 'sorted_las' path in config by appending a suffix to the filename.

    Args:
        config (JSONRegistry or dict): The config object.
        suffix (str): Suffix to insert before file extension (e.g., '_knn_pass').

    Returns:
        str: New full path to updated sorted LAS file.
    """
    if hasattr(config, "get") and hasattr(config, "set"):
        current_path = config.get("files.sorted_las")
        new_path = current_path.replace(".laz", f"{suffix}.laz")
        config.set("files.sorted_las", new_path)
        return new_path
    else:
        current_path = config["files"]["sorted_las"]
        new_path = current_path.replace(".laz", f"{suffix}.laz")
        config["files"]["sorted_las"] = new_path
        return new_path


def sorted_knn_filter(config):
    try:
        if hasattr(config, "get"):
            knn_enabled = config.get("filtering.knn_filter.enabled", False)
            input_path = config.get("files.sorted_las")
            chunk_size = config.get("filtering.knn_filter.chunk_size", 5_000_000)
            num_workers = config.get("filtering.knn_filter.num_workers", 4)
        else:
            knn_enabled = config["filtering"]["knn_filter"]["enabled"]
            input_path = config["files"]["sorted_las"]
            chunk_size = config["filtering"]["knn_filter"].get("chunk_size", 5_000_000)
            num_workers = config["filtering"]["knn_filter"].get("num_workers", 4)

        if not knn_enabled:
            logger.info("ℹ️ KNN filter is disabled; skipping.")
            return

        out_pass = update_sorted_las(config, "_knn_pass")
        out_fail = out_pass.replace("_pass.laz", "_fail.laz")

        with laspy.open(input_path) as reader:
            header = reader.header
            total = header.point_count

            with laspy.open(out_pass, mode="w", header=header) as writer_pass, \
                 laspy.open(out_fail, mode="w", header=header) as writer_fail:

                for chunk in tqdm(reader.chunk_iterator(chunk_size), total=total // chunk_size + 1, desc="🔍 KNN Filter"):
                    try:
                        # logger.info(f"📦 Chunk dtype: {chunk.array.dtype}")
                        points = chunk.array
                        mask = apply_knn_filter(points, config.get("filtering.knn_filter", {}), header)
                        writer_pass.write_points(chunk[mask])
                        writer_fail.write_points(chunk[~mask])
                    except Exception as e:
                        logger.error(f"❌ KNN filtering failed on chunk: {e}")

        logger.info(f"✅ KNN filtering complete. Output: {out_pass}")
        return out_pass

    except Exception as e:
        logger.exception(f"🔥 Exception during KNN filtering: {e}")
        raise




def sorted_kd_filter(config):
    try:
        if hasattr(config, "get"):
            kd_enabled = config.get("filtering.kd_tree_filter.enabled", False)
            input_path = config.get("files.sorted_las")
            chunk_size = config.get("filtering.kd_tree_filter.chunk_size", 500_000)
            num_workers = config.get("filtering.kd_tree_filter.num_workers", 4)
            params = config.get("filtering.kd_tree_filter", {})
        else:
            kd_enabled = config["filtering"]["kd_tree_filter"]["enabled"]
            input_path = config["files"]["sorted_las"]
            chunk_size = config["filtering"]["kd_tree_filter"].get("chunk_size", 500_000)
            num_workers = config["filtering"]["kd_tree_filter"].get("num_workers", 4)
            params = config["filtering"]["kd_tree_filter"]

        if not kd_enabled:
            logger.info("ℹ️ KD-tree filter is disabled; skipping.")
            return

        out_pass = update_sorted_las(config, "_kd_pass")
        out_fail = out_pass.replace("_pass.laz", "_fail.laz")

        with laspy.open(input_path) as reader:
            header = reader.header
            total = header.point_count

            with laspy.open(out_pass, mode="w", header=header) as writer_pass, \
                 laspy.open(out_fail, mode="w", header=header) as writer_fail:

                for chunk in tqdm(reader.chunk_iterator(chunk_size), total=total // chunk_size + 1, desc="🌲 KD Filter"):
                    try:
                        points = chunk.array
                        mask = apply_kd_tree_filter(points, params, header=header)

                        writer_pass.write_points(chunk[mask])
                        writer_fail.write_points(chunk[~mask])
                    except Exception as e:
                        logger.error(f"❌ KD-tree filtering failed on chunk: {e}")

        logger.info(f"✅ KD-tree filtering complete. Output: {out_pass}")
        return out_pass

    except Exception as e:
        logger.exception(f"🔥 Exception during KD filtering: {e}")


def write_voxelized_las(voxel_array, output_path, config):
    import laspy
    import numpy as np
    import logging
    from laspy import ExtraBytesParams

    logger = logging.getLogger(__name__)

    logger.info(f"📝 Writing LAS with {len(voxel_array)} points to {output_path}")

    offsets = config.get("transformation.t")

    # Set up LAS header
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = offsets


    logger.info("📐 Set header scales and offsets")
    logger.info(f"{header.offsets} {header.scales}")

    # Add Extra Dimensions (brute-force)
    try:
        if "NormalX" in voxel_array.dtype.names:
            try:
                header.add_extra_dim(ExtraBytesParams(name="NormalX", type=np.float32, description="Normal X"))
                logger.info("➕ Added VLR for NormalX")
            except:
                logger.error(f"❌ Failed while adding extra dims: {e}")
        if "NormalY" in voxel_array.dtype.names:
            try:
                header.add_extra_dim(ExtraBytesParams(name="NormalY", type=np.float32, description="Normal Y"))
                logger.info("➕ Added VLR for NormalY")
            except:
                logger.error(f"❌ Failed while adding extra dims: {e}")
        if "NormalZ" in voxel_array.dtype.names:
            try:
                header.add_extra_dim(ExtraBytesParams(name="NormalZ", type=np.float32, description="Normal Z"))
                logger.info("➕ Added VLR for NormalZ")
            except:
                logger.error(f"❌ Failed while adding extra dims: {e}")
        if "range" in voxel_array.dtype.names:
            header.add_extra_dim(ExtraBytesParams(name="range", type=np.float32, description="Sensor range"))
            logger.info("➕ Added VLR for range")

        if "Density" in voxel_array.dtype.names:
            header.add_extra_dim(ExtraBytesParams(name="Density", type=np.float32, description="Voxel density"))
            logger.info("➕ Added VLR for Density")
    except Exception as e:
        logger.error(f"❌ Failed while adding extra dims: {e}")
        raise

    # Create LAS object with this header
    las = laspy.LasData(header)

    # Assign required XYZ fields
    # las.X = np.round((voxel_array["X"] - header.offsets[0]) / header.scales[0]).astype(np.int32)
    # las.Y = np.round((voxel_array["Y"] - header.offsets[1]) / header.scales[1]).astype(np.int32)
    # las.Z = np.round((voxel_array["Z"] - header.offsets[2]) / header.scales[2]).astype(np.int32)
    # logger.info("✅ Assigned XYZ fields")

    las.X = voxel_array["X"]
    las.Y = voxel_array["Y"] 
    las.Z = voxel_array["Z"] 

    # Optional fields
    for field in ["intensity", "gps_time", "return_number", "point_source_id"]:
        if field in voxel_array.dtype.names:
            try:
                las[field] = voxel_array[field]
                logger.info(f"✅ Assigned field: {field}")
            except Exception as e:
                logger.warning(f"⚠️ Could not assign {field}: {e}")

    # RGB
    if all(c in voxel_array.dtype.names for c in ["red", "green", "blue"]):
        las.red = voxel_array["red"]
        las.green = voxel_array["green"]
        las.blue = voxel_array["blue"]
        logger.info("🌈 Assigned RGB fields")

    # Extra dimensions (again, brute-force)
    for name in ["NormalX", "NormalY", "NormalZ", "range", "Density"]:
        if name in voxel_array.dtype.names:
            try:
                las[name] = voxel_array[name]
                logger.info(f"✅ Set extra dimension: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to assign extra dimension {name}: {e}")

    # Save to disk
    try:
        with laspy.open(output_path, mode="w", header=header) as writer:
            writer.write_points(las.points)
        logger.info(f"💾 Successfully wrote LAS file: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to write LAS file: {e}")
        raise

      
import os
import laspy
import numpy as np
import logging
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from filters.apply_voxel_grid_filter import apply_voxel_grid_filter
from filters.post_sort_filters import write_voxelized_las  # <- your latest working version

logger = logging.getLogger(__name__)

def process_chunk(chunk_idx, chunk_array, config, header, output_dir):
    try:
        voxelized = apply_voxel_grid_filter(chunk_array, config, header)
        if voxelized.shape[0] == 0:
            return f"⚠️ Chunk {chunk_idx}: No points after voxel filtering."

        chunk_path = os.path.join(output_dir, f"voxel_chunk_{chunk_idx:04d}.las")
        write_voxelized_las(voxelized, chunk_path, config)
        return f"✅ Chunk {chunk_idx} written: {voxelized.shape[0]} points"

    except Exception as e:
        return f"❌ Chunk {chunk_idx} failed: {e}"

def sorted_voxel_grid_filter(config):
    if hasattr(config, "get"):
        voxel_enabled = config.get("filtering.voxel_filter.enabled", False)
        input_path = config.get("files.sorted_las")
        chunk_size = config.get("filtering.voxel_filter.chunk_size", 5_000_000)
        output_dir = config.get("files.output_dir", None)
        num_workers = config.get("filtering.voxel_filter.num_workers", 4)
        config_dict = config.get("filtering.voxel_filter", {})
    else:
        voxel_enabled = config["filtering"]["voxel_filter"]["enabled"]
        input_path = config["files"]["sorted_las"]
        chunk_size = config["filtering"]["voxel_filter"]["chunk_size"]
        output_dir = config["files"].get("output_dir", None)
        num_workers = config["filtering"]["voxel_filter"].get("num_workers", 4)
        config_dict = config["filtering"]["voxel_filter"]

    if not voxel_enabled:
        logger.info("ℹ️ Voxel grid filter is disabled; skipping.")
        return

    logger.info("🧪 Running post-sort Voxel Grid filter...")

    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    temp_dir = os.path.join(output_dir, "TEMP_VOXELS")
    os.makedirs(temp_dir, exist_ok=True)

    with laspy.open(input_path) as reader:
        header = reader.header
        chunks = list(reader.chunk_iterator(chunk_size))

    logger.info(f"📦 Dispatching {len(chunks)} chunks to {num_workers} workers...")

    # Package up work for each process
    task_args = [(idx, chunk.array, config, header, temp_dir)
                 for idx, chunk in enumerate(chunks)]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.starmap(process_chunk, task_args), total=len(task_args)))

    for msg in results:
        logger.info(msg)

    logger.info(f"✅ Voxel grid filtering complete. Chunks saved to: {temp_dir}")
    return temp_dir

def inject_crs_with_pdal(input_path, output_path, epsg_code):
    import json, subprocess, logging
    logger = logging.getLogger(__name__)
    logger.info(f"🌐 Injecting CRS using EPSG:{epsg_code}")

    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_path,
                # explicitly list only the dims you need
                "extra_dims": [
                    "NormalX=float32",
                    "NormalY=float32",
                    "NormalZ=float32",
                    "range=float32",
                    "Density=float32"
                ]
            },
            {
                "type": "filters.reprojection",
                "in_srs": f"EPSG:{epsg_code}",
                "out_srs": f"EPSG:{epsg_code}"
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "a_srs": f"EPSG:{epsg_code}",
                "compression": "laszip",
                "minor_version": 4,
                "dataformat_id": 7,
                "global_encoding": 17,
                # still okay to dump all non‑standard dims here
                "extra_dims": "all"
            }
        ]
    }

    proc = subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=json.dumps(pipeline).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        logger.error(proc.stderr.decode())
        raise RuntimeError("PDAL pipeline failed")
    logger.info(proc.stdout.decode())


def merge_voxel_chunks(config):
    import os
    import laspy
    import numpy as np
    import subprocess
    from laspy import ScaleAwarePointRecord

    sorted_las  = config.get("files.sorted_las")
    output_path = config.get("files.voxel_las")
    epsg        = config.get("crs.epsg")

    voxel_dir = os.path.join(os.path.dirname(sorted_las), "TEMP_VOXELS")
    chunk_files = sorted([
        os.path.join(voxel_dir, f)
        for f in os.listdir(voxel_dir)
        if f.lower().endswith(".las")
    ])

    if not chunk_files:
        raise RuntimeError(f"No LAS chunks found in {voxel_dir}")

    # Read first LAS to get header and array
    first = laspy.read(chunk_files[0])
    header = first.header
    scale = header.scales
    offset = header.offsets
    all_arrays = [first.points.array]

    for fp in chunk_files[1:]:
        chunk = laspy.read(fp)
        all_arrays.append(chunk.points.array)

    merged_array = np.concatenate(all_arrays)

    # Convert merged array into a ScaleAwarePointRecord
    point_record = ScaleAwarePointRecord(merged_array, header.point_format, scales=scale, offsets=offset)

    # Create final LAS object and assign point record
    las = laspy.LasData(header)
    las.points = point_record
    las.write(output_path)

    # Inject CRS using PDAL
    
    output_final_path = output_path.replace(".las","_final.laz")

    inject_crs_with_pdal(output_path, output_final_path, epsg)


