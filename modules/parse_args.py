import argparse

def parse_args():
    """
    Parses command-line arguments for the PLY processing script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter a huge PLY point cloud, integrate GNSS, and export LAZ files."
    )
    parser.add_argument("--ply", required=True, help="Input PLY file (binary).")
    parser.add_argument("--traj", required=True, help="Dense sensor trajectory file (header, columns: time, traj_x, traj_y, traj_z, ...).")
    parser.add_argument("--gpstraj", help="Sparse GNSS trajectory file (header, columns: time, gps_x, gps_y, gps_z, ...).")
    parser.add_argument("--epsg", help="Target EPSG code (e.g., 26912 for UTM Zone 12N).")
    parser.add_argument("--out_pass", required=True, help="Output LAZ file for accepted points.")
    parser.add_argument("--out_fail", required=True, help="Output LAZ file for rejected points.")
    parser.add_argument("--short", type=float, required=True, help="Minimum distance filter (meters).")
    parser.add_argument("--long", type=float, required=True, help="Maximum distance filter (meters).")
    parser.add_argument("--zpass", type=float, required=True, help="Z-height filter (relative to sensor).")
    parser.add_argument("--mintraj", type=float, required=True, help="Minimum sensor movement (meters).")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="Number of points per processing chunk.")
    parser.add_argument("--test", action="store_true", help="Enable test mode, limit processing to TEST_LIMIT points.")
    parser.add_argument("--no-quiet", action="store_true", help="Enable logging to console.")
    return parser.parse_args()
    