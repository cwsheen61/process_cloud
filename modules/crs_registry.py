## crs_registry.py
"""
Global CRS (Coordinate Reference System) Registry.
This dictionary keeps track of the CRS status of various datasets.
"""
import pyproj
from pyproj import CRS
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize CRS values explicitly
crs_registry = {
    "trajectory": -1,  # Default: Sensor frame
    "points": -1,      # Default: Sensor frame
    "gnss_raw": 4326,  # GNSS data is always WGS84 (Lat/Lon)
    "has_pseudo_normals": -1  # ✅ Use -1 for consistency (means "not assigned")
}

def get_crs(key):
    """Retrieve the CRS for a specific dataset key."""
    return crs_registry.get(key, -1)  # ✅ Default to -1 if key is missing

def set_crs(key, value):
    """Ensure only EPSG integer codes are stored, or handle pseudo-normal tracking."""
    if isinstance(value, str) and value.startswith("PROJCRS"):
        logger.warning(f"Attempted to store WKT instead of EPSG code for {key}.")
        return

    if key == "has_pseudo_normals":
        crs_registry[key] = value if isinstance(value, int) and value > 0 else -1  # ✅ Use EPSG if valid, otherwise -1
        logger.info(f"Tracking Pseudo-Normals: {'Enabled' if value > 0 else 'Disabled'}")
    else:
        logger.info(f"Setting CRS[{key}] → EPSG:{value}")
        crs_registry[key] = value

def has_pseudo_normals():
    """Returns True if pseudo-normals exist in the dataset."""
    return crs_registry.get("has_pseudo_normals", -1) > 0  # ✅ Now checks EPSG instead of None

def epsg_to_wkt(epsg_code):
    """Converts an EPSG code to WKT format without printing."""
    if epsg_code <= 0:
        return None  # Invalid EPSG codes return None silently
    try:
        crs = pyproj.CRS.from_epsg(epsg_code)
        return crs.to_wkt()
    except Exception:
        return None  # Error handling, but no print output

def print_crs_registry(verbose=False):
    """Prints the current CRS registry state."""
    print("📌 CRS Registry Status:")
    for key, value in crs_registry.items():
        if key == "has_pseudo_normals":
            status = f"Enabled (EPSG:{value})" if value > 0 else "Disabled"
        else:
            status = value
        print(f"  {key}: {status}")
