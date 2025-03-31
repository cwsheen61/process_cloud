import os
import shutil
import logging

logger = logging.getLogger(__name__)

def prepare_temp_directory(ply_path):
    """
    Creates a temporary working directory based on the PLY file path.
    Ensures a clean start by removing old temp files if they exist.

    Args:
        ply_path (str): Path to the PLY file being processed.

    Returns:
        str: Path to the newly created TEMP directory.
    """
    base_dir = ply_path
    temp_dir = os.path.join(base_dir, "TEMP")

    # 🔹 Ensure a clean start
    if os.path.exists(temp_dir):
        logger.info(f"🗑 Removing old TEMP directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)  # Remove old TEMP dir
        except Exception as e:
            logger.error(f"⚠️ Error removing {temp_dir}: {e}")
            return None  # Return None if unable to clean up

    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"📂 Temporary directory created: {temp_dir}")
    except Exception as e:
        logger.error(f"⚠️ Error creating TEMP directory: {e}")
        return None  # Return None if unable to create TEMP dir

    return temp_dir  # ✅ Return the new TEMP directory path
