import os
import shutil

def clean_temp_files(temp_dir):
    """Removes temporary files and directories after processing.
    
    Args:
        temp_dir (str): Path to the temporary directory to be removed.
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Temporary files cleaned up from {temp_dir}.")
    else:
        print("No temporary files found to clean up.")
