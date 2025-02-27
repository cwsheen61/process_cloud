import json
import os
import logging

logger = logging.getLogger(__name__)

class JSONRegistry:
    """ Manages JSON-based configuration for processing point clouds. """

    def __init__(self, ply_path, json_path):
        """
        Initializes the registry by loading the JSON config and setting paths.
        Args:
            ply_path (str): Full path to the PLY file.
            json_path (str): Path to the JSON configuration file.
        """
        self.ply_path = ply_path
        self.json_path = json_path
        self.config = self._load_config()

        # ✅ Extract `pathname` from PLY file & ensure it is set
        ply_dir, _ = os.path.split(ply_path)
        self.set("pathname", ply_dir)

        # ✅ Ensure file paths are correct (auto-fill based on PLY filename)
        self._auto_fill_paths()

    def _load_config(self):
        """ Loads the JSON config file into a dictionary. """
        try:
            with open(self.json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON syntax in: {self.json_path}")
            logger.error(str(e))
            raise
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {self.json_path}")
            raise

    def _auto_fill_paths(self):
        """ Automatically fills missing paths in the config based on PLY filename. """
        ply_dir = self.get("pathname")
        ply_name = os.path.splitext(os.path.basename(self.ply_path))[0]

        # ✅ Auto-fill file paths if missing
        default_files = {
            "ply": self.ply_path,
            "trajectory": f"{ply_name}_traj.txt",
            "gnss": f"{ply_name}_projection_wildcat_fit_metrics.txt",
            "output_pass": f"{ply_name}_filtered.laz",
            "output_fail": f"{ply_name}_fail.laz"
        }

        for key, default in default_files.items():
            if not self.get(f"files.{key}"):
                self.set(f"files.{key}", os.path.join(ply_dir, default))

    def get(self, key, default=None):
        """ Retrieves a value from the config using dot notation. """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key, value):
        """ Sets a value in the config using dot notation. """
        keys = key.split(".")
        target = self.config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def save(self):
        """ Saves the modified config back to the original JSON file. """
        with open(self.json_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def save_as(self, new_path):
        """ Saves the modified config to a new JSON file. """
        with open(new_path, "w") as f:
            json.dump(self.config, f, indent=2)
