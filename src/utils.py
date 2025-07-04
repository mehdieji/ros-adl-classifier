"""
Utility functions for the ROS ADL Classifier project.
"""

import sys
import os
from pathlib import Path


def setup_project_paths():
    """
    Add the src directory to Python path for imports.
    Call this at the beginning of any script that needs to import project modules.
    """
    # Get the project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from src/ to project root
    src_path = project_root / "src"
    
    # Add to Python path if not already there
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root, src_path


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def get_data_path(subdir=None):
    """
    Get path to data directory or subdirectory.
    
    Args:
        subdir (str, optional): Subdirectory name (e.g., 'raw', 'processed')
    
    Returns:
        Path: Path to data directory or subdirectory
    """
    project_root = get_project_root()
    data_path = project_root / "data"
    
    if subdir:
        data_path = data_path / subdir
    
    return data_path


def get_config_path(config_name="config.yaml"):
    """
    Get path to a configuration file.
    
    Args:
        config_name (str): Name of the config file (default: "config.yaml")
    
    Returns:
        Path: Path to the specified config file
    """
    project_root = get_project_root()
    return project_root / "config" / config_name


def load_config(config_name="config.yaml"):
    """
    Load a configuration file.
    
    Args:
        config_name (str): Name of the config file to load (default: "config.yaml")
    
    Returns:
        dict: Configuration dictionary
    """
    import yaml
    
    config_path = get_config_path(config_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

