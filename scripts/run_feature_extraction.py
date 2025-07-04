#!/usr/bin/env python3
"""
Windowing and Feature Extraction Script for ROS ADL Classifier

This script performs windowing and feature extraction on sensor data for multiple patients
and ADL events. It reads configuration from feature_config.yaml and processes each
instance of each ADL event for each patient, saving the extracted features as CSV files.

Based on the logic from 01_windowing_and_featuring.ipynb
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

# Add src to path for imports (now scripts is one level deeper)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils import get_project_root, get_data_path, load_config

# Import parsers
from data_preprocessing.m5_parser import parse_m5_sensor_data
from data_preprocessing.polar_parser import parse_polar_sensor_data
from data_preprocessing.sensomative_parser import parse_sensomative_sensor_data

# Import tsfresh
from tsfresh import extract_features


def extract_patient_id(folder_name):
    """Extract patient ID from folder name like '20250626_112001_patient01_0.synchronization'"""
    match = re.search(r'patient(\d+)', folder_name)
    return match.group(1) if match else None


def get_patient_folders(patient_id, raw_data_dir):
    """Get all folders for a specific patient ID"""
    if not raw_data_dir.exists():
        return []
    
    patient_folders = []
    for folder in raw_data_dir.iterdir():
        if folder.is_dir():
            folder_patient_id = extract_patient_id(folder.name)
            if folder_patient_id == patient_id:
                patient_folders.append(folder)
    
    return sorted(patient_folders)


def get_csv_dirs_for_adl_event(patient_folders, adl_event):
    """Get CSV directories for a specific ADL event"""
    # Filter for ADL event folders
    event_folders = [folder for folder in patient_folders if adl_event in folder.name]
    
    # Sort them (by name, which usually encodes the timestamp)
    event_folders = sorted(event_folders)
    
    # Construct the csv_dir for each ADL event
    csv_dirs = []
    for folder in event_folders:
        # Find the only subdirectory inside the event folder
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            csv_dir = subdirs[0]
            csv_dirs.append(str(csv_dir) + "/")  # Add trailing slash for consistency
        else:
            print(f"Warning: Expected 1 subdirectory in {folder}, found {len(subdirs)}")
    
    return csv_dirs


def parse_sensor_data(csv_dir):
    """Parse all sensor data from a CSV directory"""
    try:
        m5_1_csv_file = csv_dir + "M5StickC_01_data.csv"
        m5_wrist_l = parse_m5_sensor_data(m5_1_csv_file)

        m5_2_csv_file = csv_dir + "M5StickC_02_data.csv"
        m5_wrist_r = parse_m5_sensor_data(m5_2_csv_file)

        m5_3_csv_file = csv_dir + "M5StickC_03_data.csv"
        m5_wheel = parse_m5_sensor_data(m5_3_csv_file)

        polar_csv_file = csv_dir + "polar_acc.csv"
        polar_chest = parse_polar_sensor_data(polar_csv_file)

        sensomative_csv_file = csv_dir + "pressure1.csv"
        sensomative_bottom = parse_sensomative_sensor_data(sensomative_csv_file)
        
        return {
            'm5_wrist_l': m5_wrist_l,
            'm5_wrist_r': m5_wrist_r,
            'm5_wheel': m5_wheel,
            'polar_chest': polar_chest,
            'sensomative_bottom': sensomative_bottom
        }
    except Exception as e:
        print(f"Error parsing sensor data from {csv_dir}: {e}")
        return None


def gather_all_timeseries(sensor_data):
    """Gather all time series into a single DataFrame for tsfresh"""
    dfs = []
    
    # Define sensor data structure (same as in notebook)
    sensor_config = [
        ("M5 Wrist L", sensor_data['m5_wrist_l'], "linear_acceleration", 
         ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wrist L", sensor_data['m5_wrist_l'], "angular_velocity", 
         ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("M5 Wrist R", sensor_data['m5_wrist_r'], "linear_acceleration", 
         ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wrist R", sensor_data['m5_wrist_r'], "angular_velocity", 
         ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("M5 Wheel", sensor_data['m5_wheel'], "linear_acceleration", 
         ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("M5 Wheel", sensor_data['m5_wheel'], "angular_velocity", 
         ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]),
        ("Polar Chest", sensor_data['polar_chest'], "linear_acceleration", 
         ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]),
        ("Sensomative Bottom", sensor_data['sensomative_bottom'], "pressure", 
         [f"pressure_{i}" for i in range(12)]),
    ]
    
    for sensor_name, parsed, modality_key, cols in sensor_config:
        if modality_key in parsed:
            df = parsed[modality_key].copy()
            for col in cols:
                if col in df.columns:
                    dfs.append(
                        df[["datetime", col]].rename(columns={col: "value"}).assign(
                            sensor=sensor_name,
                            modality=modality_key,
                            channel=col
                        )
                    )
    
    if not dfs:
        return None
    
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def extract_features_from_data(all_df, tsfresh_features, extraction_params):
    """Extract features using tsfresh"""
    if all_df is None or all_df.empty:
        return None
    
    # Sort by datetime
    all_df = all_df.sort_values("datetime").reset_index(drop=True)
    
    # Windowing
    all_df["timestamp"] = (all_df["datetime"] - all_df["datetime"].min()).dt.total_seconds()
    all_df["window_id"] = (all_df["timestamp"] // extraction_params["step_size"]).astype(int)
    
    # Prepare for tsfresh
    all_df["id"] = (
        all_df["window_id"].astype(str) + "@" +
        all_df["sensor"] + "@" +
        all_df["modality"] + "@" +
        all_df["channel"]
    )
    
    # Feature extraction
    features = extract_features(
        all_df,
        column_id="id",
        column_sort="timestamp",
        column_value="value",
        default_fc_parameters=tsfresh_features,
        n_jobs=extraction_params["n_jobs"],
        disable_progressbar=extraction_params["disable_progressbar"]
    )
    
    # Reshape: Each row = one window, columns = sensor@modality@channel@feature
    feature_rows = []
    window_ids = sorted(all_df["window_id"].unique())
    
    for window_id in window_ids:
        # Find all ids for this window
        prefix = f"{window_id}@"
        row = {}
        for idx in features.index:
            if idx.startswith(prefix):
                parts = idx.split("@")
                sensor, modality, channel = parts[1], parts[2], parts[3]
                for feat in features.columns:
                    colname = f"{sensor}@{modality}@{channel}@{feat}"
                    row[colname] = features.loc[idx, feat]
        feature_rows.append(row)
    
    features_df = pd.DataFrame(feature_rows)
    features_df.insert(0, "window_id", window_ids)
    
    return features_df


def save_features(features_df, patient_id, adl_event, instance_id, output_config, project_root, nan_inf_handling=None):
    """Save features DataFrame to CSV file, with NaN/Inf handling"""
    if features_df is None or features_df.empty:
        print(f"No features to save for patient {patient_id}, {adl_event}, instance {instance_id}")
        return None
    
    # Add patient and ADL class columns
    features_df["patient"] = patient_id
    features_df["ADL_class"] = adl_event
    
    # NaN/Inf handling
    if nan_inf_handling is not None:
        features_df = features_df.replace({
            np.nan: nan_inf_handling.get('nan_value', -9000),
            np.inf: nan_inf_handling.get('posinf_value', 9000),
            -np.inf: nan_inf_handling.get('neginf_value', -9000)
        })
    
    # Create output directory
    features_dir = project_root / output_config["features_dir"]
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = output_config["filename_format"].format(
        patient_id=patient_id,
        adl_event=adl_event,
        instance_id=instance_id
    )
    
    filepath = features_dir / filename
    
    # Save to CSV
    features_df.to_csv(filepath, index=False)
    print(f"Saved features to: {filepath}")
    
    return filepath


def process_patient_adl_event(patient_id, adl_event, config, raw_data_dir, project_root):
    """Process all instances of a specific ADL event for a patient"""
    print(f"\nProcessing patient {patient_id}, ADL event: {adl_event}")
    
    # Get patient folders
    patient_folders = get_patient_folders(patient_id, raw_data_dir)
    if not patient_folders:
        print(f"No data found for patient {patient_id}")
        return []
    
    # Get CSV directories for this ADL event
    csv_dirs = get_csv_dirs_for_adl_event(patient_folders, adl_event)
    if not csv_dirs:
        print(f"No instances found for patient {patient_id}, ADL event: {adl_event}")
        return []
    
    print(f"Found {len(csv_dirs)} instances for {adl_event}")
    
    saved_files = []
    nan_inf_handling = config.get('nan_inf_handling', None)
    
    # Process each instance
    for instance_id, csv_dir in enumerate(csv_dirs):
        print(f"  Processing instance {instance_id + 1}/{len(csv_dirs)}: {csv_dir}")
        
        try:
            # Parse sensor data
            sensor_data = parse_sensor_data(csv_dir)
            if sensor_data is None:
                print(f"    Failed to parse sensor data")
                continue
            
            # Gather all time series
            all_df = gather_all_timeseries(sensor_data)
            if all_df is None:
                print(f"    No time series data found")
                continue
            
            # Extract features
            features_df = extract_features_from_data(
                all_df, 
                config["tsfresh_har_features"], 
                config["extraction_params"]
            )
            
            if features_df is None:
                print(f"    Failed to extract features")
                continue
            
            # Save features (with NaN/Inf handling)
            saved_file = save_features(
                features_df, 
                patient_id, 
                adl_event, 
                instance_id + 1,  # Use 1-based indexing for instance IDs
                config["output"], 
                project_root,
                nan_inf_handling=nan_inf_handling
            )
            
            if saved_file:
                saved_files.append(saved_file)
                print(f"    Extracted features shape: {features_df.shape}")
            
        except Exception as e:
            print(f"    Error processing instance {instance_id + 1}: {e}")
            continue
    
    return saved_files


def main():
    """Main function to run the feature extraction pipeline"""
    print("Starting Windowing and Feature Extraction Pipeline")
    print("=" * 60)
    
    # Load configuration
    config = load_config("feature_config.yaml")
    print(f"Loaded configuration from feature_config.yaml")
    
    # Setup paths
    project_root = get_project_root()
    raw_data_dir = get_data_path("raw")
    
    print(f"Project root: {project_root}")
    print(f"Raw data directory: {raw_data_dir}")
    
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory does not exist: {raw_data_dir}")
        return
    
    # Get patient IDs and ADL events from config
    patient_ids = config["patient_ids"]
    adl_events = config["adl_events"]
    
    print(f"Processing {len(patient_ids)} patients: {patient_ids}")
    print(f"Processing {len(adl_events)} ADL events: {adl_events}")
    
    # Process each patient and ADL event
    total_saved_files = []
    
    for patient_id in patient_ids:
        for adl_event in adl_events:
            saved_files = process_patient_adl_event(
                patient_id, 
                adl_event, 
                config, 
                raw_data_dir, 
                project_root
            )
            total_saved_files.extend(saved_files)
    
    print("\n" + "=" * 60)
    print(f"Feature extraction completed!")
    print(f"Total feature files saved: {len(total_saved_files)}")
    print(f"Features saved to: {project_root / config['output']['features_dir']}")


if __name__ == "__main__":
    main() 