import pandas as pd
from pathlib import Path

def parse_m5_sensor_data(csv_file: str) -> dict:
    """
    Parse M5 sensor raw data from a CSV file.
    Only handles: 'datetime', '_angular_velocity__x', '_angular_velocity__y', '_angular_velocity__z',
    '_linear_acceleration__x', '_linear_acceleration__y', '_linear_acceleration__z'.
    Returns a dict: {modality: DataFrame}, each with standardized column names.
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # Only keep the specified columns
    keep_cols = [
        'datetime',
        '_angular_velocity__x', '_angular_velocity__y', '_angular_velocity__z',
        '_linear_acceleration__x', '_linear_acceleration__y', '_linear_acceleration__z'
    ]
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    # Standardize column names
    rename_map = {
        'datetime': 'datetime',
        '_angular_velocity__x': 'angular_velocity_x',
        '_angular_velocity__y': 'angular_velocity_y',
        '_angular_velocity__z': 'angular_velocity_z',
        '_linear_acceleration__x': 'linear_acceleration_x',
        '_linear_acceleration__y': 'linear_acceleration_y',
        '_linear_acceleration__z': 'linear_acceleration_z',
    }
    df = df.rename(columns=rename_map)

    # Ensure datetime is parsed (ISO format)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H:%M:%S.%f")

    # Split into modalities
    modalities = {}
    ang_cols = [f'angular_velocity_{axis}' for axis in ['x', 'y', 'z'] if f'angular_velocity_{axis}' in df.columns]
    if ang_cols:
        modalities['angular_velocity'] = df[['datetime'] + ang_cols].copy()
    lin_cols = [f'linear_acceleration_{axis}' for axis in ['x', 'y', 'z'] if f'linear_acceleration_{axis}' in df.columns]
    if lin_cols:
        modalities['linear_acceleration'] = df[['datetime'] + lin_cols].copy()
    return modalities