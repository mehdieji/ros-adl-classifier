import pandas as pd
from pathlib import Path

def parse_polar_sensor_data(csv_file: str) -> dict:
    """
    Parse Polar sensor raw data from a CSV file.
    Only handles: 'datetime', '_vector__x', '_vector__y', '_vector__z'.
    Returns a dict: {modality: DataFrame}, with standardized column names.
    The only modality is 'linear_acceleration'.
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # Only keep the specified columns
    keep_cols = [
        'datetime',
        '_vector__x', '_vector__y', '_vector__z'
    ]
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    # Standardize column names
    rename_map = {
        'datetime': 'datetime',
        '_vector__x': 'linear_acceleration_x',
        '_vector__y': 'linear_acceleration_y',
        '_vector__z': 'linear_acceleration_z',
    }
    df = df.rename(columns=rename_map)

    # Ensure datetime is parsed (ISO format)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H:%M:%S.%f")

    # Only one modality: linear_acceleration
    lin_cols = [f'linear_acceleration_{axis}' for axis in ['x', 'y', 'z'] if f'linear_acceleration_{axis}' in df.columns]
    modalities = {}
    if lin_cols:
        modalities['linear_acceleration'] = df[['datetime'] + lin_cols].copy()
    return modalities 